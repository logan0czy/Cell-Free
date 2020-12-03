"""
Reinforcement learning network training realization.

This module is inspired by OpenAI.
Links: https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/td3/td3.py
"""
import time, math, itertools, gc
from copy import deepcopy
import os
from os import path as osp
# from IPython.display import clear_output
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn

from net import ActorCritic, sync
from comms import Environment, CodeBook
from utils import colorize, Decoder, OUStrategy, ReplayBuffer
from logx import EpochLogger


def timeCount(cur_time, start_time):
    """
    Calculate elapsed time.

    Returns:
        format_t (string): Elapsed time in format "D/H/M"

        abs_t (float): Elapsed time in seconds
    """
    abs_t = cur_time - start_time

    temp = abs_t
    d_unit, h_unit, m_unit = 24*3600, 3600, 60
    days = int(temp//d_unit)
    temp -= days*d_unit
    hours = int(temp//h_unit)
    temp -= hours*h_unit
    mins = int(temp//m_unit) 
    format_t = ":".join([str(days), str(hours), str(mins)])
    return format_t, abs_t

def actTranser(act, env, bs_azi_decoder, ris_ele_decoder, ris_azi_decoder):
    """
    Transform action from policy network to beamforming vector.

    Act Defaults:
        1d vector
        First 'bs_num*user_num' elements corresponds to relative  
            power for each base station to user
        The next 'bs_num*user_num' elements corresponds to 
            beam for each base station to user
        The next 'ris_num' elements corresponds to elevation 
            beam for each ris
        The last 'ris_num' elements corresponds to azimuth beam 
            for each ris

    Returns:
        bs_beam : Array with shape (bs_num, user_num, bs_atn)

        ris_beam : Array with shape (ris_num, ris_atn, ris_atn), 
            the last two dims store diagnal matrix.
    """
    bs_num, ris_num, user_num = env.getCount()

    shift = 0
    power = (act[shift:shift+bs_num*user_num].reshape(bs_num, user_num, 1) + 1) / 2
    power = np.amax(power, axis=1, keepdims=True) * env.max_power \
        * (power/(np.sum(power, axis=(1, 2), keepdims=True)+1e-8))
    shift += bs_num*user_num
    bs_beam = bs_azi_decoder.decode(act[shift:shift+bs_num*user_num]).reshape(bs_num, user_num, -1)
    shift += bs_num*user_num
    ris_ele_beam = ris_ele_decoder.decode(act[shift:shift+ris_num])
    shift += ris_num
    ris_azi_beam = ris_azi_decoder.decode(act[shift:shift+ris_num])

    bs_beam = np.sqrt(power) * bs_beam

    ris_beam = np.matmul(ris_ele_beam[:, :, np.newaxis], ris_azi_beam[:, np.newaxis]).reshape(ris_num, -1)
    ris_beam_expand = np.zeros(ris_beam.shape+ris_beam.shape[-1:], dtype=ris_beam.dtype)
    diagonals = ris_beam_expand.diagonal(axis1=-2, axis2=-1)
    diagonals.setflags(write=True)
    diagonals[:] = ris_beam.copy()

    return bs_beam, ris_beam_expand

def train(
        env_kwargs, net_kwargs, cbook_kwargs, act_noise=1e-3, tgt_noise=2e-3, noise_clip=4e-3, 
        epochs=100, steps_per_epoch=10000, start_steps=10000, update_after=1000, update_every=100,
        policy_decay=2, lr_q=3e-4, lr_policy=1e-4, lr_decay=1., q_weight_decay=1e-4, policy_weight_decay=0, 
        sync_rate=0.005, gamma=0.6, batch_size=128, buffer_size=100000, seed=24, 
        gpu='cuda:3', logger_kwargs=dict(), 
):
    """Twin Delayed Deep Deterministic Policy training process.

    Args:
        env_kwargs (dict): Environment's arguments, keys:
            'max_power', 'bs_atn', 'ris_atn'

        net_kwargs (dict): Network's arguments, keys:
            'critic_hidden_sizes', 'actor_hidden_sizes', 'act_limit'

        cbook_kwargs (dict): Codebook's arguments. keys: 
            'bs_codes', 'ris_azi_codes', 'ris_ele_codes', 'bs_phases', 
            'ris_azi_phases', 'ris_ele_phases'

        act_noise (float): Stddev of Gaussian noise. The noise is added 
            to action from policy network

        tgt_noise (float): Stddev of Gaussian smoothing noise which is added
            to action from target policy network.

        noise_clip (float): The absolute value limitaion of smoothing noise.

        epochs (int): Number of epochs to run and train agent

        steps_per_epoch (int): Number of steps of interaction (state-action pairs)
            between agent and environment

        start_steps (int): Number of steps to choose action uniform-randomly 
            from action space before using the policy. (Helps better exploration)

        update_after (int): Number of env interactions before using gradient
            descent update to update actor-critic network, this makes sure 
            ReplayBuffer have enough samples

        update_every (int): Number of env interactions that should elapse between 
            gradient descent updates. 
            Note: No matter how many the interval steps are, the times of env 
            interactions and gradient descent updates should be the same

        policy_decay (int): Number of steps of updating Q network before updating 
            policy net

        lr_q (float): Learning rate of Q network

        lr_policy (float): Learning rate of policy network

        lr_decay (float): Decrease lr after each epoch

        q_weight_decay (float): Regularization factor to Q network

        policy_weight_decay (float): Regularization factor to policy network

        sync_rate (float): The synchronize ratio between target network parameters 
            and main networks. Equation is:
            tgt_param := sync_rate*src_param + (1-sync_rate)*tgt_param 
        
        gamma (float): Reward decay

        batch_size (int): Number of samples to learn from at each gradient descent update

        buffer_size (int): ReplayBuffer storage amount

        seed (int): Random seed

        gpu (string): GPU which models are put on

        logger_kwargs (dict): Kewword arguments for EpochLogger.
    """
    logger = EpochLogger(**logger_kwargs)
    logger.saveConfig(locals(), stdout=False)

    np.random.seed(seed)
    torch.manual_seed(seed)

    # environment create
    env = Environment(**env_kwargs)
    # logger.log("env created...", 'blue')
    print("env created...")

    # beamforming codebook
    bs_cbook = CodeBook(cbook_kwargs['bs_codes'], env.bs_atn, cbook_kwargs['bs_phases'])
    ris_ele_cbook = CodeBook(cbook_kwargs['ris_ele_codes'], env.ris_atn[1], 
        cbook_kwargs['ris_ele_phases'], scale=True)
    ris_azi_cbook = CodeBook(cbook_kwargs['ris_azi_codes'], env.ris_atn[0], 
        cbook_kwargs['ris_azi_phases'], scale=True)
    # logger.log("codebook created...", 'blue')
    print("code book created...")

    # action decoders
    decoders = dict(bs_azi=Decoder((-net_kwargs['act_limit'], net_kwargs['act_limit']),
                        bs_cbook.book),
                    ris_ele=Decoder((-net_kwargs['act_limit'], net_kwargs['act_limit']),
                        ris_ele_cbook.book),
                    ris_azi=Decoder((-net_kwargs['act_limit'], net_kwargs['act_limit']),
                        ris_azi_cbook.book)
    )
    # logger.log("decoder created...", 'blue')
    print("decoder created...")

    # network
    device = torch.device(gpu if torch.cuda.is_available() else 'cpu')
    main_model = ActorCritic(env.obs_dim, env.act_dim, **net_kwargs)
    main_model.to(device)
    tgt_model = deepcopy(main_model)

    tgt_model.train(False)
    for param in tgt_model.parameters():
        param.requires_grad = False
    # logger.log(f"models created... the devices are: main model:{main_model.device}, target model:{tgt_model.device}", 'magenta')
    print(f"models created... the devices are: main model:{main_model.device}, target model:{tgt_model.device}")

    logger.setup_pytorch_saver(main_model)

    # exploration noise
    act_ous = OUStrategy(act_space={'dim': env.act_dim, 'low': -net_kwargs['act_limit'], 'high': net_kwargs['act_limit']},
        max_sigma=np.array([act_noise]*env.act_dim))
    tgt_ous = OUStrategy(act_space={'dim': env.act_dim, 'low': -net_kwargs['act_limit'], 'high': net_kwargs['act_limit']},
        max_sigma=np.array([tgt_noise]*env.act_dim), 
        noise_clip=np.array([noise_clip]*env.act_dim))
    # logger.log("noises created...", 'blue')
    print("noises created...")

    # list of parameters for both Q networks
    q_params = itertools.chain(main_model.q1.parameters(), main_model.q2.parameters())

    # optimizer and scheduler
    q_opt = torch.optim.Adam(q_params, lr_q, weight_decay=q_weight_decay)
    policy_opt = torch.optim.Adam(main_model.actor.parameters(), lr_policy, weight_decay=policy_weight_decay)
    q_scheduler = torch.optim.lr_scheduler.ExponentialLR(q_opt, lr_decay)
    policy_scheduler = torch.optim.lr_scheduler.ExponentialLR(policy_opt, lr_decay)

    # experience buffer
    replay_buffer = ReplayBuffer(env.obs_dim, env.act_dim, buffer_size)
    # logger.log("replay buffer created...", 'blue')
    print("replay buffer created...")

    def getQLoss(data):
        """
        Compute the Q-value loss according to Bellman equation
        w.r.t batches of transition samples.

        Args:
            data (dict): Include batches of state transitions and the reward,
                keys are {'obs', 'next_obs', 'act', 'rew'}
        """
        # Bellman backup for Q function
        with torch.no_grad():
            next_act = tgt_model.actor(data['next_obs']).cpu().numpy()
            next_act = torch.as_tensor(tgt_ous.getActFromRaw(next_act), 
                dtype=torch.float32, device=main_model.device)
            q1_tgt = tgt_model.q1(data['next_obs'], next_act)
            q2_tgt = tgt_model.q2(data['next_obs'], next_act)
            q_tgt = torch.min(q1_tgt, q2_tgt)
            backup = data['rew'].unsqueeze(-1) + gamma*q_tgt

        # MSE Q loss
        q1 = main_model.q1(data['obs'], data['act'])
        q2 = main_model.q2(data['obs'], data['act'])
        loss_q = ((q1-backup)**2).mean() + ((q2-backup)**2).mean()

        q_info = dict(Q1vals=q1.detach().cpu().numpy(),
                      Q2vals=q2.detach().cpu().numpy())
        return loss_q, q_info

    def getPolicyLoss(data):
        """
        Compute the mean estimate Q-value using current policy.

        Args:
            data (dict): Include batches of state transitions and the reward,
                keys are {'obs', 'next_obs', 'act', 'rew'}
        """
        main_model.q1.train(False)
        main_model.q2.train(False)
        for param in q_params:
            param.requires_grad = False

        act = main_model.actor(data['obs'])
        loss_policy = -main_model.q1(data['obs'], act).mean()

        main_model.q1.train(True)
        main_model.q2.train(True)
        for param in q_params:
            param.requires_grad = True

        return loss_policy

    def update(data, timer):
        """
        Update actor-critic network.

        Args:
            data (dict): Include batches of state transitions and the reward,
                keys are {'obs', 'next_obs', 'act', 'rew'}
        """
        data = {k: torch.as_tensor(v, dtype=torch.float32, device=main_model.device) 
                 for k, v in data.items()}

        q_opt.zero_grad()
        q_loss, q_info = getQLoss(data)
        q_loss.backward()
        nn.utils.clip_grad_norm_(q_params, 10)
        q_opt.step()

        logger.store(LossQ=q_loss.item(), **q_info)

        if timer%policy_decay == 0:
            policy_opt.zero_grad()
            policy_loss = getPolicyLoss(data)
            policy_loss.backward()
            nn.utils.clip_grad_norm_(main_model.actor.parameters(), 10)
            policy_opt.step()

            logger.store(LossPi=policy_loss.item())

            sync(main_model, tgt_model, sync_rate)

    def getAct(obs):
        action = main_model.act(torch.as_tensor([obs], dtype=torch.float32, device=main_model.device))
        action = act_ous.getActFromRaw(action.squeeze(0).cpu().numpy())
        return action

    # training process
    total_steps = epochs * steps_per_epoch
    obs = env.reset(seed)
    start_time, ep_time = time.time(), time.time()
    records = dict(Rew=[], LossQ=[])
    largest_EpRew = float('-inf')
    for step in range(total_steps):
        if step < start_steps:
            act = net_kwargs['act_limit'] * np.random.uniform(-1, 1, env.act_dim).astype(np.float32)
        else:
            act = getAct(obs)
        bs_beam, ris_beam = actTranser(act, env, 
            decoders['bs_azi'], decoders['ris_ele'], decoders['ris_azi'])
        next_obs, rew = env.step(bs_beam, ris_beam)

        replay_buffer.store(obs, act, rew, next_obs)
        obs = next_obs
        logger.store(EpRew=rew)

        if (step+1) > update_after and (step+1-update_after)%update_every==0:
            for j in range(update_every):
                batch = replay_buffer.sampleBatch(batch_size)
                update(batch, j)

            # Print infos
            # if (step+1) % 2000 == 0:
            #     avg_rew = logger.getStats('EpRew')[0]
            #     avg_lossQ = logger.getStats('LossQ')[0]
            #     print(f"step: {step}, avg. rew: {avg_rew:.2f}, avg. lossQ: {avg_lossQ:.2f}")

        if (step+1) % steps_per_epoch == 0:
            # reset some process
            # obs = env.reset(seed)
            act_ous.reset()
            tgt_ous.reset()
            q_scheduler.step()
            policy_scheduler.step()

            epoch = (step+1) // steps_per_epoch
            speed = timeCount(time.time(), ep_time)[1]/steps_per_epoch
            time_pass = timeCount(time.time(), start_time)[0]
            ep_time = time.time()
            # logger.log(f"speed--{speed:.2f}s/step")
            # print(f"speed--{speed:.2f} s/step")

            # Save the best or epoch model
            # if epoch % 50 == 0:
            #     logger.saveState(main_model.state_dict(), epoch)
            if logger.getStats('EpRew')[0] > largest_EpRew:
                logger.saveState(main_model.state_dict(), None)
                largest_EpRew = logger.getStats('EpRew')[0]

            # Record things
            if epoch <= 50:
                records['Rew'].extend(logger.epoch_dict['EpRew'])
                records['LossQ'].extend(logger.epoch_dict['LossQ'])

            # Log diagnostic
            logger.logTabular('Epoch', epoch)
            logger.logTabular('EpRew', with_min_max=True)
            logger.logTabular('Q1vals', with_min_max=True)
            logger.logTabular('Q2vals', with_min_max=True)
            logger.logTabular('LossPi', average_only=True)
            logger.logTabular('LossQ', average_only=True)
            logger.logTabular('Time', time_pass)
            logger.dumpTabular(stdout=False)

            gc.collect()

    for k, v in records.items():
        np.save(osp.join(logger.output_dir, k), np.array(v))
    print(f'best EpRew: {largest_EpRew:.2f}')

    return largest_EpRew

def test(env, model, decoders, 
        bs2user_CSI, bs2ris_CSI, ris2user_CSI
):
    """
    Args:
        bs2user_CSI : Array of shape (N, bs_num, user_num, bs_atn)

        bs2ris_CSI : Array of shape (N, bs_num, ris_num, ris_atn, bs_atn)

        ris2user_CSI : Array of shape (N, ris_num, user_num, ris_atn)
    """
    sum_rate = 0
    count = 0
    for bs2user_csi, bs2ris_csi, ris2user_csi in zip(bs2user_CSI, bs2ris_CSI, ris2user_CSI):
        obs = env.setCSI(bs2user_csi, bs2ris_csi, ris2user_csi)
        obs = torch.as_tensor([obs], dtype=torch.float32, device=model.device)
        act = model.act(obs).squeeze(0).cpu().numpy()
        bs_beam, ris_beam = actTranser(act, env, 
            decoders['bs_azi'], decoders['ris_ele'], decoders['ris_azi'])
        sum_rate += env._getRate(bs_beam, ris_beam)
        count += 1
    return sum_rate / count

if __name__=='__main__':
    main_fold = osp.join('Experiment', 'optimal')
    os.makedirs(main_fold, exist_ok=True)

    # seeds = list(range(0, 100, 10))
    seeds = [80, 90]
    powers = list(range(0, 35, 5))
    for seed in seeds:
        fold = osp.join(main_fold, 'seed%d'%(seed//10))
        os.makedirs(fold, exist_ok=True)

        results = []
        for power in powers:
            output_dir = osp.join(fold, 'power%d'%power)
            kwargs = dict(
                seed=seed,
                env_kwargs = {'max_power': power, 'bs_atn': 4, 'ris_atn': (8, 4)},
                net_kwargs = {
                    'critic_hidden_sizes': [512, 128, 32, 16], 
                    'actor_hidden_sizes': [512, 256, 128, 64],
                    'act_limit': 1},
                cbook_kwargs = {
                    'bs_codes': 50, 
                    'ris_ele_codes': 50, 
                    'ris_azi_codes': 50,
                    'bs_phases': 16,
                    'ris_azi_phases': 4,
                    'ris_ele_phases': 4},
                logger_kwargs = {'output_dir': output_dir},
                lr_decay=0.96, epochs=50, gpu='cuda:2', 
            )
            results.append(train(**kwargs))
        np.save(osp.join(fold, 'opt_res'), np.array(results))