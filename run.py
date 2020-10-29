"""
Training process module

Reference: https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/td3/td3.py
"""


import time
import math
import itertools
import gc
from copy import deepcopy
# from IPython.display import clear_output

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn

from net import ActorCritic, sync
from env import Environment
import utils


def timeCount(cur_time, start_time):
    """calculate elapsed time.

    Returns:
        format_t (string): elapsed time in format "D/H/M"
        abs_t (float): elapsed time in seconds
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

def actTranser(act, env, power_decoder, bs_azi_decoder, 
        ris_ele_decoder, ris_azi_decoder
):
    """transform action from policy network to beamforming vector
    Act Default:
        1-D vector
        first 'bs_num*user_num' elements corresponds to power for each base station to user
        then 'bs_num*user_num' elements corresponds to beam for each base station to user
        then 'ris_num' elements corresponds to elevation beam for each ris
        last 'ris_num' elements corresponds to azimuth beam for each ris

    Returns:
        bs_beam (np.array): shape is (bs_num, user_num, bs_atn)
        ris_beam (np.array): shape is (ris_num, ris_atn, ris_atn), the last two
            dims store diagnal matrix.
    """
    bs_num, ris_num, user_num = env.getCount()

    shift = 0
    power = power_decoder.decode(act[:bs_num*user_num]).reshape(bs_num, user_num, 1)
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
        env_kwargs, net_kwargs, cbook_kwargs, act_noise, tgt_noise, noise_clip, 
        epochs=100, steps_per_epoch=10000, start_steps=10000, update_after=1000, update_every=50,
        policy_decay=2, lr_q=1e-3, lr_policy=1e-3, lr_decay=1., q_weight_decay=1e-4, policy_weight_decay=1e-4, 
        sync_rate=0.005, n_powers=10, gamma=0.6, batch_size=64, buffer_size=100000, seed=24, sat_ratio=0,
):
    """Twin Delayed Deep Deterministic Policy training process.

    Parameters:
        env_kwargs (dict): environment's arguments, keys:

            'max_power', 'bs_atn', 'ris_atn'

        net_kwargs (dict): network's arguments, keys:

            'critic_hidden_sizes', 'actor_hidden_sizes', 'act_limit'

        cbook_kwargs (dict): codebook's arguments. keys: 

            'bs_codes', 'ris_ele_codes', 'ris_azi_codes', 'bs_phases', 'ris_azi_phases', 'ris_ele_phases'

        act_noise (float): ratio of stddev of Gaussian noise to the interval corresponding to
            the same action. The noise is added to action from policy network

        tgt_noise (float): ratio of stddev of Gaussian smoothing noise to the interval
            corresponding to the same action.

        noise_clip (float): ratio for the absolute value limitaion of smoothing noise to
            the interval corresponding to the same action.

        epochs (int): number of epochs to run and train agent

        steps_per_epoch (int): number of steps of interaction (state-action pairs)
            between agent and environment

        start_steps (int): number of steps to choose action uniform-randomly from
            action space before using the policy. (helps better exploration)

        update_after (int): number of env interactions before using gradient
            descent update to update actor-critic network, this makes sure ReplayBuffer
            have enough samples

        update_every (int): number of env interactions that should elapse between gradient
            descent updates. Note: no matter how many the interval steps are, the times of
            env interactions and gradient descent updates should be the same

        policy_decay (int): number of steps of updating Q network before updating policy net

        lr_q (float): learning rate of Q network

        lr_policy (float): learning rate of policy network

        lr_decay (float): decrease lr after each epoch

        q_weight_decay (float): regularization factor to Q network

        policy_weight_decay (float): regularization factor to policy network

        sync_rate (float): the synchronize ratio between target network parameters and main
            networks. Equation is:

            tgt_param := sync_rate*src_param + (1-sync_rate)*tgt_param 
        
        n_powers (int): number of power levels to choose.

        gamma (float): reward decay

        batch_size (int): number of samples to learn from at each gradient descent update

        buffer_size (int): ReplayBuffer storage amount

        seed (int): random seed

        sat_ratio (float): ratio of act val saturation range to its total range
    """
    def getQLoss(data):
        """compute the Q-value loss according to Bellman equation
        w.r.t batches of transition samples.

        Parameters:
            data (dict): include batches of state transitions and the reward,
                keys are {'obs', 'next_obs', 'act', 'rew'}

        Returns:
            loss_q (torch.tensor): average Q-value loss
            q1 (float)
            q2 (float)
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

        # q info logger

        return loss_q, q1.mean().item(), q2.mean().item()

    def getPolicyLoss(data):
        """compute the mean estimate Q-value using current policy.

        Parameters:
            data (dict): include batches of state transitions and the reward,
                keys are {'obs', 'next_obs', 'act', 'rew'}

        Returns:
            loss_policy (torch.Tensor): average policy loss
            avg_exceed (float): average exceeded power
        """
        main_model.q1.train(False)
        main_model.q2.train(False)
        for param in q_params:
            param.requires_grad = False

        act = main_model.actor(data['obs'])
        # punishment for invalid powers 
        act_pw = act[:, :env.getCount(0)*env.getCount(2)].view(batch_size, env.getCount(0), env.getCount(2))
        act_pw_decoded = decoders['power'].decode(act_pw.detach().cpu().numpy())
        invalid_idxs = np.argwhere(np.sum(act_pw_decoded, axis=2) > env.max_power)
        punish = torch.sum(act_pw[invalid_idxs[:, 0], invalid_idxs[:, 1]], dim=1).mean()
        if len(invalid_idxs)>0:
            avg_exceed = np.sum(act_pw_decoded[invalid_idxs[:, 0], invalid_idxs[:, 1]], axis=1).mean() - env.max_power 
        else:
            avg_exceed = 0

        loss_policy = -main_model.q1(data['obs'], act).mean() + punish

        main_model.q1.train(True)
        main_model.q2.train(True)
        for param in q_params:
            param.requires_grad = True

        return loss_policy, avg_exceed

    def update(data, timer):
        """update actor-critic network.

        Parameters:
            data (dict): include batches of state transitions and the reward,
                keys are {'obs', 'next_obs', 'act', 'rew'}
        """
        data = {k: torch.as_tensor(v, dtype=torch.float32, device=main_model.device) 
                 for k, v in data.items()}
        infos = {}

        q_opt.zero_grad()
        q_loss, q1, q2 = getQLoss(data)
        q_loss.backward()
        nn.utils.clip_grad_norm_(q_params, 10)
        q_opt.step()

        infos['q_loss'] = q_loss.item()
        infos['q1'] = q1
        infos['q2'] = q2

        if timer%policy_decay == 0:
            policy_opt.zero_grad()
            policy_loss, exceed_pw = getPolicyLoss(data)
            policy_loss.backward()
            nn.utils.clip_grad_norm_(main_model.actor.parameters(), 10)
            policy_opt.step()

            sync(main_model, tgt_model, sync_rate)

            infos['policy_loss'] = policy_loss.item()
            infos['exceed_pw'] = exceed_pw
            
            return infos

        return infos

    def getAct(obs):
        action = main_model.act(torch.as_tensor([obs], dtype=torch.float32, device=main_model.device))
        action = act_ous.getActFromRaw(action.squeeze(0).cpu().numpy())
        return action

    np.random.seed(seed)
    torch.manual_seed(seed)

    # environment create
    env = Environment(**env_kwargs)
    print("env created...")

    # beamforming codebook
    bs_cbook = utils.CodeBook(cbook_kwargs['bs_codes'], env.bs_atn, cbook_kwargs['bs_phases'])
    ris_ele_cbook = utils.CodeBook(cbook_kwargs['ris_ele_codes'], env.ris_atn[1], 
        cbook_kwargs['ris_ele_phases'], scale=True)
    ris_azi_cbook = utils.CodeBook(cbook_kwargs['ris_azi_codes'], env.ris_atn[0], 
        cbook_kwargs['ris_azi_phases'], scale=True)
    print("codebook created...")

    # action decoders
    decoders = dict(power=utils.Decoder((-net_kwargs['act_limit'], net_kwargs['act_limit']), 
                        np.array([i*env.max_power/n_powers for i in range(1, n_powers+1)]), sat_ratio=sat_ratio),
                    bs_azi=utils.Decoder((-net_kwargs['act_limit'], net_kwargs['act_limit']),
                        bs_cbook.book, sat_ratio=sat_ratio),
                    ris_ele=utils.Decoder((-net_kwargs['act_limit'], net_kwargs['act_limit']),
                        ris_ele_cbook.book, sat_ratio=sat_ratio),
                    ris_azi=utils.Decoder((-net_kwargs['act_limit'], net_kwargs['act_limit']),
                        ris_azi_cbook.book, sat_ratio=sat_ratio)
    )
    print("decoder created...")

    # network
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    main_model = ActorCritic(env.obs_dim, env.act_dim, **net_kwargs)
    main_model.to(device)
    tgt_model = deepcopy(main_model)

    tgt_model.train(False)
    for param in tgt_model.parameters():
        param.requires_grad = False
    print(f"models created... the devices are: main model:{main_model.device}, target model:{tgt_model.device}")

    # exploration noise
    spacings = np.array([decoders['power'].spacing]*env.getCount(0)*env.getCount(2)
        +[decoders['bs_azi'].spacing]*env.getCount(0)*env.getCount(2)
        +[decoders['ris_ele'].spacing]*env.getCount(1)
        +[decoders['ris_azi'].spacing]*env.getCount(1)).astype(np.float32)
    act_ous = utils.OUStrategy(act_space={'dim': env.act_dim, 'low': -net_kwargs['act_limit'], 'high': net_kwargs['act_limit']},
        max_sigma=act_noise*spacings)
    tgt_ous = utils.OUStrategy(act_space={'dim': env.act_dim, 'low': -net_kwargs['act_limit'], 'high': net_kwargs['act_limit']},
        max_sigma=tgt_noise*spacings, noise_clip=noise_clip*spacings)
    print("noises created...")

    # list of parameters for both Q networks
    q_params = itertools.chain(main_model.q1.parameters(), main_model.q2.parameters())

    # optimizer
    policy_opt = torch.optim.Adam(main_model.actor.parameters(), lr_policy, weight_decay=policy_weight_decay)
    q_opt = torch.optim.Adam(q_params, lr_q, weight_decay=q_weight_decay)
    policy_sheduler = torch.optim.lr_scheduler.ExponentialLR(policy_opt, lr_decay)
    q_sheduler = torch.optim.lr_scheduler.ExponentialLR(q_opt, lr_decay)

    # experience buffer
    replay_buffer = utils.ReplayBuffer(env.obs_dim, env.act_dim, buffer_size)
    print("replay buffer created...")

    # training process
    total_steps = epochs * steps_per_epoch
    obs = env.reset(seed)
    ep_rew = 0
    start_time, ep_time = time.time(), time.time()
    for step in range(total_steps):
        if step < start_steps:
            act = net_kwargs['act_limit'] * np.random.uniform(-1, 1, env.act_dim).astype(np.float32)
        else:
            act = getAct(obs)
        bs_beam, ris_beam = actTranser(act, env, decoders['power'], 
            decoders['bs_azi'], decoders['ris_ele'], decoders['ris_azi'])
        next_obs, rew = env.step(bs_beam, ris_beam)
        ep_rew += rew

        replay_buffer.store(obs, act, rew, next_obs)

        obs = next_obs

        if (step+1) > update_after and (step+1-update_after)%update_every==0:
            loss_infos = dict(q_loss=[], q1=[], q2=[], policy_loss=[], exceed_pw=[])
            for j in range(update_every):
                batch = replay_buffer.sampleBatch(batch_size)
                loss = update(batch, j)
                for k, v in loss.items():
                    loss_infos[k].append(v)

            for k, v in loss_infos.items():
                loss_infos[k] = np.mean(v)

            if (step+1)%2000==0:
                print(f"steps: {step}, loss_q: {loss_infos['q_loss']:.4f},", end=' ')
                print(f"q1: {loss_infos['q1']:.2f}, q2: {loss_infos['q2']:.2f}", end=' ')
                print(f"loss_policy: {loss_infos['policy_loss']:.2f},", end=' ')
                print(f"punish: {loss_infos['exceed_pw']:.2f},", end=' ')
                print(f"time elapse: {timeCount(time.time(), start_time)[0]}")

        if (step+1) % steps_per_epoch == 0:
            obs = env.reset(seed)
            act_ous.reset()
            tgt_ous.reset()
            if (step+1) // steps_per_epoch <= 200 and lr_decay < 1:
                q_sheduler.step()
                policy_sheduler.step()

            print(f"\nepoch: {step//steps_per_epoch}, avg_rew: {ep_rew/steps_per_epoch:.4f},", end=' ')
            print(f"speed: {timeCount(time.time(), ep_time)[1]/steps_per_epoch:.2f}\n")
            ep_time = time.time()
            ep_rew = 0
            torch.save(main_model.state_dict(), "./checkpoint/model_param.pth")
            # torch.cuda.empty_cache()
            gc.collect()

def test(env, model, decoders, 
        bs2user_CSI, bs2ris_CSI, ris2user_CSI
):
    """
    Parameters:
        bs2user_CSI (np.array): shape is (N, bs_num, user_num, bs_atn)
        bs2ris_CSI (np.array): shape is (N, bs_num, ris_num, ris_atn, bs_atn)
        ris2user_CSI (np.array): shape is (N, ris_num, user_num, ris_atn)

    Return:
        avg_sum_rate (float)
    """
    sum_rate = 0
    count = 0
    for bs2user_csi, bs2ris_csi, ris2user_csi in zip(bs2user_CSI, bs2ris_CSI, ris2user_CSI):
        obs = env.setCSI(bs2user_csi, bs2ris_csi, ris2user_csi)
        obs = torch.as_tensor([obs], dtype=torch.float32, device=model.device)
        act = model.act(obs).squeeze(0).cpu().numpy()
        bs_beam, ris_beam = actTranser(act, env, decoders['power'], 
            decoders['bs_azi'], decoders['ris_ele'], decoders['ris_azi'])
        sum_rate += env._getRate(bs_beam, ris_beam)
        count += 1
    return sum_rate / count

if __name__=='__main__':
    env_kwargs = {'max_power': 30, 'bs_atn': 4, 'ris_atn': (8, 4)}
    net_kwargs = {'critic_hidden_sizes': [512, 64], 
                  'actor_hidden_sizes': [1024, 512, 256, 128],
                  'act_limit': 1}
    cbook_kwargs = {'bs_codes': 500, 
                    'ris_ele_codes': 500, 
                    'ris_azi_codes': 500,
                    'bs_phases': 16,
                    'ris_azi_phases': 4,
                    'ris_ele_phases': 4}
    train(env_kwargs, net_kwargs, cbook_kwargs, 
        act_noise=1, tgt_noise=2, noise_clip=4, 
        q_weight_decay=1e-4, policy_weight_decay=0, lr_q=3e-4, lr_policy=1e-4, lr_decay=1,
        epochs=400, batch_size=128, update_every=100, n_powers=500, sat_ratio=0.05)