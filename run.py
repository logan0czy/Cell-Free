"""
Training process module

Reference: https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/td3/td3.py
"""


import time
import math
import itertools
from copy import deepcopy
# from IPython.display import clear_output

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn

from net import ActorCritic, sync
from env import Environment
import utils


def train(
        env_kwargs, net_kwargs, cbook_kwargs, act_noise, tgt_noise, noise_clip, epochs=100,
        steps_per_epoch=10000, start_steps=10000, update_after=1000, update_every=50,
        policy_decay=2, lr_policy=1e-3, lr_q=1e-3, sync_rate=0.005, n_powers=4, gamma=0.6,
        batch_size=64, buffer_size=100000, seed=24
):
    """Twin Delayed Deep Deterministic Policy training process.

    Parameters:
        env_kwargs (dict): environment's arguments, keys:

            'max_power', 'bs_atn', 'ris_atn'

        net_kwargs (dict): network's arguments, keys:

            'critic_hidden_sizes', 'actor_hidden_sizes', 'act_limit'

        cbook_kwargs (dict): codebook's arguments. keys: 

            'bs_codes', 'ris_codes', 'bs_phases', 'ris_azi_phases', 'ris_ele_phases'

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

        lr_policy (float): learning rate of policy network

        lr_q (float): learning rate of Q network

        sync_rate (float): the synchronize ratio between target network parameters and main
            networks. Equation is:

            tgt_param := sync_rate*src_param + (1-sync_rate)*tgt_param 
        
        n_powers (int): number of power levels to choose.

        gamma (float): reward decay

        batch_size (int): number of samples to learn from at each gradient descent update

        buffer_size (int): ReplayBuffer storage amount

        seed (int): random seed
    """
    def getQLoss(data):
        """compute the Q-value loss according to Bellman equation
        w.r.t batches of transition samples.

        Parameters:
            data (dict): include batches of state transitions and the reward,
                keys are {'obs', 'next_obs', 'act', 'rew'}

        Returns:
            loss_q (torch.tensor): average Q-value loss
        """
        # Bellman backup for Q function
        with torch.no_grad():
            next_act = tgt_model.actor(data['next_obs'])
            next_act = tgt_ous.getActFromRaw(next_act.cpu()).to(main_model.device)
            q1_tgt = tgt_model.q1(data['next_obs'], next_act)
            q2_tgt = tgt_model.q2(data['next_obs'], next_act)
            q_tgt = torch.min(q1_tgt, q2_tgt)
            backup = data['rew'].unsqueeze(-1) + gamma*q_tgt

        # MSE Q loss
        q1 = main_model.q1(data['obs'], data['act'])
        q2 = main_model.q2(data['obs'], data['act'])
        loss_q = ((q1-backup)**2).mean() + ((q2-backup)**2).mean()

        return loss_q

    def getPolicyLoss(data):
        """compute the mean estimate Q-value using current policy.

        Parameters:
            data (dict): include batches of state transitions and the reward,
                keys are {'obs', 'next_obs', 'act', 'rew'}

        Returns:
            loss_policy (torch.Tensor): average policy loss
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
        """update actor-critic network.
        Parameters:
            data (dict): include batches of state transitions and the reward,
                keys are {'obs', 'next_obs', 'act', 'rew'}
        """
        data = {k: torch.as_tensor(v, dtype=torch.float32, device=main_model.device) 
                 for k, v in data.items()}

        q_opt.zero_grad()
        q_loss = getQLoss(data)
        q_loss.backward()
        q_opt.step()

        if timer%policy_decay == 0:
            policy_opt.zero_grad()
            policy_loss = getPolicyLoss(data)
            policy_loss.backward()
            policy_opt.step()

            sync(main_model, tgt_model, sync_rate)
            
            return q_loss.item(), policy_loss.item()

        return q_loss.item()

    def getAct(obs):
        action = main_model.act(torch.as_tensor([obs], dtype=torch.float32, device=main_model.device))
        action = act_ous.getActFromRaw(action.unsqueeze(0).cpu())
        return action

    np.random.seed(seed)
    torch.manual_seed(seed)

    env = Environment(**env_kwargs)

    # beamforming codebook
    bs_cbook = utils.CodeBook(cbook_kwargs['bs_codes'], env.bs_atn, cbook_kwargs['bs_phases'])
    ris_ele_cbook = utils.CodeBook(cbook_kwargs['ris_codes'], env.ris_atn[1], cbook_kwargs['ris_ele_phases'])
    ris_azi_cbook = utils.CodeBook(cbook_kwargs['ris_codes'], env.ris_atn[0], cbook_kwargs['ris_azi_phases'])

    # action decoder
    transfer = utils.Decoder(env, -net_kwargs['act_limit'], net_kwargs['act_limit'], bs_cbook,
                            ris_azi_cbook, ris_ele_cbook, [(i+1)*env.max_power/n_powers for i in range(n_powers)])

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    main_model = ActorCritic(env.obs_dim, 2, **net_kwargs)
    main_model.to(device)
    tgt_model = deepcopy(main_model)

    tgt_model.train(False)
    for param in tgt_model.parameters():
        param.requires_grad = False

    # exploration noise
    act_ous = utils.OUStrategy(act_space={'dim': 2, 'low': -1*net_kwargs['act_limit'], 'high': net_kwargs['act_limit']},
                               max_sigma=np.array([act_noise*transfer.spacing[0], act_noise*transfer.spacing[1]], dtype=np.float32))
    tgt_ous = utils.OUStrategy(act_space={'dim': 2, 'low': -1*net_kwargs['act_limit'], 'high': net_kwargs['act_limit']},
                               max_sigma=np.array([tgt_noise*transfer.spacing[0], tgt_noise*transfer.spacing[1]], dtype=np.float32),
                               noise_clip=np.array([noise_clip*transfer.spacing[0], noise_clip*transfer.spacing[1]], dtype=np.float32))

    # list of parameters for both Q networks
    q_params = itertools.chain(main_model.q1.parameters(), main_model.q2.parameters())

    # optimizer
    policy_opt = torch.optim.Adam(main_model.actor.parameters(), lr_policy)
    q_opt = torch.optim.Adam(q_params, lr_q)

    # experience buffer
    replay_buffer = utils.ReplayBuffer(env.obs_dim, 2, buffer_size)

    # training process
    total_steps = epochs * steps_per_epoch
    obs = env.reset(seed)
    ep_rew = 0
    for step in range(total_steps):
        if step < start_steps:
            act = net_kwargs['act_limit'] * np.random.uniform(-1, 1, 2)
        else:
            act = getAct(obs)
        bs_beam, ris_beam = transfer.decode(act)
        next_obs, rew = env.step(bs_beam, ris_beam)
        ep_rew += rew

        replay_buffer.store(obs, act, rew, next_obs)

        obs = next_obs

        if (step+1) > update_after and (step+1-update_after)%update_every==0:
            loss_info = [[], []]
            for j in range(update_every):
                batch = replay_buffer.sampleBatch(batch_size)
                loss = update(batch, j)

                if not np.isscalar(loss):
                    loss_info[0].append(loss[0])
                    loss_info[1].append(loss[1])
                else:
                    loss_info[0].append(loss)

            if (step+1)%1000==0:
                print(f"epoch: {(step+1)//steps_per_epoch}, loss_q: {np.mean(loss_info[0]):.4f}, loss_policy: {np.mean(loss_info[1]):.4f}")

        if (step+1) % steps_per_epoch == 0:
            obs = env.reset(seed)
            act_ous.reset()
            tgt_ous.reset()

            print(f"\nepoch: {(step+1)//steps_per_epoch}, avg_rew: {ep_rew/steps_per_epoch:.4f}\n")

if __name__=='__main__':
    env_kwargs = {'max_power': 30, 'bs_atn': 4, 'ris_atn': (8, 4)}
    net_kwargs = {'critic_hidden_sizes': [1024, 512, 256], 
                  'actor_hidden_sizes': [1024, 512, 256],
                  'act_limit': 1}
    cbook_kwargs = {'bs_codes': 10, 'ris_codes': 8, 
                    'bs_phases': 8,
                    'ris_azi_phases': 4,
                    'ris_ele_phases': 4}
    train(env_kwargs, net_kwargs, cbook_kwargs, act_noise=2, tgt_noise=0, noise_clip=0)