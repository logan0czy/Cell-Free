"""
Training process module

Reference: https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/td3/td3.py
"""


import time
import math
# from IPython.display import clear_output

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn

from net import ActorCritic, sync
from env import Environment
import utils


def getQLoss(model, tgt_model, trans, tgt_noise, gamma):
    """compute the Q-value loss according to Bellman equation
    w.r.t batches of transition samples.

    Parameters:
        model (nn.Module): main actor-critic network

        tgt_model (nn.Module): target actor-critic network

        trans (dict): include batches of state transitions and the reward,
            keys are {'obs', 'next_obs', 'act', 'rew'}

        tgt_noise (OUStrategy): the smoothing noise adding to target action

        gamma (float): reward decay

    Returns:
        loss_q (torch.tensor): average Q-value loss
    """
    trans = {k: torch.as_tensor(v, dtype=torch.float32, device=model.device) 
             for k, v in trans.items()}
    
    # Bellman backup for Q function
    with torch.no_grad():
        next_act = tgt_model.actor(trans['next_obs'])
        next_act = tgt_noise.getActFromRaw(next_act.cpu()).to(model.device)
        q1_tgt = tgt_model.q1(trans['next_obs'], next_act)
        q2_tgt = tgt_model.q2(trans['next_obs'], next_act)
        q_tgt = torch.min(q1_tgt, q2_tgt)
        backup = trans['rew'].unsqueeze(-1) + gamma*q_tgt

    # MSE Q loss
    q1 = model.q1(trans['obs'], trans['act'])
    q2 = model.q2(trans['obs'], trans['act'])
    loss_q = ((q1-backup)**2).mean() + ((q2-backup)**2).mean()

    return loss_q

def getPolicyLoss(model, trans):
    """compute the mean estimate Q-value using current policy.

    Parameters:
        model (nn.Module): main actor-critic network
        trans (dict): include batches of state transitions and the reward,
            keys are {'obs', 'next_obs', 'act', 'rew'}
    
    Returns:
        loss_policy (torch.Tensor): average policy loss
    """
    obs = torch.as_tensor(trans['obs'], dtype=torch.float32, device=model.device)
    act = model.actor(obs)
    return -model.q1(obs, act).mean()
    

def train(
        epochs=100, steps_per_epoch=10000, start_steps, update_after, update_every, policy_decay,
        env_args, net_args, cbook_args, lr_act, lr_crt, sync_rate, act_scale=1., batch_size=64, 
        noise, tgt_noise, noise_clip
):
    """Twin Delayed Deep Deterministic Policy training process.

    Parameters:
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

        env_args (dict): environment's arguments

        net_args (dict): network's arguments

        cbook_args (dict): codebook's arguments. The keys contain: 'bs_codes', 'ris_codes',
            'bs_phases', 'ris_azi_phases', 'ris_ele_phases'.

        lr_act (float): learning rate of policy network

        lr_crt (float): learning rate of Q network

        sync_rate (float): the synchronize ratio between target network parameters and main
            networks
        
        act_scale (float): scale factor to the output of policy.

        batch_size (int): number of samples to learn from at each gradient descent update

        noise (float): stddev of Gaussian noise added to action from policy network

        tgt_noise (float): stddev 
    """
