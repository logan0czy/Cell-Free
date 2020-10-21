"""
The main Twin Delayed Deep Determinisic Policy Gradient Network frame.

Reference: https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/td3/core.py
"""
import torch
import torch.nn as nn


def genLayers(sizes, activation, output_activation=nn.Identity):
    """generate nn layers
    
    Parameters:
        sizes : each layer's size
        activation : hidden layer's activation function
    """
    layers = [nn.BatchNorm1d(sizes[0])]
    for i in range(len(sizes)-2):
        layers += [nn.Linear(sizes[i], sizes[i+1]), nn.BatchNorm1d(sizes[i+1]), activation()]
    layers += [nn.Linear(sizes[-2], sizes[-1]), output_activation()]
    return nn.Sequential(*layers)

def sync(src_module: nn.Module, tgt_module: nn.Module, sync_rate: float):
    """synchronize source and target modules' parameters."""
    for src_param, tgt_param in zip(src_module.parameters(), tgt_module.parameters()):
        tgt_param.copy_(sync_rate*src_param + (1-sync_rate)*tgt_param)

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        sizes = [obs_dim] + [*hidden_sizes] + [act_dim]
        self.L = genLayers(sizes, activation, nn.Tanh)
        self.act_limit = act_limit

    def forward(self, obs):
        # Return output from network scaled to action limits.
        return self.act_limit * self.L(obs)

class QFunc(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        sizes = [obs_dim+act_dim] + [*hidden_sizes] + [1]
        self.L = genLayers(sizes, activation)

    def forward(self, obs, act):
        # Return Q value from network
        obs_act = torch.cat((obs, act), dim=-1)
        return self.L(obs_act)

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, critic_hidden_sizes, actor_hidden_sizes,
                 act_limit=1, activation=nn.ELU):
        super().__init__()
        self.actor = Actor(obs_dim, act_dim, actor_hidden_sizes, activation, act_limit)
        self.q1 = QFunc(obs_dim, act_dim, critic_hidden_sizes, activation)
        self.q2 = QFunc(obs_dim, act_dim, critic_hidden_sizes, activation)

    @property
    def device(self):
        return next(self.parameters()).device

    def act(self, obs):
        # Get action w.r.t the observations using current parameters
        self.train(False)
        with torch.no_grad():
            actions = self.actor(obs)

        self.train(True)
        return actions
