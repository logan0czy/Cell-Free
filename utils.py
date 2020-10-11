import math
import random

import numpy as np

import torch


def combineShape(length, shape=None):
    """
    Reference: https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/td3/core.py
    """
    if shape is None:
        return (length, )
    return (length, shape) if np.isscalar(shape) else (length, *shape)

class CodeBook():
    """Generate the 2-D beamforming codebook"""

    def __init__(self, codes, antennas, phases=16):
        """
        initial class attributes:
        codes (int): the amount of codes
        antennas (int): the amount of antennas in horizontal or vertical dimension
        phases (int): the amount of available phases
        """
        self.codes = codes
        self.antennas = antennas
        self.phases = phases
        self.scaled = False

    def _element(self, code_num, antenna_num):
        temp1 = (code_num+self.codes/2) % self.codes
        temp2 = math.floor(antenna_num*temp1/(self.codes/self.phases))
        value = 1 / math.sqrt(self.antennas) * np.exp(1j*2*math.pi/self.phases*temp2)
        return value
    
    def generate(self):
        """
        Generate the codebook of shape (self.codes, self.antennas)
        """
        if hasattr(self, 'book'):
            return self.book
        book = [[self._element(code_num, antenna_num) for antenna_num in range(self.antennas)]
                for code_num in range(self.codes)
            ]
        self.book = np.array(book)
        return self.book

    def scale(self):
        if self.scaled:
            return self.book
        if not hasattr(self, 'book'):
            self.generate()
        self.book = self.book * math.sqrt(self.antennas)
        self.scaled = True
        return self.book

class ReplayBuffer():
    """
    An experience replay buffer.
    Reference: https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/td3/td3.py
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(combineShape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(combineShape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combineShape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.size, self.max_size, self.ptr = 0, size, 0

    def store(self, obs, act, rew, next_obs):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.size = min(self.size+1, self.max_size)
        self.ptr = (self.ptr+1) % self.max_size

    def sampleBatch(self, batch_size=64):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     next_obs=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}
