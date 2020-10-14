import math
import random

import numpy as np


def combineShape(length, shape=None):
    """
    Reference: https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/td3/core.py

    Return:
        tuple: (length, *shape)
    """
    if shape is None:
        return (length, )
    return (length, shape) if np.isscalar(shape) else (length, *shape)

class CodeBook():
    """Generate the 2-D beamforming codebook"""

    def __init__(self, codes: int, antennas: int, phases: int=16):
        """
        Parameters:
            codes : the amount of codes
            antennas : the amount of antennas in horizontal or vertical dimension
            phases : the amount of available phases
        """
        self.codes = codes
        self.antennas = antennas
        self.phases = phases
        self.scaled = False

    def _element(self, code_num: int, antenna_num: int):
        """get the (code_num, antenna_num) element's value."""
        temp1 = (code_num+self.codes/2) % self.codes
        temp2 = math.floor(antenna_num*temp1/(self.codes/self.phases))
        value = 1 / math.sqrt(self.antennas) * np.exp(1j*2*math.pi/self.phases*temp2)
        return value
    
    def generate(self):
        """Generate the codebook of shape (self.codes, self.antennas)

        Return:
            codebook
        """
        if hasattr(self, 'book'):
            return self.book
        book = [[self._element(code_num, antenna_num) for antenna_num in range(self.antennas)]
                for code_num in range(self.codes)
            ]
        self.book = np.array(book)
        return self.book

    def scale(self):
        """Rescale the codebook. When the codebook is used for ris,
        must call this scale function first.

        Return:
            codebook: shape (self.codes, self.antennas)
        """
        if self.scaled:
            return self.book
        if not hasattr(self, 'book'):
            self.generate()
        self.book = self.book * math.sqrt(self.antennas)
        self.scaled = True
        return self.book

class Decoder():
    """decode the action from policy to the real action to interact with environment."""

    def __init__(self, env, bs_cbook, ris_azi_cbook, ris_ele_cbook, power_levels):
        """
        Parameters:
            env (env.Environment): the cell-free network
            bs_cbook (CodeBook): codebook of base station
            ris_azi_cbook (CodeBook): codebook of ris in the azimuth dimension
            ris_ele_cbook (CodeBook): codebook of ris in the elevation dimension
            power_levels (tuple): choice of power
        """
        self.env = env
        self.bs_cbook = bs_cbook
        self.ris_azi_cbook = ris_azi_cbook
        self.ris_ele_cbook = ris_ele_cbook
        self.power_levels = power_levels
        self.genMap()

    def genMap(self):
        """generate the map between the discretized value of action from policy
        to the actual action to env.
        """
        def combine(arr1, arr2):
            """get all of possible combinations among two array instances along
            the first dimension."""
            res = []
            for i1 in arr1:
                for i2 in arr2:
                    i1 = np.array([i1]) if np.isscalar(i1) else i1.reshape(-1)
                    i2 = np.array([i2]) if np.isscalar(i2) else i2.reshape(-1)
                    res.append(np.concatenate((i1, i2)))
            res = np.stack(res, axis=0)
            return res

        def genIndexCombine(*sizes):
            """
            Parameters:
                sizes (List(int))
            
            Returns:
                res (np.array)
            """
            res = np.arange(sizes[0])
            for i in range(1, len(sizes)):
                res = combine(res, np.arange(sizes[i]))
            return res

        def genArrayCombine(*arrs):
            """
            Parameters:
                arrs (List[np.array])

            Returns:
                res (np.array)
            """
            res = arrs[0]
            for i in range(1, len(arrs)):
                res = combine(res, arrs[i])
            return res
        
        bs_num, ris_num, _ = self.env.getCount()
        self.bs_act_size = (len(self.power_levels) * self.bs_cbook.codes)**(bs_num)
        self.ris_act_size = (self.ris_azi_cbook.codes * self.ris_ele_cbook.codes)**(ris_num)

        single_bs_act = genIndexCombine(len(self.power_levels), self.bs_cbook.codes)
        single_ris_act = genIndexCombine(self.ris_azi_cbook.codes, self.ris_ele_cbook.codes)
        self.bs_map = genArrayCombine([single_bs_act for i in range(bs_num)])
        self.ris_map = genArrayCombine([single_ris_act for i in range(ris_num)])
        return

class ReplayBuffer():
    """An experience replay buffer.
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

    def sampleBatch(self, batch_size: int=64) -> dict: 
        """Uniformly sample from buffer.

        Returns:
            batch of transitions.
        """
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     next_obs=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs])
        return batch

class OUStrategy():
    """This strategy implements the Ornstein-Uhlenbeck process, which adds time-correlated 
    noise to the actions taken by the deterministic policy. The OU process satisfies the 
    following stochastic differential equation:
        dxt = theta*(mu - xt)*dt + sigma*dWt
    where Wt denotes the Wiener process.

        Reference 1: https://towardsdatascience.com/deep-deterministic-policy-gradients-explained-2d94655a9b7b
        Reference 2: https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process
        Reference 3: https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
    """

    def __init__(
            self,
            act_space: dict,
            mu: float=0,
            theta: float=0.15,
            max_sigma=None,
            min_sigma=None,
            noise_clip=None,
            decay_period: int=10000,
    ):
        """
        Parameters:
            act_space : keys are {'dim', 'low', 'high'}, the 'low' & 'high' must be np.array
                with the same dim to action
            max_sigma/min_sigma (np.array): stddevs for Gaussian policy noise, and each dim
                maybe different
            noise_clip (np.array): limit for absolute value of policy noise, each dim maybe
                different
            decay_period : time steps for stddev of noise decaying from max_sigma to min_sigma.
        """
        self.act_space = act_space
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self._max_sigma = max_sigma
        self._min_sigma = min_sigma if min_sigma is not None else max_sigma
        self._decay_period = decay_period
        self.noise_clip = noise_clip
        self.reset()

    def reset(self):
        self.state = self.mu * np.ones(self.act_space['dim'], dtype=np.float32)

    def evolveState(self):
        x = self.state
        dx = self.theta*(self.mu-x) + self.sigma*np.random.randn(*x.shape)
        self.state = x + dx
        if self.noise_clip is not None:
            self.state = np.clip(self.state, -self.noise_clip, self.noise_clip)
        return self.state

    def getActFromRaw(self, raw_act, time_step: int=0):
        ou_state = self.evolveState()
        self.sigma = (
            self._max_sigma
            - (self._max_sigma - self._min_sigma)
            * min(1.0, time_step*1.0/self._decay_period)
        )
        return np.clip(raw_act + ou_state, self.act_space['low'], self.act_space['high'])