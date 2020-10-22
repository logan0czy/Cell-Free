import math
import random

import numpy as np

from env import Environment


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

    def __init__(self, codes: int, antennas: int, phases: int=16, 
        scale: bool=False, duplicated: bool=True):
        """
        Parameters:
            codes : the amount of codes
            antennas : the amount of antennas in horizontal or vertical dimension
            phases : the amount of available phases
            scale : whether rescale the codebook. When the codebook is used for ris,
                this operation must be done.
            duplicated : if True, append the codebook's first code to end.
        """
        self.codes = self._generate(codes, antennas, phases, duplicated)
        if scale:
            self.book = self.book * math.sqrt(self.antennas)
    
    def _generate(self, codes, antennas, phases, duplicated):
        """Generate the codebook of shape (codes, antennas)"""
        def element(code_id: int, antenna_id: int):
            """get the (code_id, antenna_id) element's value."""
            temp = (code_id+codes/2) % codes
            temp = math.floor(antenna_id*temp/(codes/phases))
            value = 1 / math.sqrt(antennas) * np.exp(1j*2*math.pi/phases*temp)
            return value

        book = [[element(code_id, antenna_id) for antenna_id in range(antennas)]
                for code_id in range(codes)]
        self.book = np.array(book)
        if duplicated:
            self.book = np.concatenate((self.book, self.book[[0]]), axis=0)
        return len(self.book)

class Decoder():
    """decode the action from policy to the real action to interact with cell-free environment."""

    def __init__(self, env, act_low, act_high, bs_cbook, ris_azi_cbook, ris_ele_cbook, 
        power_levels, sat_ratio=0.05):
        """
        Parameters:
            env (env.Environment): the cell-free network

            act_low/act_high (float): the lowest/highest action value, assume these are the same for
                each dimension

            bs_cbook (CodeBook): codebook of base station

            ris_azi_cbook (CodeBook): codebook of ris in the azimuth dimension

            ris_ele_cbook (CodeBook): codebook of ris in the elevation dimension

            power_levels (tuple): choice of power

            sat_ratio (float): ratio of interval [act_low, act_high]. In case of the saturation problem
                when use 'tanh' activation function, assign the first and the last ratio of action 
                interval to one action respectively. 
        """
        self.env = env
        self.act_low, self.act_high = act_low, act_high
        self.bs_cbook = bs_cbook
        self.ris_azi_cbook = ris_azi_cbook
        self.ris_ele_cbook = ris_ele_cbook
        self.power_levels = power_levels
        self.sat_ratio = sat_ratio
        self._genMap()

        # calculate absolute values of interval corresponding to the same action
        self.spacing = ((self.act_high-self.act_low)*(1-2*self.sat_ratio) / (self.bs_act_size-2),
                        (self.act_high-self.act_low)*(1-2*self.sat_ratio) / (self.ris_act_size-2))

    def _genMap(self):
        """generate the map between the discretized value of action from policy
        to the actual action to env. The maps for a single object are:
            base station action : power_level_id, bs_cbook_id
            ris action : azi_cbook_id, ele_cbook_id
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
        single_ris_act = genIndexCombine(self.ris_ele_cbook.codes, self.ris_azi_cbook.codes)
        self.bs_map = genArrayCombine(*[single_bs_act for i in range(bs_num)])
        self.ris_map = genArrayCombine(*[single_ris_act for i in range(ris_num)])
        return

    def decode(self, action):
        """
        Parameters:
            action (np.array): shape is (2,), the first dim is for base station, the second
                dim is for ris.
        
        Returns:
            bs_beam (np.array): shape is (bs_num, bs_atn)
            ris_beam (np.array): shape is (ris_num, ris_atn, ris_atn), the last two dimension
                is diagnal matrix
        """
        bs_num, ris_num, _ = self.env.getCount()
        interval = self.act_high - self.act_low
        # base station action
        if action[0] < self.act_low+interval*self.sat_ratio:
            bs_act_id = 0
        elif action[0] > self.act_high-interval*self.sat_ratio:
            bs_act_id = self.bs_act_size - 1
        else:
            bs_act_id = (action[0] - (self.act_low+interval*self.sat_ratio)) // self.spacing[0] + 1
            bs_act_id = int(bs_act_id)
        # ris action
        if action[1] < self.act_low+interval*self.sat_ratio:
            ris_act_id = 0
        elif action[1] > self.act_high-interval*self.sat_ratio:
            ris_act_id = self.ris_act_size - 1
        else:
            ris_act_id = (action[1] - (self.act_low+interval*self.sat_ratio)) // self.spacing[1] + 1
            ris_act_id = int(ris_act_id)

        bs_beam = np.zeros((bs_num, self.env.bs_atn), dtype=np.complex64)
        ris_beam = np.zeros((ris_num, np.prod(self.env.ris_atn)), dtype=np.complex64)
        for i in range(bs_num):
            power = self.power_levels[self.bs_map[bs_act_id, i*2]]
            bs_beam[i] = math.sqrt(power) * self.bs_cbook.book[self.bs_map[bs_act_id, i*2+1]]
        for j in range(ris_num):
            ris_beam[j] = np.kron(self.ris_ele_cbook.book[self.ris_map[ris_act_id, j*2]], 
                                  self.ris_azi_cbook.book[self.ris_map[ris_act_id, j*2+1]])
        ris_beam_expand = np.zeros(ris_beam.shape+ris_beam.shape[-1:], dtype=ris_beam.dtype)
        diagonals = ris_beam_expand.diagonal(axis1=-2, axis2=-1)
        diagonals.setflags(write=True)
        diagonals[:] = ris_beam.copy()

        return bs_beam, ris_beam_expand

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
        dx = self.theta*(self.mu-x) + self.sigma*np.random.randn(*x.shape, ).astype(np.float32)
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

if __name__=='__main__':
    env = Environment(30)
    bs_cbook = CodeBook(10, env.bs_atn)
    ris_ele_cbook, ris_azi_cbook = CodeBook(8, env.ris_atn[1], phases=4), CodeBook(8, env.ris_atn[0], phases=4)
    ris_ele_cbook.scale()
    ris_azi_cbook.scale()

    transer = Decoder(env, -1, 1, bs_cbook, ris_azi_cbook, ris_ele_cbook, 
                      [env.max_power*i for i in range(1, 5)])
    bs_beam, ris_beam = transer.decode(np.array([0.3, -0.5]))
    print(f"bs_beam, shape{bs_beam.shape}\n{bs_beam}")
    print(f"ris_beam, shape{ris_beam.shape}\n{ris_beam}")
