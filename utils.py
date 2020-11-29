import random
import numpy as np


# for colorized formatting stdout
color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)

def colorize(string, color, bold=False, highlight=False):
    """
    Colorize a string.

    This function was originally written by John Schulman.
    """
    attr = []
    num = color2num[color]
    if highlight: num += 10
    attr.append(str(num))
    if bold: attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)

def combineShape(length, shape=None):
    """This function refers to OpenAI spinningup."""
    if shape is None:
        return (length, )
    return (length, shape) if np.isscalar(shape) else (length, *shape)

class Decoder():
    """Decode action to its corresponding choice."""

    def __init__(self, act_range, act_choices, sat_ratio=0.05):
        """
        Args:
            act_range (tuple): Range of action value, includes (act_low, act_high)

            act_choices : An array stores all kinds of choices with shape (N, *)

            sat_ratio (float): Saturation ratio of action range. In case of the 
                saturation problem when use 'tanh' activation function, assign 
                the first and the last ratio of action interval to one action 
                respectively (when the ratio value is larger than 0).
        """
        self.range = act_range
        self.choices = act_choices
        self.sat_ratio = sat_ratio
        if sat_ratio>0:
            self.spacing = (self.range[1]-self.range[0]) * (1-2*sat_ratio) / (len(self.choices)-2)
        else:
            self.spacing = (self.range[1]-self.range[0]) / len(self.choices)

    def decode(self, act_val):
        """
        Args:
            act_val : Array of action values with shape (d0, d1, ..., dN).

        Returns:
            acts : Array of corresponding choices with shape 
                (d0, d1, ..., dN, dN+1, ..., dN+choice_dim).
        """
        def discretize(val):
            if self.sat_ratio==0:
                return int(val // self.spacing)
            if val < (self.range[1]-self.range[0])*self.sat_ratio:
                return 0
            if val >= (self.range[1] - (self.range[1]-self.range[0])*self.sat_ratio):
                return -1
            residual = val - (self.range[1]-self.range[0])*self.sat_ratio
            return int(residual // self.spacing) + 1

        act_val_shape = act_val.shape
        choice_shape = self.choices.shape
        idxs = list(map(discretize, act_val.reshape(-1)))
        acts = self.choices[idxs].reshape((*act_val_shape, *choice_shape[1:]))
        return acts

class ReplayBuffer():
    """
    An experience replay buffer.

    Links: https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/td3/td3.py
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
        """Uniformly sample from buffer."""
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     next_obs=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs])
        return batch

class OUStrategy():
    """
    This strategy implements the Ornstein-Uhlenbeck process.
    
    The OU process adds time-correlated noise to the actions taken by 
    the deterministic policy. It satisfies the following stochastic 
    differential equation:
    
        dxt = theta*(mu - xt)*dt + sigma*dWt

    where Wt denotes the Wiener process.

    Links1: https://towardsdatascience.com/deep-deterministic-policy-gradients-explained-2d94655a9b7b

    Links2: https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process

    Links3: https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
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
        Args:
            act_space : Keys are {'dim', 'low', 'high'}, the 'low' & 'high' must be arrays
                with the same dim to action

            max_sigma/min_sigma : Array of stddevs for Gaussian policy noise, and each dim
                maybe different

            noise_clip : Array of limit for absolute value of policy noise, each dim maybe
                different
                
            decay_period : Time steps for stddev of noise decaying from max_sigma to min_sigma.
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
