"""
The communication environment which the agent interact with.
"""


import math

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def angle(tgt_loc, src_loc, src_loc_delta):
    """
    Compute the azimuth and elevation angle from source to target.
    If the source is planar, assume it is placed vertically.

    Parameters:
        tgt_loc (np.array): transmission target location, 3-D
        src_loc (np.array): transmission source location, 3-D
        src_loc_delta (np.array): shift from source

    Returns:
        theta_azi (float): azimuth angle
        theta_ele (float): elevation angle
    """
    normal_vec = np.zeros(3, dtype=np.float32)
    if src_loc_delta[0] == 0:
        normal_vec[0] = 1
    elif src_loc_delta[1] == 0:
        normal_vec[1] = 1
    else:
        normal_vec[1] = 1
        normal_vec[0] = -normal_vec[1]*src_loc_delta[1]/src_loc_delta[0]
    proj_vec = np.array([tgt_loc[0], tgt_loc[1], src_loc[2]])

    theta_ele = math.acos(np.linalg.norm(proj_vec-src_loc) / np.linalg.norm(tgt_loc-src_loc))
    inner_prod = abs(np.matmul(proj_vec-src_loc, normal_vec))
    theta_azi = math.acos(inner_prod / (np.linalg.norm(proj_vec-src_loc)*np.linalg.norm(normal_vec)))
    return theta_azi, theta_ele

class Environment():
    """Cell-Free network"""

    def __init__(self, max_power: float, bs_atn: int=4, ris_atn: tuple=(8, 4)):
        """
        Parameters:
            max_power : maximum transmission power(dB) of base station
            bs_atn : number of antennas of base station
            ris_atn : number of horizontal & vertical antennas of ris

        Default settings:
            The adjecent antenna's multual distance is half of the wave length,
            base station and RIS are at the same height,
            near field distance is 1m
        """
        self._genLoc()
        self._genAngle()
        self.bs_atn = bs_atn
        self.ris_atn = ris_atn
        self.max_power = 10**(max_power/10)
        self.pl0 = 10**(-3)  # base path loss, -30dB
        self.pl_exp = (2, 2.8, 2.8)  # path loss exponent, (bs-ris, ris-user, bs-user) respectively
        self.noise = 10**(-80/10)  # power of noise, -80dBm
        self.paths = 25  # number of multi-paths
        self.angle_spread = math.pi * 3 / 180  # anguler spreads for multi-path
        self.rho = 0.64  # channel state transition proportion

        bs_num, ris_num, user_num = self.getCount()
        self.obs_dim = bs_num*ris_num*2*np.prod(self.ris_atn)*self.bs_atn \
                        + ris_num*user_num*2*np.prod(self.ris_atn) \
                        + bs_num*user_num*2*self.bs_atn
        self.act_dim = bs_num*2*user_num + ris_num*2
    
    def _genLoc(self, radius: int=50, height: int=5):
        """generate the location of base stations, RISs and users

        Parameters:
            radius : network's radius
            height : height of base stations and RISs
        """
        self.bs_loc = np.array([[-radius, 0, height], [radius, 0, height]])
        self.bs_loc_delta = np.array([[0, 1, 0], [0, 1, 0]])
        # self.ris_loc = np.array([[0, radius, height], [0, -radius, height]])
        # self.ris_loc_delta = np.array([[1, 0, 0], [1, 0, 0]])
        self.ris_loc = np.array([[0, radius, height], ])
        self.ris_loc_delta = np.array([[1, 0, 0], ])

        # ptr, user_num = 0, 8
        # self.user_loc = np.zeros((user_num, 3), dtype=np.float32) 
        # for theta in np.arange(math.pi/4, 2*math.pi, math.pi/2):
        #     self.user_loc[ptr, 0] = radius*math.cos(theta)
        #     self.user_loc[ptr, 1] = radius*math.sin(theta)
        #     ptr += 1
        # for theta in np.arange(0, 2*math.pi, math.pi/2):
        #     self.user_loc[ptr, 0] = radius/2*math.cos(theta)
        #     self.user_loc[ptr, 1] = radius/2*math.sin(theta)
        #     ptr += 1
        ptr = 0
        self.user_loc = np.zeros((4, 3), dtype=np.float32) 
        for theta in np.arange(math.pi/4, 2*math.pi, math.pi/2):
            self.user_loc[ptr, 0] = radius / 2 * math.cos(theta)
            self.user_loc[ptr, 1] = radius / 2 * math.sin(theta)
            ptr += 1
        return
    
    def _genAngle(self):
        """generate the mutual azimuth & elevation angle among bs, RIS and users.

        the meanings of variable suffix are:
            _azi: angle in azimuth dimension
            _ele: angle in elevation dimension
        """
        bs_num, ris_num, user_num = self.getCount()
        # -------- w.r.t RIS --------
        self.bs2ris_azi = np.zeros((bs_num, ris_num), dtype=np.float32)
        self.bs2ris_ele = np.zeros((bs_num, ris_num), dtype=np.float32)
        for i in range(bs_num):
            for j in range(ris_num):
                theta_azi, theta_ele = angle(self.bs_loc[i], self.ris_loc[j], self.ris_loc_delta[j])
                self.bs2ris_azi[i, j] = theta_azi
                self.bs2ris_ele[i, j] = theta_ele

        self.user2ris_azi = np.zeros((user_num, ris_num), dtype=np.float32)
        self.user2ris_ele = np.zeros((user_num, ris_num), dtype=np.float32)
        for i in range(user_num):
            for j in range(ris_num):
                theta_azi, theta_ele = angle(self.user_loc[i], self.ris_loc[j], self.ris_loc_delta[j])
                self.user2ris_azi[i, j] = theta_azi
                self.user2ris_ele[i, j] = theta_ele

        # -------- w.r.t base station --------
        self.ris2bs_azi = np.zeros((ris_num, bs_num), dtype=np.float32)
        for i in range(ris_num):
            for j in range(bs_num):
                theta_azi, _ = angle(self.ris_loc[i], self.bs_loc[j], self.bs_loc_delta[j])
                self.ris2bs_azi[i, j] = theta_azi

        self.user2bs_azi = np.zeros((user_num, bs_num), dtype=np.float32)
        for i in range(user_num):
            for j in range(bs_num):
                theta_azi, _ = angle(self.user_loc[i], self.bs_loc[j], self.bs_loc_delta[j])
                self.user2bs_azi[i, j] = theta_azi
        return

    def _changeCSI(self):
        """get the channel state information at current time step
        
        Return:
            bs2user_csi : (bs_num, user_num, bs_atn)
            bs2ris_csi : (bs_num, ris_num, ris_atn, bs_atn)
            ris2user_csi : (ris_num, user_num, ris_atn)
        """
        bs_num, ris_num, user_num = self.getCount()
        # -------- bs to ris csi --------
        if not hasattr(self, 'bs2ris_csi'):
            pl = self.pl0 / np.linalg.norm(self.bs_loc[:, np.newaxis] - self.ris_loc[np.newaxis], axis=-1)**self.pl_exp[0]
            ris_arr_resp_ele = 1 / np.sqrt(self.ris_atn[1]) \
                * np.exp(1j*math.pi*np.arange(self.ris_atn[1])[np.newaxis, np.newaxis]*np.sin(self.bs2ris_ele)[:, :, np.newaxis])
            ris_arr_resp_azi = 1 / np.sqrt(self.ris_atn[0]) \
                * np.exp(1j*math.pi*np.arange(self.ris_atn[0])[np.newaxis, np.newaxis]*(np.cos(self.bs2ris_ele)*np.cos(self.bs2ris_azi))[:, :, np.newaxis])
            ris_arr_resp = np.matmul(ris_arr_resp_ele[:, :, :, np.newaxis], ris_arr_resp_azi[:, :, np.newaxis]).reshape(bs_num, ris_num, -1)
            bs_arr_resp_azi = 1 / np.sqrt(self.bs_atn) \
                * np.exp(1j*math.pi*np.arange(self.bs_atn)[np.newaxis, np.newaxis]*np.sin(self.ris2bs_azi)[:, :, np.newaxis])
            self.bs2ris_csi = np.sqrt(pl[:, :, np.newaxis, np.newaxis]) \
                *  np.matmul(ris_arr_resp[:, :, :, np.newaxis], bs_arr_resp_azi[:, :, np.newaxis])

        # -------- bs to user csi --------
        pl = self.pl0 / np.linalg.norm(self.bs_loc[:, np.newaxis] - self.user_loc[np.newaxis], axis=-1)**self.pl_exp[2]
        factor = np.exp(1j*np.random.uniform(-math.pi, math.pi, size=(bs_num, user_num, self.paths)))
        theta_azi = self.user2bs_azi.T[:, :, np.newaxis] + np.random.uniform(-self.angle_spread, self.angle_spread, size=(bs_num, user_num, self.paths))
        bs_arr_resp_azi = 1 / np.sqrt(self.bs_atn) \
            * np.exp(1j*math.pi*np.arange(self.bs_atn)[np.newaxis, np.newaxis, np.newaxis]*np.sin(theta_azi)[:, :, :, np.newaxis])
        bs2user_csi = np.sqrt(pl[:, :, np.newaxis]/self.paths) * np.sum(factor[:, :, :, np.newaxis]*bs_arr_resp_azi, axis=2)

        # -------- ris to user csi --------
        pl = self.pl0 / np.linalg.norm(self.ris_loc[:, np.newaxis] - self.user_loc[np.newaxis], axis=-1)**self.pl_exp[1]
        factor = np.exp(1j*np.random.uniform(-math.pi, math.pi, size=(ris_num, user_num, self.paths)))
        theta_ele = self.user2ris_ele.T[:, :, np.newaxis] + np.random.uniform(-self.angle_spread, self.angle_spread, size=(ris_num, user_num, self.paths))
        theta_azi = self.user2ris_azi.T[:, :, np.newaxis] + np.random.uniform(-self.angle_spread, self.angle_spread, size=(ris_num, user_num, self.paths))
        ris_arr_resp_ele = 1 / np.sqrt(self.ris_atn[1]) \
            * np.exp(1j*math.pi*np.arange(self.ris_atn[1])[np.newaxis, np.newaxis, np.newaxis]*np.sin(theta_ele)[:, :, :, np.newaxis])
        ris_arr_resp_azi = 1 / np.sqrt(self.ris_atn[0]) \
            * np.exp(1j*math.pi*np.arange(self.ris_atn[0])[np.newaxis, np.newaxis, np.newaxis]*(np.cos(theta_ele)*np.cos(theta_azi))[:, :, :, np.newaxis])
        ris_arr_resp = np.matmul(ris_arr_resp_ele[:, :, :, :, np.newaxis], ris_arr_resp_azi[:, :, :, np.newaxis]).reshape(ris_num, user_num, self.paths, -1)
        ris2user_csi = np.sqrt(pl[:, :, np.newaxis]/self.paths) * np.sum(factor[:, :, :, np.newaxis]*ris_arr_resp, axis=2)

        self.bs2user_csi = self.bs2user_csi*self.rho + np.sqrt(1-self.rho**2)*bs2user_csi if hasattr(self, 'bs2user_csi') else bs2user_csi
        self.ris2user_csi = self.ris2user_csi*self.rho + np.sqrt(1-self.rho**2)*ris2user_csi if hasattr(self, 'ris2user_csi') else ris2user_csi
        return self.bs2user_csi, self.bs2ris_csi, self.ris2user_csi

    def _getRate(self, bs_beam, ris_beam):
        """calculate total transmission rate.
        Parameters:
            bs_beam : (bs_num, user_num, bs_atn)
            ris_beam : (ris_num, ris_atn, ris_atn), the last two dimension is diagnal matrix
        """
        bs_num, _, user_num = self.getCount()
        signal_pw = 1  # represent each user's signal power
        reflect_ch = np.sum(np.matmul(self.ris2user_csi[np.newaxis, :, :, np.newaxis], 
                                      np.matmul(ris_beam[np.newaxis], self.bs2ris_csi)[:, :, np.newaxis]),
                            axis=1)
        combine_ch = self.bs2user_csi[:, :, np.newaxis] + reflect_ch
        rate = 0
        for user_id in range(user_num):
            receive_pws = np.abs(np.sum(np.matmul(combine_ch[:, [user_id]], bs_beam[:, :, :, np.newaxis]).squeeze((2, 3)), axis=0))**2
            idxs = [k for k in range(user_num) if k!= user_id]
            rate += np.log2(1 + signal_pw*receive_pws[user_id]/(signal_pw*np.sum(receive_pws[idxs])+self.noise))
        return rate

    def _csi2state(self):
        obs = np.concatenate((np.real(self.bs2user_csi).reshape(-1), np.imag(self.bs2user_csi).reshape(-1),
                              np.real(self.bs2ris_csi).reshape(-1), np.imag(self.bs2ris_csi).reshape(-1),
                              np.real(self.ris2user_csi).reshape(-1), np.imag(self.ris2user_csi).reshape(-1)))
        return obs

    def getCount(self, idx=None):
        """get the number of base stations, RIS and users"""
        if idx is None:
            return len(self.bs_loc), len(self.ris_loc), len(self.user_loc)
        nums = (len(self.bs_loc), len(self.ris_loc), len(self.user_loc))
        if np.isscalar(idx):
            return nums[idx]
        else:
            return [nums[i] for i in idx]

    def reset(self, seed):
        np.random.seed(seed)
        try:
            del self.bs2user_csi, self.bs2ris_csi, self.ris2user_csi
        except AttributeError:
            pass
        self._changeCSI()
        return self._csi2state()
        
    def step(self, bs_beam, ris_beam):
        """let the state of environment move forward"""
        rew = self._getRate(bs_beam, ris_beam)
        self._changeCSI()
        next_obs = self._csi2state()
        return next_obs, rew

if __name__=='__main__':
    from utils import CodeBook
    np.random.seed(66)
    sns.set()

    env = Environment(30)
    env.reset(2020)

    # -------- show locations --------
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, projection='3d')
    ax.scatter(env.bs_loc[:, 0], env.bs_loc[:, 1], env.bs_loc[:, 2], marker='^', s=80, label='base station')
    ax.scatter(env.ris_loc[:, 0], env.ris_loc[:, 1], env.ris_loc[:, 2], marker='s', s=80, label='ris')
    ax.scatter(env.user_loc[:, 0], env.user_loc[:, 1], env.user_loc[:, 2], marker='x', s=80, label='user')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.suptitle("Loc diagram", y=0.95)
    plt.legend()
    
    # -------- csi & rate demo --------
    bs_num, ris_num, user_num = env.getCount()
    bs_cbook = CodeBook(16, env.bs_atn)
    ris_azi_cbook = CodeBook(10, env.ris_atn[0], phases=4, scale=True)
    ris_ele_cbook = CodeBook(16, env.ris_atn[1], phases=4, scale=True)
    # base station beamforming
    bs_beam = np.zeros((bs_num, user_num, env.bs_atn), dtype=np.complex64)
    for i in range(bs_num):
        for j in range(user_num):
            bs_beam[i, j] = np.sqrt(env.max_power/user_num) * bs_cbook.book[np.random.randint(bs_cbook.codes)]
    # ris beamforming
    ris_beam = np.zeros((ris_num, np.prod(env.ris_atn)), dtype=np.complex64)
    for i in range(ris_num):
        ris_beam[i] = np.kron(ris_ele_cbook.book[np.random.randint(ris_ele_cbook.codes)], 
                              ris_azi_cbook.book[np.random.randint(ris_azi_cbook.codes)])
    ris_beam_expand = np.zeros(ris_beam.shape+ris_beam.shape[-1:], dtype=ris_beam.dtype)
    diagonals = ris_beam_expand.diagonal(axis1=-2, axis2=-1)
    diagonals.setflags(write=True)
    diagonals[:] = ris_beam.copy()
    
    print("demo achievable rate")
    _, rew = env.step(bs_beam, ris_beam_expand)
    _, rew2 = env.step(bs_beam, ris_beam_expand)
    print(f"rate 1: {rew}\nrate 2: {rew2}")

    plt.show()
