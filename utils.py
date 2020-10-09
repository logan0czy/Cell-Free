import math
import random

import numpy as np


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

    def _element(self, num_code, num_antenna):
        temp1 = (num_code+self.codes/2) % self.codes
        temp2 = math.floor(num_antenna*temp1/(self.codes/self.phases))
        value = 1 / math.sqrt(self.antennas) * np.exp(1j*2*math.pi/self.phases*temp2)
        return value
    
    def generate(self):
        """
        Generate the codebook of shape (self.codes, self.antennas)
        """
        if hasattr(self, 'book'):
            return self.book
        book = [[self._element(num_code, num_antenna) for num_antenna in range(self.antennas)]
                for num_code in range(self.codes)
            ]
        self.book = np.array(book)
        return self.book

    def scale(self):
        if self.scaled:
            return
        if not hasattr(self, 'book'):
            self.generate()
        self.book = self.book * math.sqrt(self.antennas)
        self.scaled = True


class Memory():
  """store (s, a, r, s_) transitions."""
  def __init__(self, buffer_size):
    self.buffer = deque(maxlen=buffer_size)

  def push(self, s, a, r, s_):
    """put new trainsitions into buffer.
    Parameters:
    s: current state, 1-D np.array
    a: action, 1-D np.array
    r: one step reward, scalar
    s_: next state, 1-D np.array
    """
    self.buffer.append((s, a, r, s_))
  
  def getBatch(self, batch_size):
    """randomly select batches of history transitions.
    Parameters: 
    batch_size (int)
    
    Return:
    state: (batch, state_size)
    action: (batch, action_size)
    reward: (batch, )
    state_next: (batch, state_size)
    """
    idxes = random.sample(range(len(self.buffer)), batch_size)
    transitions = []
    for idx in idxes:
      transitions.append(self.buffer[idx])
    state = []
    state_next = []
    action = []
    reward = []
    for s, a, r, s_ in transitions:
      state.append(s)
      state_next.append(s_)
      action.append(a)
      reward.append(r)
    state = np.stack(state)
    state_next = np.stack(state_next)
    action = np.stack(action)
    reward = np.array(reward)

    return state, action, reward, state_next

class Environment():
  """communication environment"""
  def __init__(self, power_max: float):
    """
    (Note: default settings
    the antenna's multual distance is half of the wave length,
    base station and RIS are at the same height,
    near field distance is 1m,
    )
    """
    # ---- space parameters ----
    self.antenna = (4, 8, 4)  # number of antennas, 0-base station, 1-RIS horizontal, 2-RIS vertical
    self.M = self.antenna[0]  # number of BS' antenna
    self.N = self.antenna[1]*self.antenna[2]  # number of RIS total elements
    self.radius = 50
    self.bs_loc = np.array([[-self.radius, 0, 5], [self.radius, 0, 5]])
    self.ris_loc = np.array([[0, -self.radius, 5], [0, self.radius, 5]])

    outer = [-self.radius/sqrt(2), self.radius/sqrt(2)]
    inner = [-self.radius/2, self.radius/2]
    user_outer = [[x, y, 0] for x in outer for y in outer]
    user_inner = [[x, 0, 0] for x in inner] + [[0, y, 0] for y in inner]
    self.user_loc = np.array(user_outer + user_inner)

    # ---- communication parameters ----
    self.power_max = power_max
    self.theta_aod, self.theta_aoah, self.theta_aoav = atan(1), atan(1), 0
    self.pl0 = -30  # base path loss, -30dB
    self.pl_exp = (2, 2.8, 2.8)  # path loss exponent, (bs-ris, ris-user, bs-user) respectively
    self.rice_fac = 10  # rice shadowing factor for LOS component
    self.racian_std = 1  # std for racian shadowing
    self.noise = 10**(-80/10)  # power of noise
    self.rho = 0.64  # channel state transition proportion; past 0.64

    # ---- state and action info ----
    n_bs, n_ris, n_user = self.getCount()
    self.act_space = {'dim1': n_bs*n_user*2*self.M,
              'dim2': n_ris*2*self.N,
              'bound1': sqrt(0.25*self.power_max),
              'bound2': 1}
    self.state_size = (
      n_bs*n_ris*2*self.N*self.M + n_ris*n_user*2*self.N + n_bs*n_user*2*self.M
    )
      # + self.act_space['dim1'] + self.act_space['dim2'] + n_bs*n_user + n_bs*n_user*2
    # )

  def reset(self):
    np.random.seed(2020)
    # generate fake actions
    # act1 = np.random.uniform(-0.25*self.act_space['bound1'], 0.25*self.act_space['bound1'], self.act_space['dim1'])
    # act2 = np.random.uniform(-0.25*self.act_space['bound2'], 0.25*self.act_space['bound2'], self.act_space['dim2'])
    # act2_vec = act2.reshape(-1, 2, self.N)
    # act2_vec = act2_vec / (np.sqrt(np.sum(act2_vec**2, axis=1, keepdims=True))+1e-8)
    # act2 = act2_vec.reshape(-1)
    # generate initial environment state
    csi = self.getCSI(reset_csi=True)
    state = self.getState(csi, None, None)
    return csi, state

  def getCount(self):
    """get the number of base stations, RIS and users.
    :return (tuple): corresponging to (n_bs, n_ris, n_user) respectively
    """
    return len(self.bs_loc), len(self.ris_loc), len(self.user_loc)

  def smallScaleCSI(self):
    """the small-scale Rayleigh fading component of csi.
    Return:
    G_tuda (np.array): BS-RIS csi, (BxR, 2, N, M)
    F_tuda (np.array): RIS-user csi, (RxK, 2, N)
    H_tuda (np.array): BS-user csi, (BxK, 2, M)
    """
    G_tuda, F_tuda, H_tuda = [], [], []

    # ---- get CSI of G_tuda ----
    for bs_id, bs_loc in enumerate(self.bs_loc):
      for ris_id, ris_loc in enumerate(self.ris_loc):
        g_tuda = (self.racian_std/sqrt(2)*np.random.randn(self.N, self.M)
              + 1j*self.racian_std/sqrt(2)*np.random.randn(self.N, self.M))
        g_tuda_real = np.real(g_tuda)
        g_tuda_imag = np.imag(g_tuda)
        G_tuda.append(np.stack([g_tuda_real, g_tuda_imag]))
    G_tuda = np.stack(G_tuda)

    # ---- get CSI of F_tuda ----
    for ris_id, ris_loc in enumerate(self.ris_loc):
      for user_id, user_loc in enumerate(self.user_loc):
        f_tuda = (self.racian_std/sqrt(2)*np.random.randn(self.N)
              + 1j*self.racian_std/sqrt(2)*np.random.randn(self.N))
        f_tuda_real = np.real(f_tuda)
        f_tuda_imag = np.imag(f_tuda)
        F_tuda.append(np.stack([f_tuda_real, f_tuda_imag]))
    F_tuda = np.stack(F_tuda)

    # ---- get CSI of H_tuda ----
    for bs_id, bs_loc in enumerate(self.bs_loc):
      for user_id, user_loc in enumerate(self.user_loc):
        h_tuda = (self.racian_std/sqrt(2)*np.random.randn(self.M)
              + 1j*self.racian_std/sqrt(2)*np.random.randn(self.M))
        h_tuda_real = np.real(h_tuda)
        h_tuda_imag = np.imag(h_tuda)
        H_tuda.append(np.stack([h_tuda_real, h_tuda_imag]))
    H_tuda = np.stack(H_tuda)

    return G_tuda, F_tuda, H_tuda

  def getCSI(self, reset_csi=False):
    """generate current csi following Jakes fading model.
    Return:
    G (np.array): BS-RIS csi, (BxR, 2, N, M)
    F (np.array): RIS-user csi, (RxK, 2, N)
    H (np.array): BS-user csi, (BxK, 2, M)
    """
    if not hasattr(self, 'G_bar'):
      # ---- get CSI of G_bar ----
      G_bar = []
      phi_aoav = np.array([np.exp(-1j*k*np.pi*sin(self.theta_aoav)) for k in range(self.antenna[2])])[:, np.newaxis]  # (N_v, 1)
      for bs_id, bs_loc in enumerate(self.bs_loc):
        for ris_id, ris_loc in enumerate(self.ris_loc):
          if ris_id == 0:
            phi_aod = np.array([np.exp(1j*k*np.pi*sin(self.theta_aod)) for k in range(self.M)])[np.newaxis, :]  # (1, M)
          else:
            phi_aod = np.array([np.exp(-1j*k*np.pi*sin(self.theta_aod)) for k in range(self.M)])[np.newaxis, :]  # (1, M)
          if bs_id == 0:
            phi_aoah = np.array([np.exp(-1j*k*np.pi*sin(self.theta_aoah)) for k in range(self.antenna[1])])[np.newaxis, :]  # (1, N_h)
          else:
            phi_aoah = np.array([np.exp(1j*k*np.pi*sin(self.theta_aoah)) for k in range(self.antenna[1])])[np.newaxis, :]  # (1, N_h)

          g_bar = np.matmul(phi_aoav, phi_aoah).reshape(-1, 1) @ phi_aod  # (N, M)
          g_bar_real = np.real(g_bar)
          g_bar_imag = np.imag(g_bar)
          G_bar.append(np.stack([g_bar_real, g_bar_imag]))
      G_bar = np.stack(G_bar)
      self.G_bar = G_bar  # (BxR, 2, N, M)

    if not hasattr(self, 'G_pl'):
      # ---- G_pl ----
      G_pl = []
      for bs_id, bs_loc in enumerate(self.bs_loc):
        for ris_id, ris_loc in enumerate(self.ris_loc):
          distance = np.sqrt(np.sum((bs_loc-ris_loc)**2))
          pathloss = 10**((self.pl0 - 10*self.pl_exp[0]*log10(distance)) / 10)
          G_pl.append(pathloss)
      self.G_pl = np.reshape(G_pl, (-1, 1, 1, 1))

      # ---- F_pl ----
      F_pl = []
      for ris_id, ris_loc in enumerate(self.ris_loc):
        for user_id, user_loc in enumerate(self.user_loc):
          distance = np.sqrt(np.sum((ris_loc-user_loc)**2))
          pathloss = 10**((self.pl0-10*self.pl_exp[1]*log10(distance)) / 10)
          F_pl.append(pathloss)
      self.F_pl = np.reshape(F_pl, (-1, 1, 1))

      # ---- H_pl ----
      H_pl = []
      for bs_id, bs_loc in enumerate(self.bs_loc):
        for user_id, user_loc in enumerate(self.user_loc):
          distance = np.sqrt(np.sum((bs_loc-user_loc)**2))
          pathloss = 10**((self.pl0-10*self.pl_exp[2]*log10(distance)) / 10)
          H_pl.append(pathloss)
      self.H_pl = np.reshape(H_pl, (-1, 1, 1))

    if not hasattr(self, 'G_tuda') or reset_csi:
      self.G_tuda, self.F_tuda, self.H_tuda = self.smallScaleCSI()
    elif self.rho != 1:
      G_tuda, F_tuda, H_tuda = self.smallScaleCSI()
      self.G_tuda = self.rho*self.G_tuda + sqrt(1-self.rho**2)*G_tuda
      self.F_tuda = self.rho*self.F_tuda + sqrt(1-self.rho**2)*F_tuda
      self.H_tuda = self.rho*self.H_tuda + sqrt(1-self.rho**2)*H_tuda
    
    G = np.sqrt(self.G_pl)*(sqrt(self.rice_fac/(1+self.rice_fac))*self.G_bar + sqrt(1/(1+self.rice_fac))*self.G_tuda)
    F = np.sqrt(self.F_pl)*self.F_tuda
    H = np.sqrt(self.H_pl)*self.H_tuda

    return G, F, H

  def actTrans(self, act1, act2):
    """transform action into vector W and Phi, the W is beamforming vector,
    and the Phi is reflecting vector.
    Parameters:
    act1 (np.array): shape of (dim1, ), from policy network
    act2 (np.array): shape of (dim2, ), from policy network

    Return:
    W (np.array): shape of (BxK, 2, M)
    Phi (np.array): shape of (R, 2, N, N), the last two dimension
             corresponds to diagonal matrix
    """
    assert act1.shape[0]==self.act_space['dim1'] and act2.shape[0]==self.act_space['dim2']
    n_bs, n_ris, n_user = self.getCount()
    W = act1.reshape(-1, 2, self.M)
    Phi_vec = act2.reshape(-1, 2, self.N)

    Phi = []
    for phi_vec_i in Phi_vec:
      phi_i = []
      for vec in phi_vec_i:
        phi_i.append(np.diag(vec))
      phi_i = np.stack(phi_i)
      Phi.append(phi_i)
    Phi = np.stack(Phi)

    return W, Phi

  def toComplex(self, arr):
    """transform array into complex form.
    Parameters:
    arr (np.array): with real and imaginary part splitted, shape of (dim-0, 2, *)

    Return:
    arr_complex (np.array): shape of (dim-0, *)
    """
    assert arr.shape[1] == 2
    arr_complex = []
    for item in arr:
      arr_complex.append(item[0] + 1j*item[1]) 
    arr_complex = np.stack(arr_complex)
    return arr_complex

  def getRate(self, csi, act1, act2):
    """the average rate corresponding to current csi.
    Parameters:
    csi (tuple): the current CSI, including (G, F, H)
             G (np.array): BS-RIS csi, (BxR, 2, N, M)
             F (np.array): RIS-user csi, (RxK, 2, N)
             H (np.array): BS-user csi, (BxK, 2, M)
    act1 (np.array): shape of (dim1, )
    act2 (np.array): shape of (dim2, )

    Return:
    avg_rate (float)
    """
    n_bs, n_ris, n_user = self.getCount()
    G, F, H = self.toComplex(csi[0]), self.toComplex(csi[1]), self.toComplex(csi[2])
    W, Phi = self.actTrans(act1, act2)
    W, Phi = self.toComplex(W), self.toComplex(Phi)
    G = G.reshape(n_bs, n_ris, self.N, self.M)  # (B, R, N, M)
    W = W.reshape(n_bs, n_user, self.M, 1)  # (B, K, M, 1)

    rate = []
    for user_id in range(n_user):
      # caculate the combination channel from the base stations to current user.
      direct_link = H[[user_id+bs_id*n_user for bs_id in range(n_bs)]][:, np.newaxis]  # (B, 1, M)

      F_user = F[[user_id+ris_id*n_user for ris_id in range(n_ris)]][:, np.newaxis]  # (R, 1, N)
      F_phi_prod = np.matmul(F_user, Phi)[np.newaxis, :]  # (1, R, 1, N)
      reflect_link = np.matmul(F_phi_prod, G).squeeze(2)  # (B, R, M)
      reflect_link = np.sum(reflect_link, axis=1, keepdims=True)  # (B, 1, M)

      link_bs_user = direct_link + reflect_link

      # calculate communication rate
      recieve_signal = np.matmul(link_bs_user[:, np.newaxis], W).squeeze(2)  # (B, K)
      recieve_signal = np.sum(recieve_signal, axis=0, keepdims=False)  # (K, )
      interference = np.sum(abs(recieve_signal[[i for i in range(n_user) if i != user_id]])**2)
      SINR = abs(recieve_signal[user_id])**2 / (self.noise + interference) 
      rate.append(log2(1+SINR))

    avg_rate = np.mean(rate)
    return avg_rate

  def getState(self, csi, act1, act2):
    """generate current state"""
    # n_bs, n_ris, n_user = self.getCount()    
    # G, F, H = self.toComplex(csi[0]), self.toComplex(csi[1]), self.toComplex(csi[2])
    # W, Phi = self.actTrans(act1, act2)
    # W, Phi = self.toComplex(W), self.toComplex(Phi)
    # G = G.reshape(n_bs, n_ris, self.N, self.M)  # (B, R, N, M)
    # W = W.reshape(n_bs, n_user, self.M, 1)  # (B, K, M, 1)

    # target_signal_powers, interf_signal_powers = [], []
    # for user_id in range(n_user):
    #   # caculate the combination channel from the base stations to current user.
    #   direct_link = H[[user_id+bs_id*n_user for bs_id in range(n_bs)]][:, np.newaxis]  # (B, 1, M)

    #   F_user = F[[user_id+ris_id*n_user for ris_id in range(n_ris)]][:, np.newaxis]  # (R, 1, N)
    #   F_phi_prod = np.matmul(F_user, Phi)[np.newaxis, :]  # (1, R, 1, N)
    #   reflect_link = np.matmul(F_phi_prod, G).squeeze(2)  # (B, R, M)
    #   reflect_link = np.sum(reflect_link, axis=1, keepdims=True)  # (B, 1, M)

    #   link_bs_user = direct_link + reflect_link

    #   # calculate signal powers
    #   recieve_signal = np.matmul(link_bs_user[:, np.newaxis], W).squeeze(2)  # (B, K)
    #   signal_powers = abs(recieve_signal)**2
    #   target = signal_powers[:, user_id]
    #   interf = np.sum(signal_powers[:, [i for i in range(n_user) if i!=user_id]], axis=1, keepdims=False)

    #   target_signal_powers.extend(target)
    #   interf_signal_powers.extend(interf)
    # target_signal_powers = np.array(target_signal_powers)
    # interf_signal_powers = np.array(interf_signal_powers)

    # trans_powers = np.sum(abs(W)**2, axis=(2, 3), keepdims=False)

    # return np.concatenate((csi[0].reshape(-1), csi[1].reshape(-1), csi[2].reshape(-1),
    #             act1, act2, trans_powers.reshape(-1), target_signal_powers.reshape(-1), interf_signal_powers.reshape(-1)
    #             ))

    return np.concatenate((csi[0].reshape(-1), csi[1].reshape(-1), csi[2].reshape(-1)))

