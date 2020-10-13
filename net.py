"""
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

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        sizes = [obs_dim] + [*hidden_sizes] + [act_dim]
        self.L = genLayers(sizes, activation, nn.Tanh)
        self.act_limit = act_limit

    def forward(self, obs):
        # Return output from network scaled to action limits.
        return self.act_limit * self.L(obs)


class Actor(nn.Module):
  """policy to decide action corresponding to current state."""
  def __init__(
      self, 
      state_size,
      act_size,
      hidden_size=None, 
      n_hiddens=None,
      max_act=(1, 1)
  ):
    """
    Parameters:
    act_size (tuple): size of act1 and act2
    max_act (tuple): bound of act1 and act2
    """
    super(Actor, self).__init__()
    self.dropout_rate = 0
    self.max_act = max_act

    # ---- linear layer ----
    # self.hiddens = nn.ModuleList([nn.Sequential(nn.Linear(hidden_size, hidden_size),
    #                        nn.BatchNorm1d(hidden_size),
    #                        nn.GELU(),
    #                        nn.Dropout(self.dropout_rate))
    #                 for i in range(n_hiddens-1)])
    # self.hiddens.insert(0, nn.Sequential(nn.Linear(state_size, hidden_size),
    #                    nn.BatchNorm1d(hidden_size),
    #                    nn.GELU(),
    #                    nn.Dropout(self.dropout_rate)))
    self.hiddens = nn.ModuleList(
      [
        nn.Sequential(nn.Linear(state_size, 2048), nn.BatchNorm1d(2048), nn.GELU()),
        nn.Sequential(nn.Linear(2048, 1024), nn.BatchNorm1d(1024), nn.GELU()),
        nn.Sequential(nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.GELU()),
        nn.Sequential(nn.Linear(512, 256), nn.BatchNorm1d(256), nn.GELU()),
        nn.Sequential(nn.Linear(256, 512), nn.BatchNorm1d(512), nn.GELU()),
        #nn.Sequential(nn.Linear(512, 512), nn.BatchNorm1d(512), nn.GELU()),
        #nn.Sequential(nn.Linear(512, 512), nn.BatchNorm1d(512), nn.GELU())
      ]
    )
    self.output1 = nn.Linear(512, act_size[0])
    self.output2 = nn.Linear(512, act_size[1])

    # ---- batchnorm layer ----
    self.bn_in = nn.BatchNorm1d(state_size)

  def forward(self, state):
    """
    Parameters:
    state (tensor): input state of shape (batch, state_size).

    Return:
    act1 (tensor): action of shape (batch, act_size[0]).
    act2 (tensor): action of shape (batch, act_size[1]).
    """
    x = self.bn_in(state)
    for layer in self.hiddens:
      x = layer(x)
      
    act1 = self.max_act[0] * torch.tanh(self.output1(x))
    act2 = self.max_act[1] * torch.tanh(self.output2(x))
    return act1, act2
  
  def init(self):
    """initial module parameters."""
    nn.init.xavier_normal_(self.output1.weight, gain=nn.init.calculate_gain('tanh'))
    nn.init.xavier_normal_(self.output2.weight, gain=nn.init.calculate_gain('tanh'))
    nn.init.zeros_(self.output1.bias)
    nn.init.zeros_(self.output2.bias)


class SubCritic(nn.Module):
  """predict Q value according to current state and action."""
  def __init__(
      self, 
      state_size, 
      act_size, 
      hidden_size=None, 
      n_hiddens=None
  ):
    """
    Parameters:
    state_size (int)
    act_size (int)
    """
    super(SubCritic, self).__init__()
    self.dropout_rate = 0

    # ---- linear layer ----
    self.state_proj = nn.ModuleList(
      [
        nn.Sequential(nn.Linear(state_size, 512), nn.BatchNorm1d(512), nn.GELU()),
        nn.Sequential(nn.Linear(512, 128), nn.BatchNorm1d(128), nn.GELU())
      ]
    )
    self.act_proj = nn.ModuleList(
      [
        nn.Sequential(nn.Linear(act_size, 128), nn.BatchNorm1d(128), nn.GELU())
      ]
    )
    self.abstract = nn.ModuleList(
      [
        nn.Sequential(nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU())
        # nn.Sequential(nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU())
      ]
    )

    self.output = nn.Linear(128, 1)

    # ---- batchnorm layer ----
    self.state_bn_in = nn.BatchNorm1d(state_size)
    self.act_bn_in = nn.BatchNorm1d(act_size)

  def forward(self, state, act):
    """
    Parameters:
    state (tensor): current environment state, shape (batch, state_size)
    act (tensor): action corresponding to the state, shape (batch, act_size)

    Return:
    Q_val (tensor): Q value of shape (batch, 1)
    """
    state = self.state_bn_in(state)
    act = self.act_bn_in(act)
    for s_layer in self.state_proj:
      state = s_layer(state)
    for a_layer in self.act_proj:
      act = a_layer(act)

    x = torch.cat((state, act), dim=1)

    for layer in self.abstract:
      x = layer(x)
    Q_val = self.output(x)
    return Q_val
  
  def init(self):
    """initialize module parameters."""
    nn.init.xavier_normal_(self.output.weight)
    nn.init.zeros_(self.output.bias)


class Critic(nn.Module):
  """twin critic networks in TD3"""
  def __init__(
      self, 
      state_size, 
      act_size, 
      hidden_size, 
      n_hiddens=2
  ):
    super(Critic, self).__init__()
    self.critic1 = SubCritic(state_size, act_size, hidden_size, n_hiddens)
    self.critic2 = SubCritic(state_size, act_size, hidden_size, n_hiddens)

  def forward(self, state, act):
    """
    Parameters:
    state (tensor): current environment state of shape (batch, state_size)
    act (tensor): action corresponding to the state, shape (batch, act_size)

    Return:
    Q_val (tensor): Q value of shape (batch, 1)
    """
    Q1 = self.critic1(state, act)
    Q2 = self.critic2(state, act)
    Q_val = torch.min(Q1, Q2)
    return Q_val

  def getQ(self, state, action):
    return self.critic1(state, action)
  
  def init(self):
    self.critic1.init()
    self.critic2.init()


class TD3(nn.Module):
  """wrapped actor and critic networks"""
  def __init__(
      self, 
      state_size, 
      act_size, 
      act_hidden_size, 
      critic_hidden_size, 
      n_c_hiddens, 
      n_a_hiddens,
      max_act,
  ):
    """
    Parameters:
    state_size, act_hidden_size, critic_hidden_size -> (int) 
    act_size (tuple): size of act1 and act2
    n_c_hiddens (int): number of critic hidden layers
    n_a_hiddens (int): number of actor hidden layers
    max_act (tuple): bound of act1 and act2
    """
    super(TD3, self).__init__()
    self.actor = Actor(state_size, act_size, act_hidden_size, n_a_hiddens, max_act=max_act)
    self.critic = Critic(state_size, sum(act_size), critic_hidden_size, n_c_hiddens)
    self.actor_target = Actor(state_size, act_size, act_hidden_size, n_a_hiddens, max_act=max_act)
    self.critic_target = Critic(state_size, sum(act_size), critic_hidden_size, n_c_hiddens)

  @property
  def device(self):
    return self.actor.output1.weight.device

  def init(self):
    self.actor.init()
    self.critic.init()
    self.actor_target.load_state_dict(self.actor.state_dict())
    self.critic_target.load_state_dict(self.critic.state_dict())

  def sync(self, sync_rate):
    """syncnorize target and current networks."""
    for param_target, param in zip(self.actor_target.parameters(), self.actor.parameters()):
      param_target.copy_(sync_rate*param + (1-sync_rate)*param_target)
    for param_target, param in zip(self.critic_target.parameters(), self.critic.parameters()):
      param_target.copy_(sync_rate*param + (1-sync_rate)*param_target)

  def save(self, fpath):
    torch.save(self.state_dict(), fpath+'/model.bin')
  
  def load(self, fpath):
    self.load_state_dict(torch.load(fpath+'/model.bin', map_location=lambda storage, loc: storage))

  def act2Trans(self, act2, env):
    """transform action2(reflection matrix) to satisfy the constraint.
    Parameter:
    act2 (tensor): shape is (batch, act2_size)
    env (Environment)

    Returns:
    act2_trans (tensor): transformed action2
    """
    batch_size = act2.size()[0]
    act2_vec = act2.contiguous().view(batch_size, -1, 2, env.N)
    act2_vec = act2_vec / (torch.sqrt(torch.sum(act2_vec**2, dim=2, keepdim=True))+1e-8)
    act2_trans = act2_vec.contiguous().view(batch_size, -1).contiguous()
    return act2_trans

  def act1Check(self, act1, env):
    """check whether action1 doesn't meet the constraint, if so, return the punishment"""
    batch_size = act1.size()[0]
    n_bs, n_ris, n_user = env.getCount()
    act1 = act1.view(batch_size, n_bs, n_user, 2, env.M)
    power_used = torch.sum(act1**2, dim=(2, 3, 4))
    residual = power_used - env.power_max
    punishment = torch.sum(torch.max(residual, torch.tensor(0, dtype=torch.float, device=self.device)))
    return punishment


class OUStrategy():
  """
  This strategy implements the Ornstein-Uhlenbeck process, which adds
  time-correlated noise to the actions taken by the deterministic policy.
  The OU process satisfies the following stochastic differential equation:
  dxt = theta*(mu - xt)*dt + sigma*dWt
  where Wt denotes the Wiener process
  Based on the rllab implementation.

  refered to https://towardsdatascience.com/deep-deterministic-policy-gradients-explained-2d94655a9b7b
  taken from https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
  """

  def __init__(
      self,
      device,
      act_space,
      mu=0,
      theta=0.15,
      max_sigma=(1/20, 1/10),
      min_sigma=None,
      noise_bound=None,
      decay_period=10000,
  ):
    """
    Parameters:
    device: tensor's device
    act_space (dict): contains (dim1, dim2, bound1, bound2)
    max_sigma (tuple): proportion of two actions' bound, w.r.t noise absolute std value
    min_sigma (tuple): same to 'max_sigma'
    noise_bound (tuple): proportion of two actions' bound, w.r.t noise absolute bound
    """
    self.device = device
    self.mu = mu
    self.theta = theta
    self.sigma = max_sigma
    self._max_sigma = max_sigma
    self._min_sigma = min_sigma if min_sigma else max_sigma
    self._decay_period = decay_period
    self.act_space = act_space
    self.noise_bound = noise_bound

    self.state1 = torch.ones(self.act_space['dim1'], dtype=torch.float, device=self.device)*self.mu
    self.state2 = torch.ones(self.act_space['dim2'], dtype=torch.float, device=self.device)*self.mu
    self.reset()

  def reset(self):
    self.state1 = torch.ones(self.act_space['dim1'], dtype=torch.float, device=self.device)*self.mu
    self.state2 = torch.ones(self.act_space['dim2'], dtype=torch.float, device=self.device)*self.mu

  def evolve_state(self):
    x1 = self.state1
    dx1 = self.theta*(self.mu-x1) + self.act_space['bound1']*self.sigma[0]*torch.randn(len(x1), dtype=torch.float, device=self.device)
    self.state1 = x1 + dx1

    x2 = self.state2
    dx2 = self.theta*(self.mu-x2) + self.act_space['bound2']*self.sigma[1]*torch.randn(len(x2), dtype=torch.float, device=self.device)
    self.state2 = x2 + dx2
    return self.state1, self.state2

  def getActFromRaw(self, raw_act1, raw_act2, t):
    """
    Parameters:
    raw_act1 (tensor): shape of (*, dim1)
    raw_act2 (tensor): shape of (*, dim2)
    t (int): time step
    """
    # def decompose(x): 
    #   """decomposes a float into its exponent"""
    #   assert not isnan(x), 'invalid number'
    #   x_exp = '%.2e'%(x)
    #   for p in range(len(x_exp)-1, -1, -1):
    #     if x_exp[p]=='-':
    #       flag = -1.0
    #       break
    #     elif x_exp[p]=='+':
    #       flag = 1.0
    #       break
    #   exponent = flag*int(x_exp[p+1:])
    #   return exponent+1

    ou_state1, ou_state2 = self.evolve_state()
    if self.noise_bound:
      ou_state1 = torch.clamp(ou_state1, -self.noise_bound[0]*self.act_space['bound1'], self.noise_bound[0]*self.act_space['bound1'])
      ou_state2 = torch.clamp(ou_state2, -self.noise_bound[1]*self.act_space['bound2'], self.noise_bound[1]*self.act_space['bound2'])
    sigma1 = self._max_sigma[0] - (self._max_sigma[0]-self._min_sigma[0])*min(1.0, t*1.0/self._decay_period)
    sigma2 = self._max_sigma[1] - (self._max_sigma[1]-self._min_sigma[1])*min(1.0, t*1.0/self._decay_period)
    self.sigma = (sigma1, sigma2)

    # # check shape
    # if len(raw_act1.size)==1:
    #   raw_act1 = raw_act1.unsqueeze(0)
    #   raw_act2 = raw_act2.unsqueeze(0)
    # # add noise
    # exps1 = torch.tensor([decompose(item.item()) for item in torch.median(abs(raw_act1), dim=1)]).unsqueeze(1).to(self.device)
    # exps2 = torch.tensor([decompose(item.item()) for item in torch.median(abs(raw_act2), dim=1)]).unsqueeze(1).to(self.device)
    # act1 = torch.clamp(raw_act1+ou_state1*10**exps1, -self.act_space['bound1'], self.act_space['bound1']).squeeze()
    # act2 = torch.clamp(raw_act2+ou_state2*10**exps2, -self.act_space['bound2'], self.act_space['bound2']).squeeze()

    act1 = torch.clamp(raw_act1+ou_state1, -self.act_space['bound1'], self.act_space['bound1'])
    act2 = torch.clamp(raw_act2+ou_state2, -self.act_space['bound2'], self.act_space['bound2'])

    return act1, act2    


    