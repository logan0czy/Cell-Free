from os import path as osp
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# The color plot sequence is : blue, yellow, green, red
sns.set_theme(context='paper', style='whitegrid', palette='bright', font_scale=1.2)
fig_dpi = 600
window = 400 

# ---- optimization results ----
fig = plt.figure()
ax = plt.gca()

x = np.arange(0, 35, 5)

# convex results
color = next(ax._get_lines.prop_cycler)['color']
convex_res = np.array([2.2627, 3.8542, 6.1545, 9.1211, 12.3419, 15.4189, 18.8013])
ax.plot(x, convex_res, 's-', color=color, label='heuristic')

# random results
color = next(ax._get_lines.prop_cycler)['color']
random_res = np.load(osp.join('Resdata', 'opt_compare', 'random.npy'))
ax.plot(x, np.mean(random_res, axis=0), 'x:', color=color, label='random choice')

# subopt results
color = next(ax._get_lines.prop_cycler)['color']
subopts_res = np.load(osp.join('Resdata', 'opt_compare', 'subopts.npy'))
statis_mean = np.mean(subopts_res, axis=0)
statis_std = np.std(subopts_res, axis=0)
ax.plot(x, statis_mean, 'p--', color=color, label='TD3 with equal beam power')
ax.fill_between(x, statis_mean+statis_std, statis_mean-statis_std, color=color, alpha=0.2, linewidth=0)

# opt results
color = next(ax._get_lines.prop_cycler)['color']
opts_res = np.load(osp.join('Resdata', 'opt_compare', 'opts.npy'))
statis_mean = np.mean(opts_res, axis=0)
statis_std = np.std(opts_res, axis=0)
ax.plot(x, statis_mean, 'o-', color=color, label='TD3')
ax.fill_between(x, statis_mean+statis_std, statis_mean-statis_std, color=color, alpha=0.2, linewidth=0)

ax.set_xlabel('Transmission Power $P_{max}$ (dBm)')
ax.set_ylabel('Sum Rate (bps/Hz)')
# ax.set_title('Comparison of different optimization method')
plt.legend(loc='upper left')
sns.despine()
fig.savefig(osp.join('figures', 'opt_res_compare.pdf'), dpi=fig_dpi, bbox_inches='tight')

# ---- training process ----
opts_rew = pd.DataFrame(np.load(osp.join('Resdata', 'opt_compare', 'train_process', 'opts_rew_30.npy')).T, 
    dtype=np.float32).rolling(window, min_periods=window).mean()[::window].to_numpy()
subopts_rew = pd.DataFrame(np.load(osp.join('Resdata', 'opt_compare', 'train_process', 'subopts_rew_30.npy')).T, 
    dtype=np.float32).rolling(window, min_periods=window).mean()[::window].to_numpy()

fig = plt.figure()
ax = plt.gca()

x = window * np.arange(1, opts_rew.shape[0]+1)

color = next(ax._get_lines.prop_cycler)['color']
statis_mean = np.mean(opts_rew, axis=1)
statis_std = np.std(opts_rew, axis=1)
ax.plot(x, statis_mean, '-', color=color, label='TD3')
ax.fill_between(x, statis_mean+statis_std, statis_mean-statis_std, color=color, alpha=0.2, linewidth=0)

color = next(ax._get_lines.prop_cycler)['color']
statis_mean = np.mean(subopts_rew, axis=1)
statis_std = np.std(subopts_rew, axis=1)
ax.plot(x, statis_mean, '-', color=color, label='TD3 with equal beam power')
ax.fill_between(x, statis_mean+statis_std, statis_mean-statis_std, color=color, alpha=0.2, linewidth=0)

ax.set_xlabel('Training Steps')
ax.set_ylabel('Sum Rate (bps/Hz)')
# ax.set_title('Learning curves')
plt.legend(loc='upper left')
sns.despine()
fig.savefig(osp.join('figures', 'learning_process.pdf'), dpi=fig_dpi, bbox_inches='tight')

# ---- different correlation level ----
fig = plt.figure()
ax = plt.gca()

x = [0.1*idx for idx in range(11)]
# power results
for power in [10, 20, 30]:
    color = next(ax._get_lines.prop_cycler)['color']
    res = np.load(osp.join('Resdata', 'delay', 'rhos', 'power%d.npy'%power))
    statis_mean = np.mean(res, axis=0)
    statis_std = np.std(res, axis=0)
    ax.plot(x, statis_mean, 'x:', color=color, label='power=%d'%power)
    ax.fill_between(x, statis_mean+statis_std, statis_mean-statis_std, color=color, alpha=0.2, linewidth=0)

ax.set_xlabel(r'$\rho$')
ax.set_ylabel('Sum Rate (bps/Hz)')
# ax.set_title('The performance of TD3 with delayed CSI')
plt.legend(loc='upper left')
sns.despine()
fig.savefig(osp.join('figures', 'CSI_rhos.pdf'), dpi=fig_dpi, bbox_inches='tight')
