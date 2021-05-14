from os import path as osp
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# The color plot sequence is : blue, yellow, green, red
sns.set_theme(context='paper', style='whitegrid', palette='bright', font_scale=1.2)
fig_dpi = 600
window = 400 

# ---- different correlation level ----
fig = plt.figure()
ax = plt.gca()

x = [0.1*idx for idx in range(11)]
# power results
for power, slog in zip([10, 20, 30], ['^-', 's--', 'x:']):
    color = next(ax._get_lines.prop_cycler)['color']
    res = np.load(osp.join('Res', 'rhos', 'power%d.npy'%power))
    statis_mean = np.mean(res, axis=0)
    statis_std = np.std(res, axis=0)
    ax.plot(x, statis_mean, slog, color=color, label='$P_{max}$ = %d dBm'%power)
    ax.fill_between(x, statis_mean+statis_std, statis_mean-statis_std, color=color, alpha=0.2, linewidth=0)

ax.set_xlabel(r'$\rho$')
ax.set_ylabel('Sum Rate (bps/Hz)')
# ax.set_title('The performance of TD3 with delayed CSI')
plt.legend(loc='lower left')
sns.despine()

# plt.show()
# exit()
fig.savefig(osp.join('figures', 'CSI_rhos.pdf'), dpi=fig_dpi, bbox_inches='tight')
