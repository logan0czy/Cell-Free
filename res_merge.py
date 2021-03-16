import os
import os.path as osp
import math
import numpy as np


# ---- load random results from 'process.txt' ----
random_res = []
for seed in range(10):
    main_path = osp.join('Experiment', 'optimal', 'seed%d'%seed)
    sub_p_res = []
    for power in range(0, 35, 5):
        fpath = osp.join(main_path, 'power%d'%power, 'process.txt')
        with open(fpath, 'r') as f:
            f.readline()
            sub_p_res.append(float(f.readline().split()[1]))
    random_res.append(np.array(sub_p_res))
np.save(osp.join('Resdata', 'random'), np.array(random_res))

# ---- generate fixed test channels ----
from comms import Environment
rhos = [x*0.1 for x in range(11)]
for rho in rhos:
    fold = osp.join('Testdata', 'rho%.1f'%rho)
    os.makedirs(fold, exist_ok=True)
    env = Environment(30)
    env.rho = rho
    env.reset(2020)
    bs2user_csi = [env.bs2user_csi]
    bs2ris_csi = [env.bs2ris_csi]
    ris2user_csi = [env.ris2user_csi]
    for i in range(499):
        env._changeCSI()
        bs2user_csi.append(env.bs2user_csi)
        bs2ris_csi.append(env.bs2ris_csi)
        ris2user_csi.append(env.ris2user_csi)
    np.save(osp.join(fold, 'bs2user_csi'), np.array(bs2user_csi))
    np.save(osp.join(fold, 'bs2ris_csi'), np.array(bs2ris_csi))
    np.save(osp.join(fold, 'ris2user_csi'), np.array(ris2user_csi))

# ---- merge result ----
from run import test
save_path = osp.join('Resdata', 'noise_compare')
os.makedirs(save_path, exist_ok=True)
test_fpath = osp.join('Testdata', 'rho_main')

config_path = osp.join('Experiment', 'compare_noise', 'seed%d', 'power%d', 'degree%d')
power = 30
res = []
for i in [0, 1, 5, 6]:
    sub_res = []
    for degree in range(1, 7):
        sub_res.append(test(test_fpath, config_path%(i, power, degree)))
        print(f'seed:{i}, power:{power}, exp:{-degree}, rew:{sub_res[-1]}')
    res.append(np.array(sub_res))
np.save(osp.join(save_path, 'power%d.npy'%power), np.array(res))