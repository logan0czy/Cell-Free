"""
Show the antenna array response shape in 2D/3D beamforming mode.
"""
import math
import os.path as osp
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits import mplot3d
import mpl_toolkits.mplot3d.axes3d as axes3d
from comms import CodeBook


def plot2D(ax, cbook, antenna_num):
    """Uniform linear array beamforming patterns"""
    thetas = np.linspace(0, 2*math.pi, num=200, endpoint=False)
    for code in cbook.book:
        array_response = np.array(
            [1/math.sqrt(antenna_num)*np.exp(1j*math.pi*np.arange(antenna_num)*math.sin(theta))
            for theta in thetas])
        value = abs(np.sum(code*array_response, axis=1))
        ax.plot(thetas, value)

def subplot3D(ax, code_azi, code_ele, ant_azi_num, ant_ele_num):
    """A beamforming pattern of uniform rectangular array"""
    thetas = np.linspace(0, 2*math.pi, 100, endpoint=False)
    betas = np.linspace(-math.pi/2, math.pi/2, 50)
    values = np.zeros((len(betas), len(thetas)), dtype=np.float32)
    for i, beta in enumerate(betas):
        for j, theta in enumerate(thetas):
            array_response_ele = 1/math.sqrt(ant_ele_num)*np.exp(1j*math.pi*np.arange(ant_ele_num)*math.sin(beta))
            array_response_azi = 1/math.sqrt(ant_azi_num)*np.exp(1j*math.pi*np.arange(ant_azi_num)*math.sin(theta)*math.cos(beta))
            values[i, j] = abs(np.sum(np.kron(code_ele*array_response_ele, code_azi*array_response_azi)))

    THETA, BETA = np.meshgrid(thetas, betas)
    X = values * np.cos(BETA) * np.sin(THETA)
    Y = values * np.cos(BETA) * np.cos(THETA)
    Z = values * np.sin(BETA)
    my_color = cm.jet(values / np.amax(values))
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=my_color, cmap=plt.get_cmap('jet'),
        linewidth=0, antialiased=False, alpha=0.5)
    ax.set_axis_off()
    # ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    # plt.colorbar(surf, pad=0.05)

if __name__=='__main__':
    fig_dpi = 300
    bs_cbook = CodeBook(8, 4, duplicated=False)
    ris_azi_cbook = CodeBook(3, 4, phases=4, scale=True, duplicated=False)
    ris_ele_cbook = CodeBook(3, 4, phases=4, scale=True, duplicated=False)
    # -------- show beamforming pattern --------
    sns.set_theme(context='paper', style='whitegrid', palette='bright', font_scale=2)
    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    plot2D(ax, bs_cbook, 4)
    fig.savefig(osp.join('figures', 'ULA_resp.pdf'), dpi=fig_dpi, bbox_inches='tight')
    
    fig = plt.figure(figsize=(8, 8))
    for id_h in range(ris_azi_cbook.codes):
        for id_v in range(ris_ele_cbook.codes):
            ax = plt.subplot(ris_azi_cbook.codes, ris_ele_cbook.codes, id_h*ris_azi_cbook.codes+id_v+1, projection='3d')
            subplot3D(ax, ris_azi_cbook.book[id_h], ris_ele_cbook.book[id_v], 4, 4)
            # ax.set_title(f"{id_h}-{id_v}")
    # plt.subplots_adjust(wspace=0.2, hspace=0.2)
    fig.tight_layout()
    fig.savefig(osp.join('figures', 'URA_resp.pdf'), dpi=fig_dpi, bbox_inches='tight')
    # plt.show()
