"""
Show the antenna array response shape in 2D/3D beamforming mode.
"""


import math

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits import mplot3d
import mpl_toolkits.mplot3d.axes3d as axes3d

from utils import CodeBook


def plot2D(cbook, ax, antennas):
    """unifomr linear array beamforming pattern plotting

    Parameters:
        cbook (CodeBook)
        ax (axes.SubplotBase)
    """
    for code in cbook.codes:
        thetas = np.linspace(0, 2*math.pi, num=100, endpoint=False)
        array_response = np.array(
            [1/math.sqrt(antennas)*np.exp(1j*math.pi*np.arange(antennas)*math.cos(theta))
            for theta in thetas])
        value = abs(np.sum(code*array_response, axis=1))
        ax.plot(thetas, value)

def plot3D(cbook_azi, cbook_ele, choice, ax, antennas_azi, antennas_ele):
    """uniform rectangular array beamforming pattern plotting

    Parameters:
        cbook_azi (CodeBook): azimuth beamforming codebook
        cbook_ele (CodeBook): elevation beamforming codebook
        choice (tuple): beamforming code choice
        ax (axes.SubplotBase)
    """
    code_azi, code_ele = cbook_azi.book[choice[0]], cbook_ele.book[choice[1]]

    thetas = np.linspace(0, 2*math.pi, 100, endpoint=False)
    betas = np.linspace(-math.pi/2, math.pi/2, 50)
    values = np.zeros((len(betas), len(thetas)), dtype=np.float32)
    for i, beta in enumerate(betas):
        for j, theta in enumerate(thetas):
            array_response_ele = 1/math.sqrt(antennas_ele)*np.exp(1j*math.pi*np.arange(antennas_ele)*math.sin(beta))
            array_response_azi = 1/math.sqrt(antennas_azi)*np.exp(1j*math.pi*np.arange(antennas_azi)*math.cos(theta)*math.cos(beta))
            values[i, j] = abs(np.sum(np.kron(code_ele*array_response_ele, code_azi*array_response_azi)))

    THETA, BETA = np.meshgrid(thetas, betas)
    X = values * np.cos(BETA) * np.sin(THETA)
    Y = values * np.cos(BETA) * np.cos(THETA)
    Z = values * np.sin(BETA)
    my_color = cm.jet(values / np.amax(values))
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=my_color, cmap=plt.get_cmap('jet'),
        linewidth=0, antialiased=False, alpha=0.5)
    plt.colorbar(surf, pad=0.1)

if __name__=='__main__':
    sns.set()
    bs_cbook = CodeBook(16, 4, duplicated=Fasle)
    ris_azi_cbook = CodeBook(4, 8, phases=8, scale=True, duplicated=False)
    ris_ele_cbook = CodeBook(4, 4, phases=8, scale=True, duplicated=False)
    # -------- show beamforming pattern --------
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    plot2D(bs_cbook, ax, 4)
    plt.title("2-D beamforming with %ddirections"%bs_cbook.codes)
    plt.legend()
    
    plt.figure(figsize=(12, 12))
    for id_h in range(ris_azi_cbook.codes):
        for id_v in range(ris_ele_cbook.codes):
            ax = plt.subplot(ris_azi_cbook.codes, ris_ele_cbook.codes, id_h*ris_azi_cbook.codes+id_v+1, projection='3d')
            plot3D(ris_azi_cbook, ris_ele_cbook, (id_h, id_v), ax, 8, 4)
            ax.set_title(f"azi id-{id_h}, ele id-{id_v}")
    plt.suptitle(f"3-D beamforming", y=0.95)
    plt.show()
