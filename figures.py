import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import mpl_toolkits.mplot3d.axes3d as axes3d
import seaborn as sns
sns.set()
from utils import CodeBook

def plot2D(cbook, ax):
    """unifomr linear array beamforming pattern plotting

    Parameters:
        cbook (np.array): the 2-D codebook
        ax (axes.SubplotBase)
    """
    codes = cbook.generate()
    for code in codes:
        thetas = np.linspace(0, 2*math.pi, num=100, endpoint=False)
        array_response = np.array(
            [1/math.sqrt(cbook.antennas)*np.exp(1j*math.pi*np.arange(cbook.antennas)*math.cos(theta))
            for theta in thetas])
        value = abs(np.sum(code*array_response, axis=1))
        ax.plot(thetas, value)

def plot3D(cbook_h, cbook_v, choice, ax):
    """uniform rectangular array beamforming pattern plotting

    Parameters:
        cbook_h (np.array): horizontal beamforming codebook
        cbook_v (np.array): vertical beamforming codebook
        choice (tuple): beamforming code choice
        ax (axes.SubplotBase)
    """
    cbook_h.scale()
    cbook_v.scale()
    code_h, code_v = cbook_h.book[choice[0]], cbook_v.book[choice[1]]

    thetas = np.linspace(0, 2*math.pi, 100, endpoint=False)
    betas = np.linspace(-math.pi/2, math.pi/2, 50)
    values = np.zeros((len(betas), len(thetas)))
    for i, beta in enumerate(betas):
        for j, theta in enumerate(thetas):
            array_response_v = 1/math.sqrt(cbook_v.antennas)*np.exp(1j*math.pi*np.arange(cbook_v.antennas)*math.sin(beta))
            array_response_h = 1/math.sqrt(cbook_h.antennas)*np.exp(1j*math.pi*np.arange(cbook_h.antennas)*math.cos(theta)*math.cos(beta))
            values[i, j] = abs(np.sum(np.kron(code_v*array_response_v, code_h*array_response_h)))

    THETA, BETA = np.meshgrid(thetas, betas)
    X = values * np.cos(BETA) * np.sin(THETA)
    Y = values * np.cos(BETA) * np.cos(THETA)
    Z = values * np.sin(BETA)
    picture = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('jet'),
        linewidth=0, antialiased=False, alpha=0.5)
    plt.colorbar(picture, pad=0.1)

if __name__=='__main__':
    cbook_bs = CodeBook(10, 4)
    cbook_ris_h, cbook_ris_v = CodeBook(3, 8, phases=4), CodeBook(3, 4, phases=4)
    # -------- show beamforming pattern --------
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    plot2D(cbook_bs, ax)
    plt.title("2-D beamforming with %ddirections"%cbook_bs.codes)
    
    plt.figure(figsize=(12, 12))
    for id_h in range(cbook_ris_h.codes):
        for id_v in range(cbook_ris_v.codes):
            ax = plt.subplot(cbook_ris_h.codes, cbook_ris_v.codes, id_h*cbook_ris_h.codes+id_v+1, projection='3d')
            plot3D(cbook_ris_h, cbook_ris_v, (id_h, id_v), ax)
    plt.suptitle(f"3-D beamforming with directions: h-{cbook_ris_h.codes} v-{cbook_ris_v.codes}", y=0.95)
    plt.show()
