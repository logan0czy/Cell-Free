import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import mpl_toolkits.mplot3d.axes3d as axes3d
import seaborn as sns
sns.set()
from utils import CodeBook

def plot2D(cbook):
    """linear uniform array 2-D beamforming plotting"""
    codes = cbook.generate()
    plt.figure(figsize=(8, 8))
    for code in codes:
        thetas = np.linspace(0, 2*math.pi, num=100, endpoint=False)
        phase_shift = np.array(
            [1/math.sqrt(cbook.antennas)*np.exp(1j*math.pi*np.arange(cbook.antennas)*math.cos(theta))
            for theta in thetas])
        value = abs(np.sum(code*phase_shift, axis=1))
        plt.polar(thetas, value)
    plt.show()

def plot3D(cbook_h, cbook_v, choice):
    """specifically for RIS 3-D beamforming pattern plotting
    Parameters:
    cbook_h (np.array): horizontal codebook
    cbook_v (np.array): vertical codebook
    choice (tuple): the choice of horizontal and vertical codebook
    """
    cbook_h.scale()
    cbook_v.scale()
    code_v, code_h = cbook_v.book[choice[1]], cbook_h.book[choice[0]]

    thetas = np.linspace(0, 2*math.pi, 100, endpoint=False)
    betas = np.linspace(-math.pi/2, math.pi/2, 50)
    values = np.zeros((len(betas), len(thetas)))
    for i, beta in enumerate(betas):
        for j, theta in enumerate(thetas):
            phase_shift_v = 1/math.sqrt(cbook_v.antennas)*np.exp(1j*math.pi*np.arange(cbook_v.antennas)*math.sin(beta))
            phase_shift_h = 1/math.sqrt(cbook_h.antennas)*np.exp(1j*math.pi*np.arange(cbook_h.antennas)*math.cos(theta)*math.cos(beta))
            values[i, j] = abs(np.sum(np.kron(code_v*phase_shift_v, code_h*phase_shift_h)))

    plt.figure(figsize=(8, 8))
    ax = plt.axes(projection='3d')
    THETA, BETA = np.meshgrid(thetas, betas)
    X = values * np.cos(BETA) * np.sin(THETA)
    Y = values * np.cos(BETA) * np.cos(THETA)
    Z = values * np.sin(BETA)
    plot = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('jet'),
        linewidth=0, antialiased=False, alpha=0.5)
    plt.colorbar(plot)
    plt.show()

if __name__=='__main__':
    cbook_bs = CodeBook(10, 4)
    cbook_ris_h, cbook_ris_v = CodeBook(4, 8, phases=4), CodeBook(4, 4, phases=4)
    plot2D(cbook_bs)
    plot3D(cbook_ris_h, cbook_ris_v, (2, 4))
