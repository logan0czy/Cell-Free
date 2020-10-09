import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from bases import CodeBook

# -------- 2-D beamforming -------- 
cbook_bs = CodeBook(codes=8, antennas=4)
codes_bs = cbook_bs.generate()
plt.figure(figsize=(8, 8))
for code in codes_bs:
    thetas = np.linspace(0, 2*math.pi, num=100, endpoint=False)
    phase_shift = np.array(
        [1/math.sqrt(cbook_bs.antennas)*np.exp(1j*math.pi*np.arange(cbook_bs.antennas)*math.cos(theta))
        for theta in thetas])
    value = abs(np.sum(code*phase_shift, axis=1))
    plt.polar(thetas, value)
plt.show()

# -------- 3-D beamforming --------   
cbook_v = CodeBook(10, 4, 4)
cbook_h = CodeBook(10, 8, 4)
cbook_v.generate()
cbook_h.generate()
cbook_v.book = cbook_v.book * math.sqrt(cbook_v.antennas)
cbook_h.book = cbook_h.book * math.sqrt(cbook_h.antennas)

code_v, code_h = cbook_v.book[0], cbook_h.book[0]
thetas = np.linspace(0, 2*math.pi, 100, endpoint=False)
betas = np.linspace(-math.pi/2, math.pi/2, 50)
values = np.zeros((len(betas), len(thetas)))
for i, beta in enumerate(betas):
    for j, theta in enumerate(thetas):
        phase_shift_v = 1/math.sqrt(cbook_v.antennas)*np.exp(1j*math.pi*np.arange(cbook_v.antennas)*math.sin(beta))
        phase_shift_h = 1/math.sqrt(cbook_h.antennas)*np.exp(1j*math.pi*np.arange(cbook_h.antennas)*math.cos(theta)*math.cos(beta))
        values[i, j] = abs(np.sum(np.kron(code_v*phase_shift_v, code_h*phase_shift_h)))

from mpl_toolkits import mplot3d
import mpl_toolkits.mplot3d.axes3d as axes3d
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
