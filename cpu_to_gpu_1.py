import cmath
import math as M
import time
from numba import cuda, vectorize, guvectorize, jit
from numba import void, uint8, uint32, uint64, int32, int64, float32, float64, f8
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt, animation, offsetbox
from matplotlib.offsetbox import AnchoredText
import matplotlib.cm as cm
from C import C
import cupy as cp
from matplotlib.patches import Circle
import cairo
from sympy import symbols, Eq, solve
# from scipy.fftpack import fft, fftshift
from scipy import signal
from scipy.signal import butter, lfilter, freqz
import torch.fft as fft
import torch

# from vispy import plot as vp

# mpl.use('Agg')
jit(device=True)

s = time.time()
nett_time_sum = 0
np.seterr(divide='ignore', invalid='ignore')


def data_type(data, flag):
    if flag == 1:
        return np.float32(data)
    else:
        return np.float64(data)

#
flag = 1
data_type1 = np.float32

cc = C()

IE = JE = 500  # size and PML parameter loop

npml = 8
NFREQS = 3
freq = [data_type(0, 1)] * NFREQS
for n in range(0, NFREQS):
    freq[n] = data_type(225E12 * (n + 1), flag)  # red light (666nm) + H
arg = [data_type(0, flag)] * NFREQS

# highest_er = np.float32(50)  # ~biological tissue er for 500 MHz
# ddx = data_type((cc.c0 / min(freq) / (10 * cc.nSiO2)), flag)  # Cells Size
# n=speed in vaccum/speed in medium

n_index = 10  # cc.nSiO2
n_sigma = 1.  # cc.sigmaSiO2
epsilon = data_type(n_index, flag)
sigma = data_type(n_sigma, flag)
epsilon_medium = data_type(1.003, flag)
sigma_medium = data_type(0., flag)

wavelength = cc.c0 / (n_index * (min(freq)))
vm = wavelength * (min(freq))
dx = 10
# wavelength = (vm / (min(freq)))
ddx = data_type(wavelength / dx, flag)  # Cells Size

# dt = data_type((ddx / cc.c0) *  M.sqrt(2),flag) # Time step

#   CFL stability condition- Lax Equivalence Theorem
dt = 1 / (vm * M.sqrt(1 / (ddx ** 2) + 1 / (ddx ** 2)))  # Tiem step

z_max = data_type(0, 1)
epsz = data_type(8.854E-12, flag)
spread = data_type(8, flag)
t0 = data_type(1, flag)
# print(ic.shape)
ic = IE / 2
jc = JE / 2
ia = 7  # total scattered field boundaries
ib = IE - ia - 1
ja = 7
jb = JE - ja - 1
nsteps = 2000
T = 0

medium_eps = 1. / (epsilon_medium + sigma_medium * dt / epsz)
medium_sigma = sigma_medium * dt / epsz

# print(dt)
for n in range(0, NFREQS):
    arg[n] = 2 * M.pi * freq[n] * dt

ez_inc_low_m2 = data_type(0., flag)
ez_inc_low_m1 = data_type(0., flag)

ez_inc_high_m2 = data_type(0., flag)
ez_inc_high_m1 = data_type(0., flag)

dz = np.zeros((IE, JE), dtype=data_type1)
iz = np.zeros((IE, JE), dtype=data_type1)
ez = np.zeros((IE, JE), dtype=data_type1)
hx = np.zeros((IE, JE), dtype=data_type1)
hy = np.zeros((IE, JE), dtype=data_type1)
ihx = np.zeros((IE, JE), dtype=data_type1)
ihy = np.zeros((IE, JE), dtype=data_type1)
ga = np.ones((IE, JE), dtype=data_type1) * medium_eps  # main medium epsilon
gb = np.zeros((IE, JE), dtype=data_type1) * medium_sigma  # main medium sigma
Pz = np.zeros((IE, JE), dtype=data_type1)

gi2 = np.ones(IE, dtype=data_type1)
gi3 = np.ones(IE, dtype=data_type1)
fi1 = np.zeros(IE, dtype=data_type1)
fi2 = np.ones(IE, dtype=data_type1)
fi3 = np.ones(IE, dtype=data_type1)

gj2 = np.ones(JE, dtype=data_type1)
gj3 = np.ones(JE, dtype=data_type1)
fj1 = np.zeros(JE, dtype=data_type1)
fj2 = np.ones(JE, dtype=data_type1)
fj3 = np.ones(JE, dtype=data_type1)

ez_inc = np.zeros(JE, dtype=data_type1)
hx_inc = np.zeros(JE, dtype=data_type1)

# PML Definition
for i in range(npml):
    xnum = npml - i
    xd = npml
    xxn = xnum / xd
    xn = 0.33333 * pow(xxn, 3)
    gi2[i] = 1. / (1. + xn)
    gi2[IE - 1 - i] = 1. / (1. + xn)
    gi3[i] = (1. - xn) / (1. + xn)
    gi3[IE - i - 1] = (1. - xn) / (1. + xn)
    xxn = (xnum - .5) / xd
    xn = 0.33333 * pow(xxn, 3)
    fi1[i] = xn
    fi1[IE - 2 - i] = xn
    fi2[i] = 1.0 / (1.0 + xn)
    fi2[IE - 2 - i] = 1.0 / (1.0 + xn)
    fi3[i] = (1.0 - xn) / (1.0 + xn)
    fi3[IE - 2 - i] = (1.0 - xn) / (1.0 + xn)

    gj2[i] = 1. / (1. + xn)
    gj2[JE - 1 - i] = 1. / (1. + xn)
    gj3[i] = (1.0 - xn) / (1. + xn)
    gj3[JE - i - 1] = (1. - xn) / (1. + xn)
    xxn = (xnum - .5) / xd
    xn = 0.33333 * pow(xxn, 3)
    fj1[i] = xn
    fj1[JE - 2 - i] = xn
    fj2[i] = 1. / (1. + xn)
    fj2[JE - 2 - i] = 1. / (1. + xn)
    fj3[i] = (1. - xn) / (1. + xn)
    fj3[JE - 2 - i] = (1. - xn) / (1. + xn)

x_offset = 0
y_offset = 0

fig = plt.figure(figsize=(15, 5))
grid = plt.GridSpec(20, 20, wspace=10, hspace=0.6)
ax = fig.add_subplot(grid[:, :5])
ay = fig.add_subplot(grid[:, 5:15])
az = fig.add_subplot(grid[:, 15:])
# Cyclic Number of image snapping
frame_interval = 16
ims = []

wstart = 10
fwidth = 5 + wstart
a = 2
b = 2
# for j in range(ja, jb):

x_points = []
y_points = []
data = np.zeros((IE, JE, 4), dtype=np.uint8)
surface = cairo.ImageSurface.create_for_data(
    data, cairo.FORMAT_ARGB32, IE, JE)
cr = cairo.Context(surface)

cr.set_source_rgb(1.0, 1.0, 1.0)
cr.paint()

cr.rectangle(0, 50, 200, 5)
cr.rectangle(210, 50, 50, 5)
cr.rectangle(270, 50, 50, 5)
cr.rectangle(330, 50, 300, 5)

# cr.rectangle(190, 60, 5, 200)
# CIRCLE
cr.arc(150, 250, 50, 0, 2 * M.pi)
cr.set_line_width(5)
cr.close_path()

cr.set_source_rgb(1.0, 0.0, 0.0)
cr.fill()

shape1 = data[:, :, 0].shape[0]
shape2 = data[:, :, 0].shape[1]
i = 0
j = 0
# print(data[38:48, 38:48, 0])
# print(0 in data[:, :, 0])
for j in range(0, shape2):
    for i in range(0, shape1):
        if data[i, j, 0] <= 0:
            # print(data[i, j, 0])
            ga[j, i] = data_type(1 / (epsilon + (sigma * dt / epsz)), flag)
            gb[j, i] = data_type(sigma * dt / epsz, flag)
            x_points.append(i)
            y_points.append(JE - j)
        if data[i, j, 0] > 0:
            pass
            # print(data[i, j, 0])

sr = freq[1]  # Sampling rate, or number of measurements per second

inputx = 450
inputy = 50

pwr_in_y = range(inputy, inputx)
pwr_in_x = [inputx] * len(pwr_in_y)

measx = 450
measy = 50
probey = range(measy, measx)  # [measy] * JE  # range(0,JE) #
probex = [measx] * len(probey)  # [measx] * IE# range(0, IE)  # range(0,IE)#

INTEGRATE = []
window = 10

fft_history_x = []
fft_history_y = []

# in_ie1 = 190
# in_ie2 = 210
#
# ia = in_ie1
# ib = in_ie2


k_vec = 2 * M.pi / wavelength
omega = 2 * M.pi * freq[0]


# -------------------------------- KERNELS ---------------------------
@jit(nopython=True, parallel=True)
def Ez_inc_CU(ez_inc, hx_inc):
    for j in range(1, JE):
        ez_inc[j] = ez_inc[j] + 0.5 * (hx_inc[j - 1] - hx_inc[j])
    return ez_inc


@jit(nopython=True, parallel=True)
def Dz_CU(dz, hx, hy, gi2, gi3, gj2, gj3):
    for j in range(1, JE):
        for i in range(1, IE):
            dz[i][j] = gi3[i] * gj3[j] * dz[i][j] + \
                       gi2[i] * gj2[j] * 0.5 * \
                       (hy[i][j] - hy[i - 1][j] -
                        hx[i][j] + hx[i][j - 1])
    return dz


@jit(nopython=True, parallel=True)
def Dz_inc_val_CU(dz, hx_inc):
    for i in range(ia, ib + 1):
        dz[i][ja] = dz[i][ja] + 0.5 * hx_inc[ja - 1]
        dz[i][jb] = dz[i][jb] - 0.5 * hx_inc[jb]
    return dz


@jit(nopython=True, parallel=True)
def Ez_Dz_CU(ez, ga, gb, dz, iz):
    for j in range(0, JE):
        for i in range(0, IE):
            ez[i, j] = ga[i, j] * (dz[i, j] - iz[i, j])
            iz[i, j] = iz[i, j] + gb[i, j] * ez[i, j]
    return ez, iz


@jit(nopython=True, parallel=True)
def Hx_inc_CU(hx_inc, ez_inc):
    for j in range(0, JE - 1):
        hx_inc[j] = hx_inc[j] + .5 * (ez_inc[j] - ez_inc[j + 1])
    return hx_inc


@jit(nopython=True, parallel=True)
def Hx_CU(ez, hx, ihx, fj3, fj2, fi1):
    for j in range(0, JE - 1):
        for i in range(0, IE - 1):
            curl_e = ez[i][j] - ez[i][j + 1]
            ihx[i][j] = ihx[i][j] + curl_e
            hx[i][j] = fj3[j] * hx[i][j] + fj2[j] * \
                       (.5 * curl_e + fi1[i] * ihx[i][j])
    return ihx, hx


@jit(nopython=True, parallel=True)
def Hx_inc_val_CU(hx, ez_inc):
    for i in range(ia, ib + 1):
        hx[i][ja - 1] = hx[i][ja - 1] + .5 * ez_inc[ja]
        hx[i][jb] = hx[i][jb] - .5 * ez_inc[jb]
    return hx


@jit(nopython=True, parallel=True)
def Hy_CU(hy, ez, ihy, fi3, fi2, fi1):
    for j in range(0, JE):
        for i in range(0, IE - 1):
            curl_e = ez[i][j] - ez[i + 1][j]
            ihy[i][j] = ihy[i][j] + curl_e
            hy[i][j] = fi3[i] * hy[i][j] - fi2[i] * \
                       (.5 * curl_e + fi1[j] * ihy[i][j])
    return ihy, hy


@jit(nopython=True, parallel=True)
def Hy_inc_CU(hy, ez_inc):
    for j in range(ja, jb + 1):
        hy[ia - 1][j] = hy[ia - 1][j] - .5 * ez_inc[j]
        hy[ib][j] = hy[ib][j] + .5 * ez_inc[j]
    return hy


@jit(nopython=True, parallel=True)
def Power_Calc(Pz, ez, hy, hx):
    for j in range(0, JE):
        for i in range(0, IE):
            Pz[i][j] = M.sqrt(M.pow(-ez[i][j] * hy[i][j], 2) + M.pow(ez[i][j] * hx[i][j], 2))
    return Pz


# -------------------------------- KERNELS ---------------------------
for n in range(1, nsteps):
    net = time.time()
    T = T + 1
    # MAIND FDTD LOOP
    # ez_incd, hx_incd = cuda.to_device(ez_inc, stream=stream), cuda.to_device(hx_inc, stream=stream)
    ez_inc = Ez_inc_CU(ez_inc, hx_inc)
    # ez_inc, hx_inc = ez_incd.copy_to_host(stream=stream), hx_incd.copy_to_host(stream=stream)
    ez_inc[0] = ez_inc_low_m2
    ez_inc_low_m2 = ez_inc_low_m1
    ez_inc_low_m1 = ez_inc[1]
    ez_inc[JE - 1] = ez_inc_high_m2
    ez_inc_high_m2 = ez_inc_high_m1
    ez_inc_high_m1 = ez_inc[JE - 2]
    dz = Dz_CU(dz, hx, hy, gi2, gi3, gj2, gj3)
    if T < 600:
        pulse = data_type(M.sin(2 * M.pi * freq[0] * dt * T), flag)
        # pulse = data_type(M.exp(-.5 * (pow((t0 - T * 4) / spread, 2))), flag)
        # pulse = data_type(M.exp(-(T-t0)**2/(2*(t0/10)**2)) * M.sin(2*M.pi * (cc.c0/wavelength)*T),flag)
        ez_inc[3] = pulse  # plane wave
    else:
        pass
    dz = Dz_inc_val_CU(dz, hx_inc)
    ez, iz = Ez_Dz_CU(ez, ga, gb, dz, iz)
    hx_inc = Hx_inc_CU(hx_inc, ez_inc)
    ihx, hx = Hx_CU(ez, hx, ihx, fj3, fj2, fi1)
    hx = Hx_inc_val_CU(hx, ez_inc)
    ihy, hy = Hy_CU(hy, ez, ihy, fi3, fi2, fi1)
    hy = Hy_inc_CU(hy, ez_inc)
    Pz = Power_Calc(Pz, ez, hy, hx)
    netend = time.time()
    # print("Time netto : " + str((netend - net)) + "[s]")
    nett_time_sum += netend - net
    # Drawing of the EM and FT plots
    if T % frame_interval == 0:
        x = np.linspace(0, JE, JE)
        y = np.linspace(0, IE, IE)
        values = range(len(x))
        X, Y = np.meshgrid(x, y)
        Z = Pz[:][:]  # Power - W/m^2s
        INTEGRATE.append(Z)
        YY = np.trapz(INTEGRATE, axis=0) / window
        if len(INTEGRATE) >= window:
            del INTEGRATE[0]
        title = ay.annotate("Time :" + '{:<.4e}'.format(T * dt * 1 * 10 ** 15) + " fs", (1, 0.5),
                            xycoords=ay.get_window_extent, xytext=(-round(JE * 2), IE - 5),
                            textcoords="offset points", fontsize=9, color='white')
        ims2 = ay.imshow(Z, cmap=cm.tab20c, extent=[0, JE, 0, IE])
        ims2.set_interpolation('bilinear')
        ims4 = ay.scatter(x_points, y_points, c='grey', s=70, alpha=0.01)
        ims5 = ay.scatter(probex, probey, c='red', s=5, alpha=0.05)
        ims6, = az.plot(np.abs(YY)[probey, probex], 'r')  # field distribiution
        fft_out = fft.fftshift(fft.fft(torch.from_numpy(YY[:, measx])))
        fft_out[1] = 0
        fft_out[0] = 0
        fft_res = torch.abs(fft_out)
        fft_len = fft.fftfreq(len(fft_res), 1 / sr * 10)[:len(fft_res) // 2]
        fft_result = 2.0 / len(fft_res) * torch.abs(fft_res[0:len(fft_res) // 2])
        ims7, = ax.plot(fft_len, fft_result, 'g')
        ims.append([ims2, ims4, ims5, ims6, ims7, title])
        # print("Punkt : " + str(T))

# ax.set_xscale('log')
ax.grid(True)
ay.set_xlabel("x [um]")
ay.set_ylabel("y [um]")
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("Power [W]")
az.set_xlabel("x [um]")
az.set_ylabel("Power [W]")

labels = [item.get_text() for item in ay.get_xticklabels()]
labels[0] = '0'
labels[1] = str(0.2 * IE / dx)
labels[2] = str(0.4 * IE / dx)
labels[3] = str(0.5 * IE / dx)
labels[4] = str(0.8 * IE / dx)
labels[5] = str(IE / dx)
ay.set_xticklabels(labels)
labels[1] = str(0.2 * JE / dx)
labels[2] = str(0.4 * JE / dx)
labels[3] = str(0.5 * JE / dx)
labels[4] = str(0.8 * JE / dx)
labels[5] = str(JE / dx)
ay.set_yticklabels(labels)
labelz = [item.get_text() for item in az.get_xticklabels()]
labelz[5] = str(round(probey[0] / dx, 2))
labelz[4] = str(round(0.2 * (probey[-1]) / dx, 2))
labelz[3] = str(round(0.4 * (probey[-1]) / dx, 2))
labelz[2] = str(round(0.6 * (probey[-1]) / dx, 2))
labelz[1] = str(round(0.8 * (probey[-1]) / dx, 2))
labelz[0] = str(round((probey[-1] + 1) / dx, 2))
az.set_xticklabels(labelz)

az.grid(True)
e = time.time()
print("Time brutto : " + str((e - s)) + "[s]")
print("Time netto SUM : " + str(nett_time_sum) + "[s]")
file_name = "2d_fdtd_Si_Cylinder_Measure"
# file_name = "./" + file_name + '.gif'
file_name = "./" + file_name + '.gif'
ani = animation.ArtistAnimation(fig, ims, interval=30, blit=True)
# ani.save(file_name, writer='pillow', fps=30, dpi=100)
# ani.save(file_name + '.mp4', fps = 30, extra_args = ['-vcodec', 'libx264'])
ani.save(file_name, writer="imagemagick", fps=30)
print("OK")
plt.show()

# TODO : Meas port check
# TODO : Input port definition
# TODO : Check if the result are correct
# TODO : Results in VisPy
# TODO : Index profile of the medium
# TODO : MMI Structure

