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


def data_for_cylinder_along_z(center_x, center_y, radius, height_z, ddx, z_max):
    # zmax = abs(height_z).max()
    # if z_max > zmax:
    #     pass
    # else:
    #     z_max += zmax * 2
    z_max = 5
    z = np.linspace(0, z_max, 50)
    theta = np.linspace(0, 2 * np.pi, 50)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = (radius * np.cos(theta_grid) + center_x)
    y_grid = (radius * np.sin(theta_grid) + center_y)
    return x_grid * ddx, y_grid * ddx, z_grid, z_max


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y


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
vm = cc.c0 / cc.nSiO2
dx = 10
wavelength = (vm / (min(freq)))
ddx = data_type(wavelength / dx, flag)  # Cells Size
# print(ddx)

# dt = data_type(1 / (M.sqrt(1 / pow(ddx, 2)) * cc.c0), flag)  # Time Steps (# 2 * cc.c0 is to small??)

#   CFL stability conditio - Lax Equivalence Theorem
# dt = data_type((ddx / cc.c0) * 1 / M.sqrt(2),flag)
dt = 1 / (vm * M.sqrt(1 / (ddx ** 2) + 1 / (ddx ** 2)))

# print(dt)
for n in range(0, NFREQS):
    arg[n] = 2 * M.pi * freq[n] * dt

z_max = data_type(0, 1)
epsz = data_type(8.854E-12, flag)
spread = data_type(8, flag)
t0 = data_type(20, flag)
# print(ic.shape)
ic = IE / 2
jc = JE / 2
ia = 10  # total scattered field boundaries
ib = IE - ia - 1
ja = 10
jb = JE - ja - 1
nsteps = 4800
T = 0

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
ga = np.ones((IE, JE), dtype=data_type1)
gb = np.zeros((IE, JE), dtype=data_type1)
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
    xn = 0.333 * pow(xxn, 3)
    gi2[i] = 1. / (1. + xn)
    gi2[IE - 1 - i] = 1. / (1. + xn)
    gi3[i] = (1. - xn) / (1. + xn)
    gi3[IE - i - 1] = (1. - xn) / (1. + xn)
    xxn = (xnum - .5) / xd
    xn = 0.333 * pow(xxn, 3)
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
    xn = 0.333 * pow(xxn, 3)
    fj1[i] = xn
    fj1[JE - 2 - i] = xn
    fj2[i] = 1. / (1. + xn)
    fj2[JE - 2 - i] = 1. / (1. + xn)
    fj3[i] = (1. - xn) / (1. + xn)
    fj3[JE - 2 - i] = (1. - xn) / (1. + xn)

# Dielectric cylinder Specifications
radius = data_type(50, flag)
epsilon = data_type(cc.nSiO2, flag)
sigma = data_type(cc.sigmaSiO2, flag)
x_offset = 0
y_offset = 0

# for k in range(0, 3):
# for j in range(ja, jb):
#     for i in range(ia, ib):
#         xdist = ic - i
#         ydist = jc - j
#         dist = data_type(M.sqrt(pow(xdist, 2) + pow(ydist, 2)), flag)
#         if dist <= radius:
#             ga[i + x_offset][j + y_offset] = data_type(1 / (epsilon + (sigma * dt / epsz)), flag)
#             gb[i + x_offset][j + y_offset] = data_type(sigma * dt / epsz, flag)
#             ga[i - x_offset][j - y_offset] = data_type(1 / (epsilon + (sigma * dt / epsz)), flag)
#             gb[i - x_offset][j - y_offset] = data_type(sigma * dt / epsz, flag)
#             # x_offset += 50
#             # y_offset += 0

fig = plt.figure()
grid = plt.GridSpec(20, 20, wspace=10, hspace=0.6)
# 3d plot to use
# ax = fig.add_subplot(grid[:, :5], projection='3d')
ay = fig.add_subplot(grid[:, :10])
az = fig.add_subplot(grid[:, 11:])
# az.set_title('Double Sided FFT - with FFTShift')
# az.set_xlabel('Frequency (Hz)')
# az.set_ylabel('|DFT Values|')
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

cr.rectangle(200, 100, 25, 150)

cr.arc(250, 250, 50, 0, 2 * M.pi)
cr.set_line_width(5)
cr.close_path()
cr.set_source_rgb(1.0, 0.0, 0.0)
# cr.stroke()
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

# TODO : Create GPU FFT VERSION
# TODO : DISPLAY EVERYTHING IN VISPY RESULTS


PWR_FFT = []
sr = freq[0] * 10  # Sampling rate, or number of measurements per second
NFFT = 1024
fVals = np.arange(start=-NFFT / 2, stop=NFFT / 2) * sr / NFFT

measx = 350
measy = 250
probex = [measx] * IE  # range(0,IE)#
probey = [measy] * JE  # range(0,JE) #

INTEGRATE = []
window = 40


# -------------------------------- KERNELS ---------------------------
def lowpass_torch(input, limit):
    pass1 = torch.abs(fft.rfftfreq(input.shape[-1])) < limit
    pass2 = torch.abs(fft.fftfreq(input.shape[-2])) < limit
    kernel = torch.outer(pass2, pass1)

    fft_input = fft.rfft2(input)
    return fft.irfft2(fft_input * kernel, s=input.shape[-2:])


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
    pulse = data_type(M.sin(2 * M.pi * freq[0] * dt * T), flag)
    # pulse = data_type(M.exp(-.5 * (pow((t0 - T * 4) / spread, 2))), flag)
    ez_inc[3] = pulse  # plane wave
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
        title = ay.annotate("Time :" + '{:<.4e}'.format(T * dt * 1 * 10 ** 15) + " fs", (1, 0.5),
                            xycoords=ay.get_window_extent, xytext=(-round(JE * 2), IE - 5),
                            textcoords="offset points", fontsize=9, color='white')
        x = np.linspace(0, JE, JE)
        y = np.linspace(0, IE, IE)
        # xi = np.arange(x.min(), x.max() + 1)
        # yi = np.arange(y.min(), y.max() + 1)
        X, Y = np.meshgrid(x, y)
        Z = Pz[:][:]  # Power - W/m^2s
        # Z = ez[:][:] # field E
        # filtered_Z = butter_highpass_filter(Pz[:][:], freq[0] / 10, sr)
        # PWR_FFT = filtered_Z[probey, probex] / (len(probey))
        # XX = fftshift(fft(PWR_FFT, NFFT))

        XX = lowpass_torch(torch.tensor(Z), 500)
        print(XX)
        # X = fft(PWR_FFT, NFFT)
        INTEGRATE.append(XX.numpy())
        INT = np.array(INTEGRATE)
        YY = np.trapz(INT, axis=0) / window
        # print(len(INTEGRATE))
        # print(len(INTEGRATE[0]))
        if len(INTEGRATE) >= window:
            del INTEGRATE[0]
        # 3d plot to use
        # ims1 = ax.plot_surface(X * ddx * 1 * 10 ** 6, Y * ddx * 1 * 10 ** 6, Z, rstride=5, cstride=5, cmap=cm.inferno)

        # ims1 = ax.plot_wireframe(X * ddx * 1 * 10 ** 6, Y * ddx * 1 * 10 ** 6, Z, rstride=15, cstride=15) ims1 =
        # ax.scatter(X[::10] * ddx * 1 * 10 ** 6, Y[::10] * ddx * 1 * 10 ** 6, Z[::10],zdir='z', s=20, c='red',
        # depthshade=True,)
        # ims2 = ay.imshow(Z, cmap=cm.bwr, vmin=abs(Z).min(), vmax=abs(Z).max(),
        #                  extent=[0, JE, 0, IE])
        ims2 = ay.imshow(Z, cmap=cm.bwr, extent=[0, JE, 0, IE])
        ims2.set_interpolation('bilinear')

        # circ = plt.Circle((ic * ddx * 1 * 10 ** 6, jc * ddx * 1 * 10 ** 6), radius * ddx * 1 * 10 ** 6, color='silver',alpha=0.3, fill=True)
        # ims3 = ay.add_patch(circ)
        # Xc, Yc, Zc, z_max = data_for_cylinder_along_z(ic, jc, radius, Z, ddx * 1 * 10 ** 6, z_max)

        # cyli = ax.plot_surface(Xc, Yc, Zc, alpha=0.3, color='silver', shade=False)
        # ims.append([ims1, ims2, ims3, title, cyli])
        ims4 = ay.scatter(x_points, y_points, c='grey', s=70, alpha=0.01)
        # ims5 = ay.imshow(ellipse, extent=(x[0], x[-1], y[0], y[-1]), origin="lower", alpha=0.2)
        ims5 = ay.scatter(probex, probey, c='red', s=5, alpha=0.01)
        ims6, = az.plot(fVals, np.abs(YY), 'b')

        ims.append([ims2, ims4, ims5, ims6, title])
        # print("Punkt : " + str(T))

# ax.set_xlabel("x [um]")
# ax.set_ylabel("y [um]")
# ax.set_zlabel("Power [W/m2]")
ay.set_xlabel("x [um]")
ay.set_ylabel("y [um]")

e = time.time()
print("Time brutto : " + str((e - s)) + "[s]")
print("Time netto SUM : " + str(nett_time_sum) + "[s]")
file_name = "2d_fdtd_Si_Cylinder_2"
# file_name = "./" + file_name + '.gif'
file_name = "./" + file_name + '.gif'
ani = animation.ArtistAnimation(fig, ims, interval=30, blit=True)
# ani.save(file_name, writer='pillow', fps=30, dpi=100)
# ani.save(file_name + '.mp4', fps = 30, extra_args = ['-vcodec', 'libx264'])
# ani.save(file_name, writer="imagemagick", fps=30)
# figManager = plt.get_current_fig_manager()
# figManager.window.resize(1000, 600)
print("OK")
plt.show()
# Plot the frequency response of the filter
bb, aa = butter_highpass(freq[0] / 10, sr, 5)
w, h = freqz(bb, aa, fs=sr, worN=8000)
plt.subplot(1, 1, 1)
plt.plot(w, np.abs(h), 'b')
plt.plot(freq[0] / 10, 0.5 * np.sqrt(2), 'ko')
plt.axvline(freq[0] / 10, color='k')
plt.xlim(0, 0.5 * sr)
plt.title("Highpass Filter Frequency Response")
plt.xlabel('Frequency [Hz]')
plt.grid()
plt.show()
