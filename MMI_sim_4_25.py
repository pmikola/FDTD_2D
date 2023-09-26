import math as M
import sys
import time
import cairo
import matplotlib.cm as cm
import matplotlib
import numpy as np
import torch
import torch as tf
import numba
# import tensorflow as tf
# from scipy.fftpack import fft, fftshift
import torch.fft as fft
from matplotlib import pyplot as plt, animation
from numba import prange, jit
# import matplotlib.style as mplstyle
from C import C
from numba import cuda
from numba import vectorize

# from vispy import plot as vp

# mpl.use('Agg')
jit(device=True)
s = time.time()
nett_time_sum = 0
np.seterr(divide='ignore', invalid='ignore')


# plt.switch_backend('QtCairo')

# mplstyle.use('fast')
# mplstyle.use(['dark_background', 'ggplot', 'fast'])
def data_type(data, flag):
    if flag == 1:
        return np.float32(data)
    else:
        return np.float64(data)


def in_port_len(wg_input_length, port_range, port_offset_start, port_offset_stop):
    inputy_start = int(wg_input_length - port_range) + port_offset_start
    inputy_stop = int(wg_input_length) + port_offset_stop
    input_meas_port_range = np.arange(inputy_start, inputy_stop, 1)
    return input_meas_port_range


def out_port_len(wg_output_start, port_range):
    measx_start = int(wg_output_start)
    measx_stop = int(wg_output_start + port_range)
    output_meas_port_range = np.arange(measx_start, measx_stop, 1)
    return output_meas_port_range


# -------------------------------- KERNELS ---------------------------
dx = 0.05  # each grid step is dx [um]
dx_factor = 10 * dx
z = 10 / dx_factor

@jit(nopython=True, parallel=True)
def Ez_inc_CU(ez_inc, hx_inc):
    for j in prange(1, JE):
        for i in prange(0, IE):
            if j <= IE - z:
                ez_inc[i, j] = ez_inc[i, j] + 0.5 * (hx_inc[i, j - 1] - hx_inc[i, j])
            else:
                ez_inc[i, j] = 0.
    return ez_inc


@jit(nopython=True, parallel=True)
# @numba.jit(nopython=True, parallel=True)
def Dz_CU(dz, hx, hy, gi2, gi3, gj2, gj3):
    for j in prange(1, JE):
        for i in prange(1, IE):
            if j <= IE - z:
                dz[i, j] = gi3[i] * gj3[j] * dz[i, j] + \
                           gi2[i] * gj2[j] * 0.5 * \
                           (hy[i, j] - hy[i - 1, j] -
                            hx[i, j] + hx[i, j - 1])
            else:
                dz[i, j] = 0.
    return dz


@jit(nopython=True, parallel=True)
def Hy_inc_CU(hy, ez_inc):
    # for j in prange(1, JE):
    for j in prange(ja, jb + 1):
        hy[ia - 1, j] = hy[ia - 1, j] - .5 * ez_inc[ia - 1, j]
        hy[ib, j] = hy[ib, j] + .5 * ez_inc[ib, j]
    return hy


@jit(nopython=True, parallel=True)
def Dz_inc_val_CU(dz, hx_inc):
    for i in prange(ia, ib):
        dz[i, ja] = dz[i, ja] + 0.5 * hx_inc[i, ja - 1]
        dz[i, jb] = dz[i, jb] - 0.5 * hx_inc[i, jb - 1]
    return dz


@jit(nopython=True, parallel=True)
def Ez_Dz_CU(ez, ga, gb, dz, iz):
    for j in prange(0, JE):
        for i in prange(0, IE):
            if j <= IE - z:
                ez[i, j] = ga[i, j] * (dz[i, j] - iz[i, j])
                iz[i, j] = iz[i, j] + gb[i, j] * ez[i, j]
            else:
                ez[i, j] = 0.
                iz[i, j] = 0.  # iz[i, j] + gb[i, j] * ez[i, j]
    return ez, iz


@jit(nopython=True, parallel=True)
def Hx_inc_CU(hx_inc, ez_inc):
    for j in prange(0, JE - 1):
        for i in prange(0, IE):
            if j <= IE - z:
                hx_inc[i, j] = hx_inc[i, j] + 0.5 * (ez_inc[i, j] - ez_inc[i, j + 1])
            else:
                hx_inc[i, j] = 0.
    return hx_inc


@jit(nopython=True, parallel=True)
def Hx_CU(ez, hx, ihx, fj3, fj2, fi1):
    for j in prange(0, JE - 1):
        for i in prange(0, IE - 1):
            if j <= IE - z:
                curl_e = ez[i, j] - ez[i, j + 1]
                ihx[i, j] = ihx[i, j] + curl_e
                hx[i, j] = fj3[j] * hx[i, j] + fj2[j] * \
                           (.5 * curl_e + fi1[i] * ihx[i, j])
            else:
                hx[i, j] = 0.
    return ihx, hx


@jit(nopython=True, parallel=True)
def Hx_inc_val_CU(hx, ez_inc):
    # for j in prange(0, JE):
    for i in prange(ia, ib + 1):
        hx[i, ja - 1] = hx[i, ja - 1] + 0.5 * ez_inc[i, ja]
        hx[i, jb] = hx[i, jb] - 0.5 * ez_inc[i, jb]
    return hx


@jit(nopython=True, parallel=True)
def Hy_CU(hy, ez, ihy, fi3, fi2, fi1):
    for j in prange(0, JE):
        for i in prange(0, IE - 1):
            if j <= IE - z:
                curl_e = ez[i, j] - ez[i + 1, j]
                ihy[i, j] = ihy[i, j] + curl_e
                hy[i, j] = fi3[i] * hy[i, j] - fi2[i] * \
                           (.5 * curl_e + fi1[j] * ihy[i, j])
            else:
                hy[i, j] = 0.
    return ihy, hy


@jit(nopython=True, parallel=True)
def Power_Calc(Pz, ez, hy, hx):
    for j in prange(0, JE):
        for i in prange(0, IE):
            Pz[i, j] = M.sqrt(M.pow(-ez[i, j] * hy[i, j], 2) + M.pow(ez[i, j] * hx[i, j], 2))
    return Pz


# -------------------------------- KERNELS ---------------------------
cc = C()
flag = 1
data_type1 = np.float32
IE = 1500  # y
JE = 1500  # x
npml = int(8 / dx_factor)
NFREQS = 3
freq = [data_type(0, 1)] * NFREQS
for n in range(0, NFREQS):
    freq[n] = data_type(70.5394E12 * (n + 1), flag)  # infrared light (4250nm) + H
arg = [data_type(0, flag)] * NFREQS
# highest_er = np.float32(50)  # ~biological tissue er for 500 MHz
# ddx = data_type((cc.c0 / min(freq) / (10 * cc.nSiO2)), flag)  # Cells Size
# n=speed in vaccum/speed in medium

n_index = 3.6  # cc.nGe # effective index from equation or fimmwave
n_sigma = 0.  # cc.sigmaSiO2
epsilon = data_type(n_index, flag)
sigma = data_type(n_sigma, flag)
epsilon_medium = data_type(1.003, flag)
sigma_medium = data_type(0., flag)

# cc.c0 / (n_index * (min(freq)))
A = 1.
vm = cc.wavelength * (min(freq))
# vm = cc.c0 / cc.nGe
# wavelength = (vm / (min(freq)))
ddx = data_type(cc.wavelength * dx, flag)  # Cells Size
# dt = 1 / (cc.c0 * np.sqrt((1 / ddx)**2 + (1 / ddx)**2))
# dt = data_type((ddx / cc.c0) * M.sqrt(2), flag)  # Time step
dt = ddx / (2 * cc.c0)  # Working moderate but ok
correction = 0.5
second_correction = 11.85
wave_meas = 4.25 / dx + 1 / dx_factor

#   CFL stability condition- Lax Equivalence Theorem
# dt = 1 / (vm * M.sqrt(1 / (ddx ** 2) + 1 / (ddx ** 2)))  # Time step

# dt = M.sqrt(epsilon * sigma)  # same as 2*cc.c0

z_max = data_type(0, 1)
epsz = data_type(8.854E-12, flag)
spread = data_type(8, flag)
t0 = data_type(1, flag)
# print(ic.shape)
ia = int(7 / dx_factor)  # total scattered field boundaries
ib = IE - ia - 1
ja = ia
jb = JE - ja - 1
T = 0
zero_range = int(3 / dx_factor)  # 5
zeroing_ezinc = int(2 / dx_factor)
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
gb = np.ones((IE, JE), dtype=data_type1) * medium_sigma  # main medium sigma
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

ez_inc = np.zeros((IE, JE), dtype=data_type1)
hx_inc = np.zeros((IE, JE), dtype=data_type1)

# PML Definition
alpha = 0.33333
for i in range(npml):
    xnum = npml - i
    xd = npml
    xxn = xnum / xd
    xn = alpha * xxn ** 3
    xxn_minus_half = (xnum - 0.5) / xd

    gi_value = 1. / (1. + xn)
    gi2[i] = gi_value
    gi2[IE - 1 - i] = gi_value

    gi3_value = (1. - xn) / (1. + xn)
    gi3[i] = gi3_value
    gi3[IE - i - 1] = gi3_value

    fi1_value = xn
    fi1[i] = fi1_value
    fi1[IE - 2 - i] = fi1_value

    fi2_value = 1.0 / (1.0 + xn)
    fi2[i] = fi2_value
    fi2[IE - 2 - i] = fi2_value

    fi3_value = (1.0 - xn) / (1.0 + xn)
    fi3[i] = fi3_value
    fi3[IE - 2 - i] = fi3_value

    gj_value = 1. / (1. + xn)
    gj2[i] = gj_value
    gj2[JE - 1 - i] = gj_value

    gj3_value = (1. - xn) / (1. + xn)
    gj3[i] = gj3_value
    gj3[JE - i - 1] = gj3_value

    fj1_value = xn
    fj1[i] = fj1_value
    fj1[JE - 2 - i] = fj1_value

    fj2_value = 1. / (1. + xn)
    fj2[i] = fj2_value
    fj2[JE - 2 - i] = fj2_value

    fj3_value = (1. - xn) / (1. + xn)
    fj3[i] = fj3_value
    fj3[JE - 2 - i] = fj3_value

x_offset = 0
y_offset = 0

ims = []
x_points = []
y_points = []
data = np.zeros((IE, JE, 4), dtype=np.uint8)
# surface = cairo.ImageSurface.create_for_data(data, cairo.FORMAT_ARGB32, IE, JE)
surface = cairo.ImageSurface.create_for_data(data, cairo.FORMAT_RGB24, IE, JE)
cr = cairo.Context(surface)
# cr.set_source_rgb(1.0, 0.0, 0.0)
cr.set_source_rgb(1.0, 1.0, 1.0)
cr.paint()

# 2x2 MMI 4.25um PARAMETERS
############################
waveguide_width = correction * 20 / dx_factor
mmi_width = correction * 63 / dx_factor
mmi_length = correction * 700 / dx_factor
mmi_left_corner = IE / 2 - mmi_width / 2
wg_offset = correction * 4 / dx_factor
wg_input_length = correction * 200 / dx_factor
tapers_length = correction * 50 / dx_factor
taper_width_offset = correction * 5 / dx_factor
############################
wg_top_left_corner = mmi_left_corner + wg_offset
wg_bottom_left_corner = mmi_left_corner + mmi_width - waveguide_width - wg_offset
wg_output_start = wg_input_length + mmi_length + 2 * tapers_length
wg_output_length = IE - wg_output_start
############################
taper_width_ittop_right = wg_top_left_corner - taper_width_offset
taper_width_itbottom_right = wg_top_left_corner + taper_width_offset + waveguide_width
taper_width_ibtop_right = wg_bottom_left_corner - taper_width_offset
taper_width_ibbottom_right = wg_bottom_left_corner + taper_width_offset + waveguide_width
############################
taper_width_ottop_right = wg_top_left_corner - taper_width_offset
taper_width_otbottom_right = wg_top_left_corner + taper_width_offset + waveguide_width
taper_width_obtop_right = wg_bottom_left_corner - taper_width_offset
taper_width_obbottom_right = wg_bottom_left_corner + taper_width_offset + waveguide_width
# INPUT WAVEGUIDES
cr.rectangle(wg_top_left_corner, 0, waveguide_width, wg_input_length)
cr.rectangle(wg_bottom_left_corner, 0, waveguide_width, wg_input_length)
# INPUT TAPERS
# TOP TAPER
cr.move_to(wg_top_left_corner, wg_input_length)  # top left corner
cr.line_to(taper_width_ittop_right, wg_input_length + tapers_length)  # top right corner
cr.line_to(taper_width_itbottom_right, wg_input_length + tapers_length)  # bottom right corner
cr.line_to(wg_top_left_corner + waveguide_width, wg_input_length)  # bottom left corner
# BOTTOM TAPER
cr.move_to(wg_bottom_left_corner, wg_input_length)  # top left corner
cr.line_to(taper_width_ibtop_right, wg_input_length + tapers_length)  # top right corner
cr.line_to(taper_width_ibbottom_right, wg_input_length + tapers_length)  # bottom right corner
cr.line_to(wg_bottom_left_corner + waveguide_width, wg_input_length)  # bottom left corner
# MMI SECTION
cr.rectangle(mmi_left_corner, wg_input_length + tapers_length, mmi_width, mmi_length)
# OUTPUT TAPERS
# TOP TAPER
cr.move_to(taper_width_ottop_right, wg_input_length + tapers_length + mmi_length)  # top left corner
cr.line_to(wg_top_left_corner, wg_input_length + tapers_length + tapers_length + mmi_length)  # top right corner
cr.line_to(wg_top_left_corner + waveguide_width,
           wg_input_length + tapers_length + tapers_length + mmi_length)  # bottom right corner
cr.line_to(taper_width_otbottom_right, wg_input_length + tapers_length + mmi_length)  # bottom left corner
# BOTTOM TAPER
cr.move_to(taper_width_obtop_right, wg_input_length + tapers_length + mmi_length)  # top left corner
cr.line_to(wg_bottom_left_corner, wg_input_length + tapers_length + tapers_length + mmi_length)  # top right corner
cr.line_to(wg_bottom_left_corner + waveguide_width,
           wg_input_length + tapers_length + tapers_length + mmi_length)  # bottom right corner
cr.line_to(taper_width_obbottom_right, wg_input_length + tapers_length + mmi_length)  # bottom left corner
# OUTPUT
cr.rectangle(wg_top_left_corner, wg_output_start, waveguide_width, wg_output_length)
cr.rectangle(wg_bottom_left_corner, wg_output_start, waveguide_width, wg_output_length)

cr.set_source_rgb(1.0, 0.0, 0.0)
cr.fill()
############################

shape1 = data[:, :, 0].shape[0]
shape2 = data[:, :, 0].shape[1]
print(shape1, shape2)
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
        else:  # data[i, j, 0] >= 0:
            pass
            # print(data[i, j, 0])

port_range = wave_meas
port_offset_start = 0
port_offset_stop = 0
port_width_start = int(mmi_left_corner + mmi_width / 2 - mmi_width * 2)
port_width_stop = int(mmi_left_corner + mmi_width / 2 + mmi_width * 2)

probey = range(port_width_start, port_width_stop)
input_meas_port_range = in_port_len(wg_input_length, port_range, port_offset_start, port_offset_stop)
output_meas_port_range = out_port_len(wg_output_start, port_range)
probex_out = [output_meas_port_range] * (len(probey))
probex_in = [input_meas_port_range] * (len(probey))

INTEGRATE = []
MaxField = []

fft_history_x = []
fft_history_y = []
source_start = int(wg_bottom_left_corner + 1)
source_end = int(wg_bottom_left_corner + waveguide_width - 1)
window = source_end - source_start
x = np.linspace(0, JE, JE)
y = np.linspace(0, IE, IE)
# values = range(len(x))
X, Y = np.meshgrid(x, y)

fig = plt.figure(figsize=(12, 6))
grid = plt.GridSpec(20, 20, wspace=2, hspace=0.6)
# ax = fig.add_subplot(grid[:, :5])
ay = fig.add_subplot(grid[:10, :])
az = fig.add_subplot(grid[12:, :])
# Cyclic Number of image snapping
frame_interval = 64
nsteps = 10000
tstamp = 0
stop_sim = 0
for n in range(1, nsteps + 1):
    net = time.time()
    T += 1
    # MAIND FDTD LOOP

    ez_inc = Ez_inc_CU(ez_inc, hx_inc)
    ez_inc[0:zeroing_ezinc, :] = ez_inc[-zeroing_ezinc:, :] = ez_inc[:, 0:zeroing_ezinc] = ez_inc[:,
                                                                                           -zeroing_ezinc:] = 0.0

    dz = Dz_CU(dz, hx, hy, gi2, gi3, gj2, gj3)
    if T <= int(nsteps):
        # w = 2 * np.pi * freq[0]
        # k = 2 * np.pi / wavelength
        # theta = np.pi / 2
        # kx = k * np.cos(theta)
        # ky = k * np.sin(theta)
        # source = np.sin(kx + ky - w * T * dt)
        source = data_type(M.sin(2 * np.pi * freq[0] * dt * T), flag)  # plane wave
        ez_inc[source_start:source_end, 0:zero_range] = A * source  # (source_end - source_start)
        # ez_inc[source_start:source_end, 0:zero_range] = A * source  # (source_end - source_start)

    else:
        pass
        # ez_inc[source_start:source_end, zero_range] = 0.

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

    if T % frame_interval == 0:
        imax = omax = 0.
        pending = round(T * 100. / nsteps, 2)
        INTEGRATE.append(Pz)
        YY = np.trapz(INTEGRATE, axis=0, dx=1.0) / window
        # print(YY.shape)
        # print(np.sum(YY,axis=1))
        if len(INTEGRATE) >= window:
            del INTEGRATE[0]
        # measure_port = np.abs(YY)[probey, probex]

        measure_port_out = [0.] * (len(probey))
        measure_port_in = [0.] * (len(probey))
        g = 0
        h = 0

        for g in range(0, len(np.array(probex_out)[0])):
            measure_port_out += np.abs(YY)[probey, np.array(probex_out)[:, g]]
            omax += np.max(measure_port_out)

        for h in range(0, len(np.array(probex_in)[0])):
            measure_port_in += np.abs(YY)[probey, np.array(probex_in)[:, h]]
            imax += np.max(measure_port_in)

        imax /= len(np.array(probex_in)[0])
        imax /= window * correction
        omax /= len(np.array(probex_out)[0])
        omax /= window * correction

        measure_port_out /= len(np.array(probex_out)[0])
        measure_port_out /= window * correction
        measure_port_out *= second_correction

        measure_port_in /= len(np.array(probex_in)[0])
        measure_port_in /= window * correction
        measure_port_in *= second_correction

        # title = ay.annotate("Time :" + '{:<.4e}'.format(T * dt * 1 * 10 ** 15) + " fs", (1, 0.5),
        #                     xycoords=ay.get_window_extent, xytext=(-round(JE * 2), IE - 5),
        #                     textcoords="offset points", fontsize=9, color='white')
        # ay.set(xlim=(-ic, ic), ylim=(-jc, jc))
        # input_max = np.max(measure_port_in)
        # if pending > 50 and input_max < 0.95:
        #     port_offset_start -= 2
        #     port_offset_stop -= 1
        # elif pending > 50 and input_max > 1.05:
        #     port_offset_start += 2
        #     port_offset_stop += 1
        # else:
        #     pass

        opoint = np.sum(measure_port_out)
        # print(ppoint)
        if opoint >= 10.:
            stop_sim += 1
            if stop_sim >= 15:  # int(IE / 100):
                sys.stdout.write('\r')
                sys.stdout.write("Pending...100 %")
                sys.stdout.flush()
                break
        else:
            pass

        # cm.nipy_spectral
        ims2 = ay.imshow(Pz, cmap=cm.nipy_spectral, extent=[0, JE, 0, IE], aspect='auto')
        ay.set_ylim([IE * 0.4, IE * 0.6])
        # ims2.set_interpolation('kaiser')
        ims4 = ay.scatter(x_points, y_points, c='grey', s=1, alpha=0.01)
        ims_oport_start = ay.scatter(np.array(probex_out)[:, 0], probey, c='red', s=5, alpha=0.05)
        ims_oport_stop = ay.scatter(np.array(probex_out)[:, g], probey, c='red', s=5, alpha=0.05)
        ims_iport_start = ay.scatter(np.array(probex_in)[:, 0], probey, c='g', s=5, alpha=0.05)
        ims_iport_stop = ay.scatter(np.array(probex_in)[:, h], probey, c='g', s=5, alpha=0.05)
        # ims6, = az.plot(MaxField[mf_id], 'r')  # field distribiution

        ims_out, = az.plot(measure_port_out, 'r', alpha=0.85)
        ims_in, = az.plot(measure_port_in, '-.g', alpha=0.85)
        az.set_ylim([-0.1, 1.1])

        # FFT CALCULATION
        # fft_out = fft.fftshift(fft.fft(torch.from_numpy(YY[:, measx])))
        # fft_out[1] = 0
        # fft_out[0] = 0
        # fft_res = torch.abs(fft_out)
        # fft_len = fft.fftfreq(len(fft_res), 1 / sr * 10)[:len(fft_res) // 2]
        # fft_result = 2.0 / len(fft_res) * torch.abs(fft_res[0:len(fft_res) // 2])
        # ims7, = ax.plot(fft_len, fft_result, 'g')
        # FFT CALCULATION

        ims.append([ims2, ims4, ims_oport_start, ims_oport_stop, ims_iport_start, ims_iport_stop, ims_out,
                    ims_in])  # , title])
        sys.stdout.write('\r')
        sys.stdout.write("Pending..." + str(pending) + " %")
        sys.stdout.flush()

# ax.set_xscale('log')
# ax.grid(True)
ay.set_xlabel("x [um]")
ay.set_ylabel("y [um]")
# ax.set_xlabel("Frequency [Hz]")
# ax.set_ylabel("Power [W]")
az.set_xlabel("y [um]")
az.set_ylabel("Pfwd")

xlabels = [item.get_text() for item in ay.get_xticklabels()]
ylabels = [item.get_text() for item in ay.get_yticklabels()]
xlab = [float(x) * dx / correction for x in xlabels]
ylab = [float(y) * dx / correction - (dx * IE / 2) / correction for y in ylabels]
ay.set_xticklabels(xlab)
ay.set_yticklabels(ylab)

az.grid(True)
az.locator_params(axis='x', nbins=6)
zlabels = [item.get_text() for item in az.get_xticklabels()]
zlab = [float(z) * dx / correction - (float(zlabels[-1]) * dx / 2) / correction for z in zlabels[1:]]

zfin = [float(z)*dx for z in zlabels[1:-1]]


zlab_bins = np.arange(start=-zfin[-1], stop=zlab[-1], step=2*zlab[-1]/6)
# print(zlab_bins )
# az.axis( xmin = zlab[0], xmax = zlab[-1])
# az.tick_params(axis='x', which='major', labelsize=8)
# az.set_xticks([zlab_bins[0],zlab_bins[1],zlab_bins[2],zlab_bins[3],zlab_bins[4]])
# print(zlab_bins)
az.set_xticklabels([' ',str(zlab_bins[0]), str(zlab_bins[1]), str(zlab_bins[2]), str(zlab_bins[3]), str(zlab_bins[4]), str(zlab_bins[5])])
# az.set_xticklabels(az.xaxis.get_majorticklabels())#, rotation=90)
az.legend(['out', 'in'], loc='best')

# zlab = [float(z) * dx - float(zlabels[-1]) * dx / 2 if z.isnumeric() else -float(z[1:]) * dx - float(zlabels[-1]) * dx / 2 for z in zlabels[:]]


e = time.time()
print("\nTime brutto : " + str((e - s)) + "[s]")
print("Time netto SUM : " + str(nett_time_sum) + "[s]")
file_name = "2d_fdtd_MMI_4.25um"
# file_name = "./" + file_name + '.gif'
file_name = "./" + file_name + '.gif'
ani = animation.ArtistAnimation(fig, ims, interval=30, blit=True  , repeat=False)
# ani.save(file_name, writer='pillow', fps=30, dpi=100)
# ani.save(file_name + '.mp4', fps = 30, extra_args = ['-vcodec', 'libx264'])

# ani.save(file_name, writer="imagemagick", fps=30)
print("Plotting...")
plt.show()
