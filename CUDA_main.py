import time
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt, animation, offsetbox
from matplotlib.offsetbox import AnchoredText
import matplotlib.cm as cm
from C import C
import cupy as cp
from numba import cuda
#from numba import cuda as cp
import math as mm
from matplotlib.patches import Circle

s = time.time()


def data_type(data, flag):
    if flag == 1:
        return cp.float32(data)
    elif flag == 2:
        return cp.float64(data)
    elif flag == 3:
        return cp.unt32(data)
    else:
        return cp.int32(data)


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


# mpl.use('Agg')
cp.cuda.Device(0)
device = cp.cuda.runtime.getDeviceCount()
print("Used devices :" + str(device))
# cp.cuda.Device(1).use()

flag = 1
data_type1 = cp.float32

cc = C()

IE = JE = 60  # size and PML parameter loop
# print(cp.result_type(IE))
N = IE
M = IE
npml = 8
NFREQS = 3
freq = [data_type(0., 1)] * NFREQS
for n in range(0, NFREQS):
    freq[n] = data_type(450E12 * (n + 1), flag)  # red light (666nm) + H
arg = [data_type(0, flag)] * NFREQS

# highest_er = np.float32(50)  # ~biological tissue er for 500 MHz
ddx = (cc.c0 / min(freq) / (10 * cc.nSiO2))  # Cells Size

# print(np.result_type(ddx))

# dt = data_type(1 / (M.sqrt(1 / pow(ddx, 2)) * cc.c0), flag)  # Time Steps (# 2 * cc.c0 is to small??)

#   CFL stability condition - Lax Equivalence Theorem
dt = data_type(((ddx / cc.c0) * 1 / np.sqrt(2)), flag)
print(dt)
for n in range(0, NFREQS):
    arg[n] = 2 * cp.pi * freq[n] * dt

z_max = 0
epsz = 8.854E-12
spread = 8
t0 = 20
ic = round(IE / 2)
jc = round(JE / 2)
ia = 7  # total scattered field boundaries
ib = IE - ia - 1
ja = 7
jb = JE - ja - 1
nsteps = 200
T = data_type(0, -1)
nett_time_sum = 0
pulse = data_type(0., flag)
ez_inc_low_m2 = data_type(0., flag)
ez_inc_low_m1 = data_type(0., flag)

ez_inc_high_m2 = data_type(0., flag)
ez_inc_high_m1 = data_type(0., flag)

dz = cp.zeros((N, M), dtype=data_type1)
ez = cp.zeros((N, M), dtype=data_type1)
iz = cp.zeros((N, M), dtype=data_type1)
hx = cp.zeros((N, M), dtype=data_type1)
hy = cp.zeros((N, M), dtype=data_type1)
ihx = cp.zeros((N, M), dtype=data_type1)
ihy = cp.zeros((N, M), dtype=data_type1)
ga = cp.ones((N, M), dtype=data_type1)
gb = cp.zeros((N, M), dtype=data_type1)
Pz = cp.zeros((N, M), dtype=data_type1)

gi2 = cp.ones((N,), dtype=data_type1)
gi3 = cp.ones((N,), dtype=data_type1)
fi1 = cp.zeros((N,), dtype=data_type1)
fi2 = cp.ones((N,), dtype=data_type1)
fi3 = cp.ones((N,), dtype=data_type1)

gj2 = cp.ones((M,), dtype=data_type1)
gj3 = cp.ones((M,), dtype=data_type1)
fj1 = cp.zeros((M,), dtype=data_type1)
fj2 = cp.ones((M,), dtype=data_type1)
fj3 = cp.ones((M,), dtype=data_type1)

ez_inc = cp.zeros((M,), dtype=data_type1)
hx_inc = cp.zeros((M,), dtype=data_type1)
GRID = (int(N / 1024) + 1, 1, 1)
BLOCK = (1024, 1, 1)
tpb = (8, 8)
bpg = (dz.shape[0] // tpb[0] + 1, dz.shape[1] // tpb[0] + 1)

# PML Definition
for i in range(npml):
    xnum = npml - i
    xd = npml
    xxn = xnum / xd
    xn = 0.33 * pow(xxn, 3)
    gi2[i] = 1. / (1. + xn)
    gi2[IE - 1 - i] = 1. / (1. + xn)
    gi3[i] = (1. - xn) / (1. + xn)
    gi3[IE - i - 1] = (1. - xn) / (1. + xn)
    xxn = (xnum - .5) / xd
    xn = 0.33 * pow(xxn, 3)
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
    xn = 0.33 * pow(xxn, 3)
    fj1[i] = xn
    fj1[JE - 2 - i] = xn
    fj2[i] = 1. / (1. + xn)
    fj2[JE - 2 - i] = 1. / (1. + xn)
    fj3[i] = (1. - xn) / (1. + xn)
    fj3[JE - 2 - i] = (1. - xn) / (1. + xn)

# Dielectric cylinder Specifications
radius = 7
epsilon = cc.nSiO2
sigma = cc.sigmaSiO2
for j in range(ja, jb - 1):
    for i in range(ia, ib - 1):
        xdist = ic - i
        ydist = jc - j
        dist = np.sqrt(pow(xdist, 2) + pow(ydist, 2))
        if dist <= radius:
            ga[i][j] = 1 / (epsilon + (sigma * dt / epsz))
            gb[i][j] = sigma * dt / epsz


# -------------------------------- KERNELS ---------------------------
@cuda.jit
def Ez_inc_CU(ez_inc, hx_inc, dz, hx, hy, gi2, gi3, gj2, gj3, freq, dt, T):
    i, j = cuda.grid(2)
    if i + 1 < dz.shape[0] and j + 1 < dz.shape[1]:
        ez_inc[j] = ez_inc[j] + 0.5 * (hx_inc[j - 1] - hx_inc[j])
        ez_inc[0] = ez_inc[1]
        ez_inc[JE - 1] = ez_inc[JE - 2]
        dz[i][j] = gi3[i] * gj3[j] * dz[i][j] + \
                   gi2[i] * gj2[j] * 0.5 * \
                   (hy[i][j] - hy[i - 1][j] -
                    hx[i][j] + hx[i][j - 1])
    cuda.syncthreads()
    pulse = mm.sin(2 * mm.pi * freq * dt * T)
    ez_inc[3] = pulse  # plane wave
    # pulse = data_type(M.exp(-.5 * (pow((t0 - T) / spread, 2))), flag)
    # dz[ic][5] = pulse
    # dz[ic][3] = pulse_freq


@cuda.jit
def Dz_inc_val_CU(dz, hx_inc, ia, ib):
    i = cuda.grid(1)
    if ia <= i <= ib:
        dz[i][ja] = dz[i][ja] + 0.5 * hx_inc[ja - 1]
        dz[i][jb] = dz[i][jb] - 0.5 * hx_inc[jb]
    cuda.syncthreads()



@cuda.jit
def Ez_Dz_CU(ez, ga, gb, dz, iz):
    i, j = cuda.grid(2)
    if i < ez.shape[0] and j < ez.shape[1]:
        ez[i][j] = ga[i][j] * (dz[i][j] - iz[i][j])
        iz[i][j] = iz[i][j] + gb[i][j] * ez[i][j]
    cuda.syncthreads()


@cuda.jit
def Hx_inc_CU(hx_inc, ez_inc):
    j = cuda.grid(1)
    if j < hx_inc.shape[0] - 1:
        hx_inc[j] = hx_inc[j] + .5 * (ez_inc[j] - ez_inc[j + 1])
    cuda.syncthreads()


@cuda.jit
def Hx_CU(ez, hx, ihx, fj3, fj2, fi1):
    i, j = cuda.grid(2)
    if i < hx.shape[0] - 1 and j < hx.shape[1] - 1:
        curl_e = ez[i][j] - ez[i][j + 1]
        ihx[i][j] = ihx[i][j] + curl_e
        hx[i][j] = fj3[j] * hx[i][j] + fj2[j] * \
                   (.5 * curl_e + fi1[i] * ihx[i][j])
    cuda.syncthreads()


@cuda.jit
def Hx_inc_val_CU(hx, ez_inc, ia, ib):
    i = cuda.grid(1)
    if ia <= i <= ib:
        hx[i][ja - 1] = hx[i][ja - 1] + .5 * ez_inc[ja]
        hx[i][jb] = hx[i][jb] - .5 * ez_inc[jb]
    cuda.syncthreads()


@cuda.jit
def Hy_CU(hy, ez, ihy, fi3, fi2, fi1):
    i, j = cuda.grid(2)
    if i < hy.shape[0] - 1 and j < hy.shape[1]:
        curl_e = ez[i][j] - ez[i + 1][j]
        ihy[i][j] = ihy[i][j] + curl_e
        hy[i][j] = fi3[i] * hy[i][j] - fi2[i] * \
                   (.5 * curl_e + fi1[j] * ihy[i][j])
    cuda.syncthreads()


@cuda.jit
def Hy_inc_CU(hy, ez_inc, ja, jb):
    j = cuda.grid(1)
    if ja <= j < jb:
        hy[ia - 1][j] = hy[ia - 1][j] - .5 * ez_inc[j]
        hy[ib][j] = hy[ib][j] + .5 * ez_inc[j]
    cuda.syncthreads()


@cuda.jit
def Power_Calc(Pz, ez, hy, hx):
    i, j = cuda.grid(2)
    # i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    # j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if i < Pz.shape[0] and j < Pz.shape[1]:
        Pz[i][j] = mm.sqrt(mm.pow(-ez[i][j] * hy[i][j], 2) + mm.pow(ez[i][j] * hx[i][j], 2))
    cuda.syncthreads()


# -------------------------------- KERNELS ---------------------------
fig = plt.figure()
grid = plt.GridSpec(20, 20, wspace=10, hspace=0.6)
ax = fig.add_subplot(grid[:, :8], projection='3d')
ay = fig.add_subplot(grid[:, 12:])
# Cyclic Number of image snapping
frame_interval = 1
ims = []

for n in range(1, nsteps):
    net = time.time()
    T = T + 1
    # MAIND FDTD LOOP
    # Ez incident field
    Ez_inc_CU[bpg, tpb](ez_inc, hx_inc, dz, hx, hy, gi2, gi3, gj2, gj3, freq[0], dt, T)

    Dz_inc_val_CU[bpg, tpb](dz, hx_inc, ia, ib)
    # Calculate the Ez field from Dz field
    Ez_Dz_CU[bpg, tpb](ez, ga, gb, dz, iz)

    # Calc the incident Hx
    Hx_inc_CU[bpg, tpb](hx_inc, ez_inc)
    # Calculate the Hx field
    Hx_CU[bpg, tpb](ez, hx, ihx, fj3, fj2, fi1)

    # Incident Hx val
    Hx_inc_val_CU[bpg, tpb](hx, ez_inc, ia, ib)

    # Calculate the Hy field
    Hy_CU[bpg, tpb](hy, ez, ihy, fi3, fi2, fi1)

    # Incident Hy val
    Hy_inc_CU[bpg, tpb](hy, ez_inc, ja, jb)

    Power_Calc[bpg, tpb](Pz, ez, hy, hx)

    netend = time.time()
    print("Time netto : " + str((netend - net)) + "[s]")
    nett_time_sum += netend - net
    # Drawing of the EM and FT plots
    if T % frame_interval == 0:
        title = ay.annotate("Time :" + '{:<.4e}'.format(T * dt * 1 * 10 ** 15) + " fs", (1, 0.5),
                            xycoords=ay.get_window_extent, xytext=(-round(JE * 2), IE - 5),
                            textcoords="offset points", fontsize=9, color='white')
        x = np.linspace(0, JE, JE)
        y = np.linspace(0, IE, IE)
        # xi = cp.arange(x.min(), x.max() + 1).get()
        # yi = cp.arange(y.min(), y.max() + 1).get()
        X, Y = np.meshgrid(x, y)
        Z = Pz[:][:].get()  # Power - W/m^2
        # Z = ez[:][:] # field E
        ims1 = ax.plot_surface(X * ddx * 1 * 10 ** 6, Y * ddx * 1 * 10 ** 6, Z, rstride=5, cstride=5, cmap=cm.inferno)
        # ims1 = ax.plot_wireframe(X * ddx * 1 * 10 ** 6, Y * ddx * 1 * 10 ** 6, Z, rstride=15, cstride=15) #
        # ims1 = ax.scatter(X[::10] * ddx * 1 * 10 ** 6, Y[::10] * ddx * 1 * 10 ** 6, Z[::10],zdir='z', s=20, c='red',
        # depthshade=True,)
        ims2 = ay.imshow(Z, cmap=cm.bwr, vmin=abs(Z).min(), vmax=abs(Z).max(),
                         extent=[0, JE * ddx * 1 * 10 ** 6, 0, IE * ddx * 1 * 10 ** 6])
        ims2.set_interpolation('bilinear')
        circ = plt.Circle((ic * ddx * 1 * 10 ** 6, jc * ddx * 1 * 10 ** 6), radius * ddx * 1 * 10 ** 6, color='silver',
                          alpha=0.3, fill=True)
        ims3 = ay.add_patch(circ)
        Xc, Yc, Zc, z_max = data_for_cylinder_along_z(ic, jc, radius, Z, ddx * 1 * 10 ** 6, z_max)

        cyli = ax.plot_surface(Xc, Yc, Zc, alpha=0.3, color='silver', shade=False)
        ims.append([ims1, ims2, ims3, title, cyli])
        print("Punkt : " + str(T))

cp.cuda.Stream.null.synchronize()
ax.set_xlabel("x [um]")
ax.set_ylabel("y [um]")
ax.set_zlabel("Power [W/m2]")
ay.set_xlabel("x [um]")
ay.set_ylabel("y [um]")
e = time.time()
print("Time brutto : " + str((e - s)) + "[s]")
print("Time netto SUM : " + str(nett_time_sum) + "[s]")
file_name = "2d_fdtd_Si_Cylinder"
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
