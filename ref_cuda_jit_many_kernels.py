import math as M
import time
from numba import cuda, vectorize, guvectorize
from numba import void, uint8, uint32, uint64, int32, int64, float32, float64, f8
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt, animation, offsetbox
from matplotlib.offsetbox import AnchoredText
import matplotlib.cm as cm
from C import C
import cupy as cp
from matplotlib.patches import Circle

# mpl.use('Agg')

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


flag = 1
data_type1 = np.float32

cc = C()

IE = JE = 60  # size and PML parameter loop
npml = 8
NFREQS = 3
freq = [data_type(0, 1)] * NFREQS
for n in range(0, NFREQS):
    freq[n] = data_type(450E12 * (n + 1), flag)  # red light (666nm) + H
arg = [data_type(0, flag)] * NFREQS

# highest_er = np.float32(50)  # ~biological tissue er for 500 MHz
# ddx = data_type((cc.c0 / min(freq) / (10 * cc.nSiO2)), flag)  # Cells Size
dx = 10
wavelength = (cc.c0 / (min(freq) * cc.nSiO2))
ddx = data_type(wavelength / dx, flag)  # Cells Size
print(ddx)

# dt = data_type(1 / (M.sqrt(1 / pow(ddx, 2)) * cc.c0), flag)  # Time Steps (# 2 * cc.c0 is to small??)
v = cc.c0 / cc.nSiO2
#   CFL stability conditio - Lax Equivalence Theorem
# dt = data_type((ddx / cc.c0) * 1 / M.sqrt(2),flag)
dt = 1 / (v * M.sqrt(1 / (ddx ** 2) + 1 / (ddx ** 2)))

print(dt)
for n in range(0, NFREQS):
    arg[n] = 2 * M.pi * freq[n] * dt

z_max = data_type(0, 1)
epsz = data_type(8.854E-12, flag)
spread = data_type(8, flag)
t0 = data_type(20, flag)
# print(ic.shape)
ic = IE / 2
jc = JE / 2
ia = 7  # total scattered field boundaries
ib = IE - ia - 1
ja = 7
jb = JE - ja - 1
nsteps = 200
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
radius = data_type(5, flag)
epsilon = data_type(cc.nSiO2, flag)
sigma = data_type(cc.sigmaSiO2, flag)
for j in range(ja, jb):
    for i in range(ia, ib):
        xdist = ic - i
        ydist = jc - j
        dist = data_type(M.sqrt(pow(xdist, 2) + pow(ydist, 2)), flag)
        if dist <= radius:
            ga[i][j] = data_type(1 / (epsilon + (sigma * dt / epsz)), flag)
            gb[i][j] = data_type(sigma * dt / epsz, flag)

fig = plt.figure()
grid = plt.GridSpec(20, 20, wspace=10, hspace=0.6)
ax = fig.add_subplot(grid[:, :8], projection='3d')
ay = fig.add_subplot(grid[:, 12:])
# Cyclic Number of image snapping
frame_interval = 1
ims = []

tpb = (8, 8)
bpg = (dz.shape[0] // tpb[0] + 1, dz.shape[1] // tpb[0] + 1)
#bpg = (2,60)

# -------------------------------- KERNELS ---------------------------
@cuda.jit
def Ez_inc_CU(ez_inc, hx_inc):
    j = cuda.grid(1)
    if j + 1 < ez_inc.shape[0]:
        ez_inc[j] = ez_inc[j] + 0.5 * (hx_inc[j - 1] - hx_inc[j])


@cuda.jit
def Dz_CU(dz, hx, hy, gi2, gi3, gj2, gj3):
    i, j = cuda.grid(2)
    if i + 1 < dz.shape[0] and j + 1 < dz.shape[1]:
        dz[i][j] = gi3[i] * gj3[j] * dz[i][j] + \
                   gi2[i] * gj2[j] * 0.5 * \
                   (hy[i][j] - hy[i - 1][j] -
                    hx[i][j] + hx[i][j - 1])


@cuda.jit
def Dz_inc_val_CU(dz, hx_inc):
    ia = 7  # total scattered field boundaries
    ib = 52
    ja = 7
    jb = 52
    i = cuda.grid(1)
    if ia <= i <= ib:
        dz[i][ja] = dz[i][ja] + 0.5 * hx_inc[ja - 1]
        dz[i][jb] = dz[i][jb] - 0.5 * hx_inc[jb]


@cuda.jit
def Ez_Dz_CU(ez, ga, gb, dz, iz):
    i, j = cuda.grid(2)
    if i < ez.shape[0] and j < ez.shape[1]:
        ez[i][j] = ga[i][j] * (dz[i][j] - iz[i][j])
        iz[i][j] = iz[i][j] + gb[i][j] * ez[i][j]


@cuda.jit
def Hx_inc_CU(hx_inc, ez_inc):
    j = cuda.grid(1)
    if j < hx_inc.shape[0] - 1:
        hx_inc[j] = hx_inc[j] + .5 * (ez_inc[j] - ez_inc[j + 1])


@cuda.jit
def Hx_CU(ez, hx, ihx, fj3, fj2, fi1):
    i, j = cuda.grid(2)
    if i < hx.shape[0] - 1 and j < hx.shape[1] - 1:
        curl_e = ez[i][j] - ez[i][j + 1]
        ihx[i][j] = ihx[i][j] + curl_e
        hx[i][j] = fj3[j] * hx[i][j] + fj2[j] * \
                   (.5 * curl_e + fi1[i] * ihx[i][j])


@cuda.jit
def Hx_inc_val_CU(hx, ez_inc):
    ia = 7  # total scattered field boundaries
    ib = 52
    ja = 7
    jb = 52
    i = cuda.grid(1)
    if ia <= i <= ib:
        hx[i][ja - 1] = hx[i][ja - 1] + .5 * ez_inc[ja]
        hx[i][jb] = hx[i][jb] - .5 * ez_inc[jb]


@cuda.jit
def Hy_CU(hy, ez, ihy, fi3, fi2, fi1):
    # for j in range(0, JE):
    #     for i in range(0, IE - 1):
    #         curl_e = ez[i][j] - ez[i + 1][j]
    #         ihy[i][j] = ihy[i][j] + curl_e
    #         hy[i][j] = fi3[i] * hy[i][j] - fi2[i] * \
    #                    (.5 * curl_e + fi1[j] * ihy[i][j])
    i, j = cuda.grid(2)
    if i < hy.shape[0] - 1 and j < hy.shape[1]:
        curl_e = ez[i][j] - ez[i + 1][j]
        ihy[i][j] = ihy[i][j] + curl_e
        hy[i][j] = fi3[i] * hy[i][j] - fi2[i] * \
                   (.5 * curl_e + fi1[j] * ihy[i][j])


@cuda.jit
def Hy_inc_CU(hy, ez_inc):
    ia = 7  # total scattered field boundaries
    ib = 52
    ja = 7
    jb = 52
    j = cuda.grid(1)
    if ja <= j < jb:
        hy[ia - 1][j] = hy[ia - 1][j] - .5 * ez_inc[j]
        hy[ib][j] = hy[ib][j] + .5 * ez_inc[j]


@cuda.jit
def Power_Calc(Pz, ez, hy, hx):
    i, j = cuda.grid(2)
    # i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    # j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if i < Pz.shape[0] and j < Pz.shape[1]:
        Pz[i][j] = M.sqrt(M.pow(-ez[i][j] * hy[i][j], 2) + M.pow(ez[i][j] * hx[i][j], 2))


stream = cuda.stream()
# -------------------------------- KERNELS ---------------------------
for n in range(1, nsteps):
    net = time.time()
    T = T + 1
    # MAIND FDTD LOOP
    ez_incd, hx_incd = cuda.to_device(ez_inc, stream=stream), cuda.to_device(hx_inc, stream=stream)
    Ez_inc_CU[bpg, tpb](ez_incd, hx_incd)
    ez_inc, hx_inc = ez_incd.copy_to_host(stream=stream), hx_incd.copy_to_host(stream=stream)
    stream.synchronize()

    ez_inc[0] = ez_inc_low_m2
    ez_inc_low_m2 = ez_inc_low_m1
    ez_inc_low_m1 = ez_inc[1]
    ez_inc[JE - 1] = ez_inc_high_m2
    ez_inc_high_m2 = ez_inc_high_m1
    ez_inc_high_m1 = ez_inc[JE - 2]

    dzd, hxd, hyd, gi2d, gi3d, gj2d, gj3d = cuda.to_device(dz, stream=stream), cuda.to_device(hx,stream=stream), cuda.to_device(hy, stream=stream), cuda.to_device( gi2, stream=stream), cuda.to_device(gi3, stream=stream), cuda.to_device(gj2, stream=stream), cuda.to_device(gj3,stream=stream)
    Dz_CU[bpg, tpb](dzd, hxd, hyd, gi2d, gi3d, gj2d, gj3d)
    dz, hx, hy, gi2, gi3, gj2, gj3 = dzd.copy_to_host(stream=stream), hxd.copy_to_host(stream=stream), hyd.copy_to_host(stream=stream), gi2d.copy_to_host(stream=stream), gi3d.copy_to_host(stream=stream), gj2d.copy_to_host(stream=stream), gj3d.copy_to_host(stream=stream)
    stream.synchronize()
    pulse = data_type(M.sin(2 * M.pi * freq[0] * dt * T), flag)
    # pulse = data_type(M.exp(-.5 * (pow((t0 - T) / spread, 2))), flag)
    ez_inc[3] = pulse  # plane wave
    # dz[ic][5] = pulse
    # dz[ic][3] = pulse_freq
    # for i in range(ia, ib + 1):
    #     dz[i][ja] = dz[i][ja] + 0.5 * hx_inc[ja - 1]
    #     dz[i][jb] = dz[i][jb] - 0.5 * hx_inc[jb]
    dzd, hx_incd, IEd, JEd = cuda.to_device(dz, stream=stream), cuda.to_device(hx_inc, stream=stream), cuda.to_device(IE, stream=stream), cuda.to_device(JE, stream=stream)
    # print(IEd,IEd.copy_to_host())
    Dz_inc_val_CU[bpg, tpb](dzd, hx_incd)
    dz, hx_inc = dzd.copy_to_host(stream=stream), hx_incd.copy_to_host(stream=stream)
    stream.synchronize()
    # Calculate the Ez field from Dz field
    # for j in range(0, JE):
    #     for i in range(0, IE):
    #         ez[i, j] = ga[i, j] * (dz[i, j] - iz[i, j])
    #         iz[i, j] = iz[i, j] + gb[i, j] * ez[i, j]
    ezd, gad, gbd, dzd, izd = cuda.to_device(ez, stream=stream), cuda.to_device(ga, stream=stream), cuda.to_device(gb, stream=stream), cuda.to_device(dz, stream=stream), cuda.to_device(iz, stream=stream)
    Ez_Dz_CU[bpg, tpb](ezd, gad, gbd, dzd, izd)
    ez, ga, gb, dz, iz = ezd.copy_to_host(stream=stream), gad.copy_to_host(stream=stream), gbd.copy_to_host(stream=stream), dzd.copy_to_host(stream=stream), izd.copy_to_host(stream=stream)
    stream.synchronize()
    # Calc the incident Hx
    # for j in range(0, JE - 1):
    #     hx_inc[j] = hx_inc[j] + .5 * (ez_inc[j] - ez_inc[j + 1])
    hx_incd, ez_incd = cuda.to_device(hx_inc, stream=stream), cuda.to_device(ez_inc, stream=stream)
    Hx_inc_CU[bpg, tpb](hx_incd, ez_incd)
    hx_inc, ez_inc = hx_incd.copy_to_host(stream=stream), ez_incd.copy_to_host(stream=stream)
    stream.synchronize()
    # stream.synchronize()
    # Calculate the Hx field
    # for j in range(0, JE - 1):
    #     for i in range(0, IE - 1):
    #         curl_e = ez[i][j] - ez[i][j + 1]
    #         ihx[i][j] = ihx[i][j] + curl_e
    #         hx[i][j] = fj3[j] * hx[i][j] + fj2[j] * \
    #                    (.5 * curl_e + fi1[i] * ihx[i][j])
    ezd, hxd, ihxd, fj3d, fj2d, fi1d = cuda.to_device(ez, stream=stream), cuda.to_device(hx,stream=stream), cuda.to_device(ihx, stream=stream), cuda.to_device(fj3, stream=stream), cuda.to_device(fj2, stream=stream), cuda.to_device(fi1,stream=stream)
    Hx_CU[bpg, tpb](ezd, hxd, ihxd, fj3d, fj2d, fi1d)
    ez, hx, ihx, fj3, fj2, fi1 = ezd.copy_to_host(stream=stream), hxd.copy_to_host(stream=stream), ihxd.copy_to_host(stream=stream), fj3d.copy_to_host(stream=stream), fj2d.copy_to_host(stream=stream), fi1d.copy_to_host(stream=stream)
    stream.synchronize()
    # Incident Hx val
    # for i in range(ia, ib + 1):
    #     hx[i][ja - 1] = hx[i][ja - 1] + .5 * ez_inc[ja]
    #     hx[i][jb] = hx[i][jb] - .5 * ez_inc[jb]
    hxd, ez_incd = cuda.to_device(hx, stream=stream), cuda.to_device(ez_inc, stream=stream)
    Hx_inc_val_CU[bpg, tpb](hxd, ez_incd)
    hx, ez_inc = hxd.copy_to_host(stream=stream), ez_incd.copy_to_host(stream=stream)
    stream.synchronize()
    # Calculate the Hy field
    # for j in range(0, JE):
    #     for i in range(0, IE - 1):
    #         curl_e = ez[i][j] - ez[i + 1][j]
    #         ihy[i][j] = ihy[i][j] + curl_e
    #         hy[i][j] = fi3[i] * hy[i][j] - fi2[i] * \
    #                    (.5 * curl_e + fi1[j] * ihy[i][j])
    hyd, ezd, ihyd, fi3d, fi2d, fi1d = cuda.to_device(hy, stream=stream), cuda.to_device(ez,stream=stream), cuda.to_device(ihy, stream=stream), cuda.to_device(fi3, stream=stream), cuda.to_device(fi2, stream=stream), cuda.to_device(fi1,stream=stream)
    Hy_CU[bpg, tpb](hyd, ezd, ihyd, fi3d, fi2d, fi1d)
    hy, ez, ihy, fi3, fi2, fi1 = hyd.copy_to_host(stream=stream), ezd.copy_to_host(stream=stream), ihyd.copy_to_host(stream=stream), fi3d.copy_to_host(stream=stream), fi2d.copy_to_host(stream=stream), fi1d.copy_to_host(stream=stream)
    stream.synchronize()

    # Incident Hy val
    # for j in range(ja, jb + 1):
    #     hy[ia - 1][j] = hy[ia - 1][j] - .5 * ez_inc[j]
    #     hy[ib][j] = hy[ib][j] + .5 * ez_inc[j]
    hyd, ez_incd, jad, jbd = cuda.to_device(hy, stream=stream), cuda.to_device(ez_inc, stream=stream), cuda.to_device(ja, stream=stream), cuda.to_device(jb, stream=stream)
    Hy_inc_CU[bpg, tpb](hyd, ez_incd)
    hy, ez_inc, ja, jb = hyd.copy_to_host(stream=stream), ez_incd.copy_to_host(stream=stream), jad.copy_to_host(stream=stream), jbd.copy_to_host(stream=stream)
    stream.synchronize()
    # Power Calc
    # for j in range(0, JE):
    #     for i in range(0, IE):
    #         Pz[i][j] = M.sqrt(M.pow(-ez[i][j] * hy[i][j], 2) + M.pow(ez[i][j] * hx[i][j], 2))
    Pzd, ezd, hyd, hxd = cuda.to_device(Pz, stream=stream), cuda.to_device(ez, stream=stream), cuda.to_device(hy, stream=stream), cuda.to_device(hx, stream=stream)
    Power_Calc[bpg, tpb](Pzd, ezd, hyd, hxd)
    Pz, ez, hy, hx = Pzd.copy_to_host(stream=stream), ezd.copy_to_host(stream=stream), hyd.copy_to_host(stream=stream), hxd.copy_to_host(stream=stream)
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
        # xi = np.arange(x.min(), x.max() + 1)
        # yi = np.arange(y.min(), y.max() + 1)
        X, Y = np.meshgrid(x, y)
        Z = Pz[:][:]  # Power - W/m^2
        # Z = ez[:][:] # field E
        ims1 = ax.plot_surface(X * ddx * 1 * 10 ** 6, Y * ddx * 1 * 10 ** 6, Z, rstride=5, cstride=5, cmap=cm.inferno)
        # ims1 = ax.plot_wireframe(X * ddx * 1 * 10 ** 6, Y * ddx * 1 * 10 ** 6, Z, rstride=15, cstride=15) ims1 =
        # ax.scatter(X[::10] * ddx * 1 * 10 ** 6, Y[::10] * ddx * 1 * 10 ** 6, Z[::10],zdir='z', s=20, c='red',
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
        # print("Punkt : " + str(T))

ax.set_xlabel("x [um]")
ax.set_ylabel("y [um]")
ax.set_zlabel("Power [W/m2]")
ay.set_xlabel("x [um]")
ay.set_ylabel("y [um]")

e = time.time()
print("Time brutto : " + str((e - s)) + "[s]")
print("Time netto SUM : " + str((nett_time_sum)) + "[s]")
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
