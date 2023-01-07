import cupy as cp
import numpy as np
from cupyx import jit
from numba import cuda

flag = 1
data_type1 = cp.float32


def data_type(data, flag):
    if flag == 1:
        return cp.float32(data)
    elif flag == 2:
        return cp.float64(data)
    else:
        return cp.uint32(data)


IE = JE = data_type(60, 0)
ez_inc_low_m2 = data_type(0., flag)
ez_inc_low_m1 = data_type(0., flag)

ez_inc_high_m2 = data_type(0., flag)
ez_inc_high_m1 = data_type(0., flag)

N = cp.uint32(IE)
M = cp.uint32(JE)

dz = cp.zeros((N, M), dtype=data_type1)
ez = cp.zeros((N, M), dtype=data_type1)
hx = cp.zeros((N, M), dtype=data_type1)
hy = cp.zeros((N, M), dtype=data_type1)
ihx = cp.zeros((N, M), dtype=data_type1)
ihy = cp.zeros((N, M), dtype=data_type1)
ga = cp.ones((N, M), dtype=data_type1)
gb = cp.ones((N, M), dtype=data_type1)
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

squared_diff = cp.ElementwiseKernel(
    'int32 x, int32 y',
    'int32 z',
    'z = (x - y) * (x - y)',
    'squared_diff'
)

x = cp.array([3, 2, 1, 5, 4, 3, 2])
y = cp.array([1, 2, 3, 1, 1, 1, 1])
m = squared_diff(x, y)

print(m)
print(m / 2)
print('!!!!!!!!!!!!!!!!')
Ez_inc_CU = cp.RawKernel(r'''
extern "C" __global__
void Ez_inc_kernel(float *vec1, float *vec2, int N)
    {
      int i = threadIdx.x+blockDim.x*blockIdx.x;
      if (i < N)
        vec1[i] = vec1[i] + 0.5 * (vec2[i - 1] - vec2[i]);
        // EZINC[j] += .5 * (HXINC[j - 1] - HXINC[j]);
    }
''', 'Ez_inc_kernel')

N = cp.int32(JE)
Ez_inc = cp.ones((N,), dtype=cp.float32)
Hz_inc = cp.ones((N,), dtype=cp.float32)
grid = (cp.int32(N / 1024) + 1, 1, 1)
block = (1024, 1, 1)
Ez_inc_CU(grid, block, (Ez_inc, Hz_inc, N))
K = cp.asnumpy(Ez_inc)
print(np.result_type(K))
print(K)
print('--------------')


@jit.rawkernel()
def Ez_inc_gpu(ez_inc, hx_inc, N):
    j = jit.threadIdx.x + jit.blockDim.x * jit.blockIdx.x
    ez_inc[j + 1] += 0.5 * (hx_inc[j] - hx_inc[j + 1])


Ez_inc_gpu(grid, block, (cp.array(Ez_inc), cp.array(Hz_inc), cp.array((N))))

print(Ez_inc)
print('????????????????')

Dz_inc_CU = cp.RawKernel(r'''
extern "C" 
{
__global__ void Dz_inc_kernel(float *dz, float *hx, float *hy, float *gi2, float *gi3,float *gj2,float *gj3,int N,int M) 
    {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        int j = blockDim.y * blockIdx.y + threadIdx.y;
        if (i+1 <= N && j+1 <= M)
        {
            dz[i,j] = gi3[i] * gj3[j] * dz[i,j] + gi2[i] * gj2[j] * 0.5 * (hy[i,j] - hy[i - 1,j] - hx[i,j] + hx[i,j - 1]);
        }
    }
}
''', 'Dz_inc_kernel')

# module = cp.RawModule(code=Dzinc)
# DZ = module.get_function('Dz_inc_kernel')
# print(DZ.attributes)
Dzinc(grid, block, (dz, hx, hy, gi2, gi3, gj2, gj3, N, M))
print(dz)

print('~~~~~~~~~~~~~~~~~~~~')


@cuda.jit
def Dz_inc_CU(dz, hx, hy, gi2, gi3, gj2, gj3):
    i, j = cuda.grid(2)
    if i+1 <= dz.shape[0] and j+1 <= dz.shape[1]:
        dz[i, j] = gi3[i] * gj3[j] * dz[i, j] + gi2[i] * gj2[j] * \
                   0.5 * (hy[i, j] - hy[i - 1, j] - hx[i, j] + hx[i, j - 1])


Dz_inc_CU[grid, block](dz, hx, hy, gi2, gi3, gj2, gj3)
print(dz)
