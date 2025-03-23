#include <iostream>
#include <cuda_runtime.h>

#include "data_relate.h"
using std::cout, std::endl;


__inline__ uint ceil_div(uint val1, uint val2)
{
    uint res = val1 / val2;
    if (val1 % val2 != 0) ++res;
    return res;
}

__device__ __inline__ uint flattening(uint idx_x, uint idx_y, uint y_dim)
{
    return idx_x * y_dim + idx_y;
}

__global__ void matmul_kernel(float* A, float* B, float* C, uint M, uint N, uint K)
{
    uint idx_x = blockDim.x * blockIdx.x + threadIdx.x;
    uint idx_y = blockDim.y * blockIdx.y + threadIdx.y;
    if (idx_x >= M || idx_y >= N)
        return;
    float res = 0.0;
    for (uint i = 0; i < K; ++i)
    {
        res += A[flattening(idx_x, i, K)] * B[flattening(i, idx_y, N)];
    }
    C[flattening(idx_x, idx_y, N)] = res;
}


int main()
{
    const uint M = 64;
    const uint N = 32;
    const uint K = 16;
    Surface<float> A_tensor(M * K, false);
    Surface<float> B_tensor(K * N, false);
    Surface<float> C_tensor(M * N, true);
    dim3 block(32,16,1);
    dim3 grid(ceil_div(M, block.x), ceil_div(N, block.y), 1);
    matmul_kernel<<<grid, block>>>(A_tensor.devPtr, B_tensor.devPtr, C_tensor.devPtr, M, N, K);
    cudaDeviceSynchronize();
    C_tensor.d2h();
    float* res_val = C_tensor.hostPtr;
    for (int i = 0; i < N; ++i)
    {
        cout << res_val[i] << " ";
    }
    cout << endl;
}