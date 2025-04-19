#include <iostream>
#include <cuda_runtime.h>

#include "data_relate.h"
using std::cout, std::endl;

constexpr uint M = 64;
constexpr uint N = 32;
constexpr uint K = 16;

constexpr uint smem_M = 32;
constexpr uint smem_N = 16;
constexpr uint smem_K = K;

struct SharedMem {
    float smemA[smem_M][smem_K];
    float smemB[smem_K][smem_N];
};

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
    
    extern __shared__ char shared_bytes[];
    SharedMem* smem = reinterpret_cast<SharedMem*>(shared_bytes);
    uint idx_x = blockDim.x * blockIdx.x + threadIdx.x;
    uint idx_y = blockDim.y * blockIdx.y + threadIdx.y;
    if (idx_x >= M || idx_y >= N)
        return;
    uint begin_m = blockIdx.x * smem_M;
    uint begin_n = blockIdx.y * smem_N;
    if (threadIdx.y == 0)
    {
        for (int i = 0; i < K; ++i)
        {
            smem->smemA[threadIdx.x][i] = A[flattening(begin_m + threadIdx.x, i, K)];
        }
    }
    uint tid = threadIdx.y * blockDim.x + threadIdx.x;
    if (tid < K)
    {
        for (int i = 0; i < smem_N; ++i)
        {
            smem->smemB[tid][i] = B[flattening(tid, i + begin_n, N)];
        }
    }
    __syncthreads();
    float res = 0.0;
    for (uint i = 0; i < K; ++i)
    {
        res += smem->smemA[threadIdx.x][i] * smem->smemB[i][threadIdx.y];
    }
    C[flattening(idx_x, idx_y, N)] = res;
}


int main()
{
    Surface<float> A_tensor(M * K, false);
    Surface<float> B_tensor(K * N, false);
    Surface<float> C_tensor(M * N, true);
    dim3 block(smem_M,smem_N,1);
    dim3 grid(ceil_div(M, block.x), ceil_div(N, block.y), 1);
    matmul_kernel<<<grid, block, sizeof(SharedMem)>>>(A_tensor.devPtr, B_tensor.devPtr, C_tensor.devPtr, M, N, K);
    cudaDeviceSynchronize();
    C_tensor.d2h();
    float* res_val = C_tensor.hostPtr;
    for (int i = 0; i < N; ++i)
    {
        cout << res_val[i] << " ";
    }
    cout << endl;
}