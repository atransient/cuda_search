#include <stdio.h>
#include <cuda_runtime.h>

constexpr int W = 16;
constexpr int H = 32;

constexpr int SMEM_W = 16;
constexpr int SMEM_H = 16;

constexpr int print_block = 0;

__global__ void g2s(int *origin)
{
    
    __shared__ int smem[H][W];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int by = blockIdx.y;
    int *block_visit_val = origin + by * SMEM_H * SMEM_W;
    smem[ty][tx] = block_visit_val[ty * W + tx];
    __syncthreads();

    if (tx == 0 && ty == 0 && by == print_block)
    {
        for (int y = 0; y < SMEM_H; ++y)
        {
            for (int x = 0; x < SMEM_W; ++x)
            {
                printf("%4d  ",smem[y][x]);
            }
            printf("\n");
        }
    }
}

int main()
{
    int hostval[W * H] = {0};
    for (int i = 0; i < W * H; ++i)
    {
        hostval[i] = i;
    }

    int *gmem_val = nullptr;
    cudaMalloc(&gmem_val, H * W * sizeof(int));
    cudaMemcpy(gmem_val, hostval, H * W * sizeof(int), cudaMemcpyHostToDevice);

    dim3 block(SMEM_W,SMEM_H,1);
    dim3 grid(W / SMEM_W, H / SMEM_H,1);
    g2s<<<grid, block>>>(gmem_val);
    cudaDeviceSynchronize();
    cudaFree(gmem_val);
}