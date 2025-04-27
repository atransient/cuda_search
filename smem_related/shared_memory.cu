#include <stdio.h>
#include <cuda_runtime.h>

constexpr int W = 16;
constexpr int H = 16;

__global__ void g2s(int *origin)
{
    __shared__ int smem[H][W];
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    smem[ty][tx] = origin[ty * W + tx];
    __syncthreads();

    if (tx == 0 && ty == 0)
    {
        for (int y = 0; y < H; ++y)
        {
            for (int x = 0; x < W; ++x)
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

    dim3 block(H,W,1);
    dim3 grid(1,1,1);
    g2s<<<grid, block>>>(gmem_val);
    cudaDeviceSynchronize();
    cudaFree(gmem_val);
}