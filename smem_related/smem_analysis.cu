#include <iostream>
#include <cuda_runtime.h>

__global__ void smem_test(int begin_index)
{
    __shared__ __align__(16) char smem_data[32];
    // printf("111\n");
    smem_data[threadIdx.x] = threadIdx.x;

    if (threadIdx.x == 0)
    {
        int4 vec_val = *((int4*)(&smem_data[begin_index]));
        printf("%d, %d, %d, %d\n", vec_val.x, vec_val.y, vec_val.z, vec_val.w);
    }
}


int main()
{
    cudaError_t err = cudaSuccess;
    

    smem_test<<<1,32>>>(0);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch smem_test kernel (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
      }
    smem_test<<<1,32>>>(1);
    cudaDeviceSynchronize();
    cudaError_t err1 = cudaGetLastError();
    if (err1 == cudaSuccess) {
        printf("222\n");
    }
    if (err1 != cudaSuccess) {
        fprintf(stderr, "Failed to launch smem_test kernel (error code %s)!\n",
                cudaGetErrorString(err1));
        exit(EXIT_FAILURE);
      }
    
    printf("111\n");
}