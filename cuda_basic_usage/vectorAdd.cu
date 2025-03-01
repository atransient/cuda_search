#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>


__global__ void vectorAdd(const float *A, const float *B, float *C,
                          int numElements) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < numElements) {
    C[i] = A[i] + B[i] + 0.0f;
  }
}

int main() {
    int numElements = 5000;
    int showElements = 10;
    size_t size = numElements * sizeof(float);

    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);
    for (int i = 0; i < numElements; ++i)
    {
        h_A[i] = i + 0.5;
        h_B[i] = i + 2.0;
    }

    float *d_A = NULL;
    float *d_B = NULL;
    float *d_C = NULL;

    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 32 * 4;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    int error_num = 0;
    for(int i = 0; i < numElements; ++i)
    {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-6)++error_num;
    }
    printf("error num is: %d\n", error_num);
    printf("show value for %d num\n", showElements);
    for (int i = 0; i < showElements; ++i)
    {
        printf("h_A[%d] is : %f, h_B[%d] is : %f, h_C[%d] is : %f\n", i,h_A[i],i,h_B[i],i,h_C[i]);
    }
}