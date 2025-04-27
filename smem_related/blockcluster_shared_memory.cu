#include <stdio.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

__device__ bool thread0() {
    return (!threadIdx.x && !threadIdx.y && !threadIdx.z) && (!blockIdx.x && !blockIdx.y && !blockIdx.z);
}



// Distributed Shared memory histogram kernel
__global__ void clusterHist_kernel(const int nbins)
{
    extern __shared__ int smem[];
    namespace cg = cooperative_groups;
    int tid = cg::this_grid().thread_rank();

    // Cluster initialization, size and calculating local bin offsets.
    cg::cluster_group cluster = cg::this_cluster();
    unsigned int clusterBlockRank = cluster.block_rank();
    int cluster_size = cluster.dim_blocks().x;
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        printf("blockIdx.x : %d\n",blockIdx.x);
        
        printf("tid : %d\n",tid);
        printf("clusterBlockRank :  %d\n",clusterBlockRank);
        printf("cluster_size :  %d\n",cluster_size);
        smem[0] = 1234;
        printf("smem : %d\n", smem[0]);
    }
    cluster.sync();
    if (threadIdx.x == 0 && blockIdx.x == 1)
    {
        int *dst_smem = cluster.map_shared_rank(smem, 0);
        printf("blockIdx.x : %d\n", blockIdx.x);
        printf("dst_smem : %d\n", dst_smem[0]);
    }
    cluster.sync();
}

int main()
{
    int array_size = 64, threads_per_block = 32;
    // Launch via extensible launch
    cudaLaunchConfig_t config = {0};
    config.gridDim = array_size / threads_per_block;
    config.blockDim = threads_per_block;

    // cluster_size depends on the histogram size.
    // ( cluster_size == 1 ) implies no distributed shared memory, just thread block local shared memory
    int nbins = 2;
    int cluster_size = 2; // size 2 is an example here
    int nbins_per_block = nbins / cluster_size;

    //dynamic shared memory size is per block.
    //Distributed shared memory size =  cluster_size * nbins_per_block * sizeof(int)
    config.dynamicSmemBytes = nbins_per_block * sizeof(int);

    ::cudaFuncSetAttribute((void *)clusterHist_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, config.dynamicSmemBytes);

    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeClusterDimension;
    attribute[0].val.clusterDim.x = cluster_size;
    attribute[0].val.clusterDim.y = 1;
    attribute[0].val.clusterDim.z = 1;

    config.numAttrs = 1;
    config.attrs = attribute;

    cudaLaunchKernelEx(&config, clusterHist_kernel, nbins);
    cudaDeviceSynchronize();
}