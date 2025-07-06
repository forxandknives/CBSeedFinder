#include "Random.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "float.h"

#include <stdio.h>

#include <iostream>

__device__ void dummy();

__global__ void testFunction(int* outputArray);

int main()
{
    constexpr uint64_t TOTAL_SEEDS = 512;//(1ULL << 31);
    constexpr uint32_t THREADS_PER_BLOCK = 256;
    constexpr uint32_t BLOCKS_PER_RUN = TOTAL_SEEDS / THREADS_PER_BLOCK;
  
    cudaError_t c;

    //_control87();

    const int arraySize = 20;    
    int* cudaOutput = 0;
    int* output = (int*)malloc(sizeof(int) * arraySize);   

    c = cudaMalloc(&cudaOutput, arraySize * sizeof(int));
    if (c != cudaSuccess) {
        printf("Failed to allocate cuda mem!\n");
        exit(1);
    }

    testFunction << < 1, arraySize >> > (cudaOutput);

    c = cudaGetLastError();
    if (c != cudaSuccess) {
        printf("kernel error!\n");
        exit(1);
    }

    c = cudaMemcpy(output, cudaOutput, arraySize * sizeof(int), cudaMemcpyDeviceToHost);
    if (c != cudaSuccess) {
        printf("failed to copy!\n");
        exit(1);
    }

    printf("\nOUTPUT\n");

    for (uint32_t i = 0; i < arraySize; i++) {
        printf("Thread %d: Output: %d\n", i, output[i]);
    }

    free(output);
    cudaFree(cudaOutput);

}

__global__ void testFunction(int* outputArray) {
    int block = blockIdx.x + blockIdx.y * gridDim.x;
    int threadNumber = block * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;       
    
    //god help me
    rnd_state rnd_state;    
    bbRandom bb = bbRandom();
    bb.bbSeedRnd(&rnd_state, threadNumber);

    //printf("thread: %d rnd_state: %d\n", threadNumber, bb.bbRndSeed());

    int a = bb.bbRand(&rnd_state, 0, 100);      
    
    outputArray[threadNumber] = a;
}
__device__ void dummy() {};