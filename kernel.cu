#include "Random.cuh"
#include "CreateSeed.cuh"

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

    const int arraySize = 1;    
    int* cudaOutput = 0;
    int* output = (int*)malloc(sizeof(int) * arraySize);   

    c = cudaMalloc(&cudaOutput, arraySize * sizeof(int));
    if (c != cudaSuccess) {
        printf("Failed to allocate cuda mem!\n");
        exit(1);
    }

    printf("starting!\n");

    testFunction <<<1, arraySize>>> (cudaOutput);    

    printf("ended!\n");

    c = cudaGetLastError();
    if (c != cudaSuccess) {
        printf("kernel error!\n");
        exit(1);
    }

    c = cudaMemcpy(output, cudaOutput, arraySize * sizeof(int), cudaMemcpyDeviceToHost);
    if (c != cudaSuccess) {
        printf("failed to copy!\n");
        printf("%s\n", cudaGetErrorString(c));
        exit(1);
    }

    printf("\nOUTPUT\n");

    for (uint32_t i = 0; i < arraySize; i++) {
        printf("Thread %d: Output: %d\n", i, output[i]);
    }

    printf("at end\n");

    free(output);
    cudaFree(cudaOutput);

}

__global__ void testFunction(int* outputArray) {
    int block = blockIdx.x + blockIdx.y * gridDim.x;
    int threadNumber = block * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;       
    
    //god help me
    rnd_state rnd_state;    
    bbRandom bb = bbRandom();
    bb.bbSeedRnd(&rnd_state, 100/*threadNumber*/);

    int a = threadNumber;        

    extern __shared__ RoomTemplates rts[roomTemplateAmount];

    //we want the first thread of each block to spawn the room templates;
    if (threadIdx.x == 0) {
        CreateRoomTemplates(rts);       
    }       

    __syncthreads();

    InitNewGame(&bb, &rnd_state, rts);

    outputArray[threadNumber] = a;      

    //TODO:
    //Once everything is working see if we can reduce the data type of some variables
    //from int32_t to int8_t or int16_t depending on the known max-limit of those variables.

    //See if we can just make a global rnd_state varaible instead of the stupid pointer stuff.

}
__device__ void dummy() {};