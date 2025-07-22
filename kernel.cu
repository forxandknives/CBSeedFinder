#include "Random.cuh"
#include "CreateSeed.cuh"
#include "Extents.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "float.h"

#include <stdio.h>

#include <iostream>

__device__ void dummy();

__global__ void testFunction(int* outputArray, float* extents, uint8_t* forest);

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

    int i = 0;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    printf("  Memory Clock Rate (MHz): %d\n", prop.memoryClockRate / 1024);
    printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %.1f\n", 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
    printf("  Total global memory (Gbytes) %.1f\n", (float)(prop.totalGlobalMem) / 1024.0 / 1024.0 / 1024.0);
    printf("  Shared memory per block (Kbytes) %.1f\n", (float)(prop.sharedMemPerBlock) / 1024.0);
    printf("  minor-major: %d-%d\n", prop.minor, prop.major);
    printf("  Warp-size: %d\n", prop.warpSize);
    printf("  Concurrent kernels: %s\n", prop.concurrentKernels ? "yes" : "no");
    printf("  Concurrent computation/communication: %s\n\n", prop.deviceOverlap ? "yes" : "no");        


    ////////////////////////////////////EXTENTS///////////////////////////////
    const int32_t extentSize = 48960;
    float* hostExtents = (float*)malloc(extentSize * sizeof(float));   

    PopulateRoomExtents(hostExtents);   

    float* deviceExtents;
    c = cudaMalloc((void**)&deviceExtents, extentSize * sizeof(float));
    if (c != cudaSuccess) {
        printf("Failed to allocate memory for extents!\n");
        exit(1);
    }

    c = cudaMemcpy(deviceExtents, hostExtents, extentSize * sizeof(float), cudaMemcpyHostToDevice);
    if (c != cudaSuccess) {
        printf("Failed to copy extents to gpu!\n");
        exit(1);
    }
    ////////////////////////////////////EXTENTS///////////////////////////////

    ////////////////////////////////////FOREST///////////////////////////////
    int32_t totalForestDataSize = 3380;
    uint8_t* forestData = (uint8_t*)malloc(totalForestDataSize * sizeof(uint8_t));

    PopulateForestData(forestData);

    uint8_t* deviceForestData;
    c = cudaMalloc((void**)&deviceForestData, totalForestDataSize * sizeof(uint8_t));
    if (c != cudaSuccess) {
        printf("Failed to allocate memory for forest data!\n");
        exit(1);
    }

    c = cudaMemcpy(deviceForestData, forestData, totalForestDataSize * sizeof(uint8_t), cudaMemcpyHostToDevice);
    if (c != cudaSuccess) {
        printf("Failed to copy forest data to gpu!\n");
        exit(1);
    }
    ////////////////////////////////////FOREST///////////////////////////////

    printf("starting!\n");

    testFunction <<<1, arraySize>>> (cudaOutput, deviceExtents, deviceForestData);    

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

    free(hostExtents);
    cudaFree(deviceExtents);

}

__global__ void testFunction(int* outputArray, float* extents, uint8_t* forest) {
    int block = blockIdx.x + blockIdx.y * gridDim.x;
    int threadNumber = block * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;         

    //god help me
    rnd_state rnd_state;    
    bbRandom bb = bbRandom();
    bb.bbSeedRnd(&rnd_state, 1022940408/*threadNumber*/);

    int a = threadNumber;        

    extern __shared__ RoomTemplates rts[roomTemplateAmount];

    //we want the first thread of each block to spawn the room templates;
    if (threadIdx.x == 0) {
        CreateRoomTemplates(rts);         
    }       

    __syncthreads();

    InitNewGame(&bb, &rnd_state, rts, extents, forest);

    outputArray[threadNumber] = a;      

    //TODO:
    //Once everything is working see if we can reduce the data type of some variables
    //from int32_t to int8_t or int16_t depending on the known max-limit of those variables.

    //See if we can just make a global rnd_state varaible instead of the stupid pointer stuff.

}

__device__ void dummy() {};