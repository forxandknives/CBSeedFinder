#include "Random.cuh"
#include "CreateSeed.cuh"
#include "Extents.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "float.h"

#include <stdio.h>

#include <iostream>
#include <chrono>

__device__ void dummy();

__global__ void testFunction(int32_t offset, int* outputArray, float* extents, uint8_t* forest);

int main()
{      
    cudaError_t c;      

    /*int i = 0;

    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, i);
    printf("Major revision number:         %d\n", devProp.major);
    printf("Minor revision number:         %d\n", devProp.minor);
    printf("Name:                          %s\n", devProp.name);
    printf("Total global memory:           %lu\n", devProp.totalGlobalMem);
    printf("Total shared memory per block: %lu\n", devProp.sharedMemPerBlock);
    printf("Total registers per block:     %d\n", devProp.regsPerBlock);
    printf("Warp size:                     %d\n", devProp.warpSize);
    printf("Maximum memory pitch:          %lu\n", devProp.memPitch);
    printf("Maximum threads per block:     %d\n", devProp.maxThreadsPerBlock);
    for (int i = 0; i < 3; ++i)
        printf("Maximum dimension %d of block:  %d\n", i, devProp.maxThreadsDim[i]);
    for (int i = 0; i < 3; ++i)
        printf("Maximum dimension %d of grid:   %d\n", i, devProp.maxGridSize[i]);
    printf("Clock rate:                    %d\n", devProp.clockRate);
    printf("Total constant memory:         %lu\n", devProp.totalConstMem);
    printf("Texture alignment:             %lu\n", devProp.textureAlignment);
    printf("Concurrent copy and execution: %s\n", (devProp.deviceOverlap ? "Yes" : "No"));
    printf("Number of multiprocessors:     %d\n", devProp.multiProcessorCount);
    printf("Kernel execution timeout:      %s\n", (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));*/

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

    const int arraySize = 1;
    int* cudaOutput = 0;
    int* output = (int*)malloc(sizeof(int) * arraySize);

    c = cudaMalloc(&cudaOutput, arraySize * sizeof(int));
    if (c != cudaSuccess) {
        printf("Failed to allocate cuda mem!\n");
        exit(1);
    }

    int32_t offset = 0;

    constexpr uint64_t TOTAL_SEEDS = 2147483648;    
    constexpr uint32_t THREADS_PER_BLOCK = 1024;
    constexpr uint32_t BLOCKS_PER_RUN = TOTAL_SEEDS / THREADS_PER_BLOCK;

    int gridSize;
    int minGridSize;
    int blockSize;

    int totalThreads = 524288;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, testFunction, 0, totalThreads);

    gridSize = (totalThreads + blockSize - 1) / blockSize;

    printf("MINGRIDSIZE: %d\n", minGridSize);
    printf("GRIDSIZE: %d\n", gridSize);
    printf("BLOCKSIZE: %d\n", blockSize);

    std::chrono::steady_clock::time_point start, end;

    printf("Launching Kernel!\n");

    uint32_t tempBlockAmount = 5;

    start = std::chrono::steady_clock::now();
    //DEBUG
    //testFunction << <1, 32>> > (offset, cudaOutput, deviceExtents, deviceForestData);
    
    //RELEASE
    testFunction <<<gridSize, blockSize>>> (offset, cudaOutput, deviceExtents, deviceForestData);    

    cudaDeviceSynchronize();

    end = std::chrono::steady_clock::now();

    double ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    printf("Kernel Stopped!\n");   

    printf("Took %f milliseconds for %d seeds.\n", ms, totalThreads);

    c = cudaGetLastError();
    if (c != cudaSuccess) {
        printf("kernel error!\n");
        printf("%s\n", cudaGetErrorString(c));
        exit(1);
    }

    /*c = cudaMemcpy(output, cudaOutput, arraySize * sizeof(int), cudaMemcpyDeviceToHost);
    if (c != cudaSuccess) {
        printf("failed to copy!\n");
        printf("%s\n", cudaGetErrorString(c));
        exit(1);
    }

    printf("\nOUTPUT\n");

    for (uint32_t i = 0; i < arraySize; i++) {
        printf("Thread %d: Output: %d\n", i, output[i]);
    }

    printf("at end\n");*/

    free(output);
    cudaFree(cudaOutput);

    free(hostExtents);
    cudaFree(deviceExtents);

}

__global__ void testFunction(int32_t offset, int* outputArray, float* extents, uint8_t* forest) {
    //int block = blockIdx.x + blockIdx.y * gridDim.x;
    //int threadNumber = block * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;         

    int32_t thread = blockIdx.x * blockDim.x + threadIdx.x;
       
    __shared__ RoomTemplates rts[roomTemplateAmount];

    //we want the first thread of each block to spawn the room templates;
    if (threadIdx.x == 0) {
        CreateRoomTemplates(rts);         
    }       

    __syncthreads();

    //if (thread == 0 || thread == 2147483647) return;

    InitNewGame(thread, rts, extents, forest);

    //printf("Thread %d reached end.\n", thread);

    //outputArray[threadIdx.x] = InitNewGame(thread, rts, extents, forest);   
}

__device__ void dummy() {};