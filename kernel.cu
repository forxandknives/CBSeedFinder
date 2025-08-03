#include "Random.cuh"
#include "CreateSeed.cuh"
#include "Extents.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "float.h"

#include <stdio.h>

#include <iostream>
#include <chrono>
#include <vector>

__device__ void dummy();

__global__ void testFunction(uint32_t offset, int* outputArray, float* extents);

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
    /*int32_t totalForestDataSize = 3380;
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
    }*/
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


    int32_t gridSize;
    int32_t minGridSize;
    int32_t blockSize;

    //int32_t totalThreads = 524288;
    constexpr int32_t totalThreads = 1048576*2;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, testFunction, 0, totalThreads);

    gridSize = (totalThreads + blockSize - 1) / blockSize;

    uint32_t actualNumberOfThreads = gridSize * blockSize;

    printf("MINGRIDSIZE: %d\n", minGridSize);
    printf("GRIDSIZE: %d\n", gridSize);
    printf("BLOCKSIZE: %d\n", blockSize);

    int32_t kernels = 51;
    int32_t kernels2 = ceilf(2147483648 / actualNumberOfThreads);   

    std::vector<double> times(kernels);

    printf("Launching %d Kernels with %d Threads Per Kernel!\n", kernels, actualNumberOfThreads);       

    std::chrono::steady_clock::time_point start, end, start2, end2;

    start2 = std::chrono::steady_clock::now();

    #pragma unroll
    for (int32_t i = 0; i < kernels; i++) {

        start = std::chrono::steady_clock::now();

        offset = i * actualNumberOfThreads;

        //printf("i: %d OFFSET: %d\n", i, offset);

        //DEBUG
        //testFunction << <1, 256>> > (offset, cudaOutput, deviceExtents);
        //testFunction << <1, 256 >> > (203909504, cudaOutput, deviceExtents);

        //RELEASE
        testFunction << <gridSize, blockSize >> > (offset, cudaOutput, deviceExtents);

        cudaDeviceSynchronize();

        end = std::chrono::steady_clock::now();

        double ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        //printf("Kernel Stopped!\n");

        times.push_back(ms);

        printf("Run %d took %f milliseconds.\n", i, ms);
    }

    end2 = std::chrono::steady_clock::now();

    double ms = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2).count();

    printf("Total Runtme: %f milliseconds.\n", ms);

    double total = 0.0;
    for (double t : times) {
        total += t;
    }
    total /= kernels;  
    printf("Average RunTime: %f\n", total);

    printf("Generated %d seeds.\n", (gridSize * blockSize) * kernels);


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

__global__ void testFunction(uint32_t offset, int* outputArray, float* extents) {
    //int block = blockIdx.x + blockIdx.y * gridDim.x;
    //int threadNumber = block * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;         

    //TEMPORARY
    uint32_t thread = offset + threadIdx.x;
    //uint32_t thread = offset + blockIdx.x * blockDim.x + threadIdx.x;    

    __shared__ RoomTemplates rts[roomTemplateAmount];

    //We will make 65 threads input 52 uint8's into array.
    __shared__ uint8_t forest[3380];

    //Each thread creates one RoomTemplate.
    if (threadIdx.x < 96) {
        CreateRoomTemplates(rts, threadIdx.x+1);    
    }       

    //Have 65 threads input forest data into shared memory.
    if (threadIdx.x > 95 && threadIdx.x < 162) {       
        PopulateForestData(forest, threadIdx.x - 96);       
    }
    __syncthreads();

    //Thread 0 is the same as thread 1.
    //Thread 2147483647 is broken and will infinite loop.
    //Anything above thread 2147483647 is useless.
    if (thread == 0 || thread > 2147483646) return;

    //InitNewGame(thread, rts, extents, forest);   
    CreateMap(thread, rts, extents, forest);        

    //outputArray[threadIdx.x] = InitNewGame(thread, rts, extents, forest);   
}

__device__ void dummy() {};