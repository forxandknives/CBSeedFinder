
#ifndef CONSTANTS_CUH
#define CONSTANTS_CUH

#include "stdio.h"
#include "cuda_runtime.h"

const static int32_t ROOM1  = 1;
const static int32_t ROOM2  = 2;
const static int32_t ROOM2C = 3;
const static int32_t ROOM3  = 4;
const static int32_t ROOM4  = 5;

const static int32_t MapHeight = 18;
const static int32_t MapWidth  = 18;

__device__ static int32_t MapTemp[MapWidth+1][MapHeight+1];

const static int32_t ZONEAMOUNT = 3;

__device__ char* MapRoom[ROOM4 + 1][180] = {};	

const static int32_t roomTemplateAmount = 95;

__device__ static int32_t roomIdCounter = 0;

#endif