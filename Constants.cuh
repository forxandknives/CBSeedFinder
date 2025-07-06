
#ifndef CONSTANTS_CUH
#define CONSTANTS_CUH

#include "stdio.h"
#include "cuda_runtime.h"

const static uint32_t ROOM1  = 1;
const static uint32_t ROOM2  = 2;
const static uint32_t ROOM2C = 3;
const static uint32_t ROOM3  = 4;
const static uint32_t ROOM4  = 5;

const static uint32_t MapHeight = 18;
const static uint32_t MapWidth  = 18;

__device__ static uint32_t MapTemp[MapWidth+1][MapHeight+1];

const static uint32_t ZONEAMOUNT = 3;

#endif