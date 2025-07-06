#ifndef HELPERS_CUH
#define HELPERS_CUH

#include "cuda_runtime.h"

#include "Constants.cuh"

#include "stdio.h"

__device__ inline uint32_t GetZone(const uint32_t y);

__device__ inline uint32_t GetZone(const uint32_t y) {
	return min(floorf((float(MapWidth - y) / MapWidth * ZONEAMOUNT)), float(ZONEAMOUNT - 1));
}

#endif