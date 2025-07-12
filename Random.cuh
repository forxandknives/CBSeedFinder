#pragma once

#ifndef RANDOM_CUH
#define RANDOM_CUH

#include "cuda_runtime.h"
#include "stdio.h"

typedef struct rnd_state rnd_state;
static struct rnd_state {
	int32_t rnd_state;
};

class bbRandom {

public:	
	static const int32_t RND_A = 48271;
	static const int32_t RND_M = 2147483647;
	static const int32_t RND_Q = 44488;
	static const int32_t RND_R = 3399;

	__host__ __device__ bbRandom() { }

	__device__ static inline float rnd(rnd_state* rnd_state) {
		rnd_state->rnd_state = RND_A * (rnd_state->rnd_state % RND_Q) - RND_R * (rnd_state->rnd_state / RND_Q);
		if (rnd_state->rnd_state < 0) rnd_state->rnd_state += RND_M;
		return (rnd_state->rnd_state & 65535) / 65536.0f + (.5f / 65536.0f);
	}

	__device__ inline float bbRnd(rnd_state* rnd_state, float from, float to) {
		return rnd(rnd_state) * (to - from) + from;
	}

	__device__ inline int32_t bbRand(rnd_state* rnd_state, int32_t from, int32_t to) {
		if (to < from) {
			int temp = to;
			from = to;
			to = temp;
		}

		return int32_t(rnd(rnd_state) * (to - from + 1)) + from;
	}

	__device__ inline void bbSeedRnd(rnd_state* rnd_state, int32_t seed) {
		if (seed <= 0) {
			rnd_state->rnd_state = 1;
		} else {
			rnd_state->rnd_state = seed;
		}
	}
};

#endif