
#ifndef ROOMS_CUH
#define ROOMS_CUH

#include "cuda_runtime.h";
#include "Helpers.cuh";
#include "Constants.cuh";

__device__ inline bool SetRoom(char* room_name, uint32_t room_type, uint32_t pos, uint32_t min_pos, uint32_t max_pos);

__device__ inline bool SetRoom(char* room_name, uint32_t room_type, uint32_t pos, uint32_t min_pos, uint32_t max_pos) {
	if (max_pos < min_pos) return false;

	uint32_t looped = false; 
	uint32_t can_place = true;

	while (MapRoom[room_type][pos] != "") {
		pos++;
		if (pos > max_pos) {
			looped = false;
			pos = min_pos + 1;
			looped = true;
		}
		else {
			can_place = false;
			break;
		}
	}
	if (can_place) {
		MapRoom[room_type][pos] = room_name;
		return true;
	}
	else {
		return false;
	}
}



#endif