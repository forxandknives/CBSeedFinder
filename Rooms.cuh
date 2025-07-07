
#ifndef ROOMS_CUH
#define ROOMS_CUH

#include "cuda_runtime.h";
#include "Helpers.cuh";
#include "Constants.cuh";

typedef struct RoomTemplates RoomTemplates;
static struct RoomTemplates {
	int32_t obj, id;
	//dont need objPath i think
	int32_t zone[5] = { 0 };
	int32_t shape;
	char* name;
	int32_t commonness, large;
	int32_t useLightCones;
	bool disableOverlapCheck = true;

	float MinX, MinY, MinZ;
	float MaxX, MaxY, MaxZ;
};

typedef struct Rooms Rooms;
static struct Rooms {
	int32_t zone;
	int32_t found;
	
	int32_t obj;
	float x, y, z;
	int32_t angle;
	RoomTemplates rt;

	float dist;

	//a bunch of stuff that i might need

	float MinX, MinY, MinZ;
	float MaxX, MaxY, MaxZ;

};

__device__ inline bool SetRoom(char* room_name, uint32_t room_type, uint32_t pos, uint32_t min_pos, uint32_t max_pos);

__device__ inline bool SetRoom(char* room_name, uint32_t room_type, uint32_t pos, uint32_t min_pos, uint32_t max_pos) {
	if (max_pos < min_pos) return false;

	uint32_t looped = false; 
	uint32_t can_place = true;

	while (MapRoom[room_type][pos] != '\0') {
		pos++;
		if (pos > max_pos) {
			if (!looped) {
				pos = min_pos + 1;
				looped = true;
			}
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

__device__ inline Rooms CreateRoom(int32_t zone, int32_t roomshape, float x, float y, float z, char* name) {

	Rooms r;
	RoomTemplates rt;

	r.zone = zone;

	r.x = x;
	r.y = y;
	r.z = z;

	if (name != '\0') {
		//assume all names are lowercase
		
	}
	return r;
}


#endif