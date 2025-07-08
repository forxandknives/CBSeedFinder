
#ifndef ROOMS_CUH
#define ROOMS_CUH

#include "cuda_runtime.h";
#include "Helpers.cuh";
#include "Constants.cuh";

typedef struct RoomTemplates RoomTemplates;
static struct RoomTemplates {
	int32_t obj;
	int32_t id = -1;
	//dont need objPath i think
	int32_t zone[5] = { 0 };
	int32_t shape;
	char* name;
	int32_t commonness;
	bool large;
	int32_t useLightCones;
	bool disableOverlapCheck = true;
	bool disableDecals;

	float MinX, MinY, MinZ;
	float MaxX, MaxY, MaxZ;
};

typedef struct Rooms Rooms;
static struct Rooms {
	int32_t id = -1;
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

__device__ inline void CreateRoomTemplates(RoomTemplates* rt);
__device__ inline bool SetRoom(char* room_name, uint32_t room_type, uint32_t pos, uint32_t min_pos, uint32_t max_pos);
__device__ inline Rooms CreateRoom(int32_t zone, int32_t roomshape, float x, float y, float z, char* name);
__device__ inline bool PreventRoomOverlap(Rooms* rooms, int32_t index);
__device__ inline bool CheckRoomOverlap(Rooms* r, Rooms* r2);
__device__ inline void CalculateRoomExtents(Rooms* r);

__device__ inline void CreateRoomTemplates(RoomTemplates* rt) {

	int32_t counter = 0;

	//All commonness is defined as max(min(commonness, 100), 0);

	//LIGHT CONTAINMENT

	//lockroom
	rt[counter].id = counter;
	rt[counter].name = "lockroom";
	rt[counter].shape = ROOM2C;
	rt[counter].zone[0] = 1;
	rt[counter].zone[1] = 3;
	rt[counter].commonness = 30;
	counter++;

	//173
	rt[counter].id = counter;
	rt[counter].name = "173";
	rt[counter].shape = ROOM1;
	rt[counter].commonness = 0;
	rt[counter].disableDecals = true;
	counter++;

	//start
	rt[counter].id = counter;
	rt[counter].name = "start";
	rt[counter].shape = ROOM1;
	rt[counter].commonness = 0;
	rt[counter].disableDecals = true;
	counter++;

	//room1123
	rt[counter].id = counter;
	rt[counter].name = "room1123";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 0;
	rt[counter].zone[0] = 1;
	rt[counter].disableDecals = true;
	counter++;

	//room1archive
	rt[counter].id = counter;
	rt[counter].name = "room1archive";
	rt[counter].shape = ROOM1;
	rt[counter].commonness = 80;
	rt[counter].zone[0] = 1;
	counter++;

	//room2storage
	rt[counter].id = counter;
	rt[counter].name = "room2storage";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 0;
	rt[counter].zone[0] = 1;
	rt[counter].disableDecals = true;
	counter++;

	//room3storage
	rt[counter].id = counter;
	rt[counter].name = "room3storage";
	rt[counter].shape = ROOM3;
	rt[counter].commonness = 0;
	rt[counter].zone[0] = 1;
	rt[counter].disableOverlapCheck = true;
	counter++;

	//room2tesla_lcz
	rt[counter].id = counter;
	rt[counter].name = "room2tesla_lcz";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 100;
	rt[counter].zone[0] = 1;
	counter++;

	//endroom
	rt[counter].id = counter;
	rt[counter].name = "endroom";
	rt[counter].shape = ROOM1;
	rt[counter].commonness = 100;
	rt[counter].zone[0] = 1;
	rt[counter].zone[2] = 3;
	counter++;

	//room012
	rt[counter].id = counter;
	rt[counter].name = "room012";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 0;
	rt[counter].zone[0] = 1;
	rt[counter].disableDecals = true;
	counter++;

	//room205
	rt[counter].id = counter;
	rt[counter].name = "room205";
	rt[counter].shape = ROOM1;
	rt[counter].commonness = 0;
	rt[counter].zone[0] = 1;
	rt[counter].large = true;
	counter++;

	//room2
	rt[counter].id = counter;
	rt[counter].name = "room2";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 45;
	rt[counter].zone[0] = 1;
	counter++;

	//room2_2
	rt[counter].id = counter;
	rt[counter].name = "room2_2";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 40;
	rt[counter].zone[0] = 1;
	counter++;

	//room2_3
	rt[counter].id = counter;
	rt[counter].name = "room2_3";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 35;
	rt[counter].zone[0] = 1;
	counter++;

	//room2_4
	rt[counter].id = counter;
	rt[counter].name = "room2_4";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 35;
	rt[counter].zone[0] = 1;
	counter++;

	//room2_5
	rt[counter].id = counter;
	rt[counter].name = "room2_5";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 35;
	rt[counter].zone[0] = 1;
	counter++;

	//room2c
	rt[counter].id = counter;
	rt[counter].name = "room2c";
	rt[counter].shape = ROOM2C;
	rt[counter].commonness = 30;
	rt[counter].zone[0] = 1;
	counter++;

	//room2c2
	rt[counter].id = counter;
	rt[counter].name = "room2c2";
	rt[counter].shape = ROOM2C;
	rt[counter].commonness = 30;
	rt[counter].zone[0] = 1;
	counter++;

	//room2closets
	rt[counter].id = counter;
	rt[counter].name = "room2closets";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 0;
	rt[counter].zone[0] = 1;
	rt[counter].disableDecals = true;
	rt[counter].large = true;
	counter++;

	//room2elevator
	rt[counter].id = counter;
	rt[counter].name = "room2elevator";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 20;
	rt[counter].zone[0] = 1;
	counter++;

	//room2doors
	rt[counter].id = counter;
	rt[counter].name = "room2doors";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 30;
	rt[counter].zone[0] = 1;
	counter++;

	//room2scps
	rt[counter].id = counter;
	rt[counter].name = "room2scps";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 0;
	rt[counter].zone[0] = 1;
	counter++;

	//room860
	rt[counter].id = counter;
	rt[counter].name = "room860";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 0;
	counter++;

	//room2testroom2
	rt[counter].id = counter;
	rt[counter].name = "room2testroom2";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 0;
	rt[counter].zone[0] = 1;
	counter++;

	//room3
	rt[counter].id = counter;
	rt[counter].name = "room3";
	rt[counter].shape = ROOM3;
	rt[counter].commonness = 100;
	rt[counter].zone[0] = 1;
	counter++;

	//room3_2
	rt[counter].id = counter;
	rt[counter].name = "room3_2";
	rt[counter].shape = ROOM3;
	rt[counter].commonness = 100;
	rt[counter].zone[0] = 1;
	counter++;

	//room4
	rt[counter].id = counter;
	rt[counter].name = "room4";
	rt[counter].shape = ROOM4;
	rt[counter].commonness = 100;
	rt[counter].zone[0] = 1;
	counter++;

	//room4_2
	rt[counter].id = counter;
	rt[counter].name = "room4_2";
	rt[counter].shape = ROOM4;
	rt[counter].commonness = 80;
	rt[counter].zone[0] = 1;
	counter++;

	//roompj
	rt[counter].id = counter;
	rt[counter].name = "roompj";
	rt[counter].shape = ROOM1;
	rt[counter].commonness = 0;
	rt[counter].disableDecals = true;
	rt[counter].zone[0] = 1;
	counter++;

	//914
	rt[counter].id = counter;
	rt[counter].name = "914";
	rt[counter].shape = ROOM1;
	rt[counter].commonness = 0;
	rt[counter].disableDecals = true;
	rt[counter].zone[0] = 1;
	counter++;

	//room2gw
	rt[counter].id = counter;
	rt[counter].name = "room2gw";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 10;
	rt[counter].zone[0] = 1;
	counter++;

	//room2gw_b
	rt[counter].id = counter;
	rt[counter].name = "room2gw_b";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 0;
	rt[counter].zone[0] = 1;
	counter++;

	//room1162
	rt[counter].id = counter;
	rt[counter].name = "room1162";
	rt[counter].shape = ROOM2C;
	rt[counter].commonness = 0;
	rt[counter].zone[0] = 1;
	counter++;

	//room2scps2
	rt[counter].id = counter;
	rt[counter].name = "room2scps2";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 0;
	rt[counter].zone[0] = 1;
	counter++;

	//room2sl
	rt[counter].id = counter;
	rt[counter].name = "room2sl";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 0;
	rt[counter].zone[0] = 1;
	rt[counter].large = true;
	counter++;

	//lockroom3
	rt[counter].id = counter;
	rt[counter].name = "lockroom3";
	rt[counter].shape = ROOM2C;
	rt[counter].commonness = 15;
	rt[counter].zone[0] = 1;
	counter++;

	//room4info
	rt[counter].id = counter;
	rt[counter].name = "room4info";
	rt[counter].shape = ROOM4;
	rt[counter].commonness = 0;
	rt[counter].zone[0] = 1;
	counter++;

	//room3_3
	rt[counter].id = counter;
	rt[counter].name = "room3_3";
	rt[counter].shape = ROOM3;
	rt[counter].commonness = 20;
	rt[counter].zone[0] = 1;
	counter++;

	//checkpoint1
	rt[counter].id = counter;
	rt[counter].name = "checkpoint1";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 0;
	counter++;

	//HEAVY CONTAINMENT

	//008
	rt[counter].id = counter;
	rt[counter].name = "008";
	rt[counter].shape = ROOM1;
	rt[counter].commonness = 0;
	rt[counter].disableDecals = true;
	rt[counter].zone[0] = 2;
	counter++;

	//room035
	rt[counter].id = counter;
	rt[counter].name = "room035";
	rt[counter].shape = ROOM1;
	rt[counter].commonness = 0;
	rt[counter].zone[0] = 2;
	counter++;

	//room049
	rt[counter].id = counter;
	rt[counter].name = "room049";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 0;
	rt[counter].disableDecals = true;
	rt[counter].zone[0] = 2;
	rt[counter].disableOverlapCheck = true;
	counter++;

	//room106;
	rt[counter].id = counter;
	rt[counter].name = "room106";
	rt[counter].shape = ROOM1;
	rt[counter].commonness = 0;
	rt[counter].disableDecals = true;
	rt[counter].zone[0] = 2;
	rt[counter].large = true;
	counter++;

	//rom513
	rt[counter].id = counter;
	rt[counter].name = "room513";
	rt[counter].shape = ROOM3;
	rt[counter].commonness = 0;
	rt[counter].zone[0] = 2;
	counter++;

	//coffin
	rt[counter].id = counter;
	rt[counter].name = "coffin";
	rt[counter].shape = ROOM1;
	rt[counter].commonness = 0;
	rt[counter].disableDecals = true;
	rt[counter].zone[0] = 1;
	counter++;

	//room966
	rt[counter].id = counter;
	rt[counter].name = "room966";
	rt[counter].shape = ROOM3;
	rt[counter].commonness = 0;
	rt[counter].disableOverlapCheck = true;
	counter++;

	//endroom2
	rt[counter].id = counter;
	rt[counter].name = "endroom2";
	rt[counter].shape = ROOM1;
	rt[counter].commonness = 100;
	rt[counter].zone[0] = 2;
	counter++;

	//testroom
	rt[counter].id = counter;
	rt[counter].name = "testroom";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 0;
	rt[counter].disableDecals = true;
	rt[counter].zone[0] = 2;
	counter++;

	//tunnel
	rt[counter].id = counter;
	rt[counter].name = "tunnel";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 100;
	rt[counter].zone[0] = 2;
	counter++;

	//tunnel2
	rt[counter].id = counter;
	rt[counter].name = "tunnel2";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 70;
	rt[counter].zone[0] = 2;
	counter++;

	//room2ctunnel
	rt[counter].id = counter;
	rt[counter].name = "room2ctunnel";
	rt[counter].shape = ROOM2C;
	rt[counter].commonness = 40;
	rt[counter].zone[0] = 2;
	counter++;

	//room2nuke
	rt[counter].id = counter;
	rt[counter].name = "room2nuke";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 0;
	rt[counter].zone[0] = 2;
	rt[counter].large = true;
	counter++;

	//room2pipes
	rt[counter].id = counter;
	rt[counter].name = "room2pipes";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 50;
	rt[counter].zone[0] = 2;
	counter++;

	//room2pit
	rt[counter].id = counter;
	rt[counter].name = "room2pit";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 75;
	rt[counter].disableDecals = true;
	rt[counter].zone[0] = 2;
	counter++;

	//room3pit
	rt[counter].id = counter;
	rt[counter].name = "room3pit";
	rt[counter].shape = ROOM3;
	rt[counter].commonness = 100;
	rt[counter].disableDecals = true;
	rt[counter].zone[0] = 2;
	counter++;

	//room4pit
	rt[counter].id = counter;
	rt[counter].name = "room4pit";
	rt[counter].shape = ROOM4;
	rt[counter].commonness = 100;
	rt[counter].zone[0] = 2;
	counter++;

	//room2servers
	rt[counter].id = counter;
	rt[counter].name = "room2servers";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 0;
	rt[counter].zone[0] = 2;
	rt[counter].large = true;
	counter++;

	//room2shaft
	rt[counter].id = counter;
	rt[counter].name = "room2shaft";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 0;
	rt[counter].disableDecals = true;
	counter++;

	//room2tunnel
	rt[counter].id = counter;
	rt[counter].name = "room2tunnel";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 0;
	rt[counter].disableDecals = true;
	rt[counter].zone[0] = 2;
	counter++;

	//room3tunnel
	rt[counter].id = counter;
	rt[counter].name = "room3tunnel";
	rt[counter].shape = ROOM3;
	rt[counter].commonness = 100;
	rt[counter].zone[0] = 2;
	counter++;

	//room4tunnel
	rt[counter].id = counter;
	rt[counter].name = "room4tunnel";
	rt[counter].shape = ROOM4;
	rt[counter].commonness = 100;
	rt[counter].zone[0] = 2;
	counter++;

	//room2tesla_hcz
	rt[counter].id = counter;
	rt[counter].name = "room2tesla_hcz";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 100;
	rt[counter].zone[0] = 2;
	counter++;

	//room3z2
	rt[counter].id = counter;
	rt[counter].name = "room3z2";
	rt[counter].shape = ROOM3;
	rt[counter].commonness = 100;
	rt[counter].zone[0] = 2;
	counter++;

	//room2cpit
	rt[counter].id = counter;
	rt[counter].name = "room2cpit";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 0;
	rt[counter].disableDecals = true;
	rt[counter].zone[0] = 2;
	counter++;

	//room2pipes2
	rt[counter].id = counter;
	rt[counter].name = "room2pipes2";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 70;
	rt[counter].disableDecals = true;
	rt[counter].zone[0] = 2;
	counter++;

	//checkpoint2
	rt[counter].id = counter;
	rt[counter].name = "checkpoint2";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 0;
	counter++;

	//ENTRANCE ZONE

	//room079 yea he's in the entrance zone mhm
	rt[counter].id = counter;
	rt[counter].name = "room079";
	rt[counter].shape = ROOM1;
	rt[counter].commonness = 0;
	rt[counter].disableDecals = true;
	rt[counter].zone[0] = 3;
	rt[counter].large = true;
	counter++;

	//lockroom2
	rt[counter].id = counter;
	rt[counter].name = "lockroom2";
	rt[counter].shape = ROOM2C;
	rt[counter].commonness = 0;
	rt[counter].zone[0] = 3;
	counter++;

	//exit1 
	rt[counter].id = counter;
	rt[counter].name = "exit1";
	rt[counter].shape = ROOM1;
	rt[counter].commonness = 0;
	rt[counter].zone[0] = 3;
	rt[counter].disableOverlapCheck = true;
	counter++;

	//gateaentrance
	rt[counter].id = counter;
	rt[counter].name = "gateaentrance";
	rt[counter].shape = ROOM1;
	rt[counter].commonness = 0;
	rt[counter].zone[0] = 3;
	counter++;

	//gatea
	rt[counter].id = counter;
	rt[counter].name = "gatea";
	rt[counter].shape = ROOM1;
	rt[counter].commonness = 0;
	rt[counter].disableOverlapCheck = true;
	counter++;

	//medibay
	rt[counter].id = counter;
	rt[counter].name = "medibay";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 0;
	rt[counter].zone[0] = 3;
	counter++;

	//room2z3
	rt[counter].id = counter;
	rt[counter].name = "room2z3";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 75;
	rt[counter].zone[0] = 3;
	counter++;

	//room2cafeteria
	rt[counter].id = counter;
	rt[counter].name = "room2cafeteria";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 0;
	rt[counter].zone[0] = 3;
	rt[counter].large = true;
	rt[counter].disableOverlapCheck = true;
	counter++;

	//room2cz3
	rt[counter].id = counter;
	rt[counter].name = "room2cz3";
	rt[counter].shape = ROOM2C;
	rt[counter].commonness = 100;
	rt[counter].zone[0] = 3;
	counter++;

	//room2ccont
	rt[counter].id = counter;
	rt[counter].name = "room2ccont";
	rt[counter].shape = ROOM2C;
	rt[counter].commonness = 0;
	rt[counter].zone[0] = 3;
	rt[counter].large = true;

	//room2offices
	rt[counter].id = counter;
	rt[counter].name = "room2offices";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 30;
	rt[counter].zone[0] = 3;
	counter++;

	//room2offices2
	rt[counter].id = counter;
	rt[counter].name = "room2offices2";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 20;
	rt[counter].disableDecals = true;
	rt[counter].zone[0] = 3;
	counter++;

	//room2offices3
	rt[counter].id = counter;
	rt[counter].name = "room2offices3";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 20;
	rt[counter].zone[0] = 3;
	counter++;

	//room2offices4
	rt[counter].id = counter;
	rt[counter].name = "room2offices4";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 0;
	rt[counter].zone[0] = 3;
	counter++;

	//room2poffices
	rt[counter].id = counter;
	rt[counter].name = "room2poffices";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 0;
	rt[counter].zone[0] = 3;
	counter++;

	//room2poffices2
	rt[counter].id = counter;
	rt[counter].name = "room2poffices2";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 0;
	rt[counter].zone[0] = 3;
	counter++;

	//room2sroom
	rt[counter].id = counter;
	rt[counter].name = "room2sroom";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 0;
	rt[counter].zone[0] = 3;
	counter++;

	//room2toilets
	rt[counter].id = counter;
	rt[counter].name = "room2toilets";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 30;
	rt[counter].zone[0] = 3;
	counter++;

	//room2tesla
	rt[counter].id = counter;
	rt[counter].name = "room2tesla";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 100;
	rt[counter].zone[0] = 3;
	counter++;

	//room3servers
	rt[counter].id = counter;
	rt[counter].name = "room3servers";
	rt[counter].shape = ROOM3;
	rt[counter].commonness = 0;
	rt[counter].disableDecals = true;
	rt[counter].zone[0] = 3;
	counter++;

	//room3servers2
	rt[counter].id = counter;
	rt[counter].name = "room3servers2";
	rt[counter].shape = ROOM3;
	rt[counter].commonness = 0;
	rt[counter].disableDecals = true;
	rt[counter].zone[0] = 3;
	counter++;

	//room3z3
	rt[counter].id = counter;
	rt[counter].name = "room3z3";
	rt[counter].shape = ROOM3;
	rt[counter].commonness = 100;
	rt[counter].zone[0] = 3;
	counter++;

	//room4z3
	rt[counter].id = counter;
	rt[counter].name = "room4z3";
	rt[counter].shape = ROOM4;
	rt[counter].commonness = 100;
	rt[counter].zone[0] = 3;
	counter++;

	//room1lifts
	rt[counter].id = counter;
	rt[counter].name = "room1lifts";
	rt[counter].shape = ROOM1;
	rt[counter].commonness = 0;
	rt[counter].zone[0] = 3;
	counter++;

	//room3gw
	rt[counter].id = counter;
	rt[counter].name = "room3gw";
	rt[counter].shape = ROOM3;
	rt[counter].commonness = 10;
	rt[counter].zone[0] = 3;
	counter++;

	//room2servers2
	rt[counter].id = counter;
	rt[counter].name = "room2servers2";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 0;
	rt[counter].zone[0] = 3;
	counter++;

	//room3offices
	rt[counter].id = counter;
	rt[counter].name = "room3offices";
	rt[counter].shape = ROOM3;
	rt[counter].commonness = 0;
	rt[counter].zone[0] = 3;
	counter++;

	//room2z3_2
	rt[counter].id = counter;
	rt[counter].name = "room2z3_2";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 25;
	rt[counter].zone[0] = 3;
	counter++;

	//pocketdimension
	rt[counter].id = counter;
	rt[counter].name = "pocketdimension";
	rt[counter].shape = ROOM1;
	rt[counter].commonness = 0;
	counter++;

	//dimension1499
	rt[counter].id = counter;
	rt[counter].name = "dimension1499";
	rt[counter].shape = ROOM1;
	rt[counter].commonness = 0;
	rt[counter].disableDecals = true;

}

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

__device__ inline bool PreventRoomOverlap(Rooms* rooms, int32_t index) {
	if (rooms[index].rt.disableOverlapCheck) return true;

	Rooms r = rooms[index];

	Rooms* r2 = NULL;
	Rooms* r3 = NULL;

	bool isIntersecting = false;

	if (r.rt.name == "checkpoint1\0" || r.rt.name == "checkpoint2\0" || r.rt.name == "start\0") return true;

	for (int32_t i = 0; i < 324; i++) {
		*r2 = rooms[i];
		if (r2->id != r.id && !r2->rt.disableOverlapCheck) {
			if (CheckRoomOverlap(&rooms[i], r2)) {
				isIntersecting = true;
				break;
			}
		}
	}

	if (!isIntersecting) return true;

	isIntersecting = false;

	int32_t x = r.x / 8.0;
	int32_t y = r.y / 8.0;

	if (r.rt.shape == ROOM2) {
		r.angle += 180;
		CalculateRoomExtents(&r);
	}


}

__device__ inline bool CheckRoomOverlap(Rooms* r, Rooms* r2) {

}

__device__ inline void CalculateRoomExtents(Rooms* r) {

}


#endif