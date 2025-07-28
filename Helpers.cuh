#pragma once

#ifndef HELPERS_CUH
#define HELPERS_CUH

#include "cuda_runtime.h"
#include "Random.cuh"
#include "Constants.cuh"

#include "stdio.h"

__device__ inline uint8_t GetZone(const uint8_t y);
__device__ inline void CreateDoor(bbRandom* bb, rnd_state* rnd_state, bool open, int32_t big);
__device__ inline void CreateItem(bbRandom* bb, rnd_state* rnd_state);
__host__ __device__ inline char* RoomIDToName(RoomID r);

__device__ inline uint8_t GetZone(const uint8_t y) {
	uint8_t zone = uint8_t(min(floorf((float(MapWidth - y) / MapWidth * ZONEAMOUNT)), float(ZONEAMOUNT - 1)));
	return zone;
}

__device__ inline void CreateDoor(bbRandom* bb, rnd_state* rnd_state, bool open, int32_t big) {
	//if (open && !big && bb->bbRand(rnd_state, 1, 8)) {
	//	//do nothing because yea
	//}
	bb->bbRand(rnd_state, 1, 8);
}

__device__ inline void CreateItem(bbRandom* bb, rnd_state* rnd_state) {
	bb->bbRand(rnd_state, 1, 360);
}

__host__ __device__ inline char* RoomIDToName(RoomID r) {	

	//big ass switch statement converting number to room name god help me.
	switch (r) {
	case ROOMEMPTY:
		return "EMPTY ROOM";
	case LOCKROOM:
		return "lockroom";
	case ROOM173:
		return "173";
	case START: 
		return "start";
	case ROOM1123:
		return "room1123";
	case ROOM1ARCHIVE:
		return "room1archive";
	case ROOM2STORAGE:
		return "room2storage";
	case ROOM3STORAGE:
		return "room3storage";
	case ROOM2TESLA_LCZ:
		return "room2tesla_lcz";
	case ENDROOM:
		return "endroom";
	case ROOM012:
		return "room012";
	case ROOM205:
		return "room205";
	case ROOM2ID:
		return "room2";
	case ROOM2_2:
		return "room2_2";
	case ROOM2_3: 
		return "room2_3";
	case ROOM2_4:
		return "room2_4";
	case ROOM2_5:
		return "room2_5";
	case ROOM2CID:
		return "room2c";
	case ROOM2C2:
		return "room2c2";
	case ROOM2CLOSETS:
		return "room2closets";
	case ROOM2ELEVATOR:
		return "room2elevator";
	case ROOM2DOORS:
		return "room2doors";
	case ROOM2SCPS:
		return "room2scps";
	case ROOM860:
		return "room860";
	case ROOM2TESTROOM2:
		return "room2testroom2";
	case ROOM3ID:
		return "room3";
	case ROOM3_2:
		return "room3_2";
	case ROOM4ID:
		return "room4";
	case ROOM4_2:
		return "room4_2";
	case ROOMPJ:
		return "roompj";
	case ROOM914:
		return "914";
	case ROOM2GW:
		return "room2gw";
	case ROOM2GW_B:
		return "room2gw_b";
	case ROOM1162:
		return "room1162";
	case ROOM2SCPS2:
		return "room2scps2";
	case ROOM2SL:
		return "room2sl";
	case LOCKROOM3:
		return "lockroom3";
	case ROOM4INFO:
		return "room4info";
	case ROOM3_3:
		return "room3_3";
	case CHECKPOINT1:
		return "checkpoint1";
	case ROOM008:
		return "008";
	case ROOM035:
		return "room035";
	case ROOM049:
		return "room049";
	case ROOM106:
		return "room106";
	case ROOM513:
		return "room513";
	case COFFIN:
		return "coffin";
	case ROOM966:
		return "room966";
	case ENDROOM2:
		return "endroom2";
	case TESTROOM:
		return "testroom";
	case TUNNEL:
		return "tunnel";
	case TUNNEL2:
		return "tunnel2";
	case ROOM2CTUNNEL:
		return "room2ctunnel";
	case ROOM2NUKE:
		return "room2nuke";
	case ROOM2PIPES:
		return "room2pipes";
	case ROOM2PIT:
		return "room2pit";
	case ROOM3PIT:
		return "room3pit";
	case ROOM4PIT:
		return "room4pit";
	case ROOM2SERVERS:
		return "room2servers";
	case ROOM2SHAFT:
		return "room2shaft";
	case ROOM2TUNNEL:
		return "room2tunnel";
	case ROOM3TUNNEL:
		return "room3tunnel";
	case ROOM4TUNNELS:
		return "room4tunnels";
	case ROOM2TESLA_HCZ: 
		return "room2tesla_hcz";
	case ROOM3Z2:
		return "room3z2";
	case ROOM2CPIT:
		return "room2cpit";
	case ROOM2PIPES2:
		return "room2pipes2";
	case CHECKPOINT2:
		return "checkpoint2";
	case ROOM079:
		return "room079";
	case LOCKROOM2:
		return "lockroom2";
	case EXIT1:
		return "exit1";
	case GATEAENTRANCE:
		return "gateaentrance";
	case GATEA:
		return "gatea";
	case MEDIBAY:
		return "medibay";
	case ROOM2Z3:
		return "room2z3";
	case ROOM2CAFETERIA:
		return "room2cafeteria";
	case ROOM2CZ3:
		return "room2cz3";
	case ROOM2CCONT:
		return "room2ccont";
	case ROOM2OFFICES:
		return "room2offices";
	case ROOM2OFFICES2:
		return "room2offices2";
	case ROOM2OFFICES3:
		return "room2offices3";
	case ROOM2OFFICES4:
		return "room2offices4";
	case ROOM2POFFICES:
		return "room2poffices";
	case ROOM2POFFICES2:
		return "room2poffices2";
	case ROOM2SROOM:
		return "room2sroom";
	case ROOM2TOILETS:
		return "room2toilets";
	case ROOM2TESLA:
		return "room2tesla";
	case ROOM3SERVERS:
		return "room3servers";
	case ROOM3SERVERS2:
		return "room3servers2";
	case ROOM3Z3:
		return "room3z3";
	case ROOM4Z3:
		return "room4z3";
	case ROOM1LIFTS:
		return "room1lifts";
	case ROOM3GW:
		return "room3gw";
	case ROOM2SERVERS2:
		return "room2servers2";
	case ROOM3OFFICES:
		return "room3offices";
	case ROOM2Z3_2:
		return "room2z3_2";
	case POCKETDIMENSION:
		return "pocketdimension";
	case DIMENSION1499:
		return "dimension1499";
	default:
		return "EMPTY ROOM";
	}
}

#endif