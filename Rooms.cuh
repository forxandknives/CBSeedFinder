#pragma once

#ifndef ROOMS_CUH
#define ROOMS_CUH

#include "cuda_runtime.h";
#include "Helpers.cuh";
#include "Constants.cuh";
#include "Random.cuh";
#include "Forest.cuh";

typedef struct RoomTemplates RoomTemplates;
static struct RoomTemplates {
	int8_t id = -1;
    uint8_t index = 0;	
	uint8_t zone[5] = { 0 };
	uint8_t shape;
	RoomID name = ROOMEMPTY;
	uint8_t commonness;
	uint8_t lights;
	bool disableOverlapCheck = true;
};

typedef struct Rooms Rooms;
static struct Rooms {
	int16_t id = -1;
	uint8_t zone = 0;
	
	float x, y, z;

	int16_t angle;
	RoomTemplates* rt;

	float minX, minY, minZ;
	float maxX, maxY, maxZ;

};	

__device__ inline void CreateRoomTemplates(RoomTemplates* rt, int32_t thread);
__device__ inline bool SetRoom(RoomID MapRoom[6][70], RoomID room_name, uint8_t room_type, uint8_t pos, uint8_t min_pos, uint8_t max_pos);
__device__ inline Rooms CreateRoom(uint8_t MapTemp[19][19], uint8_t& roomIdCounter, uint8_t* forest, float* e, RoomTemplates* rts, bbRandom* bb, rnd_state* rnd_sate, uint8_t zone, uint8_t roomshape, float x, float y, float z, RoomID name);
__device__ inline bool PreventRoomOverlap(Rooms* r, Rooms* rooms, float* e);
__device__ inline bool CheckRoomOverlap(Rooms* r, Rooms* r2);
//__device__ inline void CalculateRoomExtents(Rooms* r);
__device__ inline void FillRoom(uint8_t MapTemp[19][19], bbRandom* bb, rnd_state* rnd_state, Rooms* r, uint8_t* forest);
__device__ constexpr inline void GetRoomExtents(Rooms* r, float* e);

__device__ inline void CreateRoomTemplates(RoomTemplates* rt, int32_t thread) {

	//All commonness is defined as max(min(commonness, 100), 0);
	//Rooms with disableoverlapcheck = true do not have min and max extents.
	//The name variable is not a string but instead an enum for the ID of the room the roomtemplate represents.
	//This is so we do no have to do char stuff.
	
	//We start at 1 because room template 0 is just room ambience stuff.
	int32_t counter = 1;
	
	switch (thread) {
	case 1:
		//LIGHT CONTAINMENT
		//lockroom
		rt[thread].id = thread;
		rt[thread].name = LOCKROOM;//"lockroom";
		rt[thread].shape = ROOM2C;
		rt[thread].zone[0] = 1;
		rt[thread].zone[1] = 3;
		rt[thread].commonness = 30;
		rt[thread].lights = 4;		
		rt[thread].index = 0; //lockroom
		break;

	case 2:
		//173
		rt[thread].id = thread;
		rt[thread].name = ROOM173;// "173";
		rt[thread].shape = ROOM1;
		rt[thread].commonness = 0;
		rt[thread].lights = 29;
		break;
		
	case 3:
		//start
		rt[thread].id = thread;
		rt[thread].name = START;// "start";
		rt[thread].shape = ROOM1;
		rt[thread].commonness = 0;
		rt[thread].lights = 9;
		rt[thread].index = 1; //start
		break;

	case 4:
		//room1123
		rt[thread].id = thread;
		rt[thread].name = ROOM1123;//"room1123";
		rt[thread].shape = ROOM2;
		rt[thread].commonness = 0;
		rt[thread].lights = 20;
		rt[thread].zone[0] = 1;
		rt[thread].index = 2; //room1123
		break;

	case 5:
		//room1archive
		rt[thread].id = thread;
		rt[thread].name = ROOM1ARCHIVE;// "room1archive";
		rt[thread].shape = ROOM1;
		rt[thread].commonness = 80;
		rt[thread].lights = 2;
		rt[thread].zone[0] = 1;
		rt[thread].index = 3; //room1archive
		break;

	case 6:
		//room2storage
		rt[thread].id = thread;
		rt[thread].name = ROOM2STORAGE;// "room2storage";
		rt[thread].shape = ROOM2;
		rt[thread].commonness = 0;
		rt[thread].lights = 10;
		rt[thread].zone[0] = 1;
		rt[thread].index = 4; //room2storage
		break;

	case 7:
		//room3storage
		rt[thread].id = thread;
		rt[thread].name = ROOM3STORAGE;// "room3storage";
		rt[thread].shape = ROOM3;
		rt[thread].commonness = 0;
		rt[thread].lights = 33;
		rt[thread].zone[0] = 1;
		rt[thread].disableOverlapCheck = true;
		break;

	case 8:
		//room2tesla_lcz
		rt[thread].id = thread;
		rt[thread].name = ROOM2TESLA_LCZ;//"room2tesla_lcz";
		rt[thread].shape = ROOM2;
		rt[thread].commonness = 100;
		rt[thread].lights = 8;
		rt[thread].zone[0] = 1;
		rt[thread].index = 5; //room2tesla_lcz
		break;

	case 9:
		//endroom
		rt[thread].id = thread;
		rt[thread].name = ENDROOM;// "endroom";
		rt[thread].shape = ROOM1;
		rt[thread].commonness = 100;
		rt[thread].lights = 3;
		rt[thread].zone[0] = 1;
		rt[thread].zone[2] = 3;
		rt[thread].index = 6; //endroom
		break;

	case 10:
		//room012
		rt[thread].id = thread;
		rt[thread].name = ROOM012;//"room012";
		rt[thread].shape = ROOM2;
		rt[thread].commonness = 0;
		rt[thread].lights = 4;
		rt[thread].zone[0] = 1;		
		rt[thread].index = 7; //room012
		break;

	case 11:
		//room205
		rt[thread].id = thread;
		rt[thread].name = ROOM205;//"room205";
		rt[thread].shape = ROOM1;
		rt[thread].commonness = 0;
		rt[thread].lights = 5;
		rt[thread].zone[0] = 1;
		rt[thread].index = 8; //room205
		break;

	case 12:
		//room2
		rt[thread].id = thread;
		rt[thread].name = ROOM2ID;//"room2";
		rt[thread].shape = ROOM2;
		rt[thread].commonness = 45;
		rt[thread].lights = 2;
		rt[thread].zone[0] = 1;
		rt[thread].index = 9; //room2
		break;

	case 13:
		//room2_2
		rt[thread].id = thread;
		rt[thread].name = ROOM2_2;// "room2_2";
		rt[thread].shape = ROOM2;
		rt[thread].commonness = 40;
		rt[thread].lights = 4;
		rt[thread].zone[0] = 1;
		rt[thread].index = 10; //room2_2
		break;

	case 14:
		//room2_3
		rt[thread].id = thread;
		rt[thread].name = ROOM2_3;// "room2_3";
		rt[thread].shape = ROOM2;
		rt[thread].commonness = 35;
		rt[thread].lights = 0;
		rt[thread].zone[0] = 1;
		rt[thread].index = 11; //room2_3
		break;

	case 15:
		//room2_4
		rt[thread].id = thread;
		rt[thread].name = ROOM2_4;// "room2_4";
		rt[thread].shape = ROOM2;
		rt[thread].commonness = 35;
		rt[thread].lights = 3;
		rt[thread].zone[0] = 1;
		rt[thread].index = 12; //room2_4
		break;

	case 16:
		//room2_5
		rt[thread].id = thread;
		rt[thread].name = ROOM2_5;// "room2_5";
		rt[thread].shape = ROOM2;
		rt[thread].commonness = 35;
		rt[thread].lights = 3;
		rt[thread].zone[0] = 1;		
		rt[thread].index = 13; //room2_5
		break;

	case 17:
		//room2c
		rt[thread].id = thread;
		rt[thread].name = ROOM2CID;// "room2c";
		rt[thread].shape = ROOM2C;
		rt[thread].commonness = 30;
		rt[thread].lights = 3;
		rt[thread].zone[0] = 1;
		rt[thread].index = 14; //room2c
		break;

	case 18:
		//room2c2
		rt[thread].id = thread;
		rt[thread].name = ROOM2C2;// "room2c2";
		rt[thread].shape = ROOM2C;
		rt[thread].commonness = 30;
		rt[thread].lights = 3;
		rt[thread].zone[0] = 1;
		rt[thread].index = 15; //room2c2
		break;

	case 19:
		//room2closets
		rt[thread].id = thread;
		rt[thread].name = ROOM2CLOSETS;// "room2closets";
		rt[thread].shape = ROOM2;
		rt[thread].commonness = 0;
		rt[thread].lights = 6;
		rt[thread].zone[0] = 1;
		rt[thread].index = 16; //room2closets
		break;

	case 20:
		//room2elevator
		rt[thread].id = thread;
		rt[thread].name = ROOM2ELEVATOR;// "room2elevator";
		rt[thread].shape = ROOM2;
		rt[thread].commonness = 20;
		rt[thread].lights = 3;
		rt[thread].zone[0] = 1;
		rt[thread].index = 17; //room2elevator
		break;

	case 21:
		//room2doors
		rt[thread].id = thread;
		rt[thread].name = ROOM2DOORS;// "room2doors";
		rt[thread].shape = ROOM2;
		rt[thread].commonness = 30;
		rt[thread].lights = 5;
		rt[thread].zone[0] = 1;
		rt[thread].index = 18; //room2doors
		break;

	case 22:
		//room2scps
		rt[thread].id = thread;
		rt[thread].name = ROOM2SCPS;// "room2scps";
		rt[thread].shape = ROOM2;
		rt[thread].commonness = 0;
		rt[thread].lights = 6;
		rt[thread].zone[0] = 1;
		rt[thread].index = 19; //room2scps
		break;

	case 23:
		//room860
		rt[thread].id = thread;
		rt[thread].name = ROOM860;// "room860";
		rt[thread].shape = ROOM2;
		rt[thread].commonness = 0;
		rt[thread].lights = 4;
		rt[thread].index = 20; //room860
		break;

	case 24:
		//room2testroom2
		rt[thread].id = thread;
		rt[thread].name = ROOM2TESTROOM2;// "room2testroom2";
		rt[thread].shape = ROOM2;
		rt[thread].commonness = 0;
		rt[thread].lights = 6;
		rt[thread].zone[0] = 1;
		rt[thread].index = 21; //room2testroom2
		break;

	case 25:
		//room3
		rt[thread].id = thread;
		rt[thread].name = ROOM3ID;// "room3";
		rt[thread].shape = ROOM3;
		rt[thread].commonness = 100;
		rt[thread].lights = 4;
		rt[thread].zone[0] = 1;
		rt[thread].index = 22; //room3
		break;

	case 26:
		//room3_2
		rt[thread].id = thread;
		rt[thread].name = ROOM3_2;// "room3_2";
		rt[thread].shape = ROOM3;
		rt[thread].commonness = 100;
		rt[thread].lights = 3;
		rt[thread].zone[0] = 1;
		rt[thread].index = 23; //room3_2
		break;

	case 27:
		//room4
		rt[thread].id = thread;
		rt[thread].name = ROOM4ID;// "room4";
		rt[thread].shape = ROOM4;
		rt[thread].commonness = 100;
		rt[thread].lights = 4;
		rt[thread].zone[0] = 1;
		rt[thread].index = 24; //room4
		break;

	case 28:
		//room4_2
		rt[thread].id = thread;
		rt[thread].name = ROOM4_2;// "room4_2";
		rt[thread].shape = ROOM4;
		rt[thread].commonness = 80;
		rt[thread].lights = 4;
		rt[thread].zone[0] = 1;
		rt[thread].index = 25; //room4_2
		break;

	case 29:
		//roompj
		rt[thread].id = thread;
		rt[thread].name = ROOMPJ;// "roompj";
		rt[thread].shape = ROOM1;
		rt[thread].commonness = 0;
		rt[thread].lights = 8;		
		rt[thread].zone[0] = 1;
		rt[thread].index = 26; //roompj
		break;

	case 30:
		//914
		rt[thread].id = thread;
		rt[thread].name = ROOM914;// "914";
		rt[thread].shape = ROOM1;
		rt[thread].commonness = 0;
		rt[thread].lights = 9;		
		rt[thread].zone[0] = 1;		
		rt[thread].index = 27; //914
		break;

	case 31:
		//room2gw
		rt[thread].id = thread;
		rt[thread].name = ROOM2GW;// "room2gw";
		rt[thread].shape = ROOM2;
		rt[thread].commonness = 10;
		rt[thread].lights = 2;
		rt[thread].zone[0] = 1;
		rt[thread].index = 28; //room2gw
		break;

	case 32:
		//room2gw_b
		rt[thread].id = thread;
		rt[thread].name = ROOM2GW_B;// "room2gw_b";
		rt[thread].shape = ROOM2;
		rt[thread].commonness = 0;
		rt[thread].lights = 2;
		rt[thread].zone[0] = 1;
		rt[thread].index = 29; //room2gw_b
		break;

	case 33:
		//room1162
		rt[thread].id = thread;
		rt[thread].name = ROOM1162;// "room1162";
		rt[thread].shape = ROOM2C;
		rt[thread].commonness = 0;
		rt[thread].lights = 5;
		rt[thread].zone[0] = 1;
		rt[thread].index = 30; //room1162
		break;

	case 34:
		//room2scps2
		rt[thread].id = thread;
		rt[thread].name = ROOM2SCPS2;// "room2scps2";
		rt[thread].shape = ROOM2;
		rt[thread].commonness = 0;
		rt[thread].lights = 6;
		rt[thread].zone[0] = 1;
		rt[thread].index = 31; //room2scps2
		break;

	case 35:
		//room2sl
		rt[thread].id = thread;
		rt[thread].name = ROOM2SL;// "room2sl";
		rt[thread].shape = ROOM2;
		rt[thread].commonness = 0;
		rt[thread].lights = 7;
		rt[thread].zone[0] = 1;
		rt[thread].index = 32; //room2sl
		break;

	case 36:
		//lockroom3
		rt[thread].id = thread;
		rt[thread].name = LOCKROOM3;// "lockroom3";
		rt[thread].shape = ROOM2C;
		rt[thread].commonness = 15;
		rt[thread].lights = 3;
		rt[thread].zone[0] = 1;
		rt[thread].index = 33; //lockroom3
		break;

	case 37:
		//room4info
		rt[thread].id = thread;
		rt[thread].name = ROOM4INFO;// "room4info";
		rt[thread].shape = ROOM4;
		rt[thread].commonness = 0;
		rt[thread].lights = 5;
		rt[thread].zone[0] = 1;
		rt[thread].index = 34; //room4info
		break;

	case 38:
		//room3_3
		rt[thread].id = thread;
		rt[thread].name = ROOM3_3;// "room3_3";
		rt[thread].shape = ROOM3;
		rt[thread].commonness = 20;
		rt[thread].lights = 5;
		rt[thread].zone[0] = 1;
		rt[thread].index = 35; //room3_3
		break;

	case 39:
		//checkpoint1
		rt[thread].id = thread;
		rt[thread].name = CHECKPOINT1;// "checkpoint1";
		rt[thread].shape = ROOM2;
		rt[thread].commonness = 0;
		rt[thread].lights = 2;
		rt[thread].index = 36; //checkpoint1
		break;

	case 40:
		//HEAVY CONTAINMENT

		//008
		rt[thread].id = thread;
		rt[thread].name = ROOM008;// "008";
		rt[thread].shape = ROOM1;
		rt[thread].commonness = 0;
		rt[thread].lights = 5;		
		rt[thread].zone[0] = 2;
		rt[thread].index = 37; //008
		break;

	case 41:
		//room035
		rt[thread].id = thread;
		rt[thread].name = ROOM035;// "room035";
		rt[thread].shape = ROOM1;
		rt[thread].commonness = 0;
		rt[thread].lights = 5;
		rt[thread].zone[0] = 2;
		rt[thread].index = 38; //room035
		break;

	case 42:
		//room049
		rt[thread].id = thread;
		rt[thread].name = ROOM049;// "room049";
		rt[thread].shape = ROOM2;
		rt[thread].commonness = 0;
		rt[thread].lights = 26;		
		rt[thread].zone[0] = 2;
		rt[thread].disableOverlapCheck = true;
		break;

	case 43:
		//room106;
		rt[thread].id = thread;
		rt[thread].name = ROOM106;// "room106";
		rt[thread].shape = ROOM1;
		rt[thread].commonness = 0;
		rt[thread].lights = 14;	
		rt[thread].zone[0] = 2;
		rt[thread].index = 39; //room106
		break;

	case 44:
		//rom513
		rt[thread].id = thread;
		rt[thread].name = ROOM513;// "room513";
		rt[thread].shape = ROOM3;
		rt[thread].commonness = 0;
		rt[thread].lights = 6;
		rt[thread].zone[0] = 2;
		rt[thread].index = 40; //room513
		break;

	case 45:
		//coffin
		rt[thread].id = thread;
		rt[thread].name = COFFIN;// "coffin";
		rt[thread].shape = ROOM1;
		rt[thread].commonness = 0;
		rt[thread].lights = 9;		
		rt[thread].zone[0] = 1;
		rt[thread].index = 41; //coffin
		break;

	case 46:
		//room966
		rt[thread].id = thread;
		rt[thread].name = ROOM966;// "room966";
		rt[thread].shape = ROOM3;
		rt[thread].commonness = 0;
		rt[thread].lights = 5;
		rt[thread].disableOverlapCheck = true;
		break;

	case 47:
		//endroom2
		rt[thread].id = thread;
		rt[thread].name = ENDROOM2;// "endroom2";
		rt[thread].shape = ROOM1;
		rt[thread].commonness = 100;
		rt[thread].lights = 4;
		rt[thread].zone[0] = 2;
		rt[thread].index = 42; //endroom2
		break;

	case 48:
		//testroom
		rt[thread].id = thread;
		rt[thread].name = TESTROOM;// "testroom";
		rt[thread].shape = ROOM2;
		rt[thread].commonness = 0;
		rt[thread].lights = 23;		
		rt[thread].zone[0] = 2;
		rt[thread].index = 43; //testroom
		break;

	case 49:
		//tunnel
		rt[thread].id = thread;
		rt[thread].name = TUNNEL;// "tunnel";
		rt[thread].shape = ROOM2;
		rt[thread].commonness = 100;
		rt[thread].lights = 4;
		rt[thread].zone[0] = 2;
		rt[thread].index = 44; //tunnel
		break;

	case 50:
		//tunnel2
		rt[thread].id = thread;
		rt[thread].name = TUNNEL2;// "tunnel2";
		rt[thread].shape = ROOM2;
		rt[thread].commonness = 70;
		rt[thread].lights = 5;
		rt[thread].zone[0] = 2;
		rt[thread].index = 45; //tunnel2
		break;

	case 51:
		//room2ctunnel
		rt[thread].id = thread;
		rt[thread].name = ROOM2CTUNNEL;// "room2ctunnel";
		rt[thread].shape = ROOM2C;
		rt[thread].commonness = 40;
		rt[thread].lights = 3;
		rt[thread].zone[0] = 2;
		rt[thread].index = 46; //room2ctunnel
		break;

	case 52:
		//room2nuke
		rt[thread].id = thread;
		rt[thread].name = ROOM2NUKE;// "room2nuke";
		rt[thread].shape = ROOM2;
		rt[thread].commonness = 0;
		rt[thread].lights = 14;
		rt[thread].zone[0] = 2;
		rt[thread].index = 47; //room2nuke
		break;

	case 53:
		//room2pipes
		rt[thread].id = thread;
		rt[thread].name = ROOM2PIPES;// "room2pipes";
		rt[thread].shape = ROOM2;
		rt[thread].commonness = 50;
		rt[thread].lights = 6;
		rt[thread].zone[0] = 2;
		rt[thread].index = 48; //room2pipes
		break;
		
	case 54:
		//room2pit
		rt[thread].id = thread;
		rt[thread].name = ROOM2PIT;// "room2pit";
		rt[thread].shape = ROOM2;
		rt[thread].commonness = 75;
		rt[thread].lights = 6;	
		rt[thread].zone[0] = 2;
		rt[thread].index = 49; //room2pit
		break;

	case 55:
		//room3pit
		rt[thread].id = thread;
		rt[thread].name = ROOM3PIT;// "room3pit";
		rt[thread].shape = ROOM3;
		rt[thread].commonness = 100;
		rt[thread].lights = 12;		
		rt[thread].zone[0] = 2;
		rt[thread].index = 50; //room3pit
		break;

	case 56:
		//room4pit
		rt[thread].id = thread;
		rt[thread].name = ROOM4PIT;// "room4pit";
		rt[thread].shape = ROOM4;
		rt[thread].commonness = 100;
		rt[thread].lights = 16;
		rt[thread].zone[0] = 2;
		rt[thread].index = 51; //room4pit
		break;

	case 57:
		//room2servers
		rt[thread].id = thread;
		rt[thread].name = ROOM2SERVERS;// "room2servers";
		rt[thread].shape = ROOM2;
		rt[thread].commonness = 0;
		rt[thread].lights = 5;
		rt[thread].zone[0] = 2;
		rt[thread].index = 52; //room2servers
		break;

	case 58:
		//room2shaft
		rt[thread].id = thread;
		rt[thread].name = ROOM2SHAFT;// "room2shaft";
		rt[thread].shape = ROOM2;
		rt[thread].commonness = 0;
		rt[thread].lights = 5;
		rt[thread].index = 53; //room2shaft
		break;

	case 59:
		//room2tunnel
		rt[thread].id = thread;
		rt[thread].name = ROOM2TUNNEL;// "room2tunnel";
		rt[thread].shape = ROOM2;
		rt[thread].commonness = 0;
		rt[thread].lights = 4;
		rt[thread].zone[0] = 2;
		rt[thread].index = 54; //room2tunnel
		break;

	case 60:
		//room3tunnel
		rt[thread].id = thread;
		rt[thread].name = ROOM3TUNNEL;// "room3tunnel";
		rt[thread].shape = ROOM3;
		rt[thread].commonness = 100;
		rt[thread].lights = 5;
		rt[thread].zone[0] = 2;
		rt[thread].index = 55; //room3tunnel
		break;

	case 61:
		//room4tunnels
		rt[thread].id = thread;
		rt[thread].name = ROOM4TUNNELS;// "room4tunnels";
		rt[thread].shape = ROOM4;
		rt[thread].commonness = 100;
		rt[thread].lights = 6;
		rt[thread].zone[0] = 2;
		rt[thread].index = 56; //room4tunnels
		break;

	case 62:
		//room2tesla_hcz
		rt[thread].id = thread;
		rt[thread].name = ROOM2TESLA_HCZ;// "room2tesla_hcz";
		rt[thread].shape = ROOM2;
		rt[thread].commonness = 100;
		rt[thread].lights = 10;
		rt[thread].zone[0] = 2;
		rt[thread].index = 57; //room2tesla_hcz
		break;

	case 63:
		//room3z2
		rt[thread].id = thread;
		rt[thread].name = ROOM3Z2;// "room3z2";
		rt[thread].shape = ROOM3;
		rt[thread].commonness = 100;
		rt[thread].lights = 3;
		rt[thread].zone[0] = 2;
		rt[thread].index = 58; //room3z2
		break;

	case 64:
		//room2cpit
		rt[thread].id = thread;
		rt[thread].name = ROOM2CPIT;// "room2cpit";
		rt[thread].shape = ROOM2C;
		rt[thread].commonness = 0;
		rt[thread].lights = 10;		
		rt[thread].zone[0] = 2;
		rt[thread].index = 59; //room2cpit
		break;

	case 65:
		//room2pipes2
		rt[thread].id = thread;
		rt[thread].name = ROOM2PIPES2;// "room2pipes2";
		rt[thread].shape = ROOM2;
		rt[thread].commonness = 70;
		rt[thread].lights = 6;		
		rt[thread].zone[0] = 2;
		rt[thread].index = 60; //room2pipes2
		break;

	case 66:
		//checkpoint2
		rt[thread].id = thread;
		rt[thread].name = CHECKPOINT2;// "checkpoint2";
		rt[thread].shape = ROOM2;
		rt[thread].commonness = 0;
		rt[thread].lights = 2;
		rt[thread].index = 61; //checkpoint2
		break;

	case 67:
		//ENTRANCE ZONE

		//room079 yea he's in the entrance zone mhm
		rt[thread].id = thread;
		rt[thread].name = ROOM079;// "room079";
		rt[thread].shape = ROOM1;
		rt[thread].commonness = 0;
		rt[thread].lights = 6;		
		rt[thread].zone[0] = 3;
		rt[thread].index = 62; //room079
		break;

	case 68:
		//lockroom2
		rt[thread].id = thread;
		rt[thread].name = LOCKROOM2;// "lockroom2";
		rt[thread].shape = ROOM2C;
		rt[thread].commonness = 0;
		rt[thread].lights = 6;
		rt[thread].zone[0] = 3;
		rt[thread].index = 63; //lockroom2
		break;

	case 69:
		//exit1
		rt[thread].id = thread;
		rt[thread].name = EXIT1;// "exit1";
		rt[thread].shape = ROOM1;
		rt[thread].commonness = 0;
		rt[thread].lights = 15;
		rt[thread].zone[0] = 3;
		rt[thread].disableOverlapCheck = true;
		break;

	case 70:
		//gateaentrance
		rt[thread].id = thread;
		rt[thread].name = GATEAENTRANCE;// "gateaentrance";
		rt[thread].shape = ROOM1;
		rt[thread].commonness = 0;
		rt[thread].lights = 4;
		rt[thread].zone[0] = 3;
		rt[thread].index = 64; //gateaentrance
		break;

	case 71:
		//gatea
		rt[thread].id = thread;
		rt[thread].name = GATEA;// "gatea";
		rt[thread].shape = ROOM1;
		rt[thread].commonness = 0;
		rt[thread].lights = 12;
		rt[thread].disableOverlapCheck = true;
		break;

	case 72:
		//medibay
		rt[thread].id = thread;
		rt[thread].name = MEDIBAY;// "medibay";
		rt[thread].shape = ROOM2;
		rt[thread].commonness = 0;
		rt[thread].lights = 6;
		rt[thread].zone[0] = 3;
		rt[thread].index = 65; //medibay
		break;

	case 73:
		//room2z3
		rt[thread].id = thread;
		rt[thread].name = ROOM2Z3;// "room2z3";
		rt[thread].shape = ROOM2;
		rt[thread].commonness = 75;
		rt[thread].lights = 0;
		rt[thread].zone[0] = 3;
		rt[thread].index = 66; //room2z3
		break;

	case 74:
		//room2cafeteria
		rt[thread].id = thread;
		rt[thread].name = ROOM2CAFETERIA;// "room2cafeteria";
		rt[thread].shape = ROOM2;
		rt[thread].commonness = 0;
		rt[thread].lights = 7;
		rt[thread].zone[0] = 3;
		rt[thread].index = 67; //room2cafeteria
		break;

	case 75:
		//room2cz3
		rt[thread].id = thread;
		rt[thread].name = ROOM2CZ3;// "room2cz3";
		rt[thread].shape = ROOM2C;
		rt[thread].commonness = 100;
		rt[thread].lights = 4;
		rt[thread].zone[0] = 3;
		rt[thread].index = 68; //room2cz3
		break;

	case 76:
		//room2ccont
		rt[thread].id = thread;
		rt[thread].name = ROOM2CCONT;// "room2ccont";
		rt[thread].shape = ROOM2C;
		rt[thread].commonness = 0;
		rt[thread].lights = 9;
		rt[thread].zone[0] = 3;
		rt[thread].index = 69; //room2ccont
		break;

	case 77:
		//room2offices
		rt[thread].id = thread;
		rt[thread].name = ROOM2OFFICES;// "room2offices";
		rt[thread].shape = ROOM2;
		rt[thread].commonness = 30;
		rt[thread].lights = 4;
		rt[thread].zone[0] = 3;
		rt[thread].index = 70; //room2offices
		break;

	case 78:
		//room2offices2
		rt[thread].id = thread;
		rt[thread].name = ROOM2OFFICES2;// "room2offices2";
		rt[thread].shape = ROOM2;
		rt[thread].commonness = 20;
		rt[thread].lights = 6;		
		rt[thread].zone[0] = 3;
		rt[thread].index = 71; //room2offices2
		break;

	case 79:
		//room2offices3
		rt[thread].id = thread;
		rt[thread].name = ROOM2OFFICES3;// "room2offices3";
		rt[thread].shape = ROOM2;
		rt[thread].commonness = 20;
		rt[thread].lights = 6;
		rt[thread].zone[0] = 3;
		rt[thread].index = 72; //room2offices3
		break;

	case 80:
		//room2offices4
		rt[thread].id = thread;
		rt[thread].name = ROOM2OFFICES4;// "room2offices4";
		rt[thread].shape = ROOM2;
		rt[thread].commonness = 0;
		rt[thread].lights = 5;
		rt[thread].zone[0] = 3;
		rt[thread].index = 73; //room2offices4
		break;

	case 81:
		//room2poffices
		rt[thread].id = thread;
		rt[thread].name = ROOM2POFFICES;// "room2poffices";
		rt[thread].shape = ROOM2;
		rt[thread].commonness = 0;
		rt[thread].lights = 5;
		rt[thread].zone[0] = 3;
		rt[thread].index = 74; //room2poffices
		break;

	case 82:
		//room2poffices2
		rt[thread].id = thread;
		rt[thread].name = ROOM2POFFICES2;// "room2poffices2";
		rt[thread].shape = ROOM2;
		rt[thread].commonness = 0;
		rt[thread].lights = 4;
		rt[thread].zone[0] = 3;
		rt[thread].index = 75; //room2poffices2
		break;

	case 83:
		//room2sroom
		rt[thread].id = thread;
		rt[thread].name = ROOM2SROOM;// "room2sroom";
		rt[thread].shape = ROOM2;
		rt[thread].commonness = 0;
		rt[thread].lights = 4;
		rt[thread].zone[0] = 3;
		rt[thread].index = 76; //room2sroom
		break;

	case 84:
		//room2toilets
		rt[thread].id = thread;
		rt[thread].name = ROOM2TOILETS;// "room2toilets";
		rt[thread].shape = ROOM2;
		rt[thread].commonness = 30;
		rt[thread].lights = 5;
		rt[thread].zone[0] = 3;
		rt[thread].index = 77; //room2toilets
		break;

	case 85:
		//room2tesla
		rt[thread].id = thread;
		rt[thread].name = ROOM2TESLA;// "room2tesla";
		rt[thread].shape = ROOM2;
		rt[thread].commonness = 100;
		rt[thread].lights = 8;
		rt[thread].zone[0] = 3;
		rt[thread].index = 78; //room2tesla
		break;

	case 86:
		//room3servers
		rt[thread].id = thread;
		rt[thread].name = ROOM3SERVERS;// "room3servers";
		rt[thread].shape = ROOM3;
		rt[thread].commonness = 0;
		rt[thread].lights = 6;		
		rt[thread].zone[0] = 3;
		rt[thread].index = 79; //room3servers
		break;

	case 87:
		//room3servers2
		rt[thread].id = thread;
		rt[thread].name = ROOM3SERVERS2;// "room3servers2";
		rt[thread].shape = ROOM3;
		rt[thread].commonness = 0;
		rt[thread].lights = 6;		
		rt[thread].zone[0] = 3;
		rt[thread].index = 80; //room3servers2
		break;

	case 88:
		//room3z3
		rt[thread].id = thread;
		rt[thread].name = ROOM3Z3;// "room3z3";
		rt[thread].shape = ROOM3;
		rt[thread].commonness = 100;
		rt[thread].lights = 1;
		rt[thread].zone[0] = 3;
		rt[thread].index = 81; //room3z3
		break;

	case 89:
		//room4z3
		rt[thread].id = thread;
		rt[thread].name = ROOM4Z3;// "room4z3";
		rt[thread].shape = ROOM4;
		rt[thread].commonness = 100;
		rt[thread].lights = 2;
		rt[thread].zone[0] = 3;
		rt[thread].index = 82; //room4z3
		break;

	case 90:
		//room1lifts
		rt[thread].id = thread;
		rt[thread].name = ROOM1LIFTS;// "room1lifts";
		rt[thread].shape = ROOM1;
		rt[thread].commonness = 0;
		rt[thread].lights = 1;
		rt[thread].zone[0] = 3;
		rt[thread].index = 83; //room1lifts
		break;

	case 91:
		//room3gw
		rt[thread].id = thread;
		rt[thread].name = ROOM3GW;// "room3gw";
		rt[thread].shape = ROOM3;
		rt[thread].commonness = 10;
		rt[thread].lights = 1;
		rt[thread].zone[0] = 3;
		rt[thread].index = 84; //room3gw
		break;

	case 92:
		//room2servers2
		rt[thread].id = thread;
		rt[thread].name = ROOM2SERVERS2;// "room2servers2";
		rt[thread].shape = ROOM2;
		rt[thread].commonness = 0;
		rt[thread].lights = 4;
		rt[thread].zone[0] = 3;
		rt[thread].index = 85; //room2servers2
		break;

	case 93:
		//room3offices
		rt[thread].id = thread;
		rt[thread].name = ROOM3OFFICES;// "room3offices";
		rt[thread].shape = ROOM3;
		rt[thread].commonness = 0;
		rt[thread].lights = 1;
		rt[thread].zone[0] = 3;
		rt[thread].index = 86; //room3offices
		break;

	case 94:
		//room2z3_2
		rt[thread].id = thread;
		rt[thread].name = ROOM2Z3_2;// "room2z3_2";
		rt[thread].shape = ROOM2;
		rt[thread].commonness = 25;
		rt[thread].lights = 0;
		rt[thread].zone[0] = 3;
		rt[thread].index = 87; //room2z3_2
		break;

	case 95:
		//pocketdimension
		rt[thread].id = thread;
		rt[thread].name = POCKETDIMENSION;// "pocketdimension";
		rt[thread].shape = ROOM1;
		rt[thread].commonness = 0;
		rt[thread].lights = 1;
		rt[thread].index = 88; //pocketdimension
		break;

	case 96:
		//dimension1499
		rt[thread].id = thread;
		rt[thread].name = DIMENSION1499;// "dimension1499";
		rt[thread].shape = ROOM1;
		rt[thread].commonness = 0;
		rt[thread].lights = 0;	
		rt[thread].index = 89; //dimension1499
		break;
	default:
		//printf("Cannot set room template for thread: %d.\n", thread);
		break;
	}
	
}

__device__ inline bool SetRoom(RoomID MapRoom[6][70], RoomID room_name, uint8_t room_type, uint8_t pos, uint8_t min_pos, uint8_t max_pos) {
	if (max_pos < min_pos) return false;

	bool looped = false; 
	bool can_place = true;

	while (MapRoom[room_type][pos] != ROOMEMPTY) {
		pos++;
		if (pos > max_pos) {
			if (!looped) {
				pos = min_pos + 1;
				looped = true;
			}
			else {
				can_place = false;
				break;
			}
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

__device__ inline Rooms CreateRoom(uint8_t MapTemp[19][19], uint8_t& roomIdCounter, uint8_t* forest, float* e, RoomTemplates* rts, bbRandom* bb, rnd_state* rnd_state, uint8_t zone, uint8_t roomshape, float x, float y, float z, RoomID name) {

	Rooms r = Rooms();
	//RoomTemplates* rt;

	//The original game doesn't actually have a room id variable
	r.id = roomIdCounter++;

	r.zone = zone;

	r.x = x;
	r.y = y;
	r.z = z;

	if (name != ROOMEMPTY) {
		for (uint8_t i = 0; i < roomTemplateAmount; i++) {			
			if (rts[i].name == name) {
				r.rt = &rts[i];
				
				FillRoom(MapTemp, bb, rnd_state, &r, forest);

				//Don't think we need light cone stuff.
				
				//CalculateRoomExtents(&r);
				//TEMPORARY
				GetRoomExtents(&r, e);
				return r;
			}
		}
	}	

	int32_t t = 0;
	for (uint8_t i = 0; i < roomTemplateAmount; i++) {
		//5 because that is the len of the rt.zone[] array;
		for (uint8_t j = 0; j <= 4; j++) {
			if (rts[i].zone[j] == zone) {
				if (rts[i].shape == roomshape) {
					t = t + rts[i].commonness;
					break;
				}
			}
		}
	}
	

	int32_t RandomRoom = bb->bbRand(rnd_state, 1, t);
	t = 0;
	for (uint8_t i = 0; i < roomTemplateAmount; i++) {
		for (uint8_t j = 0; j <= 4; j++) {
			if (rts[i].zone[j] == zone && rts[i].shape == roomshape) {
				t = t + rts[i].commonness;
				if (RandomRoom > t - rts[i].commonness && RandomRoom <= t) {
					r.rt = &rts[i];
					
					FillRoom(MapTemp, bb, rnd_state, &r, forest);

					//Skip light cone stuff
			
					//CalculateRoomExtents(&r);
					//TEMPORARY
					GetRoomExtents(&r, e);
					return r;
				}
			}
		}
	}	
	//It seems like there is supposed to be a return r at the bottom here,
	//but it isn't there in the blitz code so idk.
}

__device__ inline bool PreventRoomOverlap(Rooms* r, Rooms* rooms, float* e) {
	if (r->rt->disableOverlapCheck) return true;

	//We might have some problems with passing the rooms array by pointer.
	//Want to make sure we pass by reference instead of making a new copy of entire rooms array.

	Rooms* r2;
	Rooms* r3;

	bool isIntersecting = false;

	if (r->rt->name == CHECKPOINT1 || r->rt->name == CHECKPOINT2 || r->rt->name == START) return true;

#pragma unroll
	for (int16_t i = 0; i < 324; i++) {
		r2 = &rooms[i];

		//-1 id means we are at the point of the array where there are no more rooms.
		if (r2->id == -1) break;

		if (r2->id != r->id && !r2->rt->disableOverlapCheck) {
			if (CheckRoomOverlap(r, r2)) {
				isIntersecting = true;
				break;
			}
		}
	}

	if (!isIntersecting) return true;

	isIntersecting = false;

	uint8_t x = uint8_t(r->x / 8.0);
	uint8_t y = uint8_t(r->y / 8.0);

	if (r->rt->shape == ROOM2) {
		r->angle += 180;
		//CalculateRoomExtents(r);
		//TEMPORARY
		GetRoomExtents(r, e);

		for (int16_t i = 0; i < 324; i++) {
			r2 = &rooms[i];
			if (r2->id == -1) break; //Id should only be -1 after all other rooms.

			if (r2->id != r->id && !r2->rt->disableOverlapCheck) {
				if (CheckRoomOverlap(r, r2)) {
					isIntersecting = true;
					r->angle -= 180;
					//CalculateRoomExtents(r);
					//TEMPORARY
					GetRoomExtents(r, e);
					break;
				}
			}
		}

	}
	else {
		isIntersecting = true;
	}

	if (!isIntersecting) {
		return true;
	}

	//Room is either not a ROOM2 or the ROOM2 is still intersecting, now trying to swap the room with another of the same type
	isIntersecting = true;
	uint8_t x2 = 0, y2 = 0;
	int16_t rot = 0, rot2 = 0;	
	for (int16_t i = 0; i < 324; i++) {
		r2 = &rooms[i];

		if (r2->id == -1) break;

		if (r2->id != r->id && !r2->rt->disableOverlapCheck) {
			if (r->rt->shape == r2->rt->shape && r->zone == r2->zone && (r2->rt->name != CHECKPOINT1 && r2->rt->name != CHECKPOINT2 && r2->rt->name != START)) {
				x = uint8_t(r->x / 8.0);
				y = uint8_t(r->z / 8.0);
				rot = r->angle;

				x2 = uint8_t(r2->x / 8.0);
				y2 = uint8_t(r2->z / 8.0);
				rot2 = r2->angle;

				isIntersecting = false;

				r->x = float(x2) * 8.0;
				r->z = float(y2) * 8.0;
				r->angle = rot2;
				//CalculateRoomExtents(r);
				//TEMPORARY
				GetRoomExtents(r, e);

				r2->x = float(x) * 8.0;
				r2->z = float(y) * 8.0;
				r2->angle = rot;
				//CalculateRoomExtents(r2);
				//TEMPORARY
				GetRoomExtents(r2, e);

				//make sure neither room overlaps with anything after the swap
				for (int16_t j = 0; j < 324; j++) {
					r3 = &rooms[j];

					if (r3->id == -1) break;

					if (!r3->rt->disableOverlapCheck) {
						if (r3->id != r->id) {
							if (CheckRoomOverlap(r, r3)) {
								isIntersecting = true;
								break;
							}
						}

						if (r3->id != r2->id) {
							if (CheckRoomOverlap(r2, r3)) {
								isIntersecting = true;
								break;
							}
						}
					}
				}
				//Either the original room or the "reposition" room is intersecting, reset the position of each room to their original one
				if (isIntersecting) {
					r->x = float(x) * 8.0;
					r->z = float(y) * 8.0;
					r->angle = rot;
					//CalculateRoomExtents(r);
					//TEMPORARY
					GetRoomExtents(r, e);

					r2->x = float(x2) * 8.0;
					r2->z = float(y2) * 8.0;
					r2->angle = rot2;
					//CalculateRoomExtents(r2);
					//TEMPORARY
					GetRoomExtents(r2, e);

					isIntersecting = false;
				}
			}
		}
	}

	if (!isIntersecting) {
		return true;
	}

	return false;
}

__device__ inline bool CheckRoomOverlap(Rooms* r, Rooms* r2) {

	if (r->maxX <= r2->minX || r->maxZ <= r2->minZ) return false;

	if (r->minX >= r2->maxX || r->minZ >= r2->maxZ) return false;

	return true;
}

//__device__ inline void CalculateRoomExtents(Rooms* r) {
//	if (r->rt->disableOverlapCheck) return;
//
//	static const float shrinkAmount = 0.05;
//
//	//We must convert TFormVector() in blitz to c++ code.
//	static float roomScale = 8.0 / 2048.0;
//
//	//First we scale.
//	r->rt->minX *= roomScale;
//	r->rt->minY *= roomScale;
//	r->rt->minZ *= roomScale;
//
//	r->rt->maxX *= roomScale;
//	r->rt->maxY *= roomScale;
//	r->rt->maxZ *= roomScale;
//
//	//Then we rotate	
//	float rad = 0.0;
//	switch (r->angle) {
//	case 90:
//		rad = 1.5708;
//		break;
//	case 180:
//		rad = 3.14159;
//		break;
//	case 270:
//		rad = 4.71239;
//		break;
//	}
//
//	r->rt->minX = r->rt->minX * cosf(rad) - r->rt->minZ * sinf(rad);
//	r->rt->minZ = r->rt->minX * sinf(rad) + r->rt->minZ * cosf(rad);
//
//	r->rt->maxX = r->rt->maxX * cosf(rad) - r->rt->maxZ * sinf(rad);
//	r->rt->maxZ = r->rt->maxX * sinf(rad) + r->rt->maxZ * cosf(rad);
//
//	//Back to blitz.
//
//	r->minX = r->rt->minX + shrinkAmount + r->x;
//	r->minY = r->rt->minY + shrinkAmount;
//	r->minZ = r->rt->minZ + shrinkAmount + r->z;
//
//	r->maxX = r->rt->maxX - shrinkAmount + r->x;
//	r->maxY = r->rt->maxY - shrinkAmount;
//	r->maxZ = r->rt->maxZ - shrinkAmount + r->z;
//
//	if (r->minX > r->maxX) {
//		float temp = r->maxX;
//		r->maxX = r->minX;
//		r->minX = temp;
//	}
//
//	if (r->minZ > r->maxZ) {
//		float temp = r->maxZ;
//		r->maxZ = r->minZ;
//		r->minZ = temp;
//	}
//
//	/*r->minX = rintf(r->minX * 1000000.0) / 1000000.0;
//	r->minY = rintf(r->minY * 1000000.0) / 1000000.0;
//	r->minX = rintf(r->minZ * 1000000.0) / 1000000.0;
//
//	r->maxX = rintf(r->maxX * 1000000.0) / 1000000.0;
//	r->maxY = rintf(r->maxY * 1000000.0) / 1000000.0;
//	r->maxZ = rintf(r->maxZ * 1000000.0) / 1000000.0;*/
//
//	//printf("NAME: %s MINX %f MINY %f MINZ %f MAXX %f MAXY %f MAXZ %f\n", RoomIDToName(r->rt.name), r->minX, r->minY, r->minZ, r->maxX, r->maxY, r->maxZ);
//
//	return;
//}

__device__ inline void FillRoom(uint8_t MapTemp[19][19], bbRandom* bb, rnd_state* rnd_state, Rooms* r, uint8_t* forest) {

	switch (r->rt->name) {
	case ROOM860:
		CreateDoor(bb, rnd_state, false, 0);
		CreateDoor(bb, rnd_state, true, 0);

		CreateDoor(bb, rnd_state, false, 0);
		CreateDoor(bb, rnd_state, false, 0);

		GenForestGrid(bb, rnd_state, forest);

		CreateItem(bb, rnd_state);
		CreateItem(bb, rnd_state);
		break;

	case LOCKROOM:
		CreateDoor(bb, rnd_state, true, 0);
		CreateDoor(bb, rnd_state, true, 0);
		break;
	case LOCKROOM2:
#pragma unroll
		for (uint8_t i = 0; i <= 5; i++) {
			bb->bbRand(rnd_state, 2, 3);
			bb->bbRnd(rnd_state, -392, 520);
			bb->bbRnd(rnd_state, 0, 0.001);
			bb->bbRnd(rnd_state, -392, 520);
			bb->bbRnd(rnd_state, 0, 360);
			bb->bbRnd(rnd_state, 0.3, 0.6);

			bb->bbRand(rnd_state, 15, 16);
			bb->bbRnd(rnd_state, -392, 520);
			bb->bbRnd(rnd_state, 0, 0.001);
			bb->bbRnd(rnd_state, -392, 520);
			bb->bbRnd(rnd_state, 0, 360);
			bb->bbRnd(rnd_state, 0.1, 0.6);

			bb->bbRand(rnd_state, 15, 16);
			bb->bbRnd(rnd_state, -0.5, 0.5);
			bb->bbRnd(rnd_state, 0, 0.001);
			bb->bbRnd(rnd_state, -0.5, 0.5);
			bb->bbRnd(rnd_state, 0, 360);
			bb->bbRnd(rnd_state, 0.1, 0.6);
		}
		break;
	case GATEA:
		CreateDoor(bb, rnd_state, false, 0);
		CreateDoor(bb, rnd_state, false, 0);
		CreateDoor(bb, rnd_state, false, 0);
		CreateDoor(bb, rnd_state, false, 0);

		//INCOMPLETE
		//This might be wrong. We'll see when we test FillRoom();
		CreateDoor(bb, rnd_state, false, 3);
		break;

	case GATEAENTRANCE:
		CreateDoor(bb, rnd_state, true, 3);
		CreateDoor(bb, rnd_state, false, 1);
		break;

	case EXIT1:
		CreateDoor(bb, rnd_state, false, 1);
		CreateDoor(bb, rnd_state, true, 3);
		CreateDoor(bb, rnd_state, false, 3);
		CreateDoor(bb, rnd_state, false, 0);
		CreateDoor(bb, rnd_state, false, 0);
		CreateDoor(bb, rnd_state, false, 0);
		CreateDoor(bb, rnd_state, false, 0);
		break;

	case ROOMPJ:
		CreateItem(bb, rnd_state);
		CreateItem(bb, rnd_state);
		CreateDoor(bb, rnd_state, true, 1);
		break;

	case ROOM079:
		CreateDoor(bb, rnd_state, false, 1);
		CreateDoor(bb, rnd_state, false, 1);
		CreateDoor(bb, rnd_state, false, 0);
		bb->bbRnd(rnd_state, 0, 360);
		break;

	case CHECKPOINT1:
	case CHECKPOINT2:
		CreateDoor(bb, rnd_state, false, 0);
		CreateDoor(bb, rnd_state, false, 0);

		if (MapTemp[uint8_t(floorf(r->x / 8.0))][uint8_t(floorf(r->z / 8.0)) - 1] == 0) {
			CreateDoor(bb, rnd_state, false, 0);
		}
		break;

	case ROOM2PIT:
		break;

	case ROOM2TESTROOM2:
		CreateDoor(bb, rnd_state, false, 0);
		CreateDoor(bb, rnd_state, false, 0);

		CreateItem(bb, rnd_state);
		CreateItem(bb, rnd_state);
		break;

	case ROOM3TUNNEL:
		break;

	case ROOM2TOILETS:
		break;

	case ROOM2STORAGE:
		CreateDoor(bb, rnd_state, false, 0);
		CreateDoor(bb, rnd_state, false, 0);
		CreateDoor(bb, rnd_state, false, 0);
		CreateDoor(bb, rnd_state, false, 0);
		CreateDoor(bb, rnd_state, false, 0);
		CreateDoor(bb, rnd_state, false, 0);

		CreateItem(bb, rnd_state);
		CreateItem(bb, rnd_state);
		CreateItem(bb, rnd_state);
		CreateItem(bb, rnd_state);
		break;

	case ROOM2SROOM:
		CreateDoor(bb, rnd_state, false, 0);

		CreateItem(bb, rnd_state);
		CreateItem(bb, rnd_state);
		CreateItem(bb, rnd_state);
		CreateItem(bb, rnd_state);
		CreateItem(bb, rnd_state);
		break;

	case ROOM2SHAFT:
		CreateDoor(bb, rnd_state, false, 0);
		CreateDoor(bb, rnd_state, false, 0);

		CreateItem(bb, rnd_state);
		CreateItem(bb, rnd_state);
		CreateItem(bb, rnd_state);
		CreateItem(bb, rnd_state);
		CreateItem(bb, rnd_state);

		bb->bbRnd(rnd_state, 0, 360);
		break;

	case ROOM2POFFICES:
		CreateDoor(bb, rnd_state, false, 0);
		CreateDoor(bb, rnd_state, false, 0);
		CreateDoor(bb, rnd_state, false, 0);

		CreateItem(bb, rnd_state);
		CreateItem(bb, rnd_state);
		CreateItem(bb, rnd_state);
		CreateItem(bb, rnd_state);
		CreateItem(bb, rnd_state);
		break;

	case ROOM2POFFICES2:
		CreateDoor(bb, rnd_state, false, 0);
		CreateDoor(bb, rnd_state, false, 0);

		bb->bbRnd(rnd_state, 0, 360);
		bb->bbRnd(rnd_state, 0, 360);
		bb->bbRnd(rnd_state, 0, 360);

		CreateItem(bb, rnd_state);
		CreateItem(bb, rnd_state);
		CreateItem(bb, rnd_state);
		break;

	case ROOM2ELEVATOR:
		CreateDoor(bb, rnd_state, false, 3);
		break;

	case ROOM2CAFETERIA:
		CreateItem(bb, rnd_state);
		CreateItem(bb, rnd_state);
		CreateItem(bb, rnd_state);
		CreateItem(bb, rnd_state);
		CreateItem(bb, rnd_state);
		break;

	case ROOM2NUKE:
		CreateDoor(bb, rnd_state, false, false);
		CreateDoor(bb, rnd_state, false, false);
		CreateDoor(bb, rnd_state, true, 3);
		CreateDoor(bb, rnd_state, false, 3);

		CreateItem(bb, rnd_state);
		CreateItem(bb, rnd_state);
		break;

	case TUNNEL:
		break;

	case ROOM2TUNNEL:
		CreateDoor(bb, rnd_state, true, 3);
		CreateDoor(bb, rnd_state, true, 3);
		CreateDoor(bb, rnd_state, false, 1);

		bb->bbRnd(rnd_state, 0, 360);

		CreateItem(bb, rnd_state);
		break;

	case ROOM2CTUNNEL:
		break;

	case ROOM008:
		CreateDoor(bb, rnd_state, true, 0);
		CreateDoor(bb, rnd_state, false, 0);
		CreateDoor(bb, rnd_state, false, 0);

		CreateItem(bb, rnd_state);
		CreateItem(bb, rnd_state);
		break;

	case ROOM035:
		CreateDoor(bb, rnd_state, true, 0);
		CreateDoor(bb, rnd_state, false, 0);
		CreateDoor(bb, rnd_state, false, 0);
		CreateDoor(bb, rnd_state, false, 0);

		CreateItem(bb, rnd_state);
		CreateItem(bb, rnd_state);
		CreateItem(bb, rnd_state);
		CreateItem(bb, rnd_state);
		CreateItem(bb, rnd_state);
		break;

	case ROOM513:
		CreateDoor(bb, rnd_state, false, 0);

		CreateItem(bb, rnd_state);
		CreateItem(bb, rnd_state);
		CreateItem(bb, rnd_state);
		break;

	case ROOM966:
		CreateDoor(bb, rnd_state, false, 0);
		CreateDoor(bb, rnd_state, false, 0);

		CreateItem(bb, rnd_state);
		break;

	case ROOM3STORAGE:
		CreateDoor(bb, rnd_state, true, 3);
		CreateDoor(bb, rnd_state, false, 3);
		CreateDoor(bb, rnd_state, true, 3);
		CreateDoor(bb, rnd_state, false, 3);

		bb->bbRand(rnd_state, 1, 3);

		CreateItem(bb, rnd_state);
		CreateItem(bb, rnd_state);

		bb->bbRnd(rnd_state, 0, 360);

		CreateDoor(bb, rnd_state, false, 2);
		CreateDoor(bb, rnd_state, false, 2);
		CreateDoor(bb, rnd_state, false, 2);
		CreateDoor(bb, rnd_state, false, 2);
		break;

	case ROOM049:
		CreateDoor(bb, rnd_state, true, 3);
		CreateDoor(bb, rnd_state, false, 3);
		CreateDoor(bb, rnd_state, true, 3);
		CreateDoor(bb, rnd_state, false, 3);
		CreateDoor(bb, rnd_state, false, 0);
		CreateDoor(bb, rnd_state, false, 0);
		CreateDoor(bb, rnd_state, false, 0);
		CreateDoor(bb, rnd_state, false, 2);

		CreateItem(bb, rnd_state);
		CreateItem(bb, rnd_state);
		CreateItem(bb, rnd_state);

		CreateDoor(bb, rnd_state, true, 1);
		CreateDoor(bb, rnd_state, false, 2);
		CreateDoor(bb, rnd_state, false, 2);
		break;

	case ROOM2ID:
	case ROOM2_2:
		break;

	case ROOM012:
		CreateDoor(bb, rnd_state, false, 0);
		CreateDoor(bb, rnd_state, false, 0);

		CreateItem(bb, rnd_state);
		CreateItem(bb, rnd_state);

		bb->bbRnd(rnd_state, 0, 360);
		break;

	case TUNNEL2:
		break;

	case ROOM2PIPES:
		break;

	case ROOM3PIT:
		break;

	case ROOM2SERVERS:
		CreateDoor(bb, rnd_state, false, 2);
		CreateDoor(bb, rnd_state, true, 0);
		CreateDoor(bb, rnd_state, true, 0);
		CreateDoor(bb, rnd_state, false, 0);
		break;

	case ROOM3SERVERS:
		CreateItem(bb, rnd_state);

		
		if (bb->bbRand(rnd_state, 1, 2) == 1) {
			CreateItem(bb, rnd_state);
		}
		if (bb->bbRand(rnd_state, 1, 2) == 1) {
			CreateItem(bb, rnd_state);
		}

		CreateItem(bb, rnd_state);
		break;

	case ROOM3SERVERS2:
		CreateItem(bb, rnd_state);
		CreateItem(bb, rnd_state);
		break;

	case TESTROOM:
		CreateDoor(bb, rnd_state, false, 2);
		CreateDoor(bb, rnd_state, true, 0);

		CreateItem(bb, rnd_state);
		break;

	case ROOM2CLOSETS:
		CreateItem(bb, rnd_state);
		CreateItem(bb, rnd_state);
		CreateItem(bb, rnd_state);

		if (bb->bbRand(rnd_state, 1, 2) == 1) {
			CreateItem(bb, rnd_state);
		}
		if (bb->bbRand(rnd_state, 1, 2) == 1) {
			CreateItem(bb, rnd_state);
		}

		CreateItem(bb, rnd_state);
		CreateItem(bb, rnd_state);
		CreateItem(bb, rnd_state);

		CreateDoor(bb, rnd_state, false, 0);
		break;

	case ROOM2OFFICES:
		CreateItem(bb, rnd_state);
		CreateItem(bb, rnd_state);
		CreateItem(bb, rnd_state);
		CreateItem(bb, rnd_state);
		break;

	case ROOM2OFFICES2:
		CreateItem(bb, rnd_state);
		CreateItem(bb, rnd_state);

		bb->bbRand(rnd_state, 1, 2);

		CreateItem(bb, rnd_state);
		CreateItem(bb, rnd_state);

		bb->bbRand(rnd_state, 1, 4);
		break;

	case ROOM2OFFICES3:
		bb->bbRand(rnd_state, 1, 2);

		CreateItem(bb, rnd_state);
		CreateItem(bb, rnd_state);
		CreateItem(bb, rnd_state);
		CreateItem(bb, rnd_state);

#pragma unroll
		for (uint8_t i = 0; i <= bb->bbRand(rnd_state, 0, 1); i++) {
			CreateItem(bb, rnd_state);
		}

		CreateItem(bb, rnd_state);

		if (bb->bbRand(rnd_state, 1, 2) == 1) {
			CreateItem(bb, rnd_state);
		}
		if (bb->bbRand(rnd_state, 1, 2) == 1) {
			CreateItem(bb, rnd_state);
		}

		CreateDoor(bb, rnd_state, true, 0);
		break;

	case START:
		CreateDoor(bb, rnd_state, true, 1);	
		CreateDoor(bb, rnd_state, false, 0);
		CreateDoor(bb, rnd_state, true, 0);
		CreateDoor(bb, rnd_state, false, 0);
		CreateDoor(bb, rnd_state, true, 0);
		CreateDoor(bb, rnd_state, false, 0);

		bb->bbRand(rnd_state, 1, 360);
		bb->bbRand(rnd_state, 1, 360); 
		break;

	case ROOM2SCPS:
		CreateDoor(bb, rnd_state, true, 0);
		CreateDoor(bb, rnd_state, true, 0);
		CreateDoor(bb, rnd_state, true, 0);
		CreateDoor(bb, rnd_state, true, 0);
		CreateDoor(bb, rnd_state, true, 0);
		CreateDoor(bb, rnd_state, true, 0);

		CreateItem(bb, rnd_state);
		CreateItem(bb, rnd_state);
		CreateItem(bb, rnd_state);
		CreateItem(bb, rnd_state);
		CreateItem(bb, rnd_state);

#pragma unroll
		for (uint8_t i = 0; i <= 14; i++) {
			bb->bbRand(rnd_state, 15, 16);
			bb->bbRand(rnd_state, 1, 360);

			if (i > 10) {
				bb->bbRnd(rnd_state, 0.2, 0.25);
			}
			else {
				bb->bbRnd(rnd_state, 0.1, 0.17);
			}
		}
		break;

	case ROOM205:
		CreateDoor(bb, rnd_state, true, 0);
		CreateDoor(bb, rnd_state, true, 0);
		break;

	case ENDROOM:
		CreateDoor(bb, rnd_state, false, 1);
		break;

	//This one might not actually be a room.
	//It is originally called endroomc.
	case ENDROOM2:
		//CreateDoor(bb, rnd_state, false, 2);
		break;

	case COFFIN:
		CreateDoor(bb, rnd_state, false, 1);

		CreateItem(bb, rnd_state);
		CreateItem(bb, rnd_state);
		CreateItem(bb, rnd_state);
		break;

	case ROOM2TESLA:
	case ROOM2TESLA_LCZ:
	case ROOM2TESLA_HCZ:
		break;

	case ROOM2DOORS:
		CreateDoor(bb, rnd_state, true, 0);
		CreateDoor(bb, rnd_state, true, 0);
		break;

	case ROOM914:
		CreateDoor(bb, rnd_state, false, 1);
		CreateDoor(bb, rnd_state, true, 0);
		CreateDoor(bb, rnd_state, true, 0);

		CreateItem(bb, rnd_state);
		CreateItem(bb, rnd_state);
		CreateItem(bb, rnd_state);
		break;

	case ROOM173:
		CreateDoor(bb, rnd_state, false, 1);

		bb->bbRand(rnd_state, 4, 5);
		
		bb->bbRnd(rnd_state, 0, 360);

#pragma unroll
		for (uint8_t x = 0; x <= 1; x++) {
#pragma unroll
			for (uint8_t z = 0; z <= 1; z++) {
				bb->bbRand(rnd_state, 4, 6);
				bb->bbRnd(rnd_state, -0.5, 0.5);
				bb->bbRnd(rnd_state, 0.001, 0.0018);
				bb->bbRnd(rnd_state, -0.5, 0.5);
				bb->bbRnd(rnd_state, 0, 360);
				bb->bbRnd(rnd_state, 0.5, 0.8);
				bb->bbRnd(rnd_state, 0.8, 1.0);
			}
		}

		CreateDoor(bb, rnd_state, true, 0);
		CreateDoor(bb, rnd_state, true, 0);
		CreateDoor(bb, rnd_state, true, 0);
		CreateDoor(bb, rnd_state, true, 0);
		CreateDoor(bb, rnd_state, false, 0);
		CreateDoor(bb, rnd_state, true, 0);
		CreateDoor(bb, rnd_state, false, 0);
		CreateDoor(bb, rnd_state, false, 0);

#pragma unroll
		for (uint8_t z = 0; z <= 1; z++) {
			CreateDoor(bb, rnd_state, false, 0);
			CreateDoor(bb, rnd_state, false, 0);
#pragma unroll
			for (uint8_t x = 0; x <= 2; x++) {
				CreateDoor(bb, rnd_state, false, 0);
			}
#pragma unroll
			for (uint8_t x = 0; x <= 4; x++) {
				CreateDoor(bb, rnd_state, false, 0);
			}
		}

		CreateItem(bb, rnd_state);
		break;

	case ROOM2CCONT:
		CreateDoor(bb, rnd_state, false, 0);

		CreateItem(bb, rnd_state);
		break;

	case ROOM106:
		CreateItem(bb, rnd_state);
		CreateItem(bb, rnd_state);
		CreateItem(bb, rnd_state);

		CreateDoor(bb, rnd_state, false, 0);
		CreateDoor(bb, rnd_state, false, 0);
		CreateDoor(bb, rnd_state, false, 0);
		break;

	case ROOM1ARCHIVE:
#pragma unroll
		for (uint8_t xtemp = 0; xtemp <= 1; xtemp++) {
#pragma unroll
			for (uint8_t ytemp = 0; ytemp <= 2; ytemp++) {
#pragma unroll
				for (uint8_t ztemp = 0; ztemp <= 2; ztemp++) {

					int8_t chance = bb->bbRand(rnd_state, -10, 100);

					if (chance < 0) {
						break;
					}
					else if (chance < 40) {
						bb->bbRand(rnd_state, 1, 6);
					}
					else if (chance < 45) {
						bb->bbRand(rnd_state, 1, 2);
					}
					else if (chance >= 95 && chance <= 100) {
						bb->bbRand(rnd_state, 1, 3);
					}

					bb->bbRnd(rnd_state, -96.0, 96.0);

					CreateItem(bb, rnd_state);
				}
			}
		}

		CreateDoor(bb, rnd_state, false, 0);
		break;

	//case ROOM2TEST1074 <--- not a room

	case ROOM1123:
		CreateItem(bb, rnd_state);
		CreateItem(bb, rnd_state);
		CreateItem(bb, rnd_state);
		CreateItem(bb, rnd_state);

		CreateDoor(bb, rnd_state, false, 0);
		CreateDoor(bb, rnd_state, false, 0);
		CreateDoor(bb, rnd_state, false, 0);
		break;

	case POCKETDIMENSION:
		CreateItem(bb, rnd_state);

		CreateDoor(bb, rnd_state, false, 0);
		CreateDoor(bb, rnd_state, false, 0);
		
		bb->bbRnd(rnd_state, 0.8, 0.8);

#pragma unroll
		for (uint8_t i = 1; i <= 8; i++) {
			if (i < 6) {
				bb->bbRnd(rnd_state, 0.5, 0.5);
			}
		}
		break;

	case ROOM3Z3:
		break;

	case ROOM2_3:
	case ROOM3ID:
	case ROOM3_2:
	case ROOM3_3:
		break;

	case ROOM1LIFTS:
		break;

	case ROOM2SERVERS2:
		CreateDoor(bb, rnd_state, false, 0);
		CreateDoor(bb, rnd_state, false, 0);
		CreateDoor(bb, rnd_state, false, 0);

		CreateItem(bb, rnd_state);

		bb->bbRand(rnd_state, 0, 245);
		break;

	case ROOM2GW:
	case ROOM2GW_B:
		if (r->rt->name == ROOM2GW_B) {
			bb->bbRnd(rnd_state, 0, 360);
		}

		CreateDoor(bb, rnd_state, false, 0);
		CreateDoor(bb, rnd_state, false, 0);

		if (r->rt->name == ROOM2GW) {
			//INCOMPLETE
			//This might be wrong
			bb->bbRand(rnd_state, 1, 2);
		}
		break;

	case ROOM3GW:
		CreateDoor(bb, rnd_state, false, 0);
		CreateDoor(bb, rnd_state, false, 0);
		CreateDoor(bb, rnd_state, false, 0);
		CreateDoor(bb, rnd_state, false, 0);
		break;

	case ROOM1162:
		CreateDoor(bb, rnd_state, false, 0);

		CreateItem(bb, rnd_state);
		break;

	case ROOM2SCPS2:
		CreateDoor(bb, rnd_state, false, 0);
		CreateDoor(bb, rnd_state, false, 0);
		CreateDoor(bb, rnd_state, false, 0);
	
		CreateItem(bb, rnd_state);
		CreateItem(bb, rnd_state);
		CreateItem(bb, rnd_state);
		CreateItem(bb, rnd_state);
		break;

	case ROOM3OFFICES:
		CreateDoor(bb, rnd_state, false, 0);
		break;

	case ROOM2OFFICES4:
		CreateDoor(bb, rnd_state, false, 0);

		CreateItem(bb, rnd_state);
		break;

	case ROOM2SL:
		CreateDoor(bb, rnd_state, false, 0);
		CreateDoor(bb, rnd_state, false, 0);
		CreateDoor(bb, rnd_state, false, 0);
		break;

	case ROOM2_4:
	case ROOM2_5:
		break;

	case ROOM3Z2:
		break;

	case LOCKROOM3:
		CreateDoor(bb, rnd_state, false, 0);
		CreateDoor(bb, rnd_state, false, 0);
		break;

	case MEDIBAY:
		CreateItem(bb, rnd_state);
		CreateItem(bb, rnd_state);
		CreateItem(bb, rnd_state);

		CreateDoor(bb, rnd_state, false, 0);
		break;

	case ROOM2CPIT:
		CreateDoor(bb, rnd_state, false, 2);

		CreateItem(bb, rnd_state);
		break;

	case DIMENSION1499:
		break;

	case ROOM4INFO:
	case ROOM4PIT:
	case ROOM4Z3:
		break;

	default:
		//printf("ROOM: %s NOT FOUND IN FillRoom().\n", RoomIDToName(name));
	}	

	//32 becase that is MaxRoomLights
#pragma unroll
	for (uint8_t i = 0; i < min(32, r->rt->lights); i++) {
		bb->bbRand(rnd_state, 1, 360);
		bb->bbRand(rnd_state, 1, 10);
	}
}

__device__ constexpr inline void GetRoomExtents(Rooms* r, float* e) {

	if (r->rt->disableOverlapCheck) return;

	int32_t xIndex = uint32_t((r->x / 8.0)) - 1;
	int32_t zIndex = uint32_t((r->z / 8.0)) - 1;
	int32_t angle = r->angle / 90;
	int32_t startIndex = r->rt->index;	

	//Bandaid fix for rooms generating in 8.0 or 0.0 positions.
	if (xIndex < 0) xIndex = 0;
	if (zIndex < 0) zIndex = 0;

	//544 because that is amount of extents per room.
	startIndex *= 544;

	//68 because that is amoung of extents per angle.
	startIndex += 68 * angle;

	//4 because the extents are in blocks of 4 (minX, maxX, minZ, maxZ).
	xIndex = startIndex + (4 * xIndex);
	//+2 because we get the x extents for this index if we don't.
	zIndex = (startIndex + (4 * zIndex)) + 2;

	r->minX = e[xIndex];
	r->maxX = e[xIndex + 1];

	r->minZ = e[zIndex];
	r->maxZ = e[zIndex + 1];
}

#endif