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
	int32_t obj;
	int32_t id = -1;
    int32_t index = 0;
	//dont need objPath i think
	int32_t zone[5] = { 0 };
	int32_t shape;
	RoomID name = ROOMEMPTY;
	int32_t commonness;
	bool large;
	int32_t lights;
	bool disableOverlapCheck = true;
	bool disableDecals;

	float minX, minY, minZ;
	float maxX, maxY, maxZ;
};

typedef struct Rooms Rooms;
static struct Rooms {
	int32_t id = -1;
	int32_t zone;
	int32_t found;
	
	int32_t obj;
	float x, y, z;
	int32_t angle;
	RoomTemplates* rt;

	float dist;	

	//a bunch of stuff that i might need

	float minX, minY, minZ;
	float maxX, maxY, maxZ;

};	

__device__ inline void CreateRoomTemplates(RoomTemplates* rt);
__device__ inline bool SetRoom(RoomID MapRoom[6][70], RoomID room_name, uint8_t room_type, uint8_t pos, uint8_t min_pos, uint8_t max_pos);
__device__ inline Rooms CreateRoom(uint8_t MapTemp[19][19], uint8_t& roomIdCounter, uint8_t* forest, float* e, RoomTemplates* rts, bbRandom* bb, rnd_state* rnd_sate, uint8_t zone, uint8_t roomshape, float x, float y, float z, RoomID name);
__device__ inline bool PreventRoomOverlap(Rooms* r, Rooms* rooms, float* e);
__device__ inline bool CheckRoomOverlap(Rooms* r, Rooms* r2);
__device__ inline void CalculateRoomExtents(Rooms* r);
__device__ inline void FillRoom(uint8_t MapTemp[19][19], bbRandom* bb, rnd_state* rnd_state, Rooms* r, uint8_t* forest);
__device__ inline void GetRoomExtents(Rooms* r, float* e);

__device__ inline void CreateRoomTemplates(RoomTemplates* rt) {

	//All commonness is defined as max(min(commonness, 100), 0);
	//Rooms with disableoverlapcheck = true do not have min and max extents.
	//The name variable is not a string but instead an enum for the ID of the room the roomtemplate represents.
	//This is so we do no have to do char stuff.
	
	//We start at 1 because room template 0 is just room ambience stuff.
	int32_t counter = 1;

	//LIGHT CONTAINMENT

	//lockroom
	rt[counter].id = counter;
	rt[counter].name = LOCKROOM;//"lockroom";
	rt[counter].shape = ROOM2C;
	rt[counter].zone[0] = 1;
	rt[counter].zone[1] = 3;
	rt[counter].commonness = 30;
	rt[counter].lights = 4;
	rt[counter].minX = -864.0;
	rt[counter].minY = 0.0;
	rt[counter].minZ = -1024.0;
	rt[counter].maxX = 1024.0;
	rt[counter].maxY = 640.0;
	rt[counter].maxZ = 864.0;
    rt[counter].index = 0; //lockroom
	counter++;

	//173
	rt[counter].id = counter;
	rt[counter].name = ROOM173;// "173";
	rt[counter].shape = ROOM1;
	rt[counter].commonness = 0;
	rt[counter].lights = 29;
	rt[counter].disableDecals = true;
	rt[counter].minX = -8705.0;
	rt[counter].minY = -768.0;
	rt[counter].minZ = -3808.0;
	rt[counter].maxX = 1792.0;
	rt[counter].maxY = 1016.0;
	rt[counter].maxZ = 1536.0;
	counter++;

	//start
	rt[counter].id = counter;
	rt[counter].name = START;// "start";
	rt[counter].shape = ROOM1;
	rt[counter].commonness = 0;
	rt[counter].lights = 9;
	rt[counter].disableDecals = true;
	rt[counter].minX = -672.0;
	rt[counter].minY = -28.0;
	rt[counter].minZ = -1024.0;
	rt[counter].maxX = 5504.0;
	rt[counter].maxY = 1400.0;
	rt[counter].maxZ = 2848.0;
    rt[counter].index = 1; //start
    counter++;

	//room1123
	rt[counter].id = counter;
	rt[counter].name = ROOM1123;//"room1123";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 0;
	rt[counter].lights = 20;
	rt[counter].zone[0] = 1;
	rt[counter].disableDecals = true;
	rt[counter].minX = -928.0;
	rt[counter].minY = -0.000061035156;
	rt[counter].minZ = -1024.0;
	rt[counter].maxX = 1024.0;
	rt[counter].maxY = 980.8511;
	rt[counter].maxZ = 1024.0;
    rt[counter].index = 2; //room1123
    counter++;

	//room1archive
	rt[counter].id = counter;
	rt[counter].name = ROOM1ARCHIVE;// "room1archive";
	rt[counter].shape = ROOM1;
	rt[counter].commonness = 80;
	rt[counter].lights = 2;
	rt[counter].zone[0] = 1;
	rt[counter].minX = -768.0;
	rt[counter].minY = -16.0;
	rt[counter].minZ = -1024.0;
	rt[counter].maxX = 288.0;
	rt[counter].maxY = 600.23865;
	rt[counter].maxZ = 752.00006;
    rt[counter].index = 3; //room1archive
    counter++;

	//room2storage
	rt[counter].id = counter;
	rt[counter].name = ROOM2STORAGE;// "room2storage";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 0;
	rt[counter].lights = 10;
	rt[counter].zone[0] = 1;
	rt[counter].disableDecals = true;
	rt[counter].minX = -1328.0001;
	rt[counter].minY = -1.0;
	rt[counter].minZ = -1024.0;
	rt[counter].maxX = 1328.0001;
	rt[counter].maxY = 708.0;
	rt[counter].maxZ = 1024.0;
    rt[counter].index = 4; //room2storage
    counter++;

	//room3storage
	rt[counter].id = counter;
	rt[counter].name = ROOM3STORAGE;// "room3storage";
	rt[counter].shape = ROOM3;
	rt[counter].commonness = 0;
	rt[counter].lights = 33;
	rt[counter].zone[0] = 1;
	rt[counter].disableOverlapCheck = true;
	counter++;

	//room2tesla_lcz
	rt[counter].id = counter;
	rt[counter].name = ROOM2TESLA_LCZ;//"room2tesla_lcz";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 100;
	rt[counter].lights = 8;
	rt[counter].zone[0] = 1;
	rt[counter].minX = -304.00012;
	rt[counter].minY = -64.0;
	rt[counter].minZ = -1024.0;
	rt[counter].maxX = 304.0001;
	rt[counter].maxY = 714.83057;
    rt[counter].maxZ = 1024.0;
    rt[counter].index = 5; //room2tesla_lcz
	counter++;

	//endroom
	rt[counter].id = counter;
	rt[counter].name = ENDROOM;// "endroom";
	rt[counter].shape = ROOM1;
	rt[counter].commonness = 100;
	rt[counter].lights = 3;
	rt[counter].zone[0] = 1;
	rt[counter].zone[2] = 3;
	rt[counter].minX = -720.0;
	rt[counter].minY = -1.0;
	rt[counter].minZ = -1024.0;
	rt[counter].maxX = 784.0;
	rt[counter].maxY = 708.0;
	rt[counter].maxZ = 1240.0;
    rt[counter].index = 6; //endroom
    counter++;

	//room012
	rt[counter].id = counter;
	rt[counter].name = ROOM012;//"room012";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 0;
	rt[counter].lights = 4;
	rt[counter].zone[0] = 1;
	rt[counter].disableDecals = true;
	rt[counter].minX = -1024.0;
	rt[counter].minY = -800.0;
	rt[counter].minZ = -1024.0;
	rt[counter].maxX = 816.00006;
	rt[counter].maxY = 708.0;
	rt[counter].maxZ = 1024.0;
    rt[counter].index = 7; //room012
    counter++;

	//room205
	rt[counter].id = counter;
	rt[counter].name = ROOM205;//"room205";
	rt[counter].shape = ROOM1;
	rt[counter].commonness = 0;
	rt[counter].lights = 5;
	rt[counter].zone[0] = 1;
	rt[counter].large = true;
	rt[counter].minX = -1792.0;
	rt[counter].minY = -160.0;
	rt[counter].minZ = -1024.0;
	rt[counter].maxX = 800.0;
	rt[counter].maxY = 1184.0;
	rt[counter].maxZ = 864.0;
    rt[counter].index = 8; //room205
    counter++;

	//room2
	rt[counter].id = counter;
	rt[counter].name = ROOM2ID;//"room2";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 45;
	rt[counter].lights = 2;
	rt[counter].zone[0] = 1;
	rt[counter].minX = -304.0;
	rt[counter].minY = -1.0;
	rt[counter].minZ = -1024.0;
	rt[counter].maxX = 304.0001;
	rt[counter].maxY = 726.83057;
	rt[counter].maxZ = 1024.0;
    rt[counter].index = 9; //room2
    counter++;

	//room2_2
	rt[counter].id = counter;
	rt[counter].name = ROOM2_2;// "room2_2";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 40;
	rt[counter].lights = 4;
	rt[counter].zone[0] = 1;
	rt[counter].minX = -800.0;
	rt[counter].minY = -1.0;
	rt[counter].minZ = -1024.0;
	rt[counter].maxX = 304.0001;
	rt[counter].maxY = 726.83057;
	rt[counter].maxZ = 1024.0;
    rt[counter].index = 10; //room2_2
    counter++;

	//room2_3
	rt[counter].id = counter;
	rt[counter].name = ROOM2_3;// "room2_3";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 35;
	rt[counter].lights = 0;
	rt[counter].zone[0] = 1;
	rt[counter].minX = -416.00003;
	rt[counter].minY = -20.0;
	rt[counter].minZ = -1024.0;
	rt[counter].maxX = 416.0;
	rt[counter].maxY = 596.23865;
	rt[counter].maxZ = 1024.0;
    rt[counter].index = 11; //room2_3
    counter++;

	//room2_4
	rt[counter].id = counter;
	rt[counter].name = ROOM2_4;// "room2_4";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 35;
	rt[counter].lights = 3;
	rt[counter].zone[0] = 1;
	rt[counter].minX = -304.0;
	rt[counter].minY = -1.0;
	rt[counter].minZ = -1024.0;
	rt[counter].maxX = 772.0;
	rt[counter].maxY = 706.86816;
	rt[counter].maxZ = 1024.0;
    rt[counter].index = 12; //room2_4
    counter++;

	//room2_5
	rt[counter].id = counter;
	rt[counter].name = ROOM2_5;// "room2_5";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 35;
	rt[counter].lights = 3;
	rt[counter].zone[0] = 1;
	rt[counter].minX = -304.0;
	rt[counter].minY = -1.0;
	rt[counter].minZ = -1024.0;
	rt[counter].maxX = 320.0;
	rt[counter].maxY = 386.86813;
	rt[counter].maxZ = 1024.0;
    rt[counter].index = 13; //room2_5
    counter++;

	//room2c
	rt[counter].id = counter;
	rt[counter].name = ROOM2CID;// "room2c";
	rt[counter].shape = ROOM2C;
	rt[counter].commonness = 30;
	rt[counter].lights = 3;
	rt[counter].zone[0] = 1;
	rt[counter].minX = -320.0;
	rt[counter].minY = 0.0;
	rt[counter].minZ = -1024.0;
	rt[counter].maxX = 1024.0;
	rt[counter].maxY = 726.83057;
	rt[counter].maxZ = 320.0;
    rt[counter].index = 14; //room2c
    counter++;

	//room2c2
	rt[counter].id = counter;
	rt[counter].name = ROOM2C2;// "room2c2";
	rt[counter].shape = ROOM2C;
	rt[counter].commonness = 30;
	rt[counter].lights = 3;
	rt[counter].zone[0] = 1;
	rt[counter].minX = -320.0;
	rt[counter].minY = 0.0;
	rt[counter].minZ = -1024.0;
	rt[counter].maxX = 1024.0;
	rt[counter].maxY = 800.8681;
	rt[counter].maxZ = 1023.99994;
    rt[counter].index = 15; //room2c2
    counter++;

	//room2closets
	rt[counter].id = counter;
	rt[counter].name = ROOM2CLOSETS;// "room2closets";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 0;
	rt[counter].lights = 6;
	rt[counter].zone[0] = 1;
	rt[counter].disableDecals = true;
	rt[counter].large = true;
	rt[counter].minX = -1972.0;
	rt[counter].minY = -416.0;
	rt[counter].minZ = -1024.0;
	rt[counter].maxX = 816.0;
	rt[counter].maxY = 708.0;
	rt[counter].maxZ = 1024.0;
    rt[counter].index = 16; //room2closets
    counter++;

	//room2elevator
	rt[counter].id = counter;
	rt[counter].name = ROOM2ELEVATOR;// "room2elevator";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 20;
	rt[counter].lights = 3;
	rt[counter].zone[0] = 1;
	rt[counter].minX = -304.00006;
	rt[counter].minY = -32.0;
	rt[counter].minZ = -1024.0;
	rt[counter].maxX = 1056.0002;
	rt[counter].maxY = 854.83057;
	rt[counter].maxZ = 1024.0002;
    rt[counter].index = 17; //room2elevator
    counter++;

	//room2doors
	rt[counter].id = counter;
	rt[counter].name = ROOM2DOORS;// "room2doors";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 30;
	rt[counter].lights = 5;
	rt[counter].zone[0] = 1;
	rt[counter].minX = -1024.0;
	rt[counter].minY = 0.0;
	rt[counter].minZ = -1024.0;
	rt[counter].maxX = 256.0;
	rt[counter].maxY = 640.0;
	rt[counter].maxZ = 1024.0;
    rt[counter].index = 18; //room2doors
    counter++;

	//room2scps
	rt[counter].id = counter;
	rt[counter].name = ROOM2SCPS;// "room2scps";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 0;
	rt[counter].lights = 6;
	rt[counter].zone[0] = 1;
	rt[counter].minX = -816.0001;
	rt[counter].minY = -1.0;
	rt[counter].minZ = -1024.0;
	rt[counter].maxX = 816.0002;
	rt[counter].maxY = 714.0;
	rt[counter].maxZ = 1024.0;
    rt[counter].index = 19; //room2scps
    counter++;

	//room860
	rt[counter].id = counter;
	rt[counter].name = ROOM860;// "room860";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 0;
	rt[counter].lights = 4;
	rt[counter].minX = -304.0;
	rt[counter].minY = -32.0;
	rt[counter].minZ = -1024.0;
	rt[counter].maxX = 1280.0;
	rt[counter].maxY = 726.83057;
	rt[counter].maxZ = 1024.0;
    rt[counter].index = 20; //room860
    counter++;

	//room2testroom2
	rt[counter].id = counter;
	rt[counter].name = ROOM2TESTROOM2;// "room2testroom2";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 0;
	rt[counter].lights = 6;
	rt[counter].zone[0] = 1;
	rt[counter].minX = -1024.0002;
	rt[counter].minY = -48.0;
	rt[counter].minZ = -1024.0;
	rt[counter].maxX = 352.0;
	rt[counter].maxY = 640.0;
	rt[counter].maxZ = 1024.0;
    rt[counter].index = 21; //room2testroom2
    counter++;

	//room3
	rt[counter].id = counter;
	rt[counter].name = ROOM3ID;// "room3";
	rt[counter].shape = ROOM3;
	rt[counter].commonness = 100;
	rt[counter].lights = 4;
	rt[counter].zone[0] = 1;
	rt[counter].minX = -1024.0;
	rt[counter].minY = 0.0;
	rt[counter].minZ = -1024.0;
	rt[counter].maxX = 1024.0;
	rt[counter].maxY = 854.83057;
	rt[counter].maxZ = 1023.99994;
    rt[counter].index = 22; //room3
    counter++;

	//room3_2
	rt[counter].id = counter;
	rt[counter].name = ROOM3_2;// "room3_2";
	rt[counter].shape = ROOM3;
	rt[counter].commonness = 100;
	rt[counter].lights = 3;
	rt[counter].zone[0] = 1;
	rt[counter].minX = -1024.0;
	rt[counter].minY = 0.0;
	rt[counter].minZ = -1024.0;
	rt[counter].maxX = 1024.0;
	rt[counter].maxY = 854.83057;
	rt[counter].maxZ = 448.0001;
    rt[counter].index = 23; //room3_2
    counter++;

	//room4
	rt[counter].id = counter;
	rt[counter].name = ROOM4ID;// "room4";
	rt[counter].shape = ROOM4;
	rt[counter].commonness = 100;
	rt[counter].lights = 4;
	rt[counter].zone[0] = 1;
	rt[counter].minX = -1024.0;
	rt[counter].minY = 0.0;
	rt[counter].minZ = -1024.0;
	rt[counter].maxX = 1024.0;
	rt[counter].maxY = 850.45544;
	rt[counter].maxZ = 1024.0;
    rt[counter].index = 24; //room4
    counter++;

	//room4_2
	rt[counter].id = counter;
	rt[counter].name = ROOM4_2;// "room4_2";
	rt[counter].shape = ROOM4;
	rt[counter].commonness = 80;
	rt[counter].lights = 4;
	rt[counter].zone[0] = 1;
	rt[counter].minX = -1024.0;
	rt[counter].minY = -0.0000038146973;
	rt[counter].minZ = -1024.0;
	rt[counter].maxX = 1024.0;
	rt[counter].maxY = 784.8511;
	rt[counter].maxZ = 1024.0;
    rt[counter].index = 25; //room4_2
    counter++;

	//roompj
	rt[counter].id = counter;
	rt[counter].name = ROOMPJ;// "roompj";
	rt[counter].shape = ROOM1;
	rt[counter].commonness = 0;
	rt[counter].lights = 8;
	rt[counter].disableDecals = true;
	rt[counter].zone[0] = 1;
	rt[counter].minX = -1024.0;
	rt[counter].minY = -0.56270504;
	rt[counter].minZ = -1024.0;
	rt[counter].maxX = 1024.0;
	rt[counter].maxY = 960.0;
	rt[counter].maxZ = 1280.0;
    rt[counter].index = 26; //roompj
    counter++;

	//914
	rt[counter].id = counter;
	rt[counter].name = ROOM914;// "914";
	rt[counter].shape = ROOM1;
	rt[counter].commonness = 0;
	rt[counter].lights = 9;
	rt[counter].disableDecals = true;
	rt[counter].zone[0] = 1;
	rt[counter].minX = -1024.0001;
	rt[counter].minY = -0.56270504;
	rt[counter].minZ = -1024.0;
	rt[counter].maxX = 1024.0001;
	rt[counter].maxY = 816.0;
	rt[counter].maxZ = 1024.0001;
    rt[counter].index = 27; //914
    counter++;

	//room2gw
	rt[counter].id = counter;
	rt[counter].name = ROOM2GW;// "room2gw";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 10;
	rt[counter].lights = 2;
	rt[counter].zone[0] = 1;
	rt[counter].minX = -544.0;
	rt[counter].minY = -18.0;
	rt[counter].minZ = -1035.9998;
	rt[counter].maxX = 544.0;
	rt[counter].maxY = 575.0;
	rt[counter].maxZ = 1024.0;
    rt[counter].index = 28; //room2gw
    counter++;

	//room2gw_b
	rt[counter].id = counter;
	rt[counter].name = ROOM2GW_B;// "room2gw_b";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 0;
	rt[counter].lights = 2;
	rt[counter].zone[0] = 1;
	rt[counter].minX = -482.00003;
	rt[counter].minY = -18.0;
	rt[counter].minZ = -1034.0;
	rt[counter].maxX = 485.0;
	rt[counter].maxY = 575.0;
	rt[counter].maxZ = 1024.0;
    rt[counter].index = 29; //room2gw_b
    counter++;

	//room1162
	rt[counter].id = counter;
	rt[counter].name = ROOM1162;// "room1162";
	rt[counter].shape = ROOM2C;
	rt[counter].commonness = 0;
	rt[counter].lights = 5;
	rt[counter].zone[0] = 1;
	rt[counter].minX = -320.0;
	rt[counter].minY = -32.0;
	rt[counter].minZ = -1027.8019;
	rt[counter].maxX = 1024.0;
	rt[counter].maxY = 752.8511;
	rt[counter].maxZ = 320.00006;
    rt[counter].index = 30; //room1162
    counter++;

	//room2scps2
	rt[counter].id = counter;
	rt[counter].name = ROOM2SCPS2;// "room2scps2";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 0;
	rt[counter].lights = 6;
	rt[counter].zone[0] = 1;
	rt[counter].minX = -304.0;
	rt[counter].minY = -9.599998;
	rt[counter].minZ = -1024.0;
	rt[counter].maxX = 1264.0;
	rt[counter].maxY = 706.8681;
	rt[counter].maxZ = 1026.3;
    rt[counter].index = 31; //room2scps2
    counter++;

	//room2sl
	rt[counter].id = counter;
	rt[counter].name = ROOM2SL;// "room2sl";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 0;
	rt[counter].lights = 7;
	rt[counter].zone[0] = 1;
	rt[counter].large = true;
	rt[counter].minX = -304.0;
	rt[counter].minY = -64.0;
	rt[counter].minZ = -1024.0001;
	rt[counter].maxX = 1792.0;
	rt[counter].maxY = 960.0;
	rt[counter].maxZ = 1024.0001;
    rt[counter].index = 32; //room2sl
    counter++;

	//lockroom3
	rt[counter].id = counter;
	rt[counter].name = LOCKROOM3;// "lockroom3";
	rt[counter].shape = ROOM2C;
	rt[counter].commonness = 15;
	rt[counter].lights = 3;
	rt[counter].zone[0] = 1;
	rt[counter].minX = -832.0;
	rt[counter].minY = 0.0;
	rt[counter].minZ = -1030.0;
	rt[counter].maxX = 1024.0;
	rt[counter].maxY = 384.0;
	rt[counter].maxZ = 864.0;
    rt[counter].index = 33; //lockroom3
    counter++;

	//room4info
	rt[counter].id = counter;
	rt[counter].name = ROOM4INFO;// "room4info";
	rt[counter].shape = ROOM4;
	rt[counter].commonness = 0;
	rt[counter].lights = 5;
	rt[counter].zone[0] = 1;
	rt[counter].minX = -1024.0;
	rt[counter].minY = -20.0;
	rt[counter].minZ = -1024.0;
	rt[counter].maxX = 1024.0;
	rt[counter].maxY = 596.23865;
	rt[counter].maxZ = 1024.0001;
    rt[counter].index = 34; //room4info
    counter++;

	//room3_3
	rt[counter].id = counter;
	rt[counter].name = ROOM3_3;// "room3_3";
	rt[counter].shape = ROOM3;
	rt[counter].commonness = 20;
	rt[counter].lights = 5;
	rt[counter].zone[0] = 1;
	rt[counter].minX = -1024.0;
	rt[counter].minY = -24.0;
	rt[counter].minZ = -1024.0682;
	rt[counter].maxX = 1024.8181;
	rt[counter].maxY = 596.23865;
	rt[counter].maxZ = 416.00003;
    rt[counter].index = 35; //room3_3
    counter++;

	//checkpoint1
	rt[counter].id = counter;
	rt[counter].name = CHECKPOINT1;// "checkpoint1";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 0;
	rt[counter].lights = 2;
	rt[counter].minX = -1104.0;
	rt[counter].minY = 0.0;
	rt[counter].minZ = -1024.0001;
	rt[counter].maxX = 1102.0;
	rt[counter].maxY = 708.0;
	rt[counter].maxZ = 1024.0001;
    rt[counter].index = 36; //checkpoint1
    counter++;

	//HEAVY CONTAINMENT

	//008
	rt[counter].id = counter;
	rt[counter].name = ROOM008;// "008";
	rt[counter].shape = ROOM1;
	rt[counter].commonness = 0;
	rt[counter].lights = 5;
	rt[counter].disableDecals = true;
	rt[counter].zone[0] = 2;
	rt[counter].minX = -608.0;
	rt[counter].minY = -32.0;
	rt[counter].minZ = -1024.0;
	rt[counter].maxX = 784.0;
	rt[counter].maxY = 1152.0;
	rt[counter].maxZ = 832.0;
    rt[counter].index = 37; //008
    counter++;

	//room035
	rt[counter].id = counter;
	rt[counter].name = ROOM035;// "room035";
	rt[counter].shape = ROOM1;
	rt[counter].commonness = 0;
	rt[counter].lights = 5;
	rt[counter].zone[0] = 2;
	rt[counter].minX = -736.0;
	rt[counter].minY = -32.0;
	rt[counter].minZ = -1024.0;
	rt[counter].maxX = 1248.0;
	rt[counter].maxY = 498.45538;
	rt[counter].maxZ = 912.0;
    rt[counter].index = 38; //room035
    counter++;

	//room049
	rt[counter].id = counter;
	rt[counter].name = ROOM049;// "room049";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 0;
	rt[counter].lights = 26;
	rt[counter].disableDecals = true;
	rt[counter].zone[0] = 2;
	rt[counter].disableOverlapCheck = true;	
	counter++;

	//room106;
	rt[counter].id = counter;
	rt[counter].name = ROOM106;// "room106";
	rt[counter].shape = ROOM1;
	rt[counter].commonness = 0;
	rt[counter].lights = 14;
	rt[counter].disableDecals = true;
	rt[counter].zone[0] = 2;
	rt[counter].large = true;
	rt[counter].minX = -1304.0;
	rt[counter].minY = -1296.0;
	rt[counter].minZ = -1024.0;
	rt[counter].maxX = 2256.0;
	rt[counter].maxY = 1687.9999;
	rt[counter].maxZ = 3120.0;
    rt[counter].index = 39; //room106
    counter++;

	//rom513
	rt[counter].id = counter;
	rt[counter].name = ROOM513;// "room513";
	rt[counter].shape = ROOM3;
	rt[counter].commonness = 0;
	rt[counter].lights = 6;
	rt[counter].zone[0] = 2;
	rt[counter].minX = -1024.0;
	rt[counter].minY = -1.0000322;
	rt[counter].minZ = -1024.0;
	rt[counter].maxX = 1024.0;
	rt[counter].maxY = 470.83054;
	rt[counter].maxZ = 1024.0;
    rt[counter].index = 40; //room513
    counter++;

	//coffin
	rt[counter].id = counter;
	rt[counter].name = COFFIN;// "coffin";
	rt[counter].shape = ROOM1;
	rt[counter].commonness = 0;
	rt[counter].lights = 9;
	rt[counter].disableDecals = true;
	rt[counter].zone[0] = 1;
	rt[counter].minX = -960.0;
	rt[counter].minY = -1537.0;
	rt[counter].minZ = -1024.0;
	rt[counter].maxX = 464.0;
	rt[counter].maxY = 704.0;
	rt[counter].maxZ = 2560.0;
    rt[counter].index = 41; //coffin
    counter++;

	//room966
	rt[counter].id = counter;
	rt[counter].name = ROOM966;// "room966";
	rt[counter].shape = ROOM3;
	rt[counter].commonness = 0;
	rt[counter].lights = 5;
	rt[counter].disableOverlapCheck = true;
	counter++;

	//endroom2
	rt[counter].id = counter;
	rt[counter].name = ENDROOM2;// "endroom2";
	rt[counter].shape = ROOM1;
	rt[counter].commonness = 100;
	rt[counter].lights = 4;
	rt[counter].zone[0] = 2;
	rt[counter].minX = -480.00003;
	rt[counter].minY = -0.56270504;
	rt[counter].minZ = -1024.0;
	rt[counter].maxX = 464.0;
	rt[counter].maxY = 1056.0;
	rt[counter].maxZ = 376.0;
    rt[counter].index = 42; //endroom2
    counter++;

	//testroom
	rt[counter].id = counter;
	rt[counter].name = TESTROOM;// "testroom";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 0;
	rt[counter].lights = 23;
	rt[counter].disableDecals = true;
	rt[counter].zone[0] = 2;
	rt[counter].minX = -1024.0;
	rt[counter].minY = -1281.0;
	rt[counter].minZ = -1024.0;
	rt[counter].maxX = 1024.0;
	rt[counter].maxY = 800.0;
	rt[counter].maxZ = 1024.0;
    rt[counter].index = 43; //testroom
    counter++;

	//tunnel
	rt[counter].id = counter;
	rt[counter].name = TUNNEL;// "tunnel";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 100;
	rt[counter].lights = 4;
	rt[counter].zone[0] = 2;
	rt[counter].minX = -288.0;
	rt[counter].minY = -0.9999924;
	rt[counter].minZ = -1024.0;
	rt[counter].maxX = 288.0;
	rt[counter].maxY = 449.0;
	rt[counter].maxZ = 1024.0;
    rt[counter].index = 44; //tunnel
    counter++;

	//tunnel2
	rt[counter].id = counter;
	rt[counter].name = TUNNEL2;// "tunnel2";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 70;
	rt[counter].lights = 5;
	rt[counter].zone[0] = 2;
	rt[counter].minX = -336.0;
	rt[counter].minY = -0.9492798;
	rt[counter].minZ = -1024.0;
	rt[counter].maxX = 528.0;
	rt[counter].maxY = 696.0;
	rt[counter].maxZ = 1024.0;
    rt[counter].index = 45; //tunnel2
    counter++;

	//room2ctunnel
	rt[counter].id = counter;
	rt[counter].name = ROOM2CTUNNEL;// "room2ctunnel";
	rt[counter].shape = ROOM2C;
	rt[counter].commonness = 40;
	rt[counter].lights = 3;
	rt[counter].zone[0] = 2;
	rt[counter].minX = -384.0;
	rt[counter].minY = -0.0000076293945;
	rt[counter].minZ = -1024.0;
	rt[counter].maxX = 1024.0;
	rt[counter].maxY = 448.00003;
	rt[counter].maxZ = 402.46378;
    rt[counter].index = 46; //room2ctunnel
    counter++;

	//room2nuke
	rt[counter].id = counter;
	rt[counter].name = ROOM2NUKE;// "room2nuke";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 0;
	rt[counter].lights = 14;
	rt[counter].zone[0] = 2;
	rt[counter].large = true;
	rt[counter].minX = -1024.0;
	rt[counter].minY = -32.0;
	rt[counter].minZ = -1024.0;
	rt[counter].maxX = 1808.0;
	rt[counter].maxY = 2016.0;
	rt[counter].maxZ = 1024.0001;
    rt[counter].index = 47; //room2nuke
    counter++;

	//room2pipes
	rt[counter].id = counter;
	rt[counter].name = ROOM2PIPES;// "room2pipes";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 50;
	rt[counter].lights = 6;
	rt[counter].zone[0] = 2;
	rt[counter].minX = -312.09503;
	rt[counter].minY = -449.0;
	rt[counter].minZ = -1026.0;
	rt[counter].maxX = 256.0;
	rt[counter].maxY = 1024.0;
	rt[counter].maxZ = 1024.0;
    rt[counter].index = 48; //room2pipes
    counter++;

	//room2pit
	rt[counter].id = counter;
	rt[counter].name = ROOM2PIT;// "room2pit";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 75;
	rt[counter].lights = 6;
	rt[counter].disableDecals = true;
	rt[counter].zone[0] = 2;
	rt[counter].minX = -1024.0001;
	rt[counter].minY = -448.0;
	rt[counter].minZ = -1024.0001;
	rt[counter].maxX = 768.00006;
	rt[counter].maxY = 448.0;
	rt[counter].maxZ = 1024.0001;
    rt[counter].index = 49; //room2pit
    counter++;

	//room3pit
	rt[counter].id = counter;
	rt[counter].name = ROOM3PIT;// "room3pit";
	rt[counter].shape = ROOM3;
	rt[counter].commonness = 100;
	rt[counter].lights = 12;
	rt[counter].disableDecals = true;
	rt[counter].zone[0] = 2;
	rt[counter].minX = -1024.0;
	rt[counter].minY = -960.99994;
	rt[counter].minZ = -1024.0;
	rt[counter].maxX = 1024.0;
	rt[counter].maxY = 385.0;
	rt[counter].maxZ = 352.0;
    rt[counter].index = 50; //room3pit
    counter++;

	//room4pit
	rt[counter].id = counter;
	rt[counter].name = ROOM4PIT;// "room4pit";
	rt[counter].shape = ROOM4;
	rt[counter].commonness = 100;
	rt[counter].lights = 16;
	rt[counter].zone[0] = 2;
	rt[counter].minX = -1024.0;
	rt[counter].minY = -960.99994;
	rt[counter].minZ = -1024.0;
	rt[counter].maxX = 1024.0;
	rt[counter].maxY = 385.0;
	rt[counter].maxZ = 1024.0001;
    rt[counter].index = 51; //room4pit
    counter++;

	//room2servers
	rt[counter].id = counter;
	rt[counter].name = ROOM2SERVERS;// "room2servers";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 0;
	rt[counter].lights = 5;
	rt[counter].zone[0] = 2;
	rt[counter].large = true;
	rt[counter].minX = -1664.0;
	rt[counter].minY = -0.021850586;
	rt[counter].minZ = -1024.0;
	rt[counter].maxX = 224.00012;
	rt[counter].maxY = 385.00006;
	rt[counter].maxZ = 1024.0;
    rt[counter].index = 52; //room2servers
    counter++;

	//room2shaft
	rt[counter].id = counter;
	rt[counter].name = ROOM2SHAFT;// "room2shaft";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 0;
	rt[counter].lights = 5;
	rt[counter].disableDecals = true;
	rt[counter].minX = -256.0;
	rt[counter].minY = -1504.0;
	rt[counter].minZ = -1024.0;
	rt[counter].maxX = 2016.0;
	rt[counter].maxY = 1775.0;
	rt[counter].maxZ = 1024.0;
    rt[counter].index = 53; //room2shaft
    counter++;

	//room2tunnel
	rt[counter].id = counter;
	rt[counter].name = ROOM2TUNNEL;// "room2tunnel";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 0;
	rt[counter].lights = 4;
	rt[counter].disableDecals = true;
	rt[counter].zone[0] = 2;
	rt[counter].minX = -880.0;
	rt[counter].minY = -32.0;
	rt[counter].minZ = -1024.0;
	rt[counter].maxX = 880.0;
	rt[counter].maxY = 512.0;
	rt[counter].maxZ = 1024.0;
    rt[counter].index = 54; //room2tunnel
    counter++;

	//room3tunnel
	rt[counter].id = counter;
	rt[counter].name = ROOM3TUNNEL;// "room3tunnel";
	rt[counter].shape = ROOM3;
	rt[counter].commonness = 100;
	rt[counter].lights = 5;
	rt[counter].zone[0] = 2;
	rt[counter].minX = -1024.0;
	rt[counter].minY = -0.0000076293945;
	rt[counter].minZ = -1024.0;
	rt[counter].maxX = 1024.0;
	rt[counter].maxY = 758.83057;
	rt[counter].maxZ = 416.00003;
    rt[counter].index = 55; //room3tunnel
    counter++;

	//room4tunnels
	rt[counter].id = counter;
	rt[counter].name = ROOM4TUNNELS;// "room4tunnels";
	rt[counter].shape = ROOM4;
	rt[counter].commonness = 100;
	rt[counter].lights = 6;
	rt[counter].zone[0] = 2;
	rt[counter].minX = -1024.0;
	rt[counter].minY = -64.0;
	rt[counter].minZ = -1024.0;
	rt[counter].maxX = 1024.0;
	rt[counter].maxY = 738.8681;
	rt[counter].maxZ = 1024.0;
    rt[counter].index = 56; //room4tunnels
    counter++;

	//room2tesla_hcz
	rt[counter].id = counter;
	rt[counter].name = ROOM2TESLA_HCZ;// "room2tesla_hcz";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 100;
	rt[counter].lights = 10;
	rt[counter].zone[0] = 2;
	rt[counter].minX = -304.0;
	rt[counter].minY = -64.0;
	rt[counter].minZ = -1024.0;
	rt[counter].maxX = 304.00003;
	rt[counter].maxY = 708.0;
	rt[counter].maxZ = 1024.0;
    rt[counter].index = 57; //room2tesla_hcz
    counter++;

	//room3z2
	rt[counter].id = counter;
	rt[counter].name = ROOM3Z2;// "room3z2";
	rt[counter].shape = ROOM3;
	rt[counter].commonness = 100;
	rt[counter].lights = 3;
	rt[counter].zone[0] = 2;
	rt[counter].minX = -1024.0;
	rt[counter].minY = -32.0;
	rt[counter].minZ = -1024.0002;
	rt[counter].maxX = 1024.0002;
	rt[counter].maxY = 416.00003;
	rt[counter].maxZ = 255.99985;
    rt[counter].index = 58; //room3z2
    counter++;

	//room2cpit
	rt[counter].id = counter;
	rt[counter].name = ROOM2CPIT;// "room2cpit";
	rt[counter].shape = ROOM2C;
	rt[counter].commonness = 0;
	rt[counter].lights = 10;
	rt[counter].disableDecals = true;
	rt[counter].zone[0] = 2;
	rt[counter].minX = -568.0001;
	rt[counter].minY = -960.99994;
	rt[counter].minZ = -1024.0;
	rt[counter].maxX = 1024.0;
	rt[counter].maxY = 874.8627;
	rt[counter].maxZ = 960.0;
    rt[counter].index = 59; //room2cpit
    counter++;

	//room2pipes2
	rt[counter].id = counter;
	rt[counter].name = ROOM2PIPES2;// "room2pipes2";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 70;
	rt[counter].lights = 6;
	rt[counter].disableDecals = true;
	rt[counter].zone[0] = 2;
	rt[counter].minX = -256.0;
	rt[counter].minY = -448.0;
	rt[counter].minZ = -1026.0;
	rt[counter].maxX = 671.5;
	rt[counter].maxY = 788.0;
	rt[counter].maxZ = 1024.0;
    rt[counter].index = 60; //room2pipes2
    counter++;

	//checkpoint2
	rt[counter].id = counter;
	rt[counter].name = CHECKPOINT2;// "checkpoint2";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 0;
	rt[counter].lights = 2;
	rt[counter].minX = -1102.0;
	rt[counter].minY = 0.0;
	rt[counter].minZ = -1024.0002;
	rt[counter].maxX = 1104.0;
	rt[counter].maxY = 708.0;
	rt[counter].maxZ = 1026.0;
    rt[counter].index = 61; //checkpoint2
    counter++;

	//ENTRANCE ZONE

	//room079 yea he's in the entrance zone mhm
	rt[counter].id = counter;
	rt[counter].name = ROOM079;// "room079";
	rt[counter].shape = ROOM1;
	rt[counter].commonness = 0;
	rt[counter].lights = 6;
	rt[counter].disableDecals = true;
	rt[counter].zone[0] = 3;
	rt[counter].large = true;
	rt[counter].minX = -416.00003;
	rt[counter].minY = -705.0;
	rt[counter].minZ = -1055.9999;
	rt[counter].maxX = 2240.0;
	rt[counter].maxY = 708.0;
	rt[counter].maxZ = 2048.0;
    rt[counter].index = 62; //room079
    counter++;

	//lockroom2
	rt[counter].id = counter;
	rt[counter].name = LOCKROOM2;// "lockroom2";
	rt[counter].shape = ROOM2C;
	rt[counter].commonness = 0;
	rt[counter].lights = 6;
	rt[counter].zone[0] = 3;
	rt[counter].minX = -864.0;
	rt[counter].minY = -1.0;
	rt[counter].minZ = -1024.0;
	rt[counter].maxX = 1024.0;
	rt[counter].maxY = 434.83054;
	rt[counter].maxZ = 864.0;
    rt[counter].index = 63; //lockroom2
    counter++;

	//exit1 
	rt[counter].id = counter;
	rt[counter].name = EXIT1;// "exit1";
	rt[counter].shape = ROOM1;
	rt[counter].commonness = 0;
	rt[counter].lights = 15;
	rt[counter].zone[0] = 3;
	rt[counter].disableOverlapCheck = true;	
	counter++;

	//gateaentrance
	rt[counter].id = counter;
	rt[counter].name = GATEAENTRANCE;// "gateaentrance";
	rt[counter].shape = ROOM1;
	rt[counter].commonness = 0;
	rt[counter].lights = 4;
	rt[counter].zone[0] = 3;
	rt[counter].minX = -720.0;
	rt[counter].minY = -32.0;
	rt[counter].minZ = -1136.0;
	rt[counter].maxX = 1360.0;
	rt[counter].maxY = 1328.0;
	rt[counter].maxZ = 1240.0;
    rt[counter].index = 64; //gateaentrance
    counter++;

	//gatea
	rt[counter].id = counter;
	rt[counter].name = GATEA;// "gatea";
	rt[counter].shape = ROOM1;
	rt[counter].commonness = 0;
	rt[counter].lights = 12;
	rt[counter].disableOverlapCheck = true;
	counter++;

	//medibay
	rt[counter].id = counter;
	rt[counter].name = MEDIBAY;// "medibay";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 0;
	rt[counter].lights = 6;
	rt[counter].zone[0] = 3;
	rt[counter].minX = -1024.0002;
	rt[counter].minY = -1.8437233;
	rt[counter].minZ = -1025.0;
	rt[counter].maxX = 288.0;
	rt[counter].maxY = 386.86813;
	rt[counter].maxZ = 1024.0;
    rt[counter].index = 65; //medibay
    counter++;

	//room2z3
	rt[counter].id = counter;
	rt[counter].name = ROOM2Z3;// "room2z3";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 75;
	rt[counter].lights = 0;
	rt[counter].zone[0] = 3;
	rt[counter].minX = -256.0;
	rt[counter].minY = -1.0;
	rt[counter].minZ = -1024.0;
	rt[counter].maxX = 256.0;
	rt[counter].maxY = 438.83054;
	rt[counter].maxZ = 1024.0001;
    rt[counter].index = 66; //room2z3
    counter++;

	//room2cafeteria
	rt[counter].id = counter;
	rt[counter].name = ROOM2CAFETERIA;// "room2cafeteria";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 0;
	rt[counter].lights = 7;
	rt[counter].zone[0] = 3;
	rt[counter].large = true;
	//TEMPORARY
	//rt[counter].disableOverlapCheck = true;
	rt[counter].minX = -1792.0;
	rt[counter].minY = -416.0;
	rt[counter].minZ = -1056.0;
	rt[counter].maxX = 1952.0;
	rt[counter].maxY = 708.0;
	rt[counter].maxZ = 1024.0001;
	rt[counter].index = 67; //room2cafeteria
	counter++;

	//room2cz3
	rt[counter].id = counter;
	rt[counter].name = ROOM2CZ3;// "room2cz3";
	rt[counter].shape = ROOM2C;
	rt[counter].commonness = 100;
	rt[counter].lights = 4;
	rt[counter].zone[0] = 3;
	rt[counter].minX = -576.0;
	rt[counter].minY = -1.0;
	rt[counter].minZ = -1024.0;
	rt[counter].maxX = 1024.0;
	rt[counter].maxY = 438.83054;
	rt[counter].maxZ = 576.0;
    rt[counter].index = 68; //room2cz3
    counter++;

	//room2ccont
	rt[counter].id = counter;
	rt[counter].name = ROOM2CCONT;// "room2ccont";
	rt[counter].shape = ROOM2C;
	rt[counter].commonness = 0;
	rt[counter].lights = 9;
	rt[counter].zone[0] = 3;
	rt[counter].large = true;
	rt[counter].minX = -2272.0;
	rt[counter].minY = -64.0;
	rt[counter].minZ = -1024.0;
	rt[counter].maxX = 1024.0;
	rt[counter].maxY = 1344.0;
	rt[counter].maxZ = 1824.0;
    rt[counter].index = 69; //room2ccont
    counter++;

	//room2offices
	rt[counter].id = counter;
	rt[counter].name = ROOM2OFFICES;// "room2offices";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 30;
	rt[counter].lights = 4;
	rt[counter].zone[0] = 3;
	rt[counter].minX = -608.0;
	rt[counter].minY = 0.0;
	rt[counter].minZ = -1024.0;
	rt[counter].maxX = 1024.0;
	rt[counter].maxY = 422.45544;
	rt[counter].maxZ = 1024.0;
    rt[counter].index = 70; //room2offices
    counter++;

	//room2offices2
	rt[counter].id = counter;
	rt[counter].name = ROOM2OFFICES2;// "room2offices2";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 20;
	rt[counter].lights = 6;
	rt[counter].disableDecals = true;
	rt[counter].zone[0] = 3;
	rt[counter].minX = -1024.0;
	rt[counter].minY = -192.0;
	rt[counter].minZ = -1024.0;
	rt[counter].maxX = 288.0;
	rt[counter].maxY = 386.86813;
	rt[counter].maxZ = 1024.0;
    rt[counter].index = 71; //room2offices2
    counter++;

	//room2offices3
	rt[counter].id = counter;
	rt[counter].name = ROOM2OFFICES3;// "room2offices3";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 20;
	rt[counter].lights = 6;
	rt[counter].zone[0] = 3;
	rt[counter].minX = -1600.0002;
	rt[counter].minY = -0.000029060417;
	rt[counter].minZ = -1024.0;
	rt[counter].maxX = 768.0;
	rt[counter].maxY = 832.0;
	rt[counter].maxZ = 1024.0002;
    rt[counter].index = 72; //room2offices3
    counter++;

	//room2offices4
	rt[counter].id = counter;
	rt[counter].name = ROOM2OFFICES4;// "room2offices4";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 0;
	rt[counter].lights = 5;
	rt[counter].zone[0] = 3;
	rt[counter].minX = -1568.0;
	rt[counter].minY = -416.0;
	rt[counter].minZ = -1024.0;
	rt[counter].maxX = 596.0;
	rt[counter].maxY = 708.0;
	rt[counter].maxZ = 1024.0;
    rt[counter].index = 73; //room2offices4
    counter++;

	//room2poffices
	rt[counter].id = counter;
	rt[counter].name = ROOM2POFFICES;// "room2poffices";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 0;
	rt[counter].lights = 5;
	rt[counter].zone[0] = 3;
	rt[counter].minX = -512.0;
	rt[counter].minY = -0.000030517578;
	rt[counter].minZ = -1024.0001;
	rt[counter].maxX = 976.0001;
	rt[counter].maxY = 438.83057;
	rt[counter].maxZ = 1024.0001;
    rt[counter].index = 74; //room2poffices
    counter++;

	//room2poffices2
	rt[counter].id = counter;
	rt[counter].name = ROOM2POFFICES2;// "room2poffices2";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 0;
	rt[counter].lights = 4;
	rt[counter].zone[0] = 3;
	rt[counter].minX = -1184.0;
	rt[counter].minY = -1.4324226;
	rt[counter].minZ = -1024.0002;
	rt[counter].maxX = 1216.0;
	rt[counter].maxY = 438.83057;
	rt[counter].maxZ = 1024.0001;
    rt[counter].index = 75; //room2poffices2
    counter++;

	//room2sroom
	rt[counter].id = counter;
	rt[counter].name = ROOM2SROOM;// "room2sroom";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 0;
	rt[counter].lights = 4;
	rt[counter].zone[0] = 3;
	rt[counter].minX = -304.0;
	rt[counter].minY = -0.000030517578;
	rt[counter].minZ = -1024.0001;
	rt[counter].maxX = 2304.0;
	rt[counter].maxY = 640.0;
	rt[counter].maxZ = 1024.0001;
    rt[counter].index = 76; //room2sroom
    counter++;

	//room2toilets
	rt[counter].id = counter;
	rt[counter].name = ROOM2TOILETS;// "room2toilets";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 30;
	rt[counter].lights = 5;
	rt[counter].zone[0] = 3;
	rt[counter].minX = -320.0;
	rt[counter].minY = 0.0;
	rt[counter].minZ = -1024.0;
	rt[counter].maxX = 1568.0;
	rt[counter].maxY = 386.86813;
	rt[counter].maxZ = 1024.0;
    rt[counter].index = 77; //room2toilets
    counter++;

	//room2tesla
	rt[counter].id = counter;
	rt[counter].name = ROOM2TESLA;// "room2tesla";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 100;
	rt[counter].lights = 8;
	rt[counter].zone[0] = 3;
	rt[counter].minX = -304.00012;
	rt[counter].minY = -64.0;
	rt[counter].minZ = -1024.0;
	rt[counter].maxX = 304.0001;
	rt[counter].maxY = 722.45544;
	rt[counter].maxZ = 1024.0;
    rt[counter].index = 78; //room2tesla
    counter++;

	//room3servers
	rt[counter].id = counter;
	rt[counter].name = ROOM3SERVERS;// "room3servers";
	rt[counter].shape = ROOM3;
	rt[counter].commonness = 0;
	rt[counter].lights = 6;
	rt[counter].disableDecals = true;
	rt[counter].zone[0] = 3;
	rt[counter].minX = -1024.0;
	rt[counter].minY = -1536.0;
	rt[counter].minZ = -1024.0;
	rt[counter].maxX = 1024.0;
	rt[counter].maxY = 385.00006;
	rt[counter].maxZ = 1032.0;
    rt[counter].index = 79; //room3servers
    counter++;

	//room3servers2
	rt[counter].id = counter;
	rt[counter].name = ROOM3SERVERS2;// "room3servers2";
	rt[counter].shape = ROOM3;
	rt[counter].commonness = 0;
	rt[counter].lights = 6;
	rt[counter].disableDecals = true;
	rt[counter].zone[0] = 3;
	rt[counter].minX = -1024.0;
	rt[counter].minY = -1536.0;
	rt[counter].minZ = -1024.0;
	rt[counter].maxX = 1024.0;
	rt[counter].maxY = 385.00006;
	rt[counter].maxZ = 1032.0;
    rt[counter].index = 80; //room3servers2
    counter++;

	//room3z3
	rt[counter].id = counter;
	rt[counter].name = ROOM3Z3;// "room3z3";
	rt[counter].shape = ROOM3;
	rt[counter].commonness = 100;
	rt[counter].lights = 1;
	rt[counter].zone[0] = 3;
	rt[counter].minX = -1024.0;
	rt[counter].minY = 0.0;
	rt[counter].minZ = -1024.0;
	rt[counter].maxX = 1024.0;
	rt[counter].maxY = 438.83054;
	rt[counter].maxZ = 576.0;
    rt[counter].index = 81; //room3z3
    counter++;

	//room4z3
	rt[counter].id = counter;
	rt[counter].name = ROOM4Z3;// "room4z3";
	rt[counter].shape = ROOM4;
	rt[counter].commonness = 100;
	rt[counter].lights = 2;
	rt[counter].zone[0] = 3;
	rt[counter].minX = -1024.0;
	rt[counter].minY = -1.0;
	rt[counter].minZ = -1024.0;
	rt[counter].maxX = 1024.0;
	rt[counter].maxY = 1440.0;
	rt[counter].maxZ = 1024.0;
    rt[counter].index = 82; //room4z3
    counter++;

	//room1lifts
	rt[counter].id = counter;
	rt[counter].name = ROOM1LIFTS;// "room1lifts";
	rt[counter].shape = ROOM1;
	rt[counter].commonness = 0;
	rt[counter].lights = 1;
	rt[counter].zone[0] = 3;
	rt[counter].minX = -464.0;
	rt[counter].minY = -36.01808;
	rt[counter].minZ = -1024.0001;
	rt[counter].maxX = 448.0;
	rt[counter].maxY = 466.85107;
	rt[counter].maxZ = 105.69403;
    rt[counter].index = 83; //room1lifts
    counter++;

	//room3gw
	rt[counter].id = counter;
	rt[counter].name = ROOM3GW;// "room3gw";
	rt[counter].shape = ROOM3;
	rt[counter].commonness = 10;
	rt[counter].lights = 1;
	rt[counter].zone[0] = 3;
	rt[counter].minX = -1024.0;
	rt[counter].minY = -18.0;
	rt[counter].minZ = -1031.0;
	rt[counter].maxX = 1038.9995;
	rt[counter].maxY = 535.0;
	rt[counter].maxZ = 547.5521;
    rt[counter].index = 84; //room3gw
    counter++;

	//room2servers2
	rt[counter].id = counter;
	rt[counter].name = ROOM2SERVERS2;// "room2servers2";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 0;
	rt[counter].lights = 4;
	rt[counter].zone[0] = 3;
	rt[counter].minX = -1081.5002;
	rt[counter].minY = -800.02185;
	rt[counter].minZ = -1048.9976;
	rt[counter].maxX = 816.00006;
	rt[counter].maxY = 464.85107;
	rt[counter].maxZ = 1024.0001;
    rt[counter].index = 85; //room2servers2
    counter++;

	//room3offices
	rt[counter].id = counter;
	rt[counter].name = ROOM3OFFICES;// "room3offices";
	rt[counter].shape = ROOM3;
	rt[counter].commonness = 0;
	rt[counter].lights = 1;
	rt[counter].zone[0] = 3;
	rt[counter].minX = -1024.0;
	rt[counter].minY = -0.0000076293945;
	rt[counter].minZ = -1024.0;
	rt[counter].maxX = 1024.0;
	rt[counter].maxY = 464.85107;
	rt[counter].maxZ = 1036.0;
    rt[counter].index = 86; //room3offices
    counter++;

	//room2z3_2
	rt[counter].id = counter;
	rt[counter].name = ROOM2Z3_2;// "room2z3_2";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 25;
	rt[counter].lights = 0;
	rt[counter].zone[0] = 3;
	rt[counter].minX = -304.0;
	rt[counter].minY = -1.0;
	rt[counter].minZ = -1024.0;
	rt[counter].maxX = 304.00018;
	rt[counter].maxY = 416.0;
	rt[counter].maxZ = 1024.0;
    rt[counter].index = 87; //room2z3_2
    counter++;

	//pocketdimension
	rt[counter].id = counter;
	rt[counter].name = POCKETDIMENSION;// "pocketdimension";
	rt[counter].shape = ROOM1;
	rt[counter].commonness = 0;
	rt[counter].lights = 1;
	rt[counter].minX = -512.0;
	rt[counter].minY = -32.0;
	rt[counter].minZ = -512.0;
	rt[counter].maxX = 512.0;
	rt[counter].maxY = 1024.0;
	rt[counter].maxZ = 512.0;
    rt[counter].index = 88; //pocketdimension
    counter++;

	//dimension1499
	rt[counter].id = counter;
	rt[counter].name = DIMENSION1499;// "dimension1499";
	rt[counter].shape = ROOM1;
	rt[counter].commonness = 0;
	rt[counter].lights = 0;
	rt[counter].disableDecals = true;
	rt[counter].minX = -7509.0;
	rt[counter].minY = -672.0001;
	rt[counter].minZ = -4207.308;
	rt[counter].maxX = 7509.2817;
	rt[counter].maxY = 8928.0;
	rt[counter].maxZ = 4207.0;
    rt[counter].index = 89; //dimension1499

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

	Rooms r;
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

	for (int32_t i = 0; i < 324; i++) {
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

	int32_t x = r->x / 8.0;
	int32_t y = r->y / 8.0;

	if (r->rt->shape == ROOM2) {
		r->angle += 180;
		//CalculateRoomExtents(r);
		//TEMPORARY
		GetRoomExtents(r, e);

		for (int32_t i = 0; i < 18 * 18; i++) {
			r2 = &rooms[i];
			if (r2->id == -1) break; //Id should only be -1 after all other rooms.

			if (r2->id != r->id && !r2->rt->disableOverlapCheck) {
				if (CheckRoomOverlap(r, r2)) {
					isIntersecting = true;
					r->angle = r->angle - 180;
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
	int32_t temp2, x2, y2, rot, rot2;	
	for (int32_t i = 0; i < 18 * 18; i++) {
		r2 = &rooms[i];

		if (r2->id == -1) break;

		if (r2->id != r->id && !r2->rt->disableOverlapCheck) {
			if (r->rt->shape == r2->rt->shape && r->zone == r2->zone && (r2->rt->name != CHECKPOINT1 && r2->rt->name != CHECKPOINT2 && r2->rt->name != START)) {
				x = r->x / 8.0;
				y = r->z / 8.0;
				rot = r->angle;

				x2 = r2->x / 8.0;
				y2 = r2->z / 8.0;
				rot2 = r2->angle;

				isIntersecting = false;

				r->x = x2 * 8.0;
				r->z = y2 * 8.0;
				r->angle = rot2;
				//CalculateRoomExtents(r);
				//TEMPORARY
				GetRoomExtents(r, e);

				r2->x = x * 8.0;
				r2->z = y * 8.0;
				r2->angle = rot;
				//CalculateRoomExtents(r2);
				//TEMPORARY
				GetRoomExtents(r2, e);

				//make sure neither room overlaps with anything after the swap
				for (int32_t j = 0; j < 18 * 18; j++) {
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
					r->x = x * 8.0;
					r->z = y * 8.0;
					r->angle = rot;
					//CalculateRoomExtents(r);
					//TEMPORARY
					GetRoomExtents(r, e);

					r2->x = x2 * 8.0;
					r2->z = y2 * 8.0;
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

__device__ inline void CalculateRoomExtents(Rooms* r) {
	if (r->rt->disableOverlapCheck) return;

	static const float shrinkAmount = 0.05;

	//We must convert TFormVector() in blitz to c++ code.
	static float roomScale = 8.0 / 2048.0;

	//First we scale.
	r->rt->minX *= roomScale;
	r->rt->minY *= roomScale;
	r->rt->minZ *= roomScale;

	r->rt->maxX *= roomScale;
	r->rt->maxY *= roomScale;
	r->rt->maxZ *= roomScale;

	//Then we rotate	
	float rad = 0.0;
	switch (r->angle) {
	case 90:
		rad = 1.5708;
		break;
	case 180:
		rad = 3.14159;
		break;
	case 270:
		rad = 4.71239;
		break;
	}

	r->rt->minX = r->rt->minX * cosf(rad) - r->rt->minZ * sinf(rad);
	r->rt->minZ = r->rt->minX * sinf(rad) + r->rt->minZ * cosf(rad);

	r->rt->maxX = r->rt->maxX * cosf(rad) - r->rt->maxZ * sinf(rad);
	r->rt->maxZ = r->rt->maxX * sinf(rad) + r->rt->maxZ * cosf(rad);

	//Back to blitz.

	r->minX = r->rt->minX + shrinkAmount + r->x;
	r->minY = r->rt->minY + shrinkAmount;
	r->minZ = r->rt->minZ + shrinkAmount + r->z;

	r->maxX = r->rt->maxX - shrinkAmount + r->x;
	r->maxY = r->rt->maxY - shrinkAmount;
	r->maxZ = r->rt->maxZ - shrinkAmount + r->z;

	if (r->minX > r->maxX) {
		float temp = r->maxX;
		r->maxX = r->minX;
		r->minX = temp;
	}

	if (r->minZ > r->maxZ) {
		float temp = r->maxZ;
		r->maxZ = r->minZ;
		r->minZ = temp;
	}

	/*r->minX = rintf(r->minX * 1000000.0) / 1000000.0;
	r->minY = rintf(r->minY * 1000000.0) / 1000000.0;
	r->minX = rintf(r->minZ * 1000000.0) / 1000000.0;

	r->maxX = rintf(r->maxX * 1000000.0) / 1000000.0;
	r->maxY = rintf(r->maxY * 1000000.0) / 1000000.0;
	r->maxZ = rintf(r->maxZ * 1000000.0) / 1000000.0;*/

	//printf("NAME: %s MINX %f MINY %f MINZ %f MAXX %f MAXY %f MAXZ %f\n", RoomIDToName(r->rt.name), r->minX, r->minY, r->minZ, r->maxX, r->maxY, r->maxZ);

	return;
}

__device__ inline void FillRoom(uint8_t MapTemp[19][19], bbRandom* bb, rnd_state* rnd_state, Rooms* r, uint8_t* forest) {
	
	RoomID name = r->rt->name;

	switch (name) {
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
		for (int32_t i = 0; i <= 5; i++) {
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
		
		int32_t x = int(floorf(r->x / 8.0));
		int32_t y = int(floorf(r->z / 8.0)) - 1;

		if (MapTemp[x][y] == 0) {
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

		for (int32_t i = 0; i <= bb->bbRand(rnd_state, 0, 1); i++) {
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

		for (int32_t i = 0; i <= 14; i++) {
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

		for (int32_t x = 0; x <= 1; x++) {
			for (int32_t z = 0; z <= 1; z++) {
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

		for (int32_t z = 0; z <= 1; z++) {
			CreateDoor(bb, rnd_state, false, 0);
			CreateDoor(bb, rnd_state, false, 0);
			for (int32_t x = 0; x <= 2; x++) {
				CreateDoor(bb, rnd_state, false, 0);
			}
			for (int32_t x = 0; x <= 4; x++) {
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
		for (int32_t xtemp = 0; xtemp <= 1; xtemp++) {
			for (int32_t ytemp = 0; ytemp <= 2; ytemp++) {
				for (int32_t ztemp = 0; ztemp <= 2; ztemp++) {

					int32_t chance = bb->bbRand(rnd_state, -10, 100);

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

		for (int32_t i = 1; i <= 8; i++) {
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
		if (name == ROOM2GW_B) {
			bb->bbRnd(rnd_state, 0, 360);
		}

		CreateDoor(bb, rnd_state, false, 0);
		CreateDoor(bb, rnd_state, false, 0);

		if (name == ROOM2GW) {	
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
	for (int32_t i = 0; i < min(32, r->rt->lights); i++) {
		bb->bbRand(rnd_state, 1, 360);
		bb->bbRand(rnd_state, 1, 10);
	}
}

__device__ inline void GetRoomExtents(Rooms* r, float* e) {

	if (r->rt->disableOverlapCheck) return;

	int32_t xIndex = int((r->x / 8.0)) - 1;
	int32_t zIndex = int((r->z / 8.0)) - 1;
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