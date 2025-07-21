#pragma once

#ifndef ROOMS_CUH
#define ROOMS_CUH

#include "cuda_runtime.h";
#include "Helpers.cuh";
#include "Constants.cuh";
#include "Random.cuh";

typedef struct RoomTemplates RoomTemplates;
static struct RoomTemplates {
	int32_t obj;
	int32_t id = -1;
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
	RoomTemplates rt;

	float dist;	

	//a bunch of stuff that i might need

	float minX, minY, minZ;
	float maxX, maxY, maxZ;

};	

__device__ inline void CreateRoomTemplates(RoomTemplates* rt);
__device__ inline bool SetRoom(RoomID room_name, uint32_t room_type, uint32_t pos, uint32_t min_pos, uint32_t max_pos);
__device__ inline Rooms CreateRoom(RoomTemplates* rts, bbRandom* bb, rnd_state* rnd_sate, int32_t zone, int32_t roomshape, float x, float y, float z, RoomID name);
__device__ inline bool PreventRoomOverlap(Rooms* r, Rooms* rooms);
__device__ inline bool CheckRoomOverlap(Rooms* r, Rooms* r2);
__device__ inline void CalculateRoomExtents(Rooms* r);
__device__ inline void FillRoom(bbRandom* bb, rnd_state* rnd_state, Rooms* r);
__device__ void GetRoomExtents(Rooms* r);

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
	counter++;

	//room2cafeteria
	rt[counter].id = counter;
	rt[counter].name = ROOM2CAFETERIA;// "room2cafeteria";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 0;
	rt[counter].lights = 7;
	rt[counter].zone[0] = 3;
	rt[counter].large = true;
	rt[counter].disableOverlapCheck = true;
	rt[counter].minX = -1792.0;
	rt[counter].minY = -416.0;
	rt[counter].minZ = -1056.0;
	rt[counter].maxX = 1952.0;
	rt[counter].maxY = 708.0;
	rt[counter].maxZ = 1024.0001;
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
	counter++;

	//room2z3_2
	rt[counter].id = counter;
	rt[counter].name = ROOM2Z3_2; "room2z3_2";
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

}

__device__ inline bool SetRoom(RoomID room_name, uint32_t room_type, uint32_t pos, uint32_t min_pos, uint32_t max_pos) {
	if (max_pos < min_pos) return false;

	uint32_t looped = false; 
	uint32_t can_place = true;

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

__device__ inline Rooms CreateRoom(RoomTemplates* rts, bbRandom* bb, rnd_state* rnd_state, int32_t zone, int32_t roomshape, float x, float y, float z, RoomID name) {

	Rooms r = Rooms();
	//RoomTemplates* rt;

	//The original game doesn't actually have a room id variable
	r.id = roomIdCounter++;

	r.zone = zone;

	r.x = x;
	r.y = y;
	r.z = z;

	if (name != ROOMEMPTY) {
		for (int32_t i = 0; i < roomTemplateAmount; i++) {			
			if (rts[i].name == name) {
				r.rt = rts[i];
				
				FillRoom(bb, rnd_state, &r);

				//Don't think we need light cone stuff.
				
				//CalculateRoomExtents(&r);
				//TEMPORARY
				GetRoomExtents(&r);
				return r;
			}
		}
	}	

	int32_t t = 0;
	for (int32_t i = 0; i < roomTemplateAmount; i++) {
		//5 because that is the len of the rt.zone[] array;
		for (int32_t j = 0; j <= 4; j++) {
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
	for (int32_t i = 0; i < roomTemplateAmount; i++) {
		for (int32_t j = 0; j <= 4; j++) {
			if (rts[i].zone[j] == zone && rts[i].shape == roomshape) {
				t = t + rts[i].commonness;
				if (RandomRoom > t - rts[i].commonness && RandomRoom <= t) {
					r.rt = rts[i];
					
					FillRoom(bb, rnd_state, &r);

					//Skip light cone stuff
			
					//CalculateRoomExtents(&r);
					//TEMPORARY
					GetRoomExtents(&r);
					return r;
				}
			}
		}
	}	
	//It seems like there is supposed to be a return r at the bottom here,
	//but it isn't there in the blitz code so idk.
}

__device__ inline bool PreventRoomOverlap(Rooms* r, Rooms* rooms) {
	if (r->rt.disableOverlapCheck) return true;

	//We might have some problems with passing the rooms array by pointer.
	//Want to make sure we pass by reference instead of making a new copy of entire rooms array.

	Rooms* r2;
	Rooms* r3;

	bool isIntersecting = false;

	if (r->rt.name == CHECKPOINT1 || r->rt.name == CHECKPOINT2 || r->rt.name == START) return true;

	for (int32_t i = 0; i < 324; i++) {
		r2 = &rooms[i];

		//-1 id means we are at the point of the array where there are no more rooms.
		if (r2->id == -1) break;

		if (r2->id != r->id && !r2->rt.disableOverlapCheck) {
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

	if (r->rt.shape == ROOM2) {
		r->angle += 180;
		//CalculateRoomExtents(r);
		//TEMPORARY
		GetRoomExtents(r);

		for (int32_t i = 0; i < 18 * 18; i++) {
			r2 = &rooms[i];
			if (r2->id == -1) break; //Id should only be -1 after all other rooms.

			if (r2->id != r->id && !r2->rt.disableOverlapCheck) {
				if (CheckRoomOverlap(r, r2)) {
					isIntersecting = true;
					r->angle = r->angle - 180;
					//CalculateRoomExtents(r);
					//TEMPORARY
					GetRoomExtents(r);
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

		if (r2->id != r->id && !r2->rt.disableOverlapCheck) {
			if (r->rt.shape == r2->rt.shape && r->zone == r2->zone && (r2->rt.name != CHECKPOINT1 && r2->rt.name != CHECKPOINT2 && r2->rt.name != START)) {
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
				GetRoomExtents(r);

				r2->x = x * 8.0;
				r2->z = y * 8.0;
				r2->angle = rot;
				//CalculateRoomExtents(r2);
				//TEMPORARY
				GetRoomExtents(r2);

				//make sure neither room overlaps with anything after the swap
				for (int32_t i = 0; i < 18 * 18; i++) {
					r3 = &rooms[i];

					if (r3->id == -1) break;

					if (!r3->rt.disableOverlapCheck) {
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
					GetRoomExtents(r);

					r2->x = x2 * 8.0;
					r2->z = y2 * 8.0;
					r2->angle = rot2;
					//CalculateRoomExtents(r2);
					//TEMPORARY
					GetRoomExtents(r2);

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

	if (r->maxX <= r2->minX || r->maxY <= r2->minY || r->maxZ <= r2->minZ) return false;

	if (r->minX >= r2->maxX || r->minY >= r2->maxY || r->minZ >= r2->maxZ) return false;

	return true;
}

__device__ inline void CalculateRoomExtents(Rooms* r) {
	if (r->rt.disableOverlapCheck) return;

	static const float shrinkAmount = 0.05;

	//We must convert TFormVector() in blitz to c++ code.
	static float roomScale = 8.0 / 2048.0;

	//First we scale.
	r->rt.minX *= roomScale;
	r->rt.minY *= roomScale;
	r->rt.minZ *= roomScale;

	r->rt.maxX *= roomScale;
	r->rt.maxY *= roomScale;
	r->rt.maxZ *= roomScale;

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

	r->rt.minX = r->rt.minX * cosf(rad) - r->rt.minZ * sinf(rad);
	r->rt.minZ = r->rt.minX * sinf(rad) + r->rt.minZ * cosf(rad);

	r->rt.maxX = r->rt.maxX * cosf(rad) - r->rt.maxZ * sinf(rad);
	r->rt.maxZ = r->rt.maxX * sinf(rad) + r->rt.maxZ * cosf(rad);

	//Back to blitz.

	r->minX = r->rt.minX + shrinkAmount + r->x;
	r->minY = r->rt.minY + shrinkAmount;
	r->minZ = r->rt.minZ + shrinkAmount + r->z;

	r->maxX = r->rt.maxX - shrinkAmount + r->x;
	r->maxY = r->rt.maxY - shrinkAmount;
	r->maxZ = r->rt.maxZ - shrinkAmount + r->z;

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

__device__ inline void FillRoom(bbRandom* bb, rnd_state* rnd_state, Rooms* r) {
	
	RoomID name = r->rt.name;

	switch (name) {
	case ROOM860:
		CreateDoor(bb, rnd_state, false, 0);
		CreateDoor(bb, rnd_state, true, 0);

		CreateDoor(bb, rnd_state, false, 0);
		CreateDoor(bb, rnd_state, false, 0);

		//GenForestGrid();

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
		printf("ROOM: %s NOT FOUND IN FillRoom().\n", RoomIDToName(name));
	}	

	//32 becase that is MaxRoomLights
	for (int32_t i = 0; i < min(32, r->rt.lights); i++) {
		bb->bbRand(rnd_state, 1, 360);
		bb->bbRand(rnd_state, 1, 10);
	}
}

__device__ void GetRoomExtents(Rooms* r) {

	//god help us

	RoomID name = r->rt.name;
	//We subtract 1 from this because the extents will be off
	//by 8.0 if we do not.
	float x = (r->x / 8.0) - 1;
	float z = (r->z / 8.0) - 1;
	int32_t angle = r->angle;

	float minX = 0.0;
	float maxX = 0.0;
	float minZ = 0.0;
	float maxZ = 0.0;

	float factor = 8.0;

	switch (name) {
	case LOCKROOM:
		switch (angle) {
		case 0:
			minX = 4.675000190734863;
			maxX = 11.949999809265137;
			minZ = 4.050000190734863;
			maxZ = 11.324999809265137;
			break;

		case 90:
			minX = 4.57499885559082;
			maxX = 12.05000114440918;
			minZ = 4.675000190734863;
			maxZ = 11.949999809265137;
			break;

		case 180:
			minX = 3.950000286102295;
			maxX = 11.424999237060547;
			minZ = 4.574999809265137;
			maxZ = 12.05000114440918;
			break;

		case 270:
			minX = 4.050000190734863;
			maxX = 11.324999809265137;
			minZ = 3.9499988555908203;
			maxZ = 11.42500114440918;
			break;

		case 360:
			minX = 4.6750006675720215;
			maxX = 11.949999809265137;
			minZ = 4.049999237060547;
			maxZ = 11.325000762939453;
			break;

		case 450:
			minX = 4.575000762939453;
			maxX = 12.049999237060547;
			minZ = 4.674999237060547;
			maxZ = 11.950000762939453;
			break;

		case 540:
			minX = 3.9499998092651367;
			maxX = 11.425000190734863;
			minZ = 4.574999809265137;
			maxZ = 12.050000190734863;
			break;

		case 630:
			minX = 4.05000114440918;
			maxX = 11.32499885559082;
			minZ = 3.9499988555908203;
			maxZ = 11.42500114440918;
			break;

		}
		break;
	case 173:
		switch (angle) {
		case 0:
			minX = -25.953907012939453;
			maxX = 14.949999809265137;
			minZ = -6.824999809265137;
			maxZ = 13.949999809265137;
			break;

		case 90:
			minX = 1.949997901916504;
			maxX = 22.925006866455078;
			minZ = -25.95391082763672;
			maxZ = 14.949999809265137;
			break;

		case 180:
			minX = 0.9500002861022949;
			maxX = 42.05390548706055;
			minZ = 1.9499993324279785;
			maxZ = 22.925003051757812;
			break;

		case 270:
			minX = -6.824997901916504;
			maxX = 13.949999809265137;
			minZ = 0.9499979019165039;
			maxZ = 42.05390930175781;
			break;

		case 360:
			minX = -25.953903198242188;
			maxX = 14.94999885559082;
			minZ = -6.825005531311035;
			maxZ = 13.950000762939453;
			break;

		case 450:
			minX = 1.9500017166137695;
			maxX = 22.92499351501465;
			minZ = -25.953907012939453;
			maxZ = 14.950000762939453;
			break;

		case 540:
			minX = 0.9499998092651367;
			maxX = 42.05390548706055;
			minZ = 1.9499998092651367;
			maxZ = 22.92500114440918;
			break;

		case 630:
			minX = -6.824986457824707;
			maxX = 13.949996948242188;
			minZ = 0.9499979019165039;
			maxZ = 42.05390548706055;
			break;

		}
		break;
	case START:
		switch (angle) {
		case 0:
			minX = 5.425000190734863;
			maxX = 29.450000762939453;
			minZ = 4.050000190734863;
			maxZ = 19.075000762939453;
			break;

		case 90:
			minX = -3.175004005432129;
			maxX = 12.05000114440918;
			minZ = 5.425000190734863;
			maxZ = 29.450000762939453;
			break;

		case 180:
			minX = -13.549997329711914;
			maxX = 10.674999237060547;
			minZ = -3.175002098083496;
			maxZ = 12.050000190734863;
			break;

		case 270:
			minX = 4.050000190734863;
			maxX = 19.074996948242188;
			minZ = -13.550003051757812;
			maxZ = 10.67500114440918;
			break;

		case 360:
			minX = 5.4250006675720215;
			maxX = 29.44999885559082;
			minZ = 4.049999237060547;
			maxZ = 19.07500457763672;
			break;

		case 450:
			minX = -3.1749954223632812;
			maxX = 12.049999237060547;
			minZ = 5.424999237060547;
			maxZ = 29.450000762939453;
			break;

		case 540:
			minX = -13.549999237060547;
			maxX = 10.675000190734863;
			minZ = -3.1750011444091797;
			maxZ = 12.050000190734863;
			break;

		case 630:
			minX = 4.05000114440918;
			maxX = 19.07499122619629;
			minZ = -13.55000114440918;
			maxZ = 10.67500114440918;
			break;

		}
		break;
	case ROOM1123:
		switch (angle) {
		case 0:
			minX = 4.425000190734863;
			maxX = 11.949999809265137;
			minZ = 4.050000190734863;
			maxZ = 11.949999809265137;
			break;

		case 90:
			minX = 3.9499988555908203;
			maxX = 12.05000114440918;
			minZ = 4.425000190734863;
			maxZ = 11.949999809265137;
			break;

		case 180:
			minX = 3.950000286102295;
			maxX = 11.674999237060547;
			minZ = 3.9499993324279785;
			maxZ = 12.05000114440918;
			break;

		case 270:
			minX = 4.050000190734863;
			maxX = 11.949999809265137;
			minZ = 3.9499988555908203;
			maxZ = 11.67500114440918;
			break;

		case 360:
			minX = 4.4250006675720215;
			maxX = 11.94999885559082;
			minZ = 4.049999237060547;
			maxZ = 11.950000762939453;
			break;

		case 450:
			minX = 3.950000762939453;
			maxX = 12.049999237060547;
			minZ = 4.424999237060547;
			maxZ = 11.950000762939453;
			break;

		case 540:
			minX = 3.9499998092651367;
			maxX = 11.675000190734863;
			minZ = 3.9499998092651367;
			maxZ = 12.050000190734863;
			break;

		case 630:
			minX = 4.05000114440918;
			maxX = 11.94999885559082;
			minZ = 3.9499988555908203;
			maxZ = 11.67500114440918;
			break;

		}
		break;
	case ROOM1ARCHIVE:
		switch (angle) {
		case 0:
			minX = 5.050000190734863;
			maxX = 9.074999809265137;
			minZ = 4.050000190734863;
			maxZ = 10.887500762939453;
			break;

		case 90:
			minX = 5.0124993324279785;
			maxX = 12.05000114440918;
			minZ = 5.050000190734863;
			maxZ = 9.074999809265137;
			break;

		case 180:
			minX = 6.825000286102295;
			maxX = 11.049999237060547;
			minZ = 5.012499809265137;
			maxZ = 12.05000114440918;
			break;

		case 270:
			minX = 4.050000190734863;
			maxX = 10.887500762939453;
			minZ = 6.824999809265137;
			maxZ = 11.05000114440918;
			break;

		case 360:
			minX = 5.0500006675720215;
			maxX = 9.074999809265137;
			minZ = 4.049999237060547;
			maxZ = 10.887500762939453;
			break;

		case 450:
			minX = 5.012500286102295;
			maxX = 12.049999237060547;
			minZ = 5.049999237060547;
			maxZ = 9.075000762939453;
			break;

		case 540:
			minX = 6.825000286102295;
			maxX = 11.050000190734863;
			minZ = 5.012499809265137;
			maxZ = 12.050000190734863;
			break;

		case 630:
			minX = 4.05000114440918;
			maxX = 10.887499809265137;
			minZ = 6.82499885559082;
			maxZ = 11.05000114440918;
			break;

		}
		break;
	case ROOM2STORAGE:
		switch (angle) {
		case 0:
			minX = 2.862499713897705;
			maxX = 13.137500762939453;
			minZ = 4.050000190734863;
			maxZ = 11.949999809265137;
			break;

		case 90:
			minX = 3.9499988555908203;
			maxX = 12.05000114440918;
			minZ = 2.862499713897705;
			maxZ = 13.137500762939453;
			break;

		case 180:
			minX = 2.7624998092651367;
			maxX = 13.237500190734863;
			minZ = 3.9499993324279785;
			maxZ = 12.05000114440918;
			break;

		case 270:
			minX = 4.050000190734863;
			maxX = 11.949999809265137;
			minZ = 2.762498378753662;
			maxZ = 13.23750114440918;
			break;

		case 360:
			minX = 2.8625001907348633;
			maxX = 13.137499809265137;
			minZ = 4.049999237060547;
			maxZ = 11.950000762939453;
			break;

		case 450:
			minX = 3.9500012397766113;
			maxX = 12.049999237060547;
			minZ = 2.862499237060547;
			maxZ = 13.137500762939453;
			break;

		case 540:
			minX = 2.7624993324279785;
			maxX = 13.23750114440918;
			minZ = 3.9499998092651367;
			maxZ = 12.050000190734863;
			break;

		case 630:
			minX = 4.050002098083496;
			maxX = 11.949997901916504;
			minZ = 2.762498378753662;
			maxZ = 13.23750114440918;
			break;

		}
		break;
	case ROOM2TESLA_LCZ:
		switch (angle) {
		case 0:
			minX = 6.862499237060547;
			maxX = 9.137500762939453;
			minZ = 4.050000190734863;
			maxZ = 11.949999809265137;
			break;

		case 90:
			minX = 3.9499993324279785;
			maxX = 12.05000114440918;
			minZ = 6.862499713897705;
			maxZ = 9.137499809265137;
			break;

		case 180:
			minX = 6.762499809265137;
			maxX = 9.237500190734863;
			minZ = 3.9499998092651367;
			maxZ = 12.050000190734863;
			break;

		case 270:
			minX = 4.049999237060547;
			maxX = 11.950000762939453;
			minZ = 6.76249885559082;
			maxZ = 9.23750114440918;
			break;

		case 360:
			minX = 6.862500190734863;
			maxX = 9.137499809265137;
			minZ = 4.050000190734863;
			maxZ = 11.949999809265137;
			break;

		case 450:
			minX = 3.950000286102295;
			maxX = 12.049999237060547;
			minZ = 6.862498760223389;
			maxZ = 9.137500762939453;
			break;

		case 540:
			minX = 6.762499809265137;
			maxX = 9.237500190734863;
			minZ = 3.9499998092651367;
			maxZ = 12.050000190734863;
			break;

		case 630:
			minX = 4.0500006675720215;
			maxX = 11.94999885559082;
			minZ = 6.762498378753662;
			maxZ = 9.237502098083496;
			break;

		}
		break;
	case ENDROOM:
		switch (angle) {
		case 0:
			minX = 5.237500190734863;
			maxX = 11.012499809265137;
			minZ = 4.050000190734863;
			maxZ = 12.793749809265137;
			break;

		case 90:
			minX = 3.1062488555908203;
			maxX = 12.05000114440918;
			minZ = 5.237500190734863;
			maxZ = 11.012499809265137;
			break;

		case 180:
			minX = 4.887500762939453;
			maxX = 10.862499237060547;
			minZ = 3.1062493324279785;
			maxZ = 12.05000114440918;
			break;

		case 270:
			minX = 4.050000190734863;
			maxX = 12.793749809265137;
			minZ = 4.88749885559082;
			maxZ = 10.86250114440918;
			break;

		case 360:
			minX = 5.2375006675720215;
			maxX = 11.01249885559082;
			minZ = 4.049999237060547;
			maxZ = 12.793750762939453;
			break;

		case 450:
			minX = 3.106250762939453;
			maxX = 12.049999237060547;
			minZ = 5.237499237060547;
			maxZ = 11.012500762939453;
			break;

		case 540:
			minX = 4.887499809265137;
			maxX = 10.862500190734863;
			minZ = 3.1062498092651367;
			maxZ = 12.050000190734863;
			break;

		case 630:
			minX = 4.05000114440918;
			maxX = 12.79374885559082;
			minZ = 4.88749885559082;
			maxZ = 10.86250114440918;
			break;

		}
		break;
	case ROOM012:
		switch (angle) {
		case 0:
			minX = 4.050000190734863;
			maxX = 11.137500762939453;
			minZ = 4.050000190734863;
			maxZ = 11.949999809265137;
			break;

		case 90:
			minX = 3.9499988555908203;
			maxX = 12.05000114440918;
			minZ = 4.050000190734863;
			maxZ = 11.137500762939453;
			break;

		case 180:
			minX = 4.762499809265137;
			maxX = 12.049999237060547;
			minZ = 3.9499993324279785;
			maxZ = 12.05000114440918;
			break;

		case 270:
			minX = 4.050000190734863;
			maxX = 11.949999809265137;
			minZ = 4.76249885559082;
			maxZ = 12.05000114440918;
			break;

		case 360:
			minX = 4.0500006675720215;
			maxX = 11.137499809265137;
			minZ = 4.049999237060547;
			maxZ = 11.950000762939453;
			break;

		case 450:
			minX = 3.950000762939453;
			maxX = 12.049999237060547;
			minZ = 4.049999237060547;
			maxZ = 11.137500762939453;
			break;

		case 540:
			minX = 4.762499809265137;
			maxX = 12.050000190734863;
			minZ = 3.9499998092651367;
			maxZ = 12.050000190734863;
			break;

		case 630:
			minX = 4.050001621246338;
			maxX = 11.94999885559082;
			minZ = 4.76249885559082;
			maxZ = 12.05000114440918;
			break;

		}
		break;
	case ROOM205:
		switch (angle) {
		case 0:
			minX = 1.0500001907348633;
			maxX = 11.074999809265137;
			minZ = 4.050000190734863;
			maxZ = 11.324999809265137;
			break;

		case 90:
			minX = 4.57499885559082;
			maxX = 12.05000114440918;
			minZ = 1.049999713897705;
			maxZ = 11.074999809265137;
			break;

		case 180:
			minX = 4.825000286102295;
			maxX = 15.049999237060547;
			minZ = 4.574999809265137;
			maxZ = 12.05000114440918;
			break;

		case 270:
			minX = 4.050000190734863;
			maxX = 11.324999809265137;
			minZ = 4.82499885559082;
			maxZ = 15.05000114440918;
			break;

		case 360:
			minX = 1.0500006675720215;
			maxX = 11.074999809265137;
			minZ = 4.0499982833862305;
			maxZ = 11.325000762939453;
			break;

		case 450:
			minX = 4.575000762939453;
			maxX = 12.049999237060547;
			minZ = 1.0500001907348633;
			maxZ = 11.075000762939453;
			break;

		case 540:
			minX = 4.824999809265137;
			maxX = 15.050000190734863;
			minZ = 4.574999809265137;
			maxZ = 12.050000190734863;
			break;

		case 630:
			minX = 4.0500030517578125;
			maxX = 11.32499885559082;
			minZ = 4.82499885559082;
			maxZ = 15.05000114440918;
			break;

		}
		break;
	case ROOM2ID:
		switch (angle) {
		case 0:
			minX = 6.862500190734863;
			maxX = 9.137500762939453;
			minZ = 4.050000190734863;
			maxZ = 11.949999809265137;
			break;

		case 90:
			minX = 3.9499993324279785;
			maxX = 12.05000114440918;
			minZ = 6.862500190734863;
			maxZ = 9.137499809265137;
			break;

		case 180:
			minX = 6.762499809265137;
			maxX = 9.237499237060547;
			minZ = 3.9499998092651367;
			maxZ = 12.050000190734863;
			break;

		case 270:
			minX = 4.049999237060547;
			maxX = 11.950000762939453;
			minZ = 6.76249885559082;
			maxZ = 9.237500190734863;
			break;

		case 360:
			minX = 6.8625006675720215;
			maxX = 9.137499809265137;
			minZ = 4.050000190734863;
			maxZ = 11.949999809265137;
			break;

		case 450:
			minX = 3.950000286102295;
			maxX = 12.049999237060547;
			minZ = 6.862499237060547;
			maxZ = 9.137500762939453;
			break;

		case 540:
			minX = 6.762499809265137;
			maxX = 9.237500190734863;
			minZ = 3.9499998092651367;
			maxZ = 12.050000190734863;
			break;

		case 630:
			minX = 4.0500006675720215;
			maxX = 11.94999885559082;
			minZ = 6.762498378753662;
			maxZ = 9.23750114440918;
			break;

		}
		break;
	case ROOM2_2:
		switch (angle) {
		case 0:
			minX = 4.925000190734863;
			maxX = 9.137500762939453;
			minZ = 4.050000190734863;
			maxZ = 11.949999809265137;
			break;

		case 90:
			minX = 3.9499993324279785;
			maxX = 12.05000114440918;
			minZ = 4.925000190734863;
			maxZ = 9.137499809265137;
			break;

		case 180:
			minX = 6.762499809265137;
			maxX = 11.174999237060547;
			minZ = 3.9499998092651367;
			maxZ = 12.05000114440918;
			break;

		case 270:
			minX = 4.050000190734863;
			maxX = 11.950000762939453;
			minZ = 6.76249885559082;
			maxZ = 11.17500114440918;
			break;

		case 360:
			minX = 4.9250006675720215;
			maxX = 9.137499809265137;
			minZ = 4.049999237060547;
			maxZ = 11.949999809265137;
			break;

		case 450:
			minX = 3.950000286102295;
			maxX = 12.049999237060547;
			minZ = 4.924999237060547;
			maxZ = 9.137500762939453;
			break;

		case 540:
			minX = 6.762499809265137;
			maxX = 11.175000190734863;
			minZ = 3.9499998092651367;
			maxZ = 12.050000190734863;
			break;

		case 630:
			minX = 4.05000114440918;
			maxX = 11.94999885559082;
			minZ = 6.762498378753662;
			maxZ = 11.17500114440918;
			break;

		}
		break;
	case ROOM2_3:
		switch (angle) {
		case 0:
			minX = 6.424999713897705;
			maxX = 9.574999809265137;
			minZ = 4.050000190734863;
			maxZ = 11.949999809265137;
			break;

		case 90:
			minX = 3.9499993324279785;
			maxX = 12.05000114440918;
			minZ = 6.425000190734863;
			maxZ = 9.574999809265137;
			break;

		case 180:
			minX = 6.325000286102295;
			maxX = 9.674999237060547;
			minZ = 3.9499998092651367;
			maxZ = 12.050000190734863;
			break;

		case 270:
			minX = 4.049999237060547;
			maxX = 11.950000762939453;
			minZ = 6.3249993324279785;
			maxZ = 9.67500114440918;
			break;

		case 360:
			minX = 6.4250006675720215;
			maxX = 9.57499885559082;
			minZ = 4.049999237060547;
			maxZ = 11.950000762939453;
			break;

		case 450:
			minX = 3.950000286102295;
			maxX = 12.049999237060547;
			minZ = 6.424999237060547;
			maxZ = 9.575000762939453;
			break;

		case 540:
			minX = 6.325000286102295;
			maxX = 9.675000190734863;
			minZ = 3.9499998092651367;
			maxZ = 12.050000190734863;
			break;

		case 630:
			minX = 4.0500006675720215;
			maxX = 11.94999885559082;
			minZ = 6.32499885559082;
			maxZ = 9.67500114440918;
			break;

		}
		break;
	case ROOM2_4:
		switch (angle) {
		case 0:
			minX = 6.862500190734863;
			maxX = 10.965624809265137;
			minZ = 4.050000190734863;
			maxZ = 11.949999809265137;
			break;

		case 90:
			minX = 3.9499988555908203;
			maxX = 12.05000114440918;
			minZ = 6.862500190734863;
			maxZ = 10.965624809265137;
			break;

		case 180:
			minX = 4.934375286102295;
			maxX = 9.237499237060547;
			minZ = 3.9499993324279785;
			maxZ = 12.050000190734863;
			break;

		case 270:
			minX = 4.049999237060547;
			maxX = 11.949999809265137;
			minZ = 4.93437385559082;
			maxZ = 9.237500190734863;
			break;

		case 360:
			minX = 6.8625006675720215;
			maxX = 10.96562385559082;
			minZ = 4.050000190734863;
			maxZ = 11.950000762939453;
			break;

		case 450:
			minX = 3.950000762939453;
			maxX = 12.049999237060547;
			minZ = 6.862499237060547;
			maxZ = 10.965625762939453;
			break;

		case 540:
			minX = 4.934374809265137;
			maxX = 9.237500190734863;
			minZ = 3.9499998092651367;
			maxZ = 12.050000190734863;
			break;

		case 630:
			minX = 4.0500006675720215;
			maxX = 11.94999885559082;
			minZ = 4.93437385559082;
			maxZ = 9.23750114440918;
			break;

		}
		break;
	case ROOM2_5:
		switch (angle) {
		case 0:
			minX = 6.862500190734863;
			maxX = 9.199999809265137;
			minZ = 4.050000190734863;
			maxZ = 11.949999809265137;
			break;

		case 90:
			minX = 3.9499993324279785;
			maxX = 12.05000114440918;
			minZ = 6.862500190734863;
			maxZ = 9.199999809265137;
			break;

		case 180:
			minX = 6.700000286102295;
			maxX = 9.237499237060547;
			minZ = 3.9499998092651367;
			maxZ = 12.050000190734863;
			break;

		case 270:
			minX = 4.049999237060547;
			maxX = 11.950000762939453;
			minZ = 6.6999993324279785;
			maxZ = 9.237500190734863;
			break;

		case 360:
			minX = 6.8625006675720215;
			maxX = 9.19999885559082;
			minZ = 4.050000190734863;
			maxZ = 11.949999809265137;
			break;

		case 450:
			minX = 3.950000286102295;
			maxX = 12.049999237060547;
			minZ = 6.862499237060547;
			maxZ = 9.200000762939453;
			break;

		case 540:
			minX = 6.700000286102295;
			maxX = 9.237500190734863;
			minZ = 3.9499998092651367;
			maxZ = 12.050000190734863;
			break;

		case 630:
			minX = 4.0500006675720215;
			maxX = 11.94999885559082;
			minZ = 6.69999885559082;
			maxZ = 9.23750114440918;
			break;

		}
		break;
	case ROOM2CID:
		switch (angle) {
		case 0:
			minX = 6.800000190734863;
			maxX = 11.949999809265137;
			minZ = 4.050000190734863;
			maxZ = 9.199999809265137;
			break;

		case 90:
			minX = 6.6999993324279785;
			maxX = 12.05000114440918;
			minZ = 6.800000190734863;
			maxZ = 11.950000762939453;
			break;

		case 180:
			minX = 3.9499998092651367;
			maxX = 9.299999237060547;
			minZ = 6.699999809265137;
			maxZ = 12.050000190734863;
			break;

		case 270:
			minX = 4.049999237060547;
			maxX = 9.199999809265137;
			minZ = 3.9499993324279785;
			maxZ = 9.300000190734863;
			break;

		case 360:
			minX = 6.8000006675720215;
			maxX = 11.949999809265137;
			minZ = 4.050000190734863;
			maxZ = 9.200000762939453;
			break;

		case 450:
			minX = 6.700000762939453;
			maxX = 12.049999237060547;
			minZ = 6.799999237060547;
			maxZ = 11.949999809265137;
			break;

		case 540:
			minX = 3.9499998092651367;
			maxX = 9.300000190734863;
			minZ = 6.699999809265137;
			maxZ = 12.050000190734863;
			break;

		case 630:
			minX = 4.0500006675720215;
			maxX = 9.19999885559082;
			minZ = 3.9499998092651367;
			maxZ = 9.30000114440918;
			break;

		}
		break;
	case ROOM2C2:
		switch (angle) {
		case 0:
			minX = 6.800000190734863;
			maxX = 11.949999809265137;
			minZ = 4.050000190734863;
			maxZ = 11.949999809265137;
			break;

		case 90:
			minX = 3.9499993324279785;
			maxX = 12.05000114440918;
			minZ = 6.800000190734863;
			maxZ = 11.949999809265137;
			break;

		case 180:
			minX = 3.950000286102295;
			maxX = 9.299999237060547;
			minZ = 3.9499998092651367;
			maxZ = 12.050000190734863;
			break;

		case 270:
			minX = 4.049999237060547;
			maxX = 11.949999809265137;
			minZ = 3.9499988555908203;
			maxZ = 9.300000190734863;
			break;

		case 360:
			minX = 6.8000006675720215;
			maxX = 11.94999885559082;
			minZ = 4.050000190734863;
			maxZ = 11.950000762939453;
			break;

		case 450:
			minX = 3.9500012397766113;
			maxX = 12.049999237060547;
			minZ = 6.799999237060547;
			maxZ = 11.950000762939453;
			break;

		case 540:
			minX = 3.9499998092651367;
			maxX = 9.300000190734863;
			minZ = 3.950000286102295;
			maxZ = 12.050000190734863;
			break;

		case 630:
			minX = 4.0500006675720215;
			maxX = 11.949997901916504;
			minZ = 3.9499988555908203;
			maxZ = 9.30000114440918;
			break;

		}
		break;
	case ROOM2CLOSETS:
		switch (angle) {
		case 0:
			minX = 0.3468751907348633;
			maxX = 11.137499809265137;
			minZ = 4.050000190734863;
			maxZ = 11.949999809265137;
			break;

		case 90:
			minX = 3.9499988555908203;
			maxX = 12.05000114440918;
			minZ = 0.3468747138977051;
			maxZ = 11.137499809265137;
			break;

		case 180:
			minX = 4.762500286102295;
			maxX = 15.753124237060547;
			minZ = 3.9499993324279785;
			maxZ = 12.05000114440918;
			break;

		case 270:
			minX = 4.050000190734863;
			maxX = 11.949999809265137;
			minZ = 4.76249885559082;
			maxZ = 15.75312614440918;
			break;

		case 360:
			minX = 0.3468756675720215;
			maxX = 11.13749885559082;
			minZ = 4.0499982833862305;
			maxZ = 11.950000762939453;
			break;

		case 450:
			minX = 3.950000762939453;
			maxX = 12.04999828338623;
			minZ = 0.3468751907348633;
			maxZ = 11.137500762939453;
			break;

		case 540:
			minX = 4.762499809265137;
			maxX = 15.753125190734863;
			minZ = 3.9499998092651367;
			maxZ = 12.050000190734863;
			break;

		case 630:
			minX = 4.0500030517578125;
			maxX = 11.94999885559082;
			minZ = 4.76249885559082;
			maxZ = 15.75312614440918;
			break;

		}
		break;
	case ROOM2ELEVATOR:
		switch (angle) {
		case 0:
			minX = 6.862499713897705;
			maxX = 12.075000762939453;
			minZ = 4.050000190734863;
			maxZ = 11.950000762939453;
			break;

		case 90:
			minX = 3.949997901916504;
			maxX = 12.05000114440918;
			minZ = 6.862500190734863;
			maxZ = 12.075000762939453;
			break;

		case 180:
			minX = 3.8249993324279785;
			maxX = 9.237500190734863;
			minZ = 3.949998378753662;
			maxZ = 12.050000190734863;
			break;

		case 270:
			minX = 4.049999237060547;
			maxX = 11.950000762939453;
			minZ = 3.824997901916504;
			maxZ = 9.23750114440918;
			break;

		case 360:
			minX = 6.862500190734863;
			maxX = 12.075000762939453;
			minZ = 4.050000190734863;
			maxZ = 11.95000171661377;
			break;

		case 450:
			minX = 3.950000286102295;
			maxX = 12.049999237060547;
			minZ = 6.862499237060547;
			maxZ = 12.075000762939453;
			break;

		case 540:
			minX = 3.8249988555908203;
			maxX = 9.237500190734863;
			minZ = 3.9499988555908203;
			maxZ = 12.050000190734863;
			break;

		case 630:
			minX = 4.0500006675720215;
			maxX = 11.94999885559082;
			minZ = 3.824997901916504;
			maxZ = 9.23750114440918;
			break;

		}
		break;
	case ROOM2DOORS:
		switch (angle) {
		case 0:
			minX = 4.050000190734863;
			maxX = 8.949999809265137;
			minZ = 4.050000190734863;
			maxZ = 11.949999809265137;
			break;

		case 90:
			minX = 3.9499993324279785;
			maxX = 12.05000114440918;
			minZ = 4.050000190734863;
			maxZ = 8.949999809265137;
			break;

		case 180:
			minX = 6.950000286102295;
			maxX = 12.049999237060547;
			minZ = 3.9499998092651367;
			maxZ = 12.05000114440918;
			break;

		case 270:
			minX = 4.050000190734863;
			maxX = 11.950000762939453;
			minZ = 6.9499993324279785;
			maxZ = 12.05000114440918;
			break;

		case 360:
			minX = 4.0500006675720215;
			maxX = 8.94999885559082;
			minZ = 4.049999237060547;
			maxZ = 11.949999809265137;
			break;

		case 450:
			minX = 3.950000286102295;
			maxX = 12.049999237060547;
			minZ = 4.049999237060547;
			maxZ = 8.950000762939453;
			break;

		case 540:
			minX = 6.950000286102295;
			maxX = 12.050000190734863;
			minZ = 3.9499998092651367;
			maxZ = 12.050000190734863;
			break;

		case 630:
			minX = 4.050001621246338;
			maxX = 11.949999809265137;
			minZ = 6.94999885559082;
			maxZ = 12.05000114440918;
			break;

		}
		break;
	case ROOM2SCPS:
		switch (angle) {
		case 0:
			minX = 4.862499237060547;
			maxX = 11.137500762939453;
			minZ = 4.050000190734863;
			maxZ = 11.949999809265137;
			break;

		case 90:
			minX = 3.9499988555908203;
			maxX = 12.05000114440918;
			minZ = 4.862499237060547;
			maxZ = 11.137500762939453;
			break;

		case 180:
			minX = 4.762499809265137;
			maxX = 11.237500190734863;
			minZ = 3.9499993324279785;
			maxZ = 12.05000114440918;
			break;

		case 270:
			minX = 4.050000190734863;
			maxX = 11.949999809265137;
			minZ = 4.762498378753662;
			maxZ = 11.23750114440918;
			break;

		case 360:
			minX = 4.862500190734863;
			maxX = 11.137499809265137;
			minZ = 4.049999237060547;
			maxZ = 11.950000762939453;
			break;

		case 450:
			minX = 3.950000762939453;
			maxX = 12.049999237060547;
			minZ = 4.862499237060547;
			maxZ = 11.137500762939453;
			break;

		case 540:
			minX = 4.7624993324279785;
			maxX = 11.237500190734863;
			minZ = 3.9499998092651367;
			maxZ = 12.050000190734863;
			break;

		case 630:
			minX = 4.05000114440918;
			maxX = 11.94999885559082;
			minZ = 4.762497901916504;
			maxZ = 11.23750114440918;
			break;

		}
		break;
	case ROOM860:
		switch (angle) {
		case 0:
			minX = 6.862500190734863;
			maxX = 12.949999809265137;
			minZ = 4.050000190734863;
			maxZ = 11.949999809265137;
			break;

		case 90:
			minX = 3.9499988555908203;
			maxX = 12.05000114440918;
			minZ = 6.862500190734863;
			maxZ = 12.949999809265137;
			break;

		case 180:
			minX = 2.950000286102295;
			maxX = 9.237499237060547;
			minZ = 3.9499993324279785;
			maxZ = 12.050000190734863;
			break;

		case 270:
			minX = 4.049999237060547;
			maxX = 11.949999809265137;
			minZ = 2.9499988555908203;
			maxZ = 9.237500190734863;
			break;

		case 360:
			minX = 6.8625006675720215;
			maxX = 12.94999885559082;
			minZ = 4.050000190734863;
			maxZ = 11.950000762939453;
			break;

		case 450:
			minX = 3.9500012397766113;
			maxX = 12.049999237060547;
			minZ = 6.862499237060547;
			maxZ = 12.949999809265137;
			break;

		case 540:
			minX = 2.9499998092651367;
			maxX = 9.237500190734863;
			minZ = 3.9499998092651367;
			maxZ = 12.050000190734863;
			break;

		case 630:
			minX = 4.0500006675720215;
			maxX = 11.949997901916504;
			minZ = 2.9499988555908203;
			maxZ = 9.23750114440918;
			break;

		}
		break;
	case ROOM2TESTROOM2:
		switch (angle) {
		case 0:
			minX = 4.049999237060547;
			maxX = 9.324999809265137;
			minZ = 4.050000190734863;
			maxZ = 11.949999809265137;
			break;

		case 90:
			minX = 3.9499993324279785;
			maxX = 12.05000114440918;
			minZ = 4.049999237060547;
			maxZ = 9.324999809265137;
			break;

		case 180:
			minX = 6.575000286102295;
			maxX = 12.05000114440918;
			minZ = 3.9499998092651367;
			maxZ = 12.05000114440918;
			break;

		case 270:
			minX = 4.050000190734863;
			maxX = 11.950000762939453;
			minZ = 6.5749993324279785;
			maxZ = 12.050002098083496;
			break;

		case 360:
			minX = 4.049999237060547;
			maxX = 9.32499885559082;
			minZ = 4.049999237060547;
			maxZ = 11.950000762939453;
			break;

		case 450:
			minX = 3.950000286102295;
			maxX = 12.049999237060547;
			minZ = 4.049999237060547;
			maxZ = 9.325000762939453;
			break;

		case 540:
			minX = 6.575000286102295;
			maxX = 12.05000114440918;
			minZ = 3.9499998092651367;
			maxZ = 12.050000190734863;
			break;

		case 630:
			minX = 4.050001621246338;
			maxX = 11.94999885559082;
			minZ = 6.57499885559082;
			maxZ = 12.050002098083496;
			break;

		}
		break;
	case ROOM3ID:
		switch (angle) {
		case 0:
			minX = 4.050000190734863;
			maxX = 11.949999809265137;
			minZ = 4.050000190734863;
			maxZ = 11.949999809265137;
			break;

		case 90:
			minX = 3.9499993324279785;
			maxX = 12.05000114440918;
			minZ = 4.050000190734863;
			maxZ = 11.949999809265137;
			break;

		case 180:
			minX = 3.950000286102295;
			maxX = 12.049999237060547;
			minZ = 3.9499998092651367;
			maxZ = 12.05000114440918;
			break;

		case 270:
			minX = 4.050000190734863;
			maxX = 11.949999809265137;
			minZ = 3.9499988555908203;
			maxZ = 12.05000114440918;
			break;

		case 360:
			minX = 4.0500006675720215;
			maxX = 11.94999885559082;
			minZ = 4.049999237060547;
			maxZ = 11.950000762939453;
			break;

		case 450:
			minX = 3.9500012397766113;
			maxX = 12.049999237060547;
			minZ = 4.049999237060547;
			maxZ = 11.950000762939453;
			break;

		case 540:
			minX = 3.9499998092651367;
			maxX = 12.050000190734863;
			minZ = 3.950000286102295;
			maxZ = 12.050000190734863;
			break;

		case 630:
			minX = 4.050001621246338;
			maxX = 11.949997901916504;
			minZ = 3.9499988555908203;
			maxZ = 12.05000114440918;
			break;

		}
		break;
	case ROOM3_2:
		switch (angle) {
		case 0:
			minX = 4.050000190734863;
			maxX = 11.949999809265137;
			minZ = 4.050000190734863;
			maxZ = 9.700000762939453;
			break;

		case 90:
			minX = 6.19999885559082;
			maxX = 12.05000114440918;
			minZ = 4.050000190734863;
			maxZ = 11.950000762939453;
			break;

		case 180:
			minX = 3.950000286102295;
			maxX = 12.049999237060547;
			minZ = 6.1999993324279785;
			maxZ = 12.05000114440918;
			break;

		case 270:
			minX = 4.050000190734863;
			maxX = 9.699999809265137;
			minZ = 3.9499993324279785;
			maxZ = 12.05000114440918;
			break;

		case 360:
			minX = 4.0500006675720215;
			maxX = 11.949999809265137;
			minZ = 4.049999237060547;
			maxZ = 9.700000762939453;
			break;

		case 450:
			minX = 6.200000762939453;
			maxX = 12.049999237060547;
			minZ = 4.049999237060547;
			maxZ = 11.949999809265137;
			break;

		case 540:
			minX = 3.9499998092651367;
			maxX = 12.050000190734863;
			minZ = 6.199999809265137;
			maxZ = 12.050000190734863;
			break;

		case 630:
			minX = 4.050001621246338;
			maxX = 9.69999885559082;
			minZ = 3.9499993324279785;
			maxZ = 12.05000114440918;
			break;

		}
		break;
	case ROOM4ID:
		switch (angle) {
		case 0:
			minX = 4.050000190734863;
			maxX = 11.949999809265137;
			minZ = 4.050000190734863;
			maxZ = 11.949999809265137;
			break;

		case 90:
			minX = 3.9499988555908203;
			maxX = 12.05000114440918;
			minZ = 4.050000190734863;
			maxZ = 11.949999809265137;
			break;

		case 180:
			minX = 3.950000286102295;
			maxX = 12.049999237060547;
			minZ = 3.9499993324279785;
			maxZ = 12.05000114440918;
			break;

		case 270:
			minX = 4.050000190734863;
			maxX = 11.949999809265137;
			minZ = 3.9499988555908203;
			maxZ = 12.05000114440918;
			break;

		case 360:
			minX = 4.0500006675720215;
			maxX = 11.94999885559082;
			minZ = 4.049999237060547;
			maxZ = 11.950000762939453;
			break;

		case 450:
			minX = 3.950000762939453;
			maxX = 12.049999237060547;
			minZ = 4.049999237060547;
			maxZ = 11.950000762939453;
			break;

		case 540:
			minX = 3.9499998092651367;
			maxX = 12.050000190734863;
			minZ = 3.9499998092651367;
			maxZ = 12.050000190734863;
			break;

		case 630:
			minX = 4.050001621246338;
			maxX = 11.94999885559082;
			minZ = 3.9499988555908203;
			maxZ = 12.05000114440918;
			break;

		}
		break;
	case ROOM4_2:
		switch (angle) {
		case 0:
			minX = 4.050000190734863;
			maxX = 11.949999809265137;
			minZ = 4.050000190734863;
			maxZ = 11.949999809265137;
			break;

		case 90:
			minX = 3.9499988555908203;
			maxX = 12.05000114440918;
			minZ = 4.050000190734863;
			maxZ = 11.949999809265137;
			break;

		case 180:
			minX = 3.950000286102295;
			maxX = 12.049999237060547;
			minZ = 3.9499993324279785;
			maxZ = 12.05000114440918;
			break;

		case 270:
			minX = 4.050000190734863;
			maxX = 11.949999809265137;
			minZ = 3.9499988555908203;
			maxZ = 12.05000114440918;
			break;

		case 360:
			minX = 4.0500006675720215;
			maxX = 11.94999885559082;
			minZ = 4.049999237060547;
			maxZ = 11.950000762939453;
			break;

		case 450:
			minX = 3.950000762939453;
			maxX = 12.049999237060547;
			minZ = 4.049999237060547;
			maxZ = 11.950000762939453;
			break;

		case 540:
			minX = 3.9499998092651367;
			maxX = 12.050000190734863;
			minZ = 3.9499998092651367;
			maxZ = 12.050000190734863;
			break;

		case 630:
			minX = 4.050001621246338;
			maxX = 11.94999885559082;
			minZ = 3.9499988555908203;
			maxZ = 12.05000114440918;
			break;

		}
		break;
	case ROOMPJ:
		switch (angle) {
		case 0:
			minX = 4.050000190734863;
			maxX = 11.949999809265137;
			minZ = 4.050000190734863;
			maxZ = 12.949999809265137;
			break;

		case 90:
			minX = 2.9499988555908203;
			maxX = 12.05000114440918;
			minZ = 4.050000190734863;
			maxZ = 11.949999809265137;
			break;

		case 180:
			minX = 3.950000286102295;
			maxX = 12.049999237060547;
			minZ = 2.9499993324279785;
			maxZ = 12.05000114440918;
			break;

		case 270:
			minX = 4.050000190734863;
			maxX = 12.949999809265137;
			minZ = 3.9499988555908203;
			maxZ = 12.05000114440918;
			break;

		case 360:
			minX = 4.0500006675720215;
			maxX = 11.94999885559082;
			minZ = 4.049999237060547;
			maxZ = 12.950000762939453;
			break;

		case 450:
			minX = 2.950000762939453;
			maxX = 12.049999237060547;
			minZ = 4.049999237060547;
			maxZ = 11.950000762939453;
			break;

		case 540:
			minX = 3.950000286102295;
			maxX = 12.050000190734863;
			minZ = 2.9499998092651367;
			maxZ = 12.050000190734863;
			break;

		case 630:
			minX = 4.050001621246338;
			maxX = 12.949997901916504;
			minZ = 3.949998378753662;
			maxZ = 12.05000114440918;
			break;

		}
		break;
	case 914:
		switch (angle) {
		case 0:
			minX = 4.049999237060547;
			maxX = 11.950000762939453;
			minZ = 4.050000190734863;
			maxZ = 11.950000762939453;
			break;

		case 90:
			minX = 3.949998378753662;
			maxX = 12.05000114440918;
			minZ = 4.049999237060547;
			maxZ = 11.950000762939453;
			break;

		case 180:
			minX = 3.9499998092651367;
			maxX = 12.050000190734863;
			minZ = 3.9499988555908203;
			maxZ = 12.05000114440918;
			break;

		case 270:
			minX = 4.050000190734863;
			maxX = 11.950000762939453;
			minZ = 3.949998378753662;
			maxZ = 12.05000114440918;
			break;

		case 360:
			minX = 4.050000190734863;
			maxX = 11.949999809265137;
			minZ = 4.049999237060547;
			maxZ = 11.950000762939453;
			break;

		case 450:
			minX = 3.950000762939453;
			maxX = 12.049999237060547;
			minZ = 4.049999237060547;
			maxZ = 11.950000762939453;
			break;

		case 540:
			minX = 3.9499993324279785;
			maxX = 12.05000114440918;
			minZ = 3.9499993324279785;
			maxZ = 12.050000190734863;
			break;

		case 630:
			minX = 4.050001621246338;
			maxX = 11.94999885559082;
			minZ = 3.949998378753662;
			maxZ = 12.05000114440918;
			break;

		}
		break;
	case ROOM2GW:
		switch (angle) {
		case 0:
			minX = 5.925000190734863;
			maxX = 10.074999809265137;
			minZ = 4.00312614440918;
			maxZ = 11.949999809265137;
			break;

		case 90:
			minX = 3.9499988555908203;
			maxX = 12.096875190734863;
			minZ = 5.925000190734863;
			maxZ = 10.074999809265137;
			break;

		case 180:
			minX = 5.825000286102295;
			maxX = 10.174999237060547;
			minZ = 3.9499998092651367;
			maxZ = 12.096874237060547;
			break;

		case 270:
			minX = 4.00312614440918;
			maxX = 11.949999809265137;
			minZ = 5.8249993324279785;
			maxZ = 10.17500114440918;
			break;

		case 360:
			minX = 5.9250006675720215;
			maxX = 10.07499885559082;
			minZ = 4.003125190734863;
			maxZ = 11.950000762939453;
			break;

		case 450:
			minX = 3.950000762939453;
			maxX = 12.09687328338623;
			minZ = 5.924999237060547;
			maxZ = 10.075000762939453;
			break;

		case 540:
			minX = 5.824999809265137;
			maxX = 10.175000190734863;
			minZ = 3.9499998092651367;
			maxZ = 12.096874237060547;
			break;

		case 630:
			minX = 4.003127098083496;
			maxX = 11.94999885559082;
			minZ = 5.82499885559082;
			maxZ = 10.17500114440918;
			break;

		}
		break;
	case ROOM2GW_B:
		switch (angle) {
		case 0:
			minX = 6.167187213897705;
			maxX = 9.844531059265137;
			minZ = 4.010937690734863;
			maxZ = 11.949999809265137;
			break;

		case 90:
			minX = 3.9499993324279785;
			maxX = 12.08906364440918;
			minZ = 6.167187690734863;
			maxZ = 9.844531059265137;
			break;

		case 180:
			minX = 6.055469036102295;
			maxX = 9.932811737060547;
			minZ = 3.9499998092651367;
			maxZ = 12.089062690734863;
			break;

		case 270:
			minX = 4.010936737060547;
			maxX = 11.950000762939453;
			minZ = 6.0554680824279785;
			maxZ = 9.93281364440918;
			break;

		case 360:
			minX = 6.1671881675720215;
			maxX = 9.84453010559082;
			minZ = 4.010936737060547;
			maxZ = 11.950000762939453;
			break;

		case 450:
			minX = 3.950000286102295;
			maxX = 12.089061737060547;
			minZ = 6.167186737060547;
			maxZ = 9.844532012939453;
			break;

		case 540:
			minX = 6.055469036102295;
			maxX = 9.932812690734863;
			minZ = 3.9499998092651367;
			maxZ = 12.089062690734863;
			break;

		case 630:
			minX = 4.01093864440918;
			maxX = 11.94999885559082;
			minZ = 6.05546760559082;
			maxZ = 9.93281364440918;
			break;

		}
		break;
	case ROOM1162:
		switch (angle) {
		case 0:
			minX = 6.800000190734863;
			maxX = 11.949999809265137;
			minZ = 4.035148620605469;
			maxZ = 9.200000762939453;
			break;

		case 90:
			minX = 6.6999993324279785;
			maxX = 12.064851760864258;
			minZ = 6.800000190734863;
			maxZ = 11.950000762939453;
			break;

		case 180:
			minX = 3.9499998092651367;
			maxX = 9.299999237060547;
			minZ = 6.6999993324279785;
			maxZ = 12.064851760864258;
			break;

		case 270:
			minX = 4.035148620605469;
			maxX = 9.199999809265137;
			minZ = 3.9499993324279785;
			maxZ = 9.300000190734863;
			break;

		case 360:
			minX = 6.8000006675720215;
			maxX = 11.949999809265137;
			minZ = 4.035148620605469;
			maxZ = 9.200000762939453;
			break;

		case 450:
			minX = 6.700000762939453;
			maxX = 12.064850807189941;
			minZ = 6.799999237060547;
			maxZ = 11.949999809265137;
			break;

		case 540:
			minX = 3.9499998092651367;
			maxX = 9.300000190734863;
			minZ = 6.699999809265137;
			maxZ = 12.064851760864258;
			break;

		case 630:
			minX = 4.035149574279785;
			maxX = 9.19999885559082;
			minZ = 3.9499998092651367;
			maxZ = 9.30000114440918;
			break;

		}
		break;
	case ROOM2SCPS2:
		switch (angle) {
		case 0:
			minX = 6.862500190734863;
			maxX = 12.887499809265137;
			minZ = 4.050000190734863;
			maxZ = 11.958984375;
			break;

		case 90:
			minX = 3.941014289855957;
			maxX = 12.05000114440918;
			minZ = 6.862500190734863;
			maxZ = 12.887499809265137;
			break;

		case 180:
			minX = 3.012500286102295;
			maxX = 9.237499237060547;
			minZ = 3.9410147666931152;
			maxZ = 12.050000190734863;
			break;

		case 270:
			minX = 4.049999237060547;
			maxX = 11.958984375;
			minZ = 3.0124988555908203;
			maxZ = 9.237500190734863;
			break;

		case 360:
			minX = 6.8625006675720215;
			maxX = 12.88749885559082;
			minZ = 4.050000190734863;
			maxZ = 11.958985328674316;
			break;

		case 450:
			minX = 3.941016674041748;
			maxX = 12.049999237060547;
			minZ = 6.862499237060547;
			maxZ = 12.887500762939453;
			break;

		case 540:
			minX = 3.0124998092651367;
			maxX = 9.237500190734863;
			minZ = 3.9410152435302734;
			maxZ = 12.050000190734863;
			break;

		case 630:
			minX = 4.0500006675720215;
			maxX = 11.958982467651367;
			minZ = 3.0124988555908203;
			maxZ = 9.23750114440918;
			break;

		}
		break;
	case ROOM2SL:
		switch (angle) {
		case 0:
			minX = 6.862500190734863;
			maxX = 14.949999809265137;
			minZ = 4.049999237060547;
			maxZ = 11.950000762939453;
			break;

		case 90:
			minX = 3.949997901916504;
			maxX = 12.05000114440918;
			minZ = 6.862500190734863;
			maxZ = 14.950000762939453;
			break;

		case 180:
			minX = 0.9500002861022949;
			maxX = 9.237499237060547;
			minZ = 3.9499988555908203;
			maxZ = 12.05000114440918;
			break;

		case 270:
			minX = 4.049999237060547;
			maxX = 11.949999809265137;
			minZ = 0.9499983787536621;
			maxZ = 9.237500190734863;
			break;

		case 360:
			minX = 6.8625006675720215;
			maxX = 14.94999885559082;
			minZ = 4.049999237060547;
			maxZ = 11.95000171661377;
			break;

		case 450:
			minX = 3.9500012397766113;
			maxX = 12.049999237060547;
			minZ = 6.862499237060547;
			maxZ = 14.950000762939453;
			break;

		case 540:
			minX = 0.9499998092651367;
			maxX = 9.237500190734863;
			minZ = 3.9499993324279785;
			maxZ = 12.05000114440918;
			break;

		case 630:
			minX = 4.050000190734863;
			maxX = 11.949997901916504;
			minZ = 0.9499988555908203;
			maxZ = 9.23750114440918;
			break;

		}
		break;
	case LOCKROOM3:
		switch (angle) {
		case 0:
			minX = 4.800000190734863;
			maxX = 11.949999809265137;
			minZ = 4.026562690734863;
			maxZ = 11.324999809265137;
			break;

		case 90:
			minX = 4.57499885559082;
			maxX = 12.07343864440918;
			minZ = 4.800000190734863;
			maxZ = 11.949999809265137;
			break;

		case 180:
			minX = 3.950000286102295;
			maxX = 11.299999237060547;
			minZ = 4.574999809265137;
			maxZ = 12.07343864440918;
			break;

		case 270:
			minX = 4.026562690734863;
			maxX = 11.324999809265137;
			minZ = 3.9499988555908203;
			maxZ = 11.30000114440918;
			break;

		case 360:
			minX = 4.8000006675720215;
			maxX = 11.949999809265137;
			minZ = 4.026561737060547;
			maxZ = 11.325000762939453;
			break;

		case 450:
			minX = 4.575000762939453;
			maxX = 12.073436737060547;
			minZ = 4.799999237060547;
			maxZ = 11.950000762939453;
			break;

		case 540:
			minX = 3.9499998092651367;
			maxX = 11.300000190734863;
			minZ = 4.574999809265137;
			maxZ = 12.073437690734863;
			break;

		case 630:
			minX = 4.02656364440918;
			maxX = 11.32499885559082;
			minZ = 3.9499988555908203;
			maxZ = 11.30000114440918;
			break;

		}
		break;
	case ROOM4INFO:
		switch (angle) {
		case 0:
			minX = 4.050000190734863;
			maxX = 11.949999809265137;
			minZ = 4.050000190734863;
			maxZ = 11.950000762939453;
			break;

		case 90:
			minX = 3.949998378753662;
			maxX = 12.05000114440918;
			minZ = 4.050000190734863;
			maxZ = 11.949999809265137;
			break;

		case 180:
			minX = 3.950000286102295;
			maxX = 12.049999237060547;
			minZ = 3.9499988555908203;
			maxZ = 12.05000114440918;
			break;

		case 270:
			minX = 4.050000190734863;
			maxX = 11.950000762939453;
			minZ = 3.9499988555908203;
			maxZ = 12.05000114440918;
			break;

		case 360:
			minX = 4.0500006675720215;
			maxX = 11.94999885559082;
			minZ = 4.049999237060547;
			maxZ = 11.950000762939453;
			break;

		case 450:
			minX = 3.950000762939453;
			maxX = 12.049999237060547;
			minZ = 4.049999237060547;
			maxZ = 11.950000762939453;
			break;

		case 540:
			minX = 3.9499998092651367;
			maxX = 12.050000190734863;
			minZ = 3.9499993324279785;
			maxZ = 12.050000190734863;
			break;

		case 630:
			minX = 4.050001621246338;
			maxX = 11.94999885559082;
			minZ = 3.949998378753662;
			maxZ = 12.05000114440918;
			break;

		}
		break;
	case ROOM3_3:
		switch (angle) {
		case 0:
			minX = 4.050000190734863;
			maxX = 11.953195571899414;
			minZ = 4.0497331619262695;
			maxZ = 9.574999809265137;
			break;

		case 90:
			minX = 6.3249993324279785;
			maxX = 12.050268173217773;
			minZ = 4.050000190734863;
			maxZ = 11.95319652557373;
			break;

		case 180:
			minX = 3.9468040466308594;
			maxX = 12.049999237060547;
			minZ = 6.324999809265137;
			maxZ = 12.050267219543457;
			break;

		case 270:
			minX = 4.0497331619262695;
			maxX = 9.574999809265137;
			minZ = 3.946803569793701;
			maxZ = 12.05000114440918;
			break;

		case 360:
			minX = 4.0500006675720215;
			maxX = 11.953195571899414;
			minZ = 4.0497331619262695;
			maxZ = 9.575000762939453;
			break;

		case 450:
			minX = 6.325000762939453;
			maxX = 12.050265312194824;
			minZ = 4.049999237060547;
			maxZ = 11.953195571899414;
			break;

		case 540:
			minX = 3.9468040466308594;
			maxX = 12.050000190734863;
			minZ = 6.324999809265137;
			maxZ = 12.05026626586914;
			break;

		case 630:
			minX = 4.049735069274902;
			maxX = 9.57499885559082;
			minZ = 3.9468040466308594;
			maxZ = 12.05000114440918;
			break;

		}
		break;
	case CHECKPOINT1:
		switch (angle) {
		case 0:
			minX = 3.7375001907348633;
			maxX = 12.254687309265137;
			minZ = 4.049999237060547;
			maxZ = 11.950000762939453;
			break;

		case 90:
			minX = 3.949998378753662;
			maxX = 12.05000114440918;
			minZ = 3.7375001907348633;
			maxZ = 12.254687309265137;
			break;

		case 180:
			minX = 3.645312786102295;
			maxX = 12.362499237060547;
			minZ = 3.9499988555908203;
			maxZ = 12.05000114440918;
			break;

		case 270:
			minX = 4.049999237060547;
			maxX = 11.950000762939453;
			minZ = 3.6453113555908203;
			maxZ = 12.36250114440918;
			break;

		case 360:
			minX = 3.7375006675720215;
			maxX = 12.25468635559082;
			minZ = 4.0499982833862305;
			maxZ = 11.95000171661377;
			break;

		case 450:
			minX = 3.950000762939453;
			maxX = 12.049999237060547;
			minZ = 3.737499713897705;
			maxZ = 12.254688262939453;
			break;

		case 540:
			minX = 3.6453123092651367;
			maxX = 12.362500190734863;
			minZ = 3.9499993324279785;
			maxZ = 12.05000114440918;
			break;

		case 630:
			minX = 4.05000114440918;
			maxX = 11.94999885559082;
			minZ = 3.6453113555908203;
			maxZ = 12.36250114440918;
			break;

		}
		break;
	case 008:
		switch (angle) {
		case 0:
			minX = 5.675000190734863;
			maxX = 11.012499809265137;
			minZ = 4.050000190734863;
			maxZ = 11.199999809265137;
			break;

		case 90:
			minX = 4.69999885559082;
			maxX = 12.05000114440918;
			minZ = 5.675000190734863;
			maxZ = 11.012499809265137;
			break;

		case 180:
			minX = 4.887500286102295;
			maxX = 10.424999237060547;
			minZ = 4.699999809265137;
			maxZ = 12.050000190734863;
			break;

		case 270:
			minX = 4.050000190734863;
			maxX = 11.199999809265137;
			minZ = 4.88749885559082;
			maxZ = 10.42500114440918;
			break;

		case 360:
			minX = 5.6750006675720215;
			maxX = 11.012499809265137;
			minZ = 4.049999237060547;
			maxZ = 11.200000762939453;
			break;

		case 450:
			minX = 4.700000762939453;
			maxX = 12.049999237060547;
			minZ = 5.674999237060547;
			maxZ = 11.012500762939453;
			break;

		case 540:
			minX = 4.887499809265137;
			maxX = 10.425000190734863;
			minZ = 4.699999809265137;
			maxZ = 12.050000190734863;
			break;

		case 630:
			minX = 4.05000114440918;
			maxX = 11.19999885559082;
			minZ = 4.88749885559082;
			maxZ = 10.42500114440918;
			break;

		}
		break;
	case ROOM035:
		switch (angle) {
		case 0:
			minX = 5.175000190734863;
			maxX = 12.824999809265137;
			minZ = 4.050000190734863;
			maxZ = 11.512499809265137;
			break;

		case 90:
			minX = 4.38749885559082;
			maxX = 12.05000114440918;
			minZ = 5.175000190734863;
			maxZ = 12.824999809265137;
			break;

		case 180:
			minX = 3.075000286102295;
			maxX = 10.924999237060547;
			minZ = 4.387499809265137;
			maxZ = 12.05000114440918;
			break;

		case 270:
			minX = 4.050000190734863;
			maxX = 11.512499809265137;
			minZ = 3.0749988555908203;
			maxZ = 10.92500114440918;
			break;

		case 360:
			minX = 5.1750006675720215;
			maxX = 12.82499885559082;
			minZ = 4.049999237060547;
			maxZ = 11.512500762939453;
			break;

		case 450:
			minX = 4.387501239776611;
			maxX = 12.049999237060547;
			minZ = 5.174999237060547;
			maxZ = 12.824999809265137;
			break;

		case 540:
			minX = 3.0749998092651367;
			maxX = 10.925000190734863;
			minZ = 4.387499809265137;
			maxZ = 12.050000190734863;
			break;

		case 630:
			minX = 4.05000114440918;
			maxX = 11.512497901916504;
			minZ = 3.0749988555908203;
			maxZ = 10.92500114440918;
			break;

		}
		break;
	case ROOM106:
		switch (angle) {
		case 0:
			minX = 2.9562501907348633;
			maxX = 16.762500762939453;
			minZ = 4.050000190734863;
			maxZ = 20.137500762939453;
			break;

		case 90:
			minX = -4.2375030517578125;
			maxX = 12.05000114440918;
			minZ = 2.9562501907348633;
			maxZ = 16.76249885559082;
			break;

		case 180:
			minX = -0.8624992370605469;
			maxX = 13.143749237060547;
			minZ = -4.23750114440918;
			maxZ = 12.05000114440918;
			break;

		case 270:
			minX = 4.050000190734863;
			maxX = 20.137500762939453;
			minZ = -0.8625030517578125;
			maxZ = 13.14375114440918;
			break;

		case 360:
			minX = 2.9562506675720215;
			maxX = 16.762496948242188;
			minZ = 4.049999237060547;
			maxZ = 20.137500762939453;
			break;

		case 450:
			minX = -4.237497329711914;
			maxX = 12.049999237060547;
			minZ = 2.9562501907348633;
			maxZ = 16.762500762939453;
			break;

		case 540:
			minX = -0.8625001907348633;
			maxX = 13.143750190734863;
			minZ = -4.237500190734863;
			maxZ = 12.050000190734863;
			break;

		case 630:
			minX = 4.050002098083496;
			maxX = 20.137496948242188;
			minZ = -0.8625040054321289;
			maxZ = 13.14375114440918;
			break;

		}
		break;
	case ROOM513:
		switch (angle) {
		case 0:
			minX = 4.050000190734863;
			maxX = 11.949999809265137;
			minZ = 4.050000190734863;
			maxZ = 11.949999809265137;
			break;

		case 90:
			minX = 3.9499988555908203;
			maxX = 12.05000114440918;
			minZ = 4.050000190734863;
			maxZ = 11.949999809265137;
			break;

		case 180:
			minX = 3.950000286102295;
			maxX = 12.049999237060547;
			minZ = 3.9499993324279785;
			maxZ = 12.05000114440918;
			break;

		case 270:
			minX = 4.050000190734863;
			maxX = 11.949999809265137;
			minZ = 3.9499988555908203;
			maxZ = 12.05000114440918;
			break;

		case 360:
			minX = 4.0500006675720215;
			maxX = 11.94999885559082;
			minZ = 4.049999237060547;
			maxZ = 11.950000762939453;
			break;

		case 450:
			minX = 3.950000762939453;
			maxX = 12.049999237060547;
			minZ = 4.049999237060547;
			maxZ = 11.950000762939453;
			break;

		case 540:
			minX = 3.9499998092651367;
			maxX = 12.050000190734863;
			minZ = 3.9499998092651367;
			maxZ = 12.050000190734863;
			break;

		case 630:
			minX = 4.050001621246338;
			maxX = 11.94999885559082;
			minZ = 3.9499988555908203;
			maxZ = 12.05000114440918;
			break;

		}
		break;
	case COFFIN:
		switch (angle) {
		case 0:
			minX = 4.300000190734863;
			maxX = 9.762499809265137;
			minZ = 4.050000190734863;
			maxZ = 17.950000762939453;
			break;

		case 90:
			minX = -2.0500011444091797;
			maxX = 12.05000114440918;
			minZ = 4.300000190734863;
			maxZ = 9.76249885559082;
			break;

		case 180:
			minX = 6.137500762939453;
			maxX = 11.799999237060547;
			minZ = -2.0500001907348633;
			maxZ = 12.05000114440918;
			break;

		case 270:
			minX = 4.050000190734863;
			maxX = 17.950000762939453;
			minZ = 6.13749885559082;
			maxZ = 11.80000114440918;
			break;

		case 360:
			minX = 4.3000006675720215;
			maxX = 9.762497901916504;
			minZ = 4.049999237060547;
			maxZ = 17.950000762939453;
			break;

		case 450:
			minX = -2.049999237060547;
			maxX = 12.049999237060547;
			minZ = 4.299999237060547;
			maxZ = 9.76250171661377;
			break;

		case 540:
			minX = 6.137500286102295;
			maxX = 11.800000190734863;
			minZ = -2.0500001907348633;
			maxZ = 12.050000190734863;
			break;

		case 630:
			minX = 4.050001621246338;
			maxX = 17.949996948242188;
			minZ = 6.137496471405029;
			maxZ = 11.80000114440918;
			break;

		}
		break;
	case ENDROOM2:
		switch (angle) {
		case 0:
			minX = 6.174999713897705;
			maxX = 9.762499809265137;
			minZ = 4.050000190734863;
			maxZ = 9.418749809265137;
			break;

		case 90:
			minX = 6.481249809265137;
			maxX = 12.05000114440918;
			minZ = 6.175000190734863;
			maxZ = 9.762499809265137;
			break;

		case 180:
			minX = 6.137500286102295;
			maxX = 9.924999237060547;
			minZ = 6.481249809265137;
			maxZ = 12.050000190734863;
			break;

		case 270:
			minX = 4.049999237060547;
			maxX = 9.418749809265137;
			minZ = 6.137499809265137;
			maxZ = 9.92500114440918;
			break;

		case 360:
			minX = 6.1750006675720215;
			maxX = 9.762499809265137;
			minZ = 4.049999237060547;
			maxZ = 9.418750762939453;
			break;

		case 450:
			minX = 6.481250762939453;
			maxX = 12.049999237060547;
			minZ = 6.174999237060547;
			maxZ = 9.762499809265137;
			break;

		case 540:
			minX = 6.137499809265137;
			maxX = 9.925000190734863;
			minZ = 6.481249809265137;
			maxZ = 12.050000190734863;
			break;

		case 630:
			minX = 4.05000114440918;
			maxX = 9.41874885559082;
			minZ = 6.137499809265137;
			maxZ = 9.92500114440918;
			break;

		}
		break;
	case TESTROOM:
		switch (angle) {
		case 0:
			minX = 4.050000190734863;
			maxX = 11.949999809265137;
			minZ = 4.050000190734863;
			maxZ = 11.949999809265137;
			break;

		case 90:
			minX = 3.9499988555908203;
			maxX = 12.05000114440918;
			minZ = 4.050000190734863;
			maxZ = 11.949999809265137;
			break;

		case 180:
			minX = 3.950000286102295;
			maxX = 12.049999237060547;
			minZ = 3.9499993324279785;
			maxZ = 12.05000114440918;
			break;

		case 270:
			minX = 4.050000190734863;
			maxX = 11.949999809265137;
			minZ = 3.9499988555908203;
			maxZ = 12.05000114440918;
			break;

		case 360:
			minX = 4.0500006675720215;
			maxX = 11.94999885559082;
			minZ = 4.049999237060547;
			maxZ = 11.950000762939453;
			break;

		case 450:
			minX = 3.950000762939453;
			maxX = 12.049999237060547;
			minZ = 4.049999237060547;
			maxZ = 11.950000762939453;
			break;

		case 540:
			minX = 3.9499998092651367;
			maxX = 12.050000190734863;
			minZ = 3.9499998092651367;
			maxZ = 12.050000190734863;
			break;

		case 630:
			minX = 4.050001621246338;
			maxX = 11.94999885559082;
			minZ = 3.9499988555908203;
			maxZ = 12.05000114440918;
			break;

		}
		break;
	case TUNNEL:
		switch (angle) {
		case 0:
			minX = 6.925000190734863;
			maxX = 9.074999809265137;
			minZ = 4.050000190734863;
			maxZ = 11.949999809265137;
			break;

		case 90:
			minX = 3.9499993324279785;
			maxX = 12.05000114440918;
			minZ = 6.925000190734863;
			maxZ = 9.074999809265137;
			break;

		case 180:
			minX = 6.825000286102295;
			maxX = 9.174999237060547;
			minZ = 3.9499998092651367;
			maxZ = 12.050000190734863;
			break;

		case 270:
			minX = 4.049999237060547;
			maxX = 11.950000762939453;
			minZ = 6.8249993324279785;
			maxZ = 9.175000190734863;
			break;

		case 360:
			minX = 6.9250006675720215;
			maxX = 9.07499885559082;
			minZ = 4.050000190734863;
			maxZ = 11.949999809265137;
			break;

		case 450:
			minX = 3.950000286102295;
			maxX = 12.049999237060547;
			minZ = 6.924999237060547;
			maxZ = 9.075000762939453;
			break;

		case 540:
			minX = 6.825000286102295;
			maxX = 9.175000190734863;
			minZ = 3.9499998092651367;
			maxZ = 12.050000190734863;
			break;

		case 630:
			minX = 4.0500006675720215;
			maxX = 11.94999885559082;
			minZ = 6.82499885559082;
			maxZ = 9.17500114440918;
			break;

		}
		break;
	case TUNNEL2:
		switch (angle) {
		case 0:
			minX = 6.737500190734863;
			maxX = 10.012499809265137;
			minZ = 4.050000190734863;
			maxZ = 11.949999809265137;
			break;

		case 90:
			minX = 3.9499988555908203;
			maxX = 12.05000114440918;
			minZ = 6.737500190734863;
			maxZ = 10.012499809265137;
			break;

		case 180:
			minX = 5.887500286102295;
			maxX = 9.362499237060547;
			minZ = 3.9499998092651367;
			maxZ = 12.050000190734863;
			break;

		case 270:
			minX = 4.049999237060547;
			maxX = 11.949999809265137;
			minZ = 5.8874993324279785;
			maxZ = 9.362500190734863;
			break;

		case 360:
			minX = 6.7375006675720215;
			maxX = 10.01249885559082;
			minZ = 4.050000190734863;
			maxZ = 11.950000762939453;
			break;

		case 450:
			minX = 3.950000762939453;
			maxX = 12.049999237060547;
			minZ = 6.737499237060547;
			maxZ = 10.012500762939453;
			break;

		case 540:
			minX = 5.887499809265137;
			maxX = 9.362500190734863;
			minZ = 3.9499998092651367;
			maxZ = 12.050000190734863;
			break;

		case 630:
			minX = 4.0500006675720215;
			maxX = 11.94999885559082;
			minZ = 5.88749885559082;
			maxZ = 9.36250114440918;
			break;

		}
		break;
	case ROOM2CTUNNEL:
		switch (angle) {
		case 0:
			minX = 6.550000190734863;
			maxX = 11.949999809265137;
			minZ = 4.050000190734863;
			maxZ = 9.522124290466309;
			break;

		case 90:
			minX = 6.377875328063965;
			maxX = 12.05000114440918;
			minZ = 6.550000190734863;
			maxZ = 11.950000762939453;
			break;

		case 180:
			minX = 3.950000286102295;
			maxX = 9.549999237060547;
			minZ = 6.377875328063965;
			maxZ = 12.050000190734863;
			break;

		case 270:
			minX = 4.049999237060547;
			maxX = 9.522124290466309;
			minZ = 3.9499993324279785;
			maxZ = 9.55000114440918;
			break;

		case 360:
			minX = 6.5500006675720215;
			maxX = 11.949999809265137;
			minZ = 4.049999237060547;
			maxZ = 9.522125244140625;
			break;

		case 450:
			minX = 6.3778767585754395;
			maxX = 12.049999237060547;
			minZ = 6.549999237060547;
			maxZ = 11.949999809265137;
			break;

		case 540:
			minX = 3.9499998092651367;
			maxX = 9.550000190734863;
			minZ = 6.377875804901123;
			maxZ = 12.050000190734863;
			break;

		case 630:
			minX = 4.0500006675720215;
			maxX = 9.522122383117676;
			minZ = 3.9499993324279785;
			maxZ = 9.55000114440918;
			break;

		}
		break;
	case ROOM2NUKE:
		switch (angle) {
		case 0:
			minX = 4.050000190734863;
			maxX = 15.012499809265137;
			minZ = 4.050000190734863;
			maxZ = 11.950000762939453;
			break;

		case 90:
			minX = 3.949997901916504;
			maxX = 12.05000114440918;
			minZ = 4.050000190734863;
			maxZ = 15.012500762939453;
			break;

		case 180:
			minX = 0.8875002861022949;
			maxX = 12.049999237060547;
			minZ = 3.9499988555908203;
			maxZ = 12.05000114440918;
			break;

		case 270:
			minX = 4.050000190734863;
			maxX = 11.949999809265137;
			minZ = 0.8874983787536621;
			maxZ = 12.05000114440918;
			break;

		case 360:
			minX = 4.0500006675720215;
			maxX = 15.01249885559082;
			minZ = 4.049999237060547;
			maxZ = 11.95000171661377;
			break;

		case 450:
			minX = 3.9500012397766113;
			maxX = 12.049999237060547;
			minZ = 4.049999237060547;
			maxZ = 15.012500762939453;
			break;

		case 540:
			minX = 0.8874998092651367;
			maxX = 12.050000190734863;
			minZ = 3.9499993324279785;
			maxZ = 12.050000190734863;
			break;

		case 630:
			minX = 4.050001621246338;
			maxX = 11.949996948242188;
			minZ = 0.8874988555908203;
			maxZ = 12.05000114440918;
			break;

		}
		break;
	case ROOM2PIPES:
		switch (angle) {
		case 0:
			minX = 6.830878734588623;
			maxX = 8.949999809265137;
			minZ = 4.042187690734863;
			maxZ = 11.949999809265137;
			break;

		case 90:
			minX = 3.9499993324279785;
			maxX = 12.05781364440918;
			minZ = 6.830879211425781;
			maxZ = 8.949999809265137;
			break;

		case 180:
			minX = 6.950000286102295;
			maxX = 9.269121170043945;
			minZ = 3.9499998092651367;
			maxZ = 12.057812690734863;
			break;

		case 270:
			minX = 4.042186737060547;
			maxX = 11.950000762939453;
			minZ = 6.9499993324279785;
			maxZ = 9.269122123718262;
			break;

		case 360:
			minX = 6.830879211425781;
			maxX = 8.94999885559082;
			minZ = 4.042187690734863;
			maxZ = 11.949999809265137;
			break;

		case 450:
			minX = 3.950000286102295;
			maxX = 12.057811737060547;
			minZ = 6.830878257751465;
			maxZ = 8.950000762939453;
			break;

		case 540:
			minX = 6.950000286102295;
			maxX = 9.269121170043945;
			minZ = 3.9499998092651367;
			maxZ = 12.057812690734863;
			break;

		case 630:
			minX = 4.04218864440918;
			maxX = 11.949999809265137;
			minZ = 6.94999885559082;
			maxZ = 9.269122123718262;
			break;

		}
		break;
	case ROOM2PIT:
		switch (angle) {
		case 0:
			minX = 4.049999237060547;
			maxX = 10.950000762939453;
			minZ = 4.049999237060547;
			maxZ = 11.950000762939453;
			break;

		case 90:
			minX = 3.949998378753662;
			maxX = 12.05000114440918;
			minZ = 4.049999237060547;
			maxZ = 10.950000762939453;
			break;

		case 180:
			minX = 4.949999809265137;
			maxX = 12.050000190734863;
			minZ = 3.9499988555908203;
			maxZ = 12.05000114440918;
			break;

		case 270:
			minX = 4.049999237060547;
			maxX = 11.950000762939453;
			minZ = 4.94999885559082;
			maxZ = 12.05000114440918;
			break;

		case 360:
			minX = 4.050000190734863;
			maxX = 10.949999809265137;
			minZ = 4.049999237060547;
			maxZ = 11.950000762939453;
			break;

		case 450:
			minX = 3.950000286102295;
			maxX = 12.049999237060547;
			minZ = 4.049999237060547;
			maxZ = 10.950000762939453;
			break;

		case 540:
			minX = 4.949999809265137;
			maxX = 12.05000114440918;
			minZ = 3.9499993324279785;
			maxZ = 12.05000114440918;
			break;

		case 630:
			minX = 4.05000114440918;
			maxX = 11.94999885559082;
			minZ = 4.94999885559082;
			maxZ = 12.05000114440918;
			break;

		}
		break;
	case ROOM3PIT:
		switch (angle) {
		case 0:
			minX = 4.050000190734863;
			maxX = 11.949999809265137;
			minZ = 4.050000190734863;
			maxZ = 9.324999809265137;
			break;

		case 90:
			minX = 6.5749993324279785;
			maxX = 12.05000114440918;
			minZ = 4.050000190734863;
			maxZ = 11.950000762939453;
			break;

		case 180:
			minX = 3.950000286102295;
			maxX = 12.049999237060547;
			minZ = 6.574999809265137;
			maxZ = 12.05000114440918;
			break;

		case 270:
			minX = 4.050000190734863;
			maxX = 9.324999809265137;
			minZ = 3.9499993324279785;
			maxZ = 12.05000114440918;
			break;

		case 360:
			minX = 4.0500006675720215;
			maxX = 11.949999809265137;
			minZ = 4.049999237060547;
			maxZ = 9.325000762939453;
			break;

		case 450:
			minX = 6.575000762939453;
			maxX = 12.049999237060547;
			minZ = 4.049999237060547;
			maxZ = 11.949999809265137;
			break;

		case 540:
			minX = 3.9499998092651367;
			maxX = 12.050000190734863;
			minZ = 6.574999809265137;
			maxZ = 12.050000190734863;
			break;

		case 630:
			minX = 4.050001621246338;
			maxX = 9.32499885559082;
			minZ = 3.9499993324279785;
			maxZ = 12.05000114440918;
			break;

		}
		break;
	case ROOM4PIT:
		switch (angle) {
		case 0:
			minX = 4.050000190734863;
			maxX = 11.949999809265137;
			minZ = 4.050000190734863;
			maxZ = 11.950000762939453;
			break;

		case 90:
			minX = 3.949998378753662;
			maxX = 12.05000114440918;
			minZ = 4.050000190734863;
			maxZ = 11.949999809265137;
			break;

		case 180:
			minX = 3.950000286102295;
			maxX = 12.049999237060547;
			minZ = 3.9499988555908203;
			maxZ = 12.05000114440918;
			break;

		case 270:
			minX = 4.050000190734863;
			maxX = 11.950000762939453;
			minZ = 3.9499988555908203;
			maxZ = 12.05000114440918;
			break;

		case 360:
			minX = 4.0500006675720215;
			maxX = 11.94999885559082;
			minZ = 4.049999237060547;
			maxZ = 11.950000762939453;
			break;

		case 450:
			minX = 3.950000762939453;
			maxX = 12.049999237060547;
			minZ = 4.049999237060547;
			maxZ = 11.950000762939453;
			break;

		case 540:
			minX = 3.9499998092651367;
			maxX = 12.050000190734863;
			minZ = 3.9499993324279785;
			maxZ = 12.050000190734863;
			break;

		case 630:
			minX = 4.050001621246338;
			maxX = 11.94999885559082;
			minZ = 3.949998378753662;
			maxZ = 12.05000114440918;
			break;

		}
		break;
	case ROOM2SERVERS:
		switch (angle) {
		case 0:
			minX = 1.5500001907348633;
			maxX = 8.825000762939453;
			minZ = 4.050000190734863;
			maxZ = 11.949999809265137;
			break;

		case 90:
			minX = 3.9499993324279785;
			maxX = 12.05000114440918;
			minZ = 1.549999713897705;
			maxZ = 8.824999809265137;
			break;

		case 180:
			minX = 7.074999809265137;
			maxX = 14.549999237060547;
			minZ = 3.9499998092651367;
			maxZ = 12.05000114440918;
			break;

		case 270:
			minX = 4.050000190734863;
			maxX = 11.950000762939453;
			minZ = 7.07499885559082;
			maxZ = 14.55000114440918;
			break;

		case 360:
			minX = 1.5500006675720215;
			maxX = 8.824999809265137;
			minZ = 4.049999237060547;
			maxZ = 11.949999809265137;
			break;

		case 450:
			minX = 3.950000286102295;
			maxX = 12.049999237060547;
			minZ = 1.5500001907348633;
			maxZ = 8.825000762939453;
			break;

		case 540:
			minX = 7.074999809265137;
			maxX = 14.550000190734863;
			minZ = 3.9499998092651367;
			maxZ = 12.050000190734863;
			break;

		case 630:
			minX = 4.050002574920654;
			maxX = 11.949999809265137;
			minZ = 7.074997901916504;
			maxZ = 14.55000114440918;
			break;

		}
		break;
	case ROOM2SHAFT:
		switch (angle) {
		case 0:
			minX = 7.050000190734863;
			maxX = 15.824999809265137;
			minZ = 4.050000190734863;
			maxZ = 11.949999809265137;
			break;

		case 90:
			minX = 3.949998378753662;
			maxX = 12.05000114440918;
			minZ = 7.050000190734863;
			maxZ = 15.825000762939453;
			break;

		case 180:
			minX = 0.07500028610229492;
			maxX = 9.049999237060547;
			minZ = 3.9499993324279785;
			maxZ = 12.050000190734863;
			break;

		case 270:
			minX = 4.049999237060547;
			maxX = 11.949999809265137;
			minZ = 0.07499837875366211;
			maxZ = 9.050000190734863;
			break;

		case 360:
			minX = 7.0500006675720215;
			maxX = 15.82499885559082;
			minZ = 4.050000190734863;
			maxZ = 11.95000171661377;
			break;

		case 450:
			minX = 3.9500017166137695;
			maxX = 12.049999237060547;
			minZ = 7.049999237060547;
			maxZ = 15.824999809265137;
			break;

		case 540:
			minX = 0.07499980926513672;
			maxX = 9.050000190734863;
			minZ = 3.9499998092651367;
			maxZ = 12.050000190734863;
			break;

		case 630:
			minX = 4.050000190734863;
			maxX = 11.949996948242188;
			minZ = 0.07499885559082031;
			maxZ = 9.05000114440918;
			break;

		}
		break;
	case ROOM2TUNNEL:
		switch (angle) {
		case 0:
			minX = 4.612500190734863;
			maxX = 11.387499809265137;
			minZ = 4.050000190734863;
			maxZ = 11.949999809265137;
			break;

		case 90:
			minX = 3.9499988555908203;
			maxX = 12.05000114440918;
			minZ = 4.612500190734863;
			maxZ = 11.387499809265137;
			break;

		case 180:
			minX = 4.512500286102295;
			maxX = 11.487499237060547;
			minZ = 3.9499993324279785;
			maxZ = 12.05000114440918;
			break;

		case 270:
			minX = 4.050000190734863;
			maxX = 11.949999809265137;
			minZ = 4.51249885559082;
			maxZ = 11.48750114440918;
			break;

		case 360:
			minX = 4.6125006675720215;
			maxX = 11.38749885559082;
			minZ = 4.049999237060547;
			maxZ = 11.950000762939453;
			break;

		case 450:
			minX = 3.950000762939453;
			maxX = 12.049999237060547;
			minZ = 4.612499237060547;
			maxZ = 11.387500762939453;
			break;

		case 540:
			minX = 4.512499809265137;
			maxX = 11.487500190734863;
			minZ = 3.9499998092651367;
			maxZ = 12.050000190734863;
			break;

		case 630:
			minX = 4.05000114440918;
			maxX = 11.94999885559082;
			minZ = 4.51249885559082;
			maxZ = 11.48750114440918;
			break;

		}
		break;
	case ROOM3TUNNEL:
		switch (angle) {
		case 0:
			minX = 4.050000190734863;
			maxX = 11.949999809265137;
			minZ = 4.050000190734863;
			maxZ = 9.574999809265137;
			break;

		case 90:
			minX = 6.3249993324279785;
			maxX = 12.05000114440918;
			minZ = 4.050000190734863;
			maxZ = 11.950000762939453;
			break;

		case 180:
			minX = 3.950000286102295;
			maxX = 12.049999237060547;
			minZ = 6.324999809265137;
			maxZ = 12.05000114440918;
			break;

		case 270:
			minX = 4.050000190734863;
			maxX = 9.574999809265137;
			minZ = 3.9499993324279785;
			maxZ = 12.05000114440918;
			break;

		case 360:
			minX = 4.0500006675720215;
			maxX = 11.949999809265137;
			minZ = 4.049999237060547;
			maxZ = 9.575000762939453;
			break;

		case 450:
			minX = 6.325000762939453;
			maxX = 12.049999237060547;
			minZ = 4.049999237060547;
			maxZ = 11.949999809265137;
			break;

		case 540:
			minX = 3.9499998092651367;
			maxX = 12.050000190734863;
			minZ = 6.324999809265137;
			maxZ = 12.050000190734863;
			break;

		case 630:
			minX = 4.050001621246338;
			maxX = 9.57499885559082;
			minZ = 3.9499993324279785;
			maxZ = 12.05000114440918;
			break;

		}
		break;
	case ROOM4TUNNELS:
		switch (angle) {
		case 0:
			minX = 4.050000190734863;
			maxX = 11.949999809265137;
			minZ = 4.050000190734863;
			maxZ = 11.949999809265137;
			break;

		case 90:
			minX = 3.9499988555908203;
			maxX = 12.05000114440918;
			minZ = 4.050000190734863;
			maxZ = 11.949999809265137;
			break;

		case 180:
			minX = 3.950000286102295;
			maxX = 12.049999237060547;
			minZ = 3.9499993324279785;
			maxZ = 12.05000114440918;
			break;

		case 270:
			minX = 4.050000190734863;
			maxX = 11.949999809265137;
			minZ = 3.9499988555908203;
			maxZ = 12.05000114440918;
			break;

		case 360:
			minX = 4.0500006675720215;
			maxX = 11.94999885559082;
			minZ = 4.049999237060547;
			maxZ = 11.950000762939453;
			break;

		case 450:
			minX = 3.950000762939453;
			maxX = 12.049999237060547;
			minZ = 4.049999237060547;
			maxZ = 11.950000762939453;
			break;

		case 540:
			minX = 3.9499998092651367;
			maxX = 12.050000190734863;
			minZ = 3.9499998092651367;
			maxZ = 12.050000190734863;
			break;

		case 630:
			minX = 4.050001621246338;
			maxX = 11.94999885559082;
			minZ = 3.9499988555908203;
			maxZ = 12.05000114440918;
			break;

		}
		break;
	case ROOM2TESLA_HCZ:
		switch (angle) {
		case 0:
			minX = 6.862500190734863;
			maxX = 9.137499809265137;
			minZ = 4.050000190734863;
			maxZ = 11.949999809265137;
			break;

		case 90:
			minX = 3.9499993324279785;
			maxX = 12.05000114440918;
			minZ = 6.862500190734863;
			maxZ = 9.137499809265137;
			break;

		case 180:
			minX = 6.762500286102295;
			maxX = 9.237499237060547;
			minZ = 3.9499998092651367;
			maxZ = 12.050000190734863;
			break;

		case 270:
			minX = 4.049999237060547;
			maxX = 11.950000762939453;
			minZ = 6.7624993324279785;
			maxZ = 9.237500190734863;
			break;

		case 360:
			minX = 6.8625006675720215;
			maxX = 9.137499809265137;
			minZ = 4.050000190734863;
			maxZ = 11.949999809265137;
			break;

		case 450:
			minX = 3.950000286102295;
			maxX = 12.049999237060547;
			minZ = 6.862499237060547;
			maxZ = 9.137500762939453;
			break;

		case 540:
			minX = 6.762499809265137;
			maxX = 9.237500190734863;
			minZ = 3.9499998092651367;
			maxZ = 12.050000190734863;
			break;

		case 630:
			minX = 4.0500006675720215;
			maxX = 11.94999885559082;
			minZ = 6.76249885559082;
			maxZ = 9.23750114440918;
			break;

		}
		break;
	case ROOM3Z2:
		switch (angle) {
		case 0:
			minX = 4.050000190734863;
			maxX = 11.950000762939453;
			minZ = 4.049999237060547;
			maxZ = 8.949999809265137;
			break;

		case 90:
			minX = 6.949999809265137;
			maxX = 12.050002098083496;
			minZ = 4.050000190734863;
			maxZ = 11.95000171661377;
			break;

		case 180:
			minX = 3.9499988555908203;
			maxX = 12.049999237060547;
			minZ = 6.950000286102295;
			maxZ = 12.05000114440918;
			break;

		case 270:
			minX = 4.049999237060547;
			maxX = 8.94999885559082;
			minZ = 3.949998378753662;
			maxZ = 12.05000114440918;
			break;

		case 360:
			minX = 4.0500006675720215;
			maxX = 11.950000762939453;
			minZ = 4.0499982833862305;
			maxZ = 8.949999809265137;
			break;

		case 450:
			minX = 6.950001239776611;
			maxX = 12.049999237060547;
			minZ = 4.049999237060547;
			maxZ = 11.950000762939453;
			break;

		case 540:
			minX = 3.9499988555908203;
			maxX = 12.050000190734863;
			minZ = 6.950000762939453;
			maxZ = 12.05000114440918;
			break;

		case 630:
			minX = 4.05000114440918;
			maxX = 8.949997901916504;
			minZ = 3.9499988555908203;
			maxZ = 12.05000114440918;
			break;

		}
		break;
	case ROOM2CPIT:
		switch (angle) {
		case 0:
			minX = 5.831249237060547;
			maxX = 11.949999809265137;
			minZ = 4.050000190734863;
			maxZ = 11.699999809265137;
			break;

		case 90:
			minX = 4.19999885559082;
			maxX = 12.05000114440918;
			minZ = 5.831249713897705;
			maxZ = 11.949999809265137;
			break;

		case 180:
			minX = 3.950000286102295;
			maxX = 10.268750190734863;
			minZ = 4.199999809265137;
			maxZ = 12.050000190734863;
			break;

		case 270:
			minX = 4.050000190734863;
			maxX = 11.699999809265137;
			minZ = 3.9499988555908203;
			maxZ = 10.26875114440918;
			break;

		case 360:
			minX = 5.831250190734863;
			maxX = 11.94999885559082;
			minZ = 4.049999237060547;
			maxZ = 11.700000762939453;
			break;

		case 450:
			minX = 4.200000762939453;
			maxX = 12.049999237060547;
			minZ = 5.831249237060547;
			maxZ = 11.950000762939453;
			break;

		case 540:
			minX = 3.9499998092651367;
			maxX = 10.268750190734863;
			minZ = 4.199999809265137;
			maxZ = 12.050000190734863;
			break;

		case 630:
			minX = 4.05000114440918;
			maxX = 11.69999885559082;
			minZ = 3.9499988555908203;
			maxZ = 10.26875114440918;
			break;

		}
		break;
	case ROOM2PIPES2:
		switch (angle) {
		case 0:
			minX = 7.050000190734863;
			maxX = 10.573046684265137;
			minZ = 4.042187690734863;
			maxZ = 11.949999809265137;
			break;

		case 90:
			minX = 3.9499988555908203;
			maxX = 12.05781364440918;
			minZ = 7.050000190734863;
			maxZ = 10.573046684265137;
			break;

		case 180:
			minX = 5.326953411102295;
			maxX = 9.049999237060547;
			minZ = 3.9499998092651367;
			maxZ = 12.057812690734863;
			break;

		case 270:
			minX = 4.042186737060547;
			maxX = 11.949999809265137;
			minZ = 5.3269524574279785;
			maxZ = 9.050000190734863;
			break;

		case 360:
			minX = 7.0500006675720215;
			maxX = 10.57304573059082;
			minZ = 4.042187690734863;
			maxZ = 11.950000762939453;
			break;

		case 450:
			minX = 3.950000762939453;
			maxX = 12.057811737060547;
			minZ = 7.049999237060547;
			maxZ = 10.573047637939453;
			break;

		case 540:
			minX = 5.326952934265137;
			maxX = 9.050000190734863;
			minZ = 3.9499998092651367;
			maxZ = 12.057812690734863;
			break;

		case 630:
			minX = 4.04218864440918;
			maxX = 11.94999885559082;
			minZ = 5.32695198059082;
			maxZ = 9.05000114440918;
			break;

		}
		break;
	case CHECKPOINT2:
		switch (angle) {
		case 0:
			minX = 3.7453126907348633;
			maxX = 12.262499809265137;
			minZ = 4.049999237060547;
			maxZ = 11.957812309265137;
			break;

		case 90:
			minX = 3.9421863555908203;
			maxX = 12.050002098083496;
			minZ = 3.7453126907348633;
			maxZ = 12.262499809265137;
			break;

		case 180:
			minX = 3.637500286102295;
			maxX = 12.354686737060547;
			minZ = 3.9421868324279785;
			maxZ = 12.05000114440918;
			break;

		case 270:
			minX = 4.049999237060547;
			maxX = 11.957812309265137;
			minZ = 3.6374988555908203;
			maxZ = 12.35468864440918;
			break;

		case 360:
			minX = 3.7453131675720215;
			maxX = 12.26249885559082;
			minZ = 4.0499982833862305;
			maxZ = 11.957813262939453;
			break;

		case 450:
			minX = 3.9421887397766113;
			maxX = 12.049999237060547;
			minZ = 3.745312213897705;
			maxZ = 12.262500762939453;
			break;

		case 540:
			minX = 3.6374998092651367;
			maxX = 12.354687690734863;
			minZ = 3.9421873092651367;
			maxZ = 12.05000114440918;
			break;

		case 630:
			minX = 4.05000114440918;
			maxX = 11.957810401916504;
			minZ = 3.6374988555908203;
			maxZ = 12.35468864440918;
			break;

		}
		break;
	case ROOM079:
		switch (angle) {
		case 0:
			minX = 6.424999713897705;
			maxX = 16.700000762939453;
			minZ = 3.9250006675720215;
			maxZ = 15.949999809265137;
			break;

		case 90:
			minX = -0.050002098083496094;
			maxX = 12.175000190734863;
			minZ = 6.425000190734863;
			maxZ = 16.700000762939453;
			break;

		case 180:
			minX = -0.7999992370605469;
			maxX = 9.674999237060547;
			minZ = -0.05000114440917969;
			maxZ = 12.174999237060547;
			break;

		case 270:
			minX = 3.9250001907348633;
			maxX = 15.949999809265137;
			minZ = -0.8000020980834961;
			maxZ = 9.67500114440918;
			break;

		case 360:
			minX = 6.4250006675720215;
			maxX = 16.69999885559082;
			minZ = 3.9250001907348633;
			maxZ = 15.95000171661377;
			break;

		case 450:
			minX = -0.04999828338623047;
			maxX = 12.174999237060547;
			minZ = 6.424999237060547;
			maxZ = 16.700000762939453;
			break;

		case 540:
			minX = -0.8000001907348633;
			maxX = 9.675000190734863;
			minZ = -0.05000019073486328;
			maxZ = 12.174999237060547;
			break;

		case 630:
			minX = 3.925001621246338;
			maxX = 15.949995994567871;
			minZ = -0.8000020980834961;
			maxZ = 9.67500114440918;
			break;

		}
		break;
	case LOCKROOM2:
		switch (angle) {
		case 0:
			minX = 4.675000190734863;
			maxX = 11.949999809265137;
			minZ = 4.050000190734863;
			maxZ = 11.324999809265137;
			break;

		case 90:
			minX = 4.57499885559082;
			maxX = 12.05000114440918;
			minZ = 4.675000190734863;
			maxZ = 11.949999809265137;
			break;

		case 180:
			minX = 3.950000286102295;
			maxX = 11.424999237060547;
			minZ = 4.574999809265137;
			maxZ = 12.05000114440918;
			break;

		case 270:
			minX = 4.050000190734863;
			maxX = 11.324999809265137;
			minZ = 3.9499988555908203;
			maxZ = 11.42500114440918;
			break;

		case 360:
			minX = 4.6750006675720215;
			maxX = 11.949999809265137;
			minZ = 4.049999237060547;
			maxZ = 11.325000762939453;
			break;

		case 450:
			minX = 4.575000762939453;
			maxX = 12.049999237060547;
			minZ = 4.674999237060547;
			maxZ = 11.950000762939453;
			break;

		case 540:
			minX = 3.9499998092651367;
			maxX = 11.425000190734863;
			minZ = 4.574999809265137;
			maxZ = 12.050000190734863;
			break;

		case 630:
			minX = 4.05000114440918;
			maxX = 11.32499885559082;
			minZ = 3.9499988555908203;
			maxZ = 11.42500114440918;
			break;

		}
		break;
	case GATEAENTRANCE:
		switch (angle) {
		case 0:
			minX = 5.237500190734863;
			maxX = 13.262499809265137;
			minZ = 3.6125001907348633;
			maxZ = 12.793749809265137;
			break;

		case 90:
			minX = 3.1062488555908203;
			maxX = 12.48750114440918;
			minZ = 5.237500190734863;
			maxZ = 13.262499809265137;
			break;

		case 180:
			minX = 2.637500286102295;
			maxX = 10.862499237060547;
			minZ = 3.1062493324279785;
			maxZ = 12.48750114440918;
			break;

		case 270:
			minX = 3.6125001907348633;
			maxX = 12.793749809265137;
			minZ = 2.6374988555908203;
			maxZ = 10.86250114440918;
			break;

		case 360:
			minX = 5.2375006675720215;
			maxX = 13.26249885559082;
			minZ = 3.612499713897705;
			maxZ = 12.793750762939453;
			break;

		case 450:
			minX = 3.1062512397766113;
			maxX = 12.487499237060547;
			minZ = 5.237499237060547;
			maxZ = 13.262500762939453;
			break;

		case 540:
			minX = 2.6374998092651367;
			maxX = 10.862500190734863;
			minZ = 3.1062498092651367;
			maxZ = 12.487500190734863;
			break;

		case 630:
			minX = 3.612501621246338;
			maxX = 12.793746948242188;
			minZ = 2.637498378753662;
			maxZ = 10.86250114440918;
			break;

		}
		break;
	case MEDIBAY:
		switch (angle) {
		case 0:
			minX = 4.049999237060547;
			maxX = 9.074999809265137;
			minZ = 4.046093940734863;
			maxZ = 11.949999809265137;
			break;

		case 90:
			minX = 3.9499993324279785;
			maxX = 12.05390739440918;
			minZ = 4.049999237060547;
			maxZ = 9.074999809265137;
			break;

		case 180:
			minX = 6.825000286102295;
			maxX = 12.05000114440918;
			minZ = 3.9499998092651367;
			maxZ = 12.05390739440918;
			break;

		case 270:
			minX = 4.046093940734863;
			maxX = 11.950000762939453;
			minZ = 6.8249993324279785;
			maxZ = 12.050002098083496;
			break;

		case 360:
			minX = 4.049999237060547;
			maxX = 9.07499885559082;
			minZ = 4.046092987060547;
			maxZ = 11.949999809265137;
			break;

		case 450:
			minX = 3.950000286102295;
			maxX = 12.053905487060547;
			minZ = 4.0499982833862305;
			maxZ = 9.075000762939453;
			break;

		case 540:
			minX = 6.825000286102295;
			maxX = 12.05000114440918;
			minZ = 3.9499998092651367;
			maxZ = 12.053906440734863;
			break;

		case 630:
			minX = 4.046095848083496;
			maxX = 11.94999885559082;
			minZ = 6.82499885559082;
			maxZ = 12.050002098083496;
			break;

		}
		break;
	case ROOM2Z3:
		switch (angle) {
		case 0:
			minX = 7.050000190734863;
			maxX = 8.949999809265137;
			minZ = 4.050000190734863;
			maxZ = 11.950000762939453;
			break;

		case 90:
			minX = 3.9499988555908203;
			maxX = 12.05000114440918;
			minZ = 7.050000190734863;
			maxZ = 8.949999809265137;
			break;

		case 180:
			minX = 6.950000286102295;
			maxX = 9.049999237060547;
			minZ = 3.9499993324279785;
			maxZ = 12.050000190734863;
			break;

		case 270:
			minX = 4.049999237060547;
			maxX = 11.950000762939453;
			minZ = 6.9499993324279785;
			maxZ = 9.050000190734863;
			break;

		case 360:
			minX = 7.0500006675720215;
			maxX = 8.94999885559082;
			minZ = 4.050000190734863;
			maxZ = 11.950000762939453;
			break;

		case 450:
			minX = 3.950000286102295;
			maxX = 12.049999237060547;
			minZ = 7.049999237060547;
			maxZ = 8.950000762939453;
			break;

		case 540:
			minX = 6.950000286102295;
			maxX = 9.050000190734863;
			minZ = 3.9499993324279785;
			maxZ = 12.050000190734863;
			break;

		case 630:
			minX = 4.050000190734863;
			maxX = 11.949999809265137;
			minZ = 6.94999885559082;
			maxZ = 9.05000114440918;
			break;

		}
		break;
	case ROOM2CAFETERIA:
		switch (angle) {
		case 0:
			minX = 1.0500001907348633;
			maxX = 15.574999809265137;
			minZ = 3.9875001907348633;
			maxZ = 11.950000762939453;
			break;

		case 90:
			minX = 3.949997901916504;
			maxX = 12.11250114440918;
			minZ = 1.049999713897705;
			maxZ = 15.575000762939453;
			break;

		case 180:
			minX = 0.3250002861022949;
			maxX = 15.049999237060547;
			minZ = 3.9499988555908203;
			maxZ = 12.11250114440918;
			break;

		case 270:
			minX = 3.9875006675720215;
			maxX = 11.949999809265137;
			minZ = 0.3249983787536621;
			maxZ = 15.05000114440918;
			break;

		case 360:
			minX = 1.0500006675720215;
			maxX = 15.57499885559082;
			minZ = 3.9874987602233887;
			maxZ = 11.95000171661377;
			break;

		case 450:
			minX = 3.9500012397766113;
			maxX = 12.11249828338623;
			minZ = 1.049999713897705;
			maxZ = 15.575000762939453;
			break;

		case 540:
			minX = 0.3249998092651367;
			maxX = 15.050000190734863;
			minZ = 3.9499993324279785;
			maxZ = 12.112500190734863;
			break;

		case 630:
			minX = 3.9875030517578125;
			maxX = 11.949996948242188;
			minZ = 0.3249988555908203;
			maxZ = 15.05000114440918;
			break;

		}
		break;
	case ROOM2CZ3:
		switch (angle) {
		case 0:
			minX = 5.800000190734863;
			maxX = 11.949999809265137;
			minZ = 4.050000190734863;
			maxZ = 10.199999809265137;
			break;

		case 90:
			minX = 5.6999993324279785;
			maxX = 12.05000114440918;
			minZ = 5.800000190734863;
			maxZ = 11.949999809265137;
			break;

		case 180:
			minX = 3.950000286102295;
			maxX = 10.299999237060547;
			minZ = 5.699999809265137;
			maxZ = 12.050000190734863;
			break;

		case 270:
			minX = 4.050000190734863;
			maxX = 10.199999809265137;
			minZ = 3.9499988555908203;
			maxZ = 10.30000114440918;
			break;

		case 360:
			minX = 5.8000006675720215;
			maxX = 11.949999809265137;
			minZ = 4.049999237060547;
			maxZ = 10.200000762939453;
			break;

		case 450:
			minX = 5.700000762939453;
			maxX = 12.049999237060547;
			minZ = 5.799999237060547;
			maxZ = 11.949999809265137;
			break;

		case 540:
			minX = 3.9499998092651367;
			maxX = 10.300000190734863;
			minZ = 5.699999809265137;
			maxZ = 12.050000190734863;
			break;

		case 630:
			minX = 4.05000114440918;
			maxX = 10.19999885559082;
			minZ = 3.9499993324279785;
			maxZ = 10.30000114440918;
			break;

		}
		break;
	case ROOM2CCONT:
		switch (angle) {
		case 0:
			minX = -0.8249998092651367;
			maxX = 11.949999809265137;
			minZ = 4.050000190734863;
			maxZ = 15.074999809265137;
			break;

		case 90:
			minX = 0.8249983787536621;
			maxX = 12.05000114440918;
			minZ = -0.8249998092651367;
			maxZ = 11.949999809265137;
			break;

		case 180:
			minX = 3.950000762939453;
			maxX = 16.924999237060547;
			minZ = 0.8249993324279785;
			maxZ = 12.05000114440918;
			break;

		case 270:
			minX = 4.050000190734863;
			maxX = 15.075000762939453;
			minZ = 3.949998378753662;
			maxZ = 16.925003051757812;
			break;

		case 360:
			minX = -0.8249988555908203;
			maxX = 11.94999885559082;
			minZ = 4.0499982833862305;
			maxZ = 15.075000762939453;
			break;

		case 450:
			minX = 0.8250007629394531;
			maxX = 12.04999828338623;
			minZ = -0.8249998092651367;
			maxZ = 11.950000762939453;
			break;

		case 540:
			minX = 3.950000286102295;
			maxX = 16.924999237060547;
			minZ = 0.8249998092651367;
			maxZ = 12.050000190734863;
			break;

		case 630:
			minX = 4.0500030517578125;
			maxX = 15.074997901916504;
			minZ = 3.9499974250793457;
			maxZ = 16.924999237060547;
			break;

		}
		break;
	case ROOM2OFFICES:
		switch (angle) {
		case 0:
			minX = 5.675000190734863;
			maxX = 11.949999809265137;
			minZ = 4.050000190734863;
			maxZ = 11.949999809265137;
			break;

		case 90:
			minX = 3.9499988555908203;
			maxX = 12.05000114440918;
			minZ = 5.675000190734863;
			maxZ = 11.949999809265137;
			break;

		case 180:
			minX = 3.950000286102295;
			maxX = 10.424999237060547;
			minZ = 3.9499993324279785;
			maxZ = 12.050000190734863;
			break;

		case 270:
			minX = 4.050000190734863;
			maxX = 11.949999809265137;
			minZ = 3.9499988555908203;
			maxZ = 10.42500114440918;
			break;

		case 360:
			minX = 5.6750006675720215;
			maxX = 11.94999885559082;
			minZ = 4.049999237060547;
			maxZ = 11.950000762939453;
			break;

		case 450:
			minX = 3.950000762939453;
			maxX = 12.049999237060547;
			minZ = 5.674999237060547;
			maxZ = 11.950000762939453;
			break;

		case 540:
			minX = 3.9499998092651367;
			maxX = 10.425000190734863;
			minZ = 3.9499998092651367;
			maxZ = 12.050000190734863;
			break;

		case 630:
			minX = 4.05000114440918;
			maxX = 11.94999885559082;
			minZ = 3.9499988555908203;
			maxZ = 10.42500114440918;
			break;

		}
		break;
	case ROOM2OFFICES2:
		switch (angle) {
		case 0:
			minX = 4.050000190734863;
			maxX = 9.074999809265137;
			minZ = 4.050000190734863;
			maxZ = 11.949999809265137;
			break;

		case 90:
			minX = 3.9499993324279785;
			maxX = 12.05000114440918;
			minZ = 4.050000190734863;
			maxZ = 9.074999809265137;
			break;

		case 180:
			minX = 6.825000286102295;
			maxX = 12.049999237060547;
			minZ = 3.9499998092651367;
			maxZ = 12.05000114440918;
			break;

		case 270:
			minX = 4.050000190734863;
			maxX = 11.950000762939453;
			minZ = 6.8249993324279785;
			maxZ = 12.05000114440918;
			break;

		case 360:
			minX = 4.0500006675720215;
			maxX = 9.07499885559082;
			minZ = 4.049999237060547;
			maxZ = 11.949999809265137;
			break;

		case 450:
			minX = 3.950000286102295;
			maxX = 12.049999237060547;
			minZ = 4.049999237060547;
			maxZ = 9.075000762939453;
			break;

		case 540:
			minX = 6.825000286102295;
			maxX = 12.050000190734863;
			minZ = 3.9499998092651367;
			maxZ = 12.050000190734863;
			break;

		case 630:
			minX = 4.050001621246338;
			maxX = 11.94999885559082;
			minZ = 6.82499885559082;
			maxZ = 12.05000114440918;
			break;

		}
		break;
	case ROOM2OFFICES3:
		switch (angle) {
		case 0:
			minX = 1.7999992370605469;
			maxX = 10.949999809265137;
			minZ = 4.050000190734863;
			maxZ = 11.950000762939453;
			break;

		case 90:
			minX = 3.949997901916504;
			maxX = 12.05000114440918;
			minZ = 1.7999987602233887;
			maxZ = 10.949999809265137;
			break;

		case 180:
			minX = 4.950000286102295;
			maxX = 14.30000114440918;
			minZ = 3.949998378753662;
			maxZ = 12.05000114440918;
			break;

		case 270:
			minX = 4.050000190734863;
			maxX = 11.950000762939453;
			minZ = 4.94999885559082;
			maxZ = 14.300003051757812;
			break;

		case 360:
			minX = 1.799999713897705;
			maxX = 10.94999885559082;
			minZ = 4.049999237060547;
			maxZ = 11.95000171661377;
			break;

		case 450:
			minX = 3.9499998092651367;
			maxX = 12.049999237060547;
			minZ = 1.7999992370605469;
			maxZ = 10.950000762939453;
			break;

		case 540:
			minX = 4.949999809265137;
			maxX = 14.30000114440918;
			minZ = 3.9499988555908203;
			maxZ = 12.050000190734863;
			break;

		case 630:
			minX = 4.050002098083496;
			maxX = 11.949999809265137;
			minZ = 4.94999885559082;
			maxZ = 14.300002098083496;
			break;

		}
		break;
	case ROOM2OFFICES4:
		switch (angle) {
		case 0:
			minX = 1.9250001907348633;
			maxX = 10.278124809265137;
			minZ = 4.050000190734863;
			maxZ = 11.949999809265137;
			break;

		case 90:
			minX = 3.9499988555908203;
			maxX = 12.05000114440918;
			minZ = 1.924999713897705;
			maxZ = 10.278124809265137;
			break;

		case 180:
			minX = 5.621875286102295;
			maxX = 14.174999237060547;
			minZ = 3.9499998092651367;
			maxZ = 12.05000114440918;
			break;

		case 270:
			minX = 4.050000190734863;
			maxX = 11.949999809265137;
			minZ = 5.6218743324279785;
			maxZ = 14.17500114440918;
			break;

		case 360:
			minX = 1.9250006675720215;
			maxX = 10.27812385559082;
			minZ = 4.049999237060547;
			maxZ = 11.950000762939453;
			break;

		case 450:
			minX = 3.950000762939453;
			maxX = 12.049999237060547;
			minZ = 1.9250001907348633;
			maxZ = 10.278125762939453;
			break;

		case 540:
			minX = 5.621874809265137;
			maxX = 14.175000190734863;
			minZ = 3.9499998092651367;
			maxZ = 12.050000190734863;
			break;

		case 630:
			minX = 4.050002098083496;
			maxX = 11.94999885559082;
			minZ = 5.62187385559082;
			maxZ = 14.17500114440918;
			break;

		}
		break;
	case ROOM2POFFICES:
		switch (angle) {
		case 0:
			minX = 6.050000190734863;
			maxX = 11.762500762939453;
			minZ = 4.049999237060547;
			maxZ = 11.950000762939453;
			break;

		case 90:
			minX = 3.949998378753662;
			maxX = 12.05000114440918;
			minZ = 6.050000190734863;
			maxZ = 11.762500762939453;
			break;

		case 180:
			minX = 4.137499809265137;
			maxX = 10.049999237060547;
			minZ = 3.9499988555908203;
			maxZ = 12.05000114440918;
			break;

		case 270:
			minX = 4.049999237060547;
			maxX = 11.950000762939453;
			minZ = 4.13749885559082;
			maxZ = 10.05000114440918;
			break;

		case 360:
			minX = 6.0500006675720215;
			maxX = 11.762499809265137;
			minZ = 4.049999237060547;
			maxZ = 11.950000762939453;
			break;

		case 450:
			minX = 3.950000762939453;
			maxX = 12.049999237060547;
			minZ = 6.049999237060547;
			maxZ = 11.762500762939453;
			break;

		case 540:
			minX = 4.137499809265137;
			maxX = 10.050000190734863;
			minZ = 3.9499993324279785;
			maxZ = 12.05000114440918;
			break;

		case 630:
			minX = 4.0500006675720215;
			maxX = 11.94999885559082;
			minZ = 4.137498378753662;
			maxZ = 10.05000114440918;
			break;

		}
		break;
	case ROOM2POFFICES2:
		switch (angle) {
		case 0:
			minX = 3.4250001907348633;
			maxX = 12.699999809265137;
			minZ = 4.049999237060547;
			maxZ = 11.950000762939453;
			break;

		case 90:
			minX = 3.949998378753662;
			maxX = 12.050002098083496;
			minZ = 3.4250001907348633;
			maxZ = 12.699999809265137;
			break;

		case 180:
			minX = 3.200000286102295;
			maxX = 12.674999237060547;
			minZ = 3.9499988555908203;
			maxZ = 12.05000114440918;
			break;

		case 270:
			minX = 4.049999237060547;
			maxX = 11.950000762939453;
			minZ = 3.1999988555908203;
			maxZ = 12.67500114440918;
			break;

		case 360:
			minX = 3.4250006675720215;
			maxX = 12.69999885559082;
			minZ = 4.0499982833862305;
			maxZ = 11.95000171661377;
			break;

		case 450:
			minX = 3.950000762939453;
			maxX = 12.049999237060547;
			minZ = 3.424999713897705;
			maxZ = 12.700000762939453;
			break;

		case 540:
			minX = 3.1999998092651367;
			maxX = 12.675000190734863;
			minZ = 3.9499993324279785;
			maxZ = 12.05000114440918;
			break;

		case 630:
			minX = 4.05000114440918;
			maxX = 11.94999885559082;
			minZ = 3.1999988555908203;
			maxZ = 12.67500114440918;
			break;

		}
		break;
	case ROOM2SROOM:
		switch (angle) {
		case 0:
			minX = 6.862500190734863;
			maxX = 16.950000762939453;
			minZ = 4.049999237060547;
			maxZ = 11.950000762939453;
			break;

		case 90:
			minX = 3.949997901916504;
			maxX = 12.05000114440918;
			minZ = 6.862500190734863;
			maxZ = 16.950000762939453;
			break;

		case 180:
			minX = -1.0500001907348633;
			maxX = 9.237499237060547;
			minZ = 3.949998378753662;
			maxZ = 12.05000114440918;
			break;

		case 270:
			minX = 4.049999237060547;
			maxX = 11.949999809265137;
			minZ = -1.050002098083496;
			maxZ = 9.237500190734863;
			break;

		case 360:
			minX = 6.8625006675720215;
			maxX = 16.94999885559082;
			minZ = 4.049999237060547;
			maxZ = 11.95000171661377;
			break;

		case 450:
			minX = 3.9500017166137695;
			maxX = 12.049999237060547;
			minZ = 6.862499237060547;
			maxZ = 16.950000762939453;
			break;

		case 540:
			minX = -1.0500001907348633;
			maxX = 9.237500190734863;
			minZ = 3.9499993324279785;
			maxZ = 12.05000114440918;
			break;

		case 630:
			minX = 4.050000190734863;
			maxX = 11.949996948242188;
			minZ = -1.0500011444091797;
			maxZ = 9.23750114440918;
			break;

		}
		break;
	case ROOM2TOILETS:
		switch (angle) {
		case 0:
			minX = 6.800000190734863;
			maxX = 14.074999809265137;
			minZ = 4.050000190734863;
			maxZ = 11.949999809265137;
			break;

		case 90:
			minX = 3.949998378753662;
			maxX = 12.05000114440918;
			minZ = 6.800000190734863;
			maxZ = 14.075000762939453;
			break;

		case 180:
			minX = 1.825000286102295;
			maxX = 9.299999237060547;
			minZ = 3.9499993324279785;
			maxZ = 12.050000190734863;
			break;

		case 270:
			minX = 4.049999237060547;
			maxX = 11.949999809265137;
			minZ = 1.824998378753662;
			maxZ = 9.300000190734863;
			break;

		case 360:
			minX = 6.8000006675720215;
			maxX = 14.07499885559082;
			minZ = 4.050000190734863;
			maxZ = 11.950000762939453;
			break;

		case 450:
			minX = 3.9500012397766113;
			maxX = 12.049999237060547;
			minZ = 6.799999237060547;
			maxZ = 14.074999809265137;
			break;

		case 540:
			minX = 1.8249998092651367;
			maxX = 9.300000190734863;
			minZ = 3.9499998092651367;
			maxZ = 12.050000190734863;
			break;

		case 630:
			minX = 4.0500006675720215;
			maxX = 11.949997901916504;
			minZ = 1.8249988555908203;
			maxZ = 9.30000114440918;
			break;

		}
		break;
	case ROOM2TESLA:
		switch (angle) {
		case 0:
			minX = 6.862499237060547;
			maxX = 9.137500762939453;
			minZ = 4.050000190734863;
			maxZ = 11.949999809265137;
			break;

		case 90:
			minX = 3.9499993324279785;
			maxX = 12.05000114440918;
			minZ = 6.862499713897705;
			maxZ = 9.137499809265137;
			break;

		case 180:
			minX = 6.762499809265137;
			maxX = 9.237500190734863;
			minZ = 3.9499998092651367;
			maxZ = 12.050000190734863;
			break;

		case 270:
			minX = 4.049999237060547;
			maxX = 11.950000762939453;
			minZ = 6.76249885559082;
			maxZ = 9.23750114440918;
			break;

		case 360:
			minX = 6.862500190734863;
			maxX = 9.137499809265137;
			minZ = 4.050000190734863;
			maxZ = 11.949999809265137;
			break;

		case 450:
			minX = 3.950000286102295;
			maxX = 12.049999237060547;
			minZ = 6.862498760223389;
			maxZ = 9.137500762939453;
			break;

		case 540:
			minX = 6.762499809265137;
			maxX = 9.237500190734863;
			minZ = 3.9499998092651367;
			maxZ = 12.050000190734863;
			break;

		case 630:
			minX = 4.0500006675720215;
			maxX = 11.94999885559082;
			minZ = 6.762498378753662;
			maxZ = 9.237502098083496;
			break;

		}
		break;
	case ROOM3SERVERS:
		switch (angle) {
		case 0:
			minX = 4.050000190734863;
			maxX = 11.949999809265137;
			minZ = 4.050000190734863;
			maxZ = 11.981249809265137;
			break;

		case 90:
			minX = 3.9187488555908203;
			maxX = 12.05000114440918;
			minZ = 4.050000190734863;
			maxZ = 11.949999809265137;
			break;

		case 180:
			minX = 3.950000286102295;
			maxX = 12.049999237060547;
			minZ = 3.9187493324279785;
			maxZ = 12.05000114440918;
			break;

		case 270:
			minX = 4.050000190734863;
			maxX = 11.981249809265137;
			minZ = 3.9499988555908203;
			maxZ = 12.05000114440918;
			break;

		case 360:
			minX = 4.0500006675720215;
			maxX = 11.94999885559082;
			minZ = 4.049999237060547;
			maxZ = 11.981250762939453;
			break;

		case 450:
			minX = 3.918750762939453;
			maxX = 12.049999237060547;
			minZ = 4.049999237060547;
			maxZ = 11.950000762939453;
			break;

		case 540:
			minX = 3.9499998092651367;
			maxX = 12.050000190734863;
			minZ = 3.9187498092651367;
			maxZ = 12.050000190734863;
			break;

		case 630:
			minX = 4.050001621246338;
			maxX = 11.981247901916504;
			minZ = 3.949998378753662;
			maxZ = 12.05000114440918;
			break;

		}
		break;
	case ROOM3SERVERS2:
		switch (angle) {
		case 0:
			minX = 4.050000190734863;
			maxX = 11.949999809265137;
			minZ = 4.050000190734863;
			maxZ = 11.981249809265137;
			break;

		case 90:
			minX = 3.9187488555908203;
			maxX = 12.05000114440918;
			minZ = 4.050000190734863;
			maxZ = 11.949999809265137;
			break;

		case 180:
			minX = 3.950000286102295;
			maxX = 12.049999237060547;
			minZ = 3.9187493324279785;
			maxZ = 12.05000114440918;
			break;

		case 270:
			minX = 4.050000190734863;
			maxX = 11.981249809265137;
			minZ = 3.9499988555908203;
			maxZ = 12.05000114440918;
			break;

		case 360:
			minX = 4.0500006675720215;
			maxX = 11.94999885559082;
			minZ = 4.049999237060547;
			maxZ = 11.981250762939453;
			break;

		case 450:
			minX = 3.918750762939453;
			maxX = 12.049999237060547;
			minZ = 4.049999237060547;
			maxZ = 11.950000762939453;
			break;

		case 540:
			minX = 3.9499998092651367;
			maxX = 12.050000190734863;
			minZ = 3.9187498092651367;
			maxZ = 12.050000190734863;
			break;

		case 630:
			minX = 4.050001621246338;
			maxX = 11.981247901916504;
			minZ = 3.949998378753662;
			maxZ = 12.05000114440918;
			break;

		}
		break;
	case ROOM3Z3:
		switch (angle) {
		case 0:
			minX = 4.050000190734863;
			maxX = 11.949999809265137;
			minZ = 4.050000190734863;
			maxZ = 10.199999809265137;
			break;

		case 90:
			minX = 5.6999993324279785;
			maxX = 12.05000114440918;
			minZ = 4.050000190734863;
			maxZ = 11.949999809265137;
			break;

		case 180:
			minX = 3.950000286102295;
			maxX = 12.049999237060547;
			minZ = 5.699999809265137;
			maxZ = 12.05000114440918;
			break;

		case 270:
			minX = 4.050000190734863;
			maxX = 10.199999809265137;
			minZ = 3.9499988555908203;
			maxZ = 12.05000114440918;
			break;

		case 360:
			minX = 4.0500006675720215;
			maxX = 11.949999809265137;
			minZ = 4.049999237060547;
			maxZ = 10.200000762939453;
			break;

		case 450:
			minX = 5.700000762939453;
			maxX = 12.049999237060547;
			minZ = 4.049999237060547;
			maxZ = 11.949999809265137;
			break;

		case 540:
			minX = 3.9499998092651367;
			maxX = 12.050000190734863;
			minZ = 5.699999809265137;
			maxZ = 12.050000190734863;
			break;

		case 630:
			minX = 4.050001621246338;
			maxX = 10.19999885559082;
			minZ = 3.9499993324279785;
			maxZ = 12.05000114440918;
			break;

		}
		break;
	case ROOM4Z3:
		switch (angle) {
		case 0:
			minX = 4.050000190734863;
			maxX = 11.949999809265137;
			minZ = 4.050000190734863;
			maxZ = 11.949999809265137;
			break;

		case 90:
			minX = 3.9499988555908203;
			maxX = 12.05000114440918;
			minZ = 4.050000190734863;
			maxZ = 11.949999809265137;
			break;

		case 180:
			minX = 3.950000286102295;
			maxX = 12.049999237060547;
			minZ = 3.9499993324279785;
			maxZ = 12.05000114440918;
			break;

		case 270:
			minX = 4.050000190734863;
			maxX = 11.949999809265137;
			minZ = 3.9499988555908203;
			maxZ = 12.05000114440918;
			break;

		case 360:
			minX = 4.0500006675720215;
			maxX = 11.94999885559082;
			minZ = 4.049999237060547;
			maxZ = 11.950000762939453;
			break;

		case 450:
			minX = 3.950000762939453;
			maxX = 12.049999237060547;
			minZ = 4.049999237060547;
			maxZ = 11.950000762939453;
			break;

		case 540:
			minX = 3.9499998092651367;
			maxX = 12.050000190734863;
			minZ = 3.9499998092651367;
			maxZ = 12.050000190734863;
			break;

		case 630:
			minX = 4.050001621246338;
			maxX = 11.94999885559082;
			minZ = 3.9499988555908203;
			maxZ = 12.05000114440918;
			break;

		}
		break;
	case ROOM1LIFTS:
		switch (angle) {
		case 0:
			minX = 6.237500190734863;
			maxX = 9.699999809265137;
			minZ = 4.049999237060547;
			maxZ = 8.36286735534668;
			break;

		case 90:
			minX = 7.537132263183594;
			maxX = 12.05000114440918;
			minZ = 6.237500190734863;
			maxZ = 9.700000762939453;
			break;

		case 180:
			minX = 6.199999809265137;
			maxX = 9.862499237060547;
			minZ = 7.537132740020752;
			maxZ = 12.05000114440918;
			break;

		case 270:
			minX = 4.049999237060547;
			maxX = 8.36286735534668;
			minZ = 6.199999809265137;
			maxZ = 9.86250114440918;
			break;

		case 360:
			minX = 6.2375006675720215;
			maxX = 9.699999809265137;
			minZ = 4.049999237060547;
			maxZ = 8.36286735534668;
			break;

		case 450:
			minX = 7.53713321685791;
			maxX = 12.049999237060547;
			minZ = 6.237499237060547;
			maxZ = 9.699999809265137;
			break;

		case 540:
			minX = 6.199999809265137;
			maxX = 9.862500190734863;
			minZ = 7.537132740020752;
			maxZ = 12.05000114440918;
			break;

		case 630:
			minX = 4.0500006675720215;
			maxX = 8.362866401672363;
			minZ = 6.199999809265137;
			maxZ = 9.86250114440918;
			break;

		}
		break;
	case ROOM3GW:
		switch (angle) {
		case 0:
			minX = 4.050000190734863;
			maxX = 12.008591651916504;
			minZ = 4.022656440734863;
			maxZ = 10.088875770568848;
			break;

		case 90:
			minX = 5.811123847961426;
			maxX = 12.07734489440918;
			minZ = 4.050000190734863;
			maxZ = 12.008591651916504;
			break;

		case 180:
			minX = 3.8914079666137695;
			maxX = 12.049999237060547;
			minZ = 5.811124324798584;
			maxZ = 12.07734489440918;
			break;

		case 270:
			minX = 4.022656440734863;
			maxX = 10.088874816894531;
			minZ = 3.891407012939453;
			maxZ = 12.05000114440918;
			break;

		case 360:
			minX = 4.0500006675720215;
			maxX = 12.008590698242188;
			minZ = 4.022655487060547;
			maxZ = 10.088876724243164;
			break;

		case 450:
			minX = 5.811125755310059;
			maxX = 12.077342987060547;
			minZ = 4.049999237060547;
			maxZ = 12.008591651916504;
			break;

		case 540:
			minX = 3.8914079666137695;
			maxX = 12.050000190734863;
			minZ = 5.811124801635742;
			maxZ = 12.077343940734863;
			break;

		case 630:
			minX = 4.022658348083496;
			maxX = 10.088873863220215;
			minZ = 3.8914074897766113;
			maxZ = 12.05000114440918;
			break;

		}
		break;
	case ROOM2SERVERS2:
		switch (angle) {
		case 0:
			minX = 3.825389862060547;
			maxX = 11.137500762939453;
			minZ = 3.9523534774780273;
			maxZ = 11.950000762939453;
			break;

		case 90:
			minX = 3.949998378753662;
			maxX = 12.147647857666016;
			minZ = 3.825389862060547;
			maxZ = 11.137500762939453;
			break;

		case 180:
			minX = 4.762499809265137;
			maxX = 12.27461051940918;
			minZ = 3.9499988555908203;
			maxZ = 12.147647857666016;
			break;

		case 270:
			minX = 3.9523534774780273;
			maxX = 11.950000762939453;
			minZ = 4.76249885559082;
			maxZ = 12.274611473083496;
			break;

		case 360:
			minX = 3.8253908157348633;
			maxX = 11.137499809265137;
			minZ = 3.952352523803711;
			maxZ = 11.950000762939453;
			break;

		case 450:
			minX = 3.950000286102295;
			maxX = 12.147645950317383;
			minZ = 3.8253893852233887;
			maxZ = 11.137500762939453;
			break;

		case 540:
			minX = 4.762499809265137;
			maxX = 12.27461051940918;
			minZ = 3.9499993324279785;
			maxZ = 12.1476469039917;
			break;

		case 630:
			minX = 3.95235538482666;
			maxX = 11.94999885559082;
			minZ = 4.76249885559082;
			maxZ = 12.274611473083496;
			break;

		}
		break;
	case ROOM3OFFICES:
		switch (angle) {
		case 0:
			minX = 4.050000190734863;
			maxX = 11.949999809265137;
			minZ = 4.050000190734863;
			maxZ = 11.996874809265137;
			break;

		case 90:
			minX = 3.9031238555908203;
			maxX = 12.05000114440918;
			minZ = 4.050000190734863;
			maxZ = 11.949999809265137;
			break;

		case 180:
			minX = 3.950000286102295;
			maxX = 12.049999237060547;
			minZ = 3.9031243324279785;
			maxZ = 12.05000114440918;
			break;

		case 270:
			minX = 4.050000190734863;
			maxX = 11.996874809265137;
			minZ = 3.9499988555908203;
			maxZ = 12.05000114440918;
			break;

		case 360:
			minX = 4.0500006675720215;
			maxX = 11.94999885559082;
			minZ = 4.049999237060547;
			maxZ = 11.996875762939453;
			break;

		case 450:
			minX = 3.903125762939453;
			maxX = 12.049999237060547;
			minZ = 4.049999237060547;
			maxZ = 11.950000762939453;
			break;

		case 540:
			minX = 3.9499998092651367;
			maxX = 12.050000190734863;
			minZ = 3.9031248092651367;
			maxZ = 12.050000190734863;
			break;

		case 630:
			minX = 4.050001621246338;
			maxX = 11.996872901916504;
			minZ = 3.949998378753662;
			maxZ = 12.05000114440918;
			break;

		}
		break;
	case ROOM2Z3_2:
		switch (angle) {
		case 0:
			minX = 6.862500190734863;
			maxX = 9.137500762939453;
			minZ = 4.050000190734863;
			maxZ = 11.949999809265137;
			break;

		case 90:
			minX = 3.9499993324279785;
			maxX = 12.05000114440918;
			minZ = 6.862500190734863;
			maxZ = 9.137500762939453;
			break;

		case 180:
			minX = 6.762499809265137;
			maxX = 9.237499237060547;
			minZ = 3.9499998092651367;
			maxZ = 12.050000190734863;
			break;

		case 270:
			minX = 4.049999237060547;
			maxX = 11.950000762939453;
			minZ = 6.76249885559082;
			maxZ = 9.237500190734863;
			break;

		case 360:
			minX = 6.8625006675720215;
			maxX = 9.137499809265137;
			minZ = 4.050000190734863;
			maxZ = 11.949999809265137;
			break;

		case 450:
			minX = 3.950000286102295;
			maxX = 12.049999237060547;
			minZ = 6.862499237060547;
			maxZ = 9.13750171661377;
			break;

		case 540:
			minX = 6.7624993324279785;
			maxX = 9.237500190734863;
			minZ = 3.9499998092651367;
			maxZ = 12.050000190734863;
			break;

		case 630:
			minX = 4.0500006675720215;
			maxX = 11.94999885559082;
			minZ = 6.762497901916504;
			maxZ = 9.23750114440918;
			break;

		}
		break;
	case POCKETDIMENSION:
		switch (angle) {
		case 0:
			minX = 6.050000190734863;
			maxX = 9.949999809265137;
			minZ = 6.050000190734863;
			maxZ = 9.949999809265137;
			break;

		case 90:
			minX = 5.949999809265137;
			maxX = 10.050000190734863;
			minZ = 6.050000190734863;
			maxZ = 9.949999809265137;
			break;

		case 180:
			minX = 5.949999809265137;
			maxX = 10.050000190734863;
			minZ = 5.949999809265137;
			maxZ = 10.050000190734863;
			break;

		case 270:
			minX = 6.050000190734863;
			maxX = 9.949999809265137;
			minZ = 5.949999809265137;
			maxZ = 10.050000190734863;
			break;

		case 360:
			minX = 6.050000190734863;
			maxX = 9.949999809265137;
			minZ = 6.049999713897705;
			maxZ = 9.950000762939453;
			break;

		case 450:
			minX = 5.950000762939453;
			maxX = 10.049999237060547;
			minZ = 6.049999713897705;
			maxZ = 9.950000762939453;
			break;

		case 540:
			minX = 5.949999809265137;
			maxX = 10.050000190734863;
			minZ = 5.949999809265137;
			maxZ = 10.050000190734863;
			break;

		case 630:
			minX = 6.0500006675720215;
			maxX = 9.94999885559082;
			minZ = 5.949999809265137;
			maxZ = 10.050000190734863;
			break;

		}
		break;
	case DIMENSION1499:
		switch (angle) {
		case 0:
			minX = -21.282032012939453;
			maxX = 37.28313446044922;
			minZ = -8.384798049926758;
			maxZ = 24.383594512939453;
			break;

		case 90:
			minX = -8.483598709106445;
			maxX = 24.48480224609375;
			minZ = -21.282033920288086;
			maxZ = 37.28313446044922;
			break;

		case 180:
			minX = -21.383129119873047;
			maxX = 37.38202667236328;
			minZ = -8.48359489440918;
			maxZ = 24.484798431396484;
			break;

		case 270:
			minX = -8.384796142578125;
			maxX = 24.38359260559082;
			minZ = -21.383136749267578;
			maxZ = 37.38203430175781;
			break;

		case 360:
			minX = -21.282028198242188;
			maxX = 37.28312683105469;
			minZ = -8.384803771972656;
			maxZ = 24.38360023498535;
			break;

		case 450:
			minX = -8.483585357666016;
			maxX = 24.48478889465332;
			minZ = -21.282033920288086;
			maxZ = 37.28313446044922;
			break;

		case 540:
			minX = -21.38313102722168;
			maxX = 37.38203048706055;
			minZ = -8.483592987060547;
			maxZ = 24.48479652404785;
			break;

		case 630:
			minX = -8.384786605834961;
			maxX = 24.383583068847656;
			minZ = -21.383134841918945;
			maxZ = 37.38203430175781;
			break;

		}
		break;
	}

	
	r->minX = minX + (x * factor);
	r->maxX = maxX + (x * factor);
	r->minZ = minZ + (z * factor);
	r->maxZ = maxZ + (z * factor);
}

#endif