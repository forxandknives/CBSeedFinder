
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
	char* name;
	int32_t commonness;
	bool large;
	int32_t useLightCones;
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
__device__ inline bool SetRoom(char* room_name, uint32_t room_type, uint32_t pos, uint32_t min_pos, uint32_t max_pos);
__device__ inline Rooms CreateRoom(RoomTemplates* rts, bbRandom bb, rnd_state rnd_sate, int32_t zone, int32_t roomshape, float x, float y, float z, char* name);
__device__ inline bool PreventRoomOverlap(Rooms* rooms, int32_t index);
__device__ inline bool CheckRoomOverlap(Rooms* r, Rooms* r2);
__device__ inline void CalculateRoomExtents(Rooms* r);
__device__ inline void FillRoom(bbRandom bb, rnd_state* rnd_state, Rooms* r);

__device__ inline void CreateRoomTemplates(RoomTemplates* rt) {

	int32_t counter = 1;

	//All commonness is defined as max(min(commonness, 100), 0);
	//Rooms with disableoverlapcheck = true do not have min and max extents.
	//The name variable is not a string but instead an enum for the ID of the room the roomtemplate represents.
	//This is so we do no have to do char stuff.

	//LIGHT CONTAINMENT

	//lockroom
	rt[counter].id = counter;
	rt[counter].name = "lockroom";
	rt[counter].shape = ROOM2C;
	rt[counter].zone[0] = 1;
	rt[counter].zone[1] = 3;
	rt[counter].commonness = 30;
	rt[counter].minX = -864.0;
	rt[counter].minY = 0.0;
	rt[counter].minZ = -1024.0;
	rt[counter].maxX = 1024.0;
	rt[counter].maxY = 640.0;
	rt[counter].maxZ = 864.0;
	counter++;

	//173
	rt[counter].id = counter;
	rt[counter].name = "173";
	rt[counter].shape = ROOM1;
	rt[counter].commonness = 0;
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
	rt[counter].name = "start";
	rt[counter].shape = ROOM1;
	rt[counter].commonness = 0;
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
	rt[counter].name = "room1123";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 0;
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
	rt[counter].name = "room1archive";
	rt[counter].shape = ROOM1;
	rt[counter].commonness = 80;
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
	rt[counter].name = "room2storage";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 0;
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
	rt[counter].minX = -304.00012;
	rt[counter].minY = -64.0;
	rt[counter].minZ = -1024.0;
	rt[counter].maxX = 304.0001;
	rt[counter].maxY = 714.83057;
	rt[counter].maxZ = 1024.0;
	counter++;

	//endroom
	rt[counter].id = counter;
	rt[counter].name = "endroom";
	rt[counter].shape = ROOM1;
	rt[counter].commonness = 100;
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
	rt[counter].name = "room012";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 0;
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
	rt[counter].name = "room205";
	rt[counter].shape = ROOM1;
	rt[counter].commonness = 0;
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
	rt[counter].name = "room2";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 45;
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
	rt[counter].name = "room2_2";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 40;
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
	rt[counter].name = "room2_3";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 35;
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
	rt[counter].name = "room2_4";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 35;
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
	rt[counter].name = "room2_5";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 35;
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
	rt[counter].name = "room2c";
	rt[counter].shape = ROOM2C;
	rt[counter].commonness = 30;
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
	rt[counter].name = "room2c2";
	rt[counter].shape = ROOM2C;
	rt[counter].commonness = 30;
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
	rt[counter].name = "room2closets";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 0;
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
	rt[counter].name = "room2elevator";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 20;
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
	rt[counter].name = "room2doors";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 30;
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
	rt[counter].name = "room2scps";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 0;
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
	rt[counter].name = "room860";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 0;
	rt[counter].minX = -304.0;
	rt[counter].minY = -32.0;
	rt[counter].minZ = -1024.0;
	rt[counter].maxX = 1280.0;
	rt[counter].maxY = 726.83057;
	rt[counter].maxZ = 1024.0;
	counter++;

	//room2testroom2
	rt[counter].id = counter;
	rt[counter].name = "room2testroom2";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 0;
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
	rt[counter].name = "room3";
	rt[counter].shape = ROOM3;
	rt[counter].commonness = 100;
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
	rt[counter].name = "room3_2";
	rt[counter].shape = ROOM3;
	rt[counter].commonness = 100;
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
	rt[counter].name = "room4";
	rt[counter].shape = ROOM4;
	rt[counter].commonness = 100;
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
	rt[counter].name = "room4_2";
	rt[counter].shape = ROOM4;
	rt[counter].commonness = 80;
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
	rt[counter].name = "roompj";
	rt[counter].shape = ROOM1;
	rt[counter].commonness = 0;
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
	rt[counter].name = "914";
	rt[counter].shape = ROOM1;
	rt[counter].commonness = 0;
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
	rt[counter].name = "room2gw";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 10;
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
	rt[counter].name = "room2gw_b";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 0;
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
	rt[counter].name = "room1162";
	rt[counter].shape = ROOM2C;
	rt[counter].commonness = 0;
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
	rt[counter].name = "room2scps2";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 0;
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
	rt[counter].name = "room2sl";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 0;
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
	rt[counter].name = "lockroom3";
	rt[counter].shape = ROOM2C;
	rt[counter].commonness = 15;
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
	rt[counter].name = "room4info";
	rt[counter].shape = ROOM4;
	rt[counter].commonness = 0;
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
	rt[counter].name = "room3_3";
	rt[counter].shape = ROOM3;
	rt[counter].commonness = 20;
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
	rt[counter].name = "checkpoint1";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 0;
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
	rt[counter].name = "008";
	rt[counter].shape = ROOM1;
	rt[counter].commonness = 0;
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
	rt[counter].name = "room035";
	rt[counter].shape = ROOM1;
	rt[counter].commonness = 0;
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
	rt[counter].minX = -1304.0;
	rt[counter].minY = -1296.0;
	rt[counter].minZ = -1024.0;
	rt[counter].maxX = 2256.0;
	rt[counter].maxY = 1687.9999;
	rt[counter].maxZ = 3120.0;
	counter++;

	//rom513
	rt[counter].id = counter;
	rt[counter].name = "room513";
	rt[counter].shape = ROOM3;
	rt[counter].commonness = 0;
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
	rt[counter].name = "coffin";
	rt[counter].shape = ROOM1;
	rt[counter].commonness = 0;
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
	rt[counter].minX = -480.00003;
	rt[counter].minY = -0.56270504;
	rt[counter].minZ = -1024.0;
	rt[counter].maxX = 464.0;
	rt[counter].maxY = 1056.0;
	rt[counter].maxZ = 376.0;
	counter++;

	//testroom
	rt[counter].id = counter;
	rt[counter].name = "testroom";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 0;
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
	rt[counter].name = "tunnel";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 100;
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
	rt[counter].name = "tunnel2";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 70;
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
	rt[counter].name = "room2ctunnel";
	rt[counter].shape = ROOM2C;
	rt[counter].commonness = 40;
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
	rt[counter].name = "room2nuke";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 0;
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
	rt[counter].name = "room2pipes";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 50;
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
	rt[counter].name = "room2pit";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 75;
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
	rt[counter].name = "room3pit";
	rt[counter].shape = ROOM3;
	rt[counter].commonness = 100;
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
	rt[counter].name = "room4pit";
	rt[counter].shape = ROOM4;
	rt[counter].commonness = 100;
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
	rt[counter].name = "room2servers";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 0;
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
	rt[counter].name = "room2shaft";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 0;
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
	rt[counter].name = "room2tunnel";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 0;
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
	rt[counter].name = "room3tunnel";
	rt[counter].shape = ROOM3;
	rt[counter].commonness = 100;
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
	rt[counter].name = "room4tunnels";
	rt[counter].shape = ROOM4;
	rt[counter].commonness = 100;
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
	rt[counter].name = "room2tesla_hcz";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 100;
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
	rt[counter].name = "room3z2";
	rt[counter].shape = ROOM3;
	rt[counter].commonness = 100;
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
	rt[counter].name = "room2cpit";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 0;
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
	rt[counter].name = "room2pipes2";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 70;
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
	rt[counter].name = "checkpoint2";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 0;
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
	rt[counter].name = "room079";
	rt[counter].shape = ROOM1;
	rt[counter].commonness = 0;
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
	rt[counter].name = "lockroom2";
	rt[counter].shape = ROOM2C;
	rt[counter].commonness = 0;
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
	rt[counter].minX = -720.0;
	rt[counter].minY = -32.0;
	rt[counter].minZ = -1136.0;
	rt[counter].maxX = 1360.0;
	rt[counter].maxY = 1328.0;
	rt[counter].maxZ = 1240.0;
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
	rt[counter].minX = -1024.0002;
	rt[counter].minY = -1.8437233;
	rt[counter].minZ = -1025.0;
	rt[counter].maxX = 288.0;
	rt[counter].maxY = 386.86813;
	rt[counter].maxZ = 1024.0;
	counter++;

	//room2z3
	rt[counter].id = counter;
	rt[counter].name = "room2z3";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 75;
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
	rt[counter].name = "room2cafeteria";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 0;
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
	rt[counter].name = "room2cz3";
	rt[counter].shape = ROOM2C;
	rt[counter].commonness = 100;
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
	rt[counter].name = "room2ccont";
	rt[counter].shape = ROOM2C;
	rt[counter].commonness = 0;
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
	rt[counter].name = "room2offices";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 30;
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
	rt[counter].name = "room2offices2";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 20;
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
	rt[counter].name = "room2offices3";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 20;
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
	rt[counter].name = "room2offices4";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 0;
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
	rt[counter].name = "room2poffices";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 0;
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
	rt[counter].name = "room2poffices2";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 0;
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
	rt[counter].name = "room2sroom";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 0;
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
	rt[counter].name = "room2toilets";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 30;
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
	rt[counter].name = "room2tesla";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 100;
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
	rt[counter].name = "room3servers";
	rt[counter].shape = ROOM3;
	rt[counter].commonness = 0;
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
	rt[counter].name = "room3servers2";
	rt[counter].shape = ROOM3;
	rt[counter].commonness = 0;
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
	rt[counter].name = "room3z3";
	rt[counter].shape = ROOM3;
	rt[counter].commonness = 100;
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
	rt[counter].name = "room4z3";
	rt[counter].shape = ROOM4;
	rt[counter].commonness = 100;
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
	rt[counter].name = "room1lifts";
	rt[counter].shape = ROOM1;
	rt[counter].commonness = 0;
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
	rt[counter].name = "room3gw";
	rt[counter].shape = ROOM3;
	rt[counter].commonness = 10;
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
	rt[counter].name = "room2servers2";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 0;
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
	rt[counter].name = "room3offices";
	rt[counter].shape = ROOM3;
	rt[counter].commonness = 0;
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
	rt[counter].name = "room2z3_2";
	rt[counter].shape = ROOM2;
	rt[counter].commonness = 25;
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
	rt[counter].name = "pocketdimension";
	rt[counter].shape = ROOM1;
	rt[counter].commonness = 0;
	rt[counter].minX = -512.0;
	rt[counter].minY = -32.0;
	rt[counter].minZ = -512.0;
	rt[counter].maxX = 512.0;
	rt[counter].maxY = 1024.0;
	rt[counter].maxZ = 512.0;
	counter++;

	//dimension1499
	rt[counter].id = counter;
	rt[counter].name = "dimension1499";
	rt[counter].shape = ROOM1;
	rt[counter].commonness = 0;
	rt[counter].disableDecals = true;
	rt[counter].minX = -7509.0;
	rt[counter].minY = -672.0001;
	rt[counter].minZ = -4207.308;
	rt[counter].maxX = 7509.2817;
	rt[counter].maxY = 8928.0;
	rt[counter].maxZ = 4207.0;
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

__device__ inline Rooms CreateRoom(RoomTemplates* rts, bbRandom bb, rnd_state rnd_state, int32_t zone, int32_t roomshape, float x, float y, float z, char* name) {

	Rooms r = Rooms();
	//RoomTemplates* rt;

	//TEMPORARY
	r.id = roomIdCounter++;

	r.zone = zone;

	r.x = x;
	r.y = y;
	r.z = z;

	if (name != "") {
		for (int32_t i = 0; i < roomTemplateAmount; i++) {
			if (rts[i].name == name) {
				r.rt = rts[i];
				if (r.obj == 0) {
					//INCOMPLETE
					//LoadRoomMesh(rts[i]);				
				}
				//INCOMPLETE
				//FillRoom(r);

				//Don't think we need light cone stuff.

				CalculateRoomExtents(&r);
				return r;
			}
		}
	}


	int32_t t = 0;
	for (int32_t i = 0; i < roomTemplateAmount; i++) {
		//5 because that is the len of the rt.zone[] array;
		for (int32_t j = 0; j < 5; j++) {
			if (rts[i].zone[j] == zone) {
				if (rts[i].shape == roomshape) {
					t = t + rts[i].commonness;
					break;
				}
			}
		}
	}
	

	int32_t RandomRoom = bb.bbRand(&rnd_state, 0, t);
	t = 0;
	for (int32_t i = 0; i < roomTemplateAmount; i++) {
		RoomTemplates* rt = &rts[i];
		for (int32_t j = 0; j <= 4; j++) {
			if (rt->zone[j] == zone && rt->shape == roomshape) {
				t = t + rt->commonness;
				if (RandomRoom > t - rt->commonness && RandomRoom <= t) {
					r.rt = *rt;

					if (rt->obj == 0) {
						//INCOMPLETE
						//LoadRoomMesh(rt);
					}

					//INCOMPLETE
					FillRoom(bb, &rnd_state, &r);

					//Skip light cone stuff

					CalculateRoomExtents(&r);
					return r;
				}
			}
		}
	}	
	//It seems like there is supposed to be a return r at the bottom here,
	//but it isn't there in the blitz code so idk.
}

__device__ inline bool PreventRoomOverlap(Rooms* rooms, int32_t index) {
	if (rooms[index].rt.disableOverlapCheck) return true;


	//CLEANUP
	//We should probably make this a pointer instead of a copy.
	Rooms* r = &rooms[index];

	Rooms* r2;
	Rooms* r3;

	bool isIntersecting = false;

	if (r->rt.name == "checkpoint1" || r->rt.name == "checkpoint2" || r->rt.name == "start") return true;

	for (int32_t i = 0; i < 324; i++) {
		r2 = &rooms[i];
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
		CalculateRoomExtents(r);

		for (int32_t i = 0; i < 18 * 18; i++) {
			if (rooms[i].id == -1) break; //Id should only be -1 after all other rooms.
			r2 = &rooms[i];

			if (r2->id != r->id && !r2->rt.disableOverlapCheck) {
				if (CheckRoomOverlap(r, r2)) {
					isIntersecting = true;
					r->angle = r->angle - 180;
					CalculateRoomExtents(r);
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
		if (rooms[i].id == -1) break;
		r2 = &rooms[i];

		if (r2->id != r->id && !r2->rt.disableOverlapCheck) {
			if (r->rt.shape == r2->rt.shape && r->zone == r2->zone && (r2->rt.name != "checkpoint1" && r2->rt.name != "checkpoint2" && r2->rt.name != "start")) {
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
				CalculateRoomExtents(r);

				r2->x = x * 8.0;
				r2->z = y * 8.0;
				r2->angle = rot;
				CalculateRoomExtents(r2);

				//make sure neither room overlaps with anything after the swap
				for (int32_t i = 0; i < 18 * 18; i++) {
					if (rooms[i].id == -1) break;
					r3 = &rooms[i];

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
					CalculateRoomExtents(r);

					r2->x = x2 * 8.0;
					r2->z = y2 * 8.0;
					r2->angle = rot2;
					CalculateRoomExtents(r2);

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
	return;
}

__device__ inline void FillRoom(bbRandom bb, rnd_state* rnd_state, Rooms* r) {
	

}

__device__ inline void CreateDoor(rnd_state* rnd_state, bool open, int32_t big) {

}

#endif