#pragma once

#ifndef CONSTANTS_CUH
#define CONSTANTS_CUH

#include "stdio.h"
#include "cuda_runtime.h"

//Each room gets a number so we dont need it's name since chars no good in cuda.
__device__ __constant__ enum RoomID : uint8_t {
	ROOMEMPTY = 0,
	LOCKROOM = 1,
	ROOM173 = 2,
	START = 3,
	ROOM1123 = 4,
	ROOM1ARCHIVE = 5,
	ROOM2STORAGE = 6,
	ROOM3STORAGE = 7,
	ROOM2TESLA_LCZ = 8,
	ENDROOM = 9,
	ROOM012 = 10,
	ROOM205 = 11,
	ROOM2ID = 12,
	ROOM2_2 = 13,
	ROOM2_3 = 14,
	ROOM2_4 = 15,
	ROOM2_5 = 16,
	ROOM2CID = 17,
	ROOM2C2 = 18,
	ROOM2CLOSETS = 19,
	ROOM2ELEVATOR = 20,
	ROOM2DOORS = 21,
	ROOM2SCPS = 22,
	ROOM860 = 23,
	ROOM2TESTROOM2 = 24,
	ROOM3ID = 25,
	ROOM3_2 = 26,
	ROOM4ID = 27,
	ROOM4_2 = 28,
	ROOMPJ = 29,
	ROOM914 = 30,
	ROOM2GW = 31,
	ROOM2GW_B = 32,
	ROOM1162 = 33,
	ROOM2SCPS2 = 34,
	ROOM2SL = 35,
	LOCKROOM3 = 36,
	ROOM4INFO = 37,
	ROOM3_3 = 38,
	CHECKPOINT1 = 39,
	ROOM008 = 40,
	ROOM035 = 41,
	ROOM049 = 42,
	ROOM106 = 43,
	ROOM513 = 44,
	COFFIN = 45,
	ROOM966 = 46,
	ENDROOM2 = 47,
	TESTROOM = 48,
	TUNNEL = 49,
	TUNNEL2 = 50,
	ROOM2CTUNNEL = 51,
	ROOM2NUKE = 52,
	ROOM2PIPES = 53,
	ROOM2PIT = 54,
	ROOM3PIT = 55,
	ROOM4PIT = 56,
	ROOM2SERVERS = 57,
	ROOM2SHAFT = 58,
	ROOM2TUNNEL = 59,
	ROOM3TUNNEL = 60,
	ROOM4TUNNELS = 61,
	ROOM2TESLA_HCZ = 62,
	ROOM3Z2 = 63,
	ROOM2CPIT = 64,
	ROOM2PIPES2 = 65,
	CHECKPOINT2 = 66,
	ROOM079 = 67,
	LOCKROOM2 = 68,
	EXIT1 = 69,
	GATEAENTRANCE = 70,
	GATEA = 71,
	MEDIBAY = 72,
	ROOM2Z3 = 73,
	ROOM2CAFETERIA = 74,
	ROOM2CZ3 = 75,
	ROOM2CCONT = 76,
	ROOM2OFFICES = 77,
	ROOM2OFFICES2 = 78,
	ROOM2OFFICES3 = 79,
	ROOM2OFFICES4 = 80,
	ROOM2POFFICES = 81,
	ROOM2POFFICES2 = 82,
	ROOM2SROOM = 83,
	ROOM2TOILETS = 84,
	ROOM2TESLA = 85,
	ROOM3SERVERS = 86,
	ROOM3SERVERS2 = 87,
	ROOM3Z3 = 88,
	ROOM4Z3 = 89,
	ROOM1LIFTS = 90,
	ROOM3GW = 91,
	ROOM2SERVERS2 = 92,
	ROOM3OFFICES = 93,
	ROOM2Z3_2 = 94,
	POCKETDIMENSION = 95,
	DIMENSION1499 = 96,
};

__device__ __constant__ const static uint8_t ROOM1  = 1;
__device__ __constant__ const static uint8_t ROOM2  = 2;
__device__ __constant__ const static uint8_t ROOM2C = 3;
__device__ __constant__ const static uint8_t ROOM3  = 4;
__device__ __constant__ const static uint8_t ROOM4  = 5;

__device__ __constant__ const uint8_t MapHeight = 18;
__device__ __constant__ const uint8_t MapWidth  = 18;

__device__ __constant__ const static uint8_t ZONEAMOUNT = 3;

__device__ __constant__ const uint8_t roomTemplateAmount = 97;

#endif