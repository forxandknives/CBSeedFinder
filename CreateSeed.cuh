#pragma once

#include "Random.cuh"
#include "Constants.cuh"
#include "Helpers.cuh"
#include "Rooms.cuh"

#include "stdio.h"

#ifndef CREATE_SEED_CUH
#define CREATE_SEED_CUH

__device__ inline void InitNewGame(bbRandom bb, rnd_state rnd_state, RoomTemplates* rts);
__device__ inline void CreateMap(bbRandom bb, rnd_state rnd_state, RoomTemplates* rts);

__device__ inline void InitNewGame(bbRandom bb, rnd_state rnd_state, RoomTemplates* rts) {

	CreateMap(bb, rnd_state, rts);
}

__device__ inline void CreateMap(bbRandom bb, rnd_state rnd_state, RoomTemplates* rts) {
	int32_t x = 0, y = 0, temp = 0;
	int32_t i = 0, x2 = 0, y2 = 0;
	int32_t width = 0, height = 0;

	int32_t zone = 0;

	RoomID MapName[MapWidth][MapHeight] = { {ROOMEMPTY, ROOMEMPTY} };
	int32_t MapRoomID[ROOM4 + 1] = { 0 };

	x = floorf(float(MapWidth / 2));
	y = MapHeight - 2;

	for (i = y; i <= MapHeight - 1; i++) {
		MapTemp[x][i] = true;
	}

	int32_t tempheight = 0;
	int32_t yhallways = 0;


	do {
		width = bb.bbRand(&rnd_state, 10, 15);

		if (x > MapWidth * 0.6) {
			width = -width;
		}
		else if (x > MapWidth * 0.4) {
			x = x - (width / 2);
		}

		if (x + width > MapWidth - 3) {
			width = MapWidth - 3 - x;
		}
		else if (x + width < 2) {
			width = -x + 2;
		}

		x = min(x, x + width);
		width = abs((int32_t)width);

		for (i = x; i <= x + width; i++) {
			MapTemp[min(i, MapWidth)][y] = true;
		}

		height = bb.bbRand(&rnd_state, 3, 4);
		if (y - height < 1) {
			height = y - 1;
		}

		yhallways = bb.bbRand(&rnd_state, 4, 5);

		if (GetZone(y - height) != GetZone(y - height + 1)) {
			height = height - 1;
		}

		for (i = 1; i <= yhallways; i++) {
			x2 = max(min(bb.bbRand(&rnd_state, x, x + width - 1), MapWidth - 2), 2);
			while (MapTemp[x2][y - 1] || MapTemp[x2 - 1][y - 1] || MapTemp[x2 + 1][y - 1]) {
				x2++;
			}

			if (x2 < x + width) {
				if (i == 1) {
					tempheight = height;				
					if (bb.bbRand(&rnd_state, 1, 2) == 1) {
						x2 = x;
					}
					else {
						x2 = x + width;
					}
				}
				else {
					tempheight = bb.bbRand(&rnd_state, 1, height);
				}

				for (y2 = y - tempheight; y2 <= y; y2++) {
					if (GetZone(y2) != GetZone(y2 + 1)) {
						MapTemp[x2][y2] = 255;
					}
					else {
						MapTemp[x2][y2] = true;
					}
				}

				if (tempheight == height) {
					temp = x2;
				}
			}
		}
		x = temp;
		y = y - height;				

	} while (y >= 2);			

	int32_t ZoneAmount = 3;
	int32_t Room1Amount[3] = { 0 };
	int32_t Room2Amount[3] = { 0 };
	int32_t Room2CAmount[3] = { 0 };
	int32_t Room3Amount[3] = { 0 };
	int32_t Room4Amount[3] = { 0 };

	//count the amount of rooms
	//might be problems with adding 1 to end number
	//or reducing starting number by 1.
	for (y = 1; y <= MapHeight - 1; y++) {
		uint32_t zone = GetZone(y);
		for (x = 1; x <= MapWidth - 1; x++) {
			if (MapTemp[x][y] > 0) {
				temp = min(MapTemp[x + 1][y], 1) + min(MapTemp[x - 1][y], 1);
				temp = temp + min(MapTemp[x][y + 1], 1) + min(MapTemp[x][y - 1], 1);
				if (MapTemp[x][y] < 255) {
					MapTemp[x][y] = temp;
				}
				switch (MapTemp[x][y]) {
				case 1:
					Room1Amount[zone] = Room1Amount[zone] + 1;
					break;
				case 2:
					if (min(MapTemp[x + 1][y], 1) + min(MapTemp[x - 1][y], 1) == 2) {
						Room2Amount[zone] = Room2Amount[zone] + 1;
					}				
					else if (min(MapTemp[x][y + 1], 1) + min(MapTemp[x][y - 1], 1) == 2) {
						Room2Amount[zone] = Room2Amount[zone] + 1;
					}
					else {
						Room2CAmount[zone] = Room2CAmount[zone] + 1;
					}
					break;
				case 3:
					Room3Amount[zone] = Room3Amount[zone] + 1;
					break;
				case 4:
					Room4Amount[zone] = Room4Amount[zone] + 1;
					break;
				}
			}
		}
	}				

	//Force more room1s.
	for (i = 0; i <= 2; i++) {
		temp = -Room1Amount[i] + 5;
		if (temp > 0) {
			for (y = (MapHeight / ZoneAmount) * (2 - i) + 1; y <= ((MapHeight / ZoneAmount) * ((2 - i) + 1.0)) - 2; y++) {
				for (x = 2; x <= MapWidth - 2; x++) {
					if (MapTemp[x][y] == 0) {
						if (min(MapTemp[x + 1][y], 1) + min(MapTemp[x - 1][y], 1) + min(MapTemp[x][y + 1], 1) + min(MapTemp[x][y - 1], 1) == 1) {
							if (MapTemp[x + 1][y]) {
								x2 = x + 1;
								y2 = y;
							}
							else if (MapTemp[x - 1][y]) {
								x2 = x - 1;
								y2 = y;
							}
							else if (MapTemp[x][y + 1]) {
								x2 = x;
								y2 = y + 1;
							}
							else if (MapTemp[x][y - 1]) {
								x2 = x;
								y2 = y - 1;
							}
							int32_t placed = false;
							if (MapTemp[x2][y2] > 1 && MapTemp[x2][y2] < 4) {
								switch (MapTemp[x2][y2]) {
								case 2:
									if (min(MapTemp[x2 + 1][y2], 1) + min(MapTemp[x2 - 1][y2], 1) == 2) {
										Room2Amount[i] = Room2Amount[i] - 1;
										Room3Amount[i] = Room3Amount[i] + 1;
										placed = true;
									}
									else if (min(MapTemp[x2][y2 + 1], 1) + min(MapTemp[x2][y2 - 1], 1) == 2) {
										Room2Amount[i] = Room2Amount[i] - 1;
										Room3Amount[i] = Room3Amount[i] + 1;
										placed = true;
									}
									break;
								case 3:
									Room3Amount[i] = Room3Amount[i] - 1;
									Room4Amount[i] = Room4Amount[i] + 1;
									placed = true;
									break;
								}
								if (placed) {
									MapTemp[x2][y2] = MapTemp[x2][y2] + 1;

									MapTemp[x][y] = 1;
									Room1Amount[i] = Room1Amount[i] + 1;
									temp = temp - 1;
								}
							}
						}
					}
					if (temp == 0) break;
				}
				if (temp == 0) break;
			}
		}
	}			

	int32_t temp2;

	//force more room4s and room2cs
	for (i = 0; i <= 2; i++) {

		switch (i) {
		case 2:
			zone = 2;
			temp2 = MapHeight / 3;
			break;
		case 1:
			zone = MapHeight / 3 + 1;
			temp2 = MapHeight * (2.0 / 3.0) - 1;
			break;
		case 0:
			zone = MapHeight * (2.0 / 3.0) + 1;
			temp2 = MapHeight - 2;
			break;
		}

		if (Room4Amount[i] < 1) {
			temp = 0;

			for (y = zone; y <= temp2; y++) {
				for (x = 2; x <= MapWidth - 2; x++) {
					if (MapTemp[x][y] == 3) {
						//switch (0)
						//We have to invert each if
						if (!(MapTemp[x + 1][y] || MapTemp[x + 1][y + 1] || MapTemp[x + 1][y - 1] || MapTemp[x + 2][y])) {
							MapTemp[x + 1][y] = 1;
							temp = 1;						
						}
						else if (!(MapTemp[x - 1][y] || MapTemp[x - 1][y + 1] || MapTemp[x - 1][y - 1] || MapTemp[x - 2][y])) {
							MapTemp[x - 1][y] = 1;
							temp = 1;						
						}
						else if (!(MapTemp[x][y + 1] || MapTemp[x + 1][y + 1] || MapTemp[x - 1][y + 1] || MapTemp[x][y + 2])) {
							MapTemp[x][y + 1] = 1;
							temp = 1;
						}
						else if (!(MapTemp[x][y - 1] || MapTemp[x + 1][y - 1] || MapTemp[x - 1][y - 1] || MapTemp[x][y - 2])) {
							MapTemp[x][y - 1] = 1;
							temp = 1;
						}

						if (temp == 1) {
							MapTemp[x][y] = 4;
							Room4Amount[i] = Room4Amount[i] + 1;
							Room3Amount[i] = Room3Amount[i] - 1;
							Room1Amount[i] = Room1Amount[i] + 1;
						}
					}
					if (temp == 1) break;
				}
				if (temp == 1) break;
			}
		}

		if (Room2CAmount[i] < 1) {
			temp = 0;

			zone = zone + 1;
			temp2 = temp2 - 1;

			for (y = zone; y <= temp2; y++) {
				for (x = 3; x <= MapWidth - 3; x++) {
					if (MapTemp[x][y] == 1) {
						if (true) {
							if (MapTemp[x - 1][y] > 0) {
								if ((MapTemp[x][y - 1] + MapTemp[x][y + 1] + MapTemp[x + 2][y]) == 0) {
									if ((MapTemp[x + 1][y - 2] + MapTemp[x + 2][y - 1] + MapTemp[x + 1][y - 1]) == 0) {
										MapTemp[x][y] = 2;
										MapTemp[x + 1][y] = 2;
										MapTemp[x + 1][y - 1] = 1;
										temp = 1;
									}
									else if ((MapTemp[x + 1][y + 2] + MapTemp[x + 2][y + 1] + MapTemp[x + 1][y + 1]) == 0) {
										MapTemp[x][y] = 2;
										MapTemp[x + 1][y] = 2;
										MapTemp[x + 1][y + 1] = 1;
										temp = 1;
									}
								}
							}
							else if (MapTemp[x + 1][y] > 0) {
								if ((MapTemp[x][y - 1] + MapTemp[x][y + 1] + MapTemp[x - 2][y]) == 0) {
									if ((MapTemp[x - 1][y - 2] + MapTemp[x - 2][y - 1] + MapTemp[x - 1][y - 1]) == 0) {
										MapTemp[x][y] = 2;
										MapTemp[x - 1][y] = 2;
										MapTemp[x - 1][y - 1] = 1;
										temp = 1;
									}
									else if ((MapTemp[x - 1][y + 2] + MapTemp[x - 2][y + 1] + MapTemp[x - 1][y + 1]) == 0) {
										MapTemp[x][y] = 2;
										MapTemp[x - 1][y] = 2;
										MapTemp[x - 1][y + 1] = 1;
										temp = 1;
									}
								}
							}
							else if (MapTemp[x][y - 1] > 0) {
								if ((MapTemp[x - 1][y] + MapTemp[x + 1][y] + MapTemp[x][y + 2]) == 0) {
									if ((MapTemp[x - 2][y + 1] + MapTemp[x - 1][y + 2] + MapTemp[x - 1][y + 1]) == 0) {
										MapTemp[x][y] = 2;
										MapTemp[x][y + 1] = 2;
										MapTemp[x - 1][y + 1] = 1;
										temp = 1;
									}
									else if ((MapTemp[x + 2][y + 1] + MapTemp[x + 1][y + 2] + MapTemp[x + 1][y + 1]) == 0) {
										MapTemp[x][y] = 2;
										MapTemp[x][y + 1] = 2;
										MapTemp[x + 1][y + 1] = 1;
										temp = 1;

									}
								}
							}
							else if (MapTemp[x][y + 1] > 0) {
								if ((MapTemp[x - 1][y] + MapTemp[x + 1][y] + MapTemp[x][y - 2]) == 0) {
									if ((MapTemp[x - 2][y - 1] + MapTemp[x - 1][y - 2] + MapTemp[x - 1][y - 1]) == 0) {
										MapTemp[x][y] = 2;
										MapTemp[x][y - 1] = 2;
										MapTemp[x - 1][y - 1] = 1;
										temp = 1;
									}
									else if ((MapTemp[x + 2][y - 1] + MapTemp[x + 1][y - 2] + MapTemp[x + 1][y - 1]) == 0) {
										MapTemp[x][y] = 2;
										MapTemp[x][y - 1] = 2;
										MapTemp[x + 1][y - 1] = 1;
										temp = 1;
									}
								}
							}
						}
						if (temp == 1) {
							Room2CAmount[i] = Room2CAmount[i] + 1;
							Room2Amount[i] = Room2Amount[i] + 1;
						}
					}
					if (temp == 1) break;
				}
				if (temp == 1) break;
			}
		}
	}	

	//EVERYTHING UP TO THIS POINT HAS BEEN VERIFIED TO MATCH
	//THE ORIGINAL GAME 1:1.

	for (int32_t i = 0; i < MapWidth + 1; i++) {
		for (int32_t j = 0; j < MapHeight + 1; j++) {
			printf("MapTemp (%d, %d) %d\n", i, j, MapTemp[i][j]);
		}
	}
	printf("Room1Amount:  %d, %d, %d\n", Room1Amount[0], Room1Amount[1], Room1Amount[2]);
	printf("Room2Amount:  %d, %d, %d\n", Room2Amount[0], Room2Amount[1], Room2Amount[2]);
	printf("Room2CAmount: %d, %d, %d\n", Room2CAmount[0], Room2CAmount[1], Room2CAmount[2]);
	printf("Room3Amount:  %d, %d, %d\n", Room3Amount[0], Room3Amount[1], Room3Amount[2]);
	printf("Room4Amount:  %d, %d, %d\n", Room4Amount[0], Room4Amount[1], Room4Amount[2]);
	printf("RND_STATE: %d\n", rnd_state.rnd_state);

	int32_t MaxRooms = 55 * MapWidth / 20;
	MaxRooms = max(MaxRooms, Room1Amount[0] + Room1Amount[1] + Room1Amount[2] + 1);
	MaxRooms = max(MaxRooms, Room2Amount[0] + Room2Amount[1] + Room2Amount[2] + 1);
	MaxRooms = max(MaxRooms, Room2CAmount[0] + Room2CAmount[1] + Room2CAmount[2] + 1);
	MaxRooms = max(MaxRooms, Room3Amount[0] + Room3Amount[1] + Room3Amount[2] + 1);
	MaxRooms = max(MaxRooms, Room4Amount[0] + Room4Amount[1] + Room4Amount[2] + 1);
	//I am just guessing the max possible rooms here.
	//I dont think it will go above ~150 but 180 just to be safe.
	// Im also moving this into constants.cuh.
	//char* MapRoom[ROOM4 + 1][180] = { "" };

	//zone 1 --------------------------------------------------------------------------------------
	int32_t min_pos = 1;
	int32_t max_pos = Room1Amount[0] - 1;

	MapRoom[ROOM1][0] = START;
	int32_t asd = static_cast<int32_t>(MapRoom[ROOM1][0]);
	SetRoom(ROOMPJ, ROOM1, floorf(0.1 * float(Room1Amount[0])), min_pos, max_pos);
	SetRoom(ROOM914, ROOM1, floorf(0.3 * float(Room1Amount[0])), min_pos, max_pos);
	SetRoom(ROOM1ARCHIVE, ROOM1, floorf(0.5 * float(Room1Amount[0])), min_pos, max_pos);
	SetRoom(ROOM205, ROOM1, floorf(0.6 * float(Room1Amount[0])), min_pos, max_pos);

	MapRoom[ROOM2C][0] = LOCKROOM;

	min_pos = 1;
	max_pos = Room2Amount[0] - 1;

	MapRoom[ROOM2][0] = ROOM2CLOSETS;
	SetRoom(ROOM2TESTROOM2, ROOM2, floorf(0.1 * float(Room2Amount[0])), min_pos, max_pos);
	SetRoom(ROOM2SCPS, ROOM2, floorf(0.2 * float(Room2Amount[0])), min_pos, max_pos);
	SetRoom(ROOM2STORAGE, ROOM2, floorf(0.3 * float(Room2Amount[0])), min_pos, max_pos);
	SetRoom(ROOM2GW_B, ROOM2, floorf(0.4 * float(Room2Amount[0])), min_pos, max_pos);
	SetRoom(ROOM2SL, ROOM2, floorf(0.5 * float(Room2Amount[0])), min_pos, max_pos);
	SetRoom(ROOM012, ROOM2, floorf(0.55 * float(Room2Amount[0])), min_pos, max_pos);
	SetRoom(ROOM2SCPS2, ROOM2, floorf(0.6 * float(Room2Amount[0])), min_pos, max_pos);
	SetRoom(ROOM1123, ROOM2, floorf(0.7 * float(Room2Amount[0])), min_pos, max_pos);
	SetRoom(ROOM2ELEVATOR, ROOM2, floorf(0.85 * float(Room2Amount[0])), min_pos, max_pos);

	int32_t tempIndex = floorf(bb.bbRnd(&rnd_state, 0.2, 0.8) * float(Room3Amount[0]));
	MapRoom[ROOM3][tempIndex] = ROOM3STORAGE;

	tempIndex = floorf(0.5 * float(Room2CAmount[0]));
	MapRoom[ROOM2C][tempIndex] = ROOM1162;

	tempIndex = floorf(0.3 * float(Room4Amount[0]));
	MapRoom[ROOM4][tempIndex] = ROOM4INFO;

	//zone 2 --------------------------------------------------------------------------------------

	min_pos = Room1Amount[0];
	max_pos = Room1Amount[0] + Room1Amount[1] - 1;

	SetRoom(ROOM079, ROOM1, Room1Amount[0] + floorf(0.15 * float(Room1Amount[1])), min_pos, max_pos);
	SetRoom(ROOM106, ROOM1, Room1Amount[0] + floorf(0.3 * float(Room1Amount[1])), min_pos, max_pos);
	SetRoom(ROOM008, ROOM1, Room1Amount[0] + floorf(0.4 * float(Room1Amount[1])), min_pos, max_pos);
	SetRoom(ROOM035, ROOM1, Room1Amount[0] + floorf(0.5 * float(Room1Amount[1])), min_pos, max_pos);
	SetRoom(COFFIN, ROOM1, Room1Amount[0] + floorf(0.7 * float(Room1Amount[1])), min_pos, max_pos);

	min_pos = Room2Amount[0];
	max_pos = Room2Amount[0] + Room2Amount[1] - 1;


	tempIndex = Room2Amount[0] + floorf(0.1 * float(Room2Amount[1]));
	MapRoom[ROOM2][tempIndex] = ROOM2NUKE;
	SetRoom(ROOM2TUNNEL, ROOM2, Room2Amount[0] + floorf(0.25 * float(Room2Amount[1])), min_pos, max_pos);
	SetRoom(ROOM049, ROOM2, Room2Amount[0] + floorf(0.4 * float(Room2Amount[1])), min_pos, max_pos);
	SetRoom(ROOM2SHAFT, ROOM2, Room2Amount[0] + floorf(0.6 * float(Room2Amount[1])), min_pos, max_pos);
	SetRoom(TESTROOM, ROOM2, Room2Amount[0] + floorf(0.7 * float(Room2Amount[1])), min_pos, max_pos);
	SetRoom(ROOM2SERVERS, ROOM2, Room2Amount[0] + floorf(0.9 * Room2Amount[1]), min_pos, max_pos);

	tempIndex = Room3Amount[0] + floorf(0.3 * float(Room3Amount[1]));
	MapRoom[ROOM3][tempIndex] = ROOM513;
	tempIndex = Room3Amount[0] + floorf(0.6 * float(Room3Amount[1]));
	MapRoom[ROOM3][tempIndex] = ROOM966;

	tempIndex = Room2CAmount[0] + floorf(0.5 * float(Room2CAmount[1]));
	MapRoom[ROOM2C][tempIndex] = ROOM2CPIT;	

	//zone 3 --------------------------------------------------------------------------------------

	MapRoom[ROOM1][Room1Amount[0] + Room1Amount[1] + Room1Amount[2] - 2] = EXIT1;
	MapRoom[ROOM1][Room1Amount[0] + Room1Amount[1] + Room1Amount[2] - 1] = GATEAENTRANCE;
	MapRoom[ROOM1][Room1Amount[0] + Room1Amount[1]] = ROOM1LIFTS;

	min_pos = Room2Amount[0] + Room2Amount[1];
	max_pos = Room2Amount[0] + Room2Amount[1] + Room2Amount[2] - 1;

	tempIndex = min_pos + floorf(0.1 * float(Room2Amount[2]));
	MapRoom[ROOM2][tempIndex] = ROOM2POFFICES;
	SetRoom(ROOM2CAFETERIA, ROOM2, min_pos + floorf(0.2 * float(Room2Amount[2])), min_pos, max_pos);
	SetRoom(ROOM2SROOM, ROOM2, min_pos + floorf(0.3 * float(Room2Amount[2])), min_pos, max_pos);
	SetRoom(ROOM2SERVERS2, ROOM2, min_pos + floorf(0.4 * Room2Amount[2]), min_pos, max_pos);
	SetRoom(ROOM2OFFICES, ROOM2, min_pos + floorf(0.45 * Room2Amount[2]), min_pos, max_pos);
	SetRoom(ROOM2OFFICES4, ROOM2, min_pos + floorf(0.5 * Room2Amount[2]), min_pos, max_pos);
	SetRoom(ROOM860, ROOM2, min_pos + floorf(0.6 * Room2Amount[2]), min_pos, max_pos);
	SetRoom(MEDIBAY, ROOM2, min_pos + floorf(0.7 * float(Room2Amount[2])), min_pos, max_pos);
	SetRoom(ROOM2POFFICES2, ROOM2, min_pos + floorf(0.8 * Room2Amount[2]), min_pos, max_pos);
	SetRoom(ROOM2OFFICES2, ROOM2, min_pos + floorf(0.9 * float(Room2Amount[2])), min_pos, max_pos);

	MapRoom[ROOM2C][Room2CAmount[0] + Room2CAmount[1]] = ROOM2CCONT;
	MapRoom[ROOM2C][Room2CAmount[0] + Room2CAmount[1] + 1] = LOCKROOM2;

	tempIndex = Room3Amount[0] + Room3Amount[1] + floorf(0.3 * float(Room3Amount[2]));
	MapRoom[ROOM3][tempIndex] = ROOM3SERVERS;
	tempIndex = Room3Amount[0] + Room3Amount[1] + floorf(0.7 * float(Room3Amount[2]));
	MapRoom[ROOM3][tempIndex] = ROOM3SERVERS2;
	tempIndex = Room3Amount[0] + Room3Amount[1] + floorf(0.5 * float(Room3Amount[2]));
	MapRoom[ROOM3][tempIndex] = ROOM3OFFICES;

	//-----------------------------------------------------------------------------------------
	temp = 0;
	//Rooms r;
	float spacing = 8.0;

	//we are going to attempt to use an array of Rooms to store the created rooms.
	//Guessing that we never go past 180 rooms
	Rooms rooms[MapHeight*MapWidth] = { NULL };
	int32_t roomsIndex = 0; //we must increment this after creating each room;		

	for (y = MapHeight-1; y >= 1; y--) {
		if (y < MapHeight / 3 + 1) {
			zone = 3;
		}
		else if (y < MapHeight * (2.0 / 3.0)) {
			zone = 2;
		}
		else {
			zone = 1;
		}

		for (x = 1; x <= MapWidth - 2; x++) {
			if (MapTemp[x][y] == 255) {
				if (y > MapHeight / 2) {
					rooms[roomsIndex++] = CreateRoom(rts, bb, rnd_state, zone, ROOM2, x * 8, 0, y * 8, CHECKPOINT1);
				}
				else {
					rooms[roomsIndex++] = CreateRoom(rts, bb, rnd_state, zone, ROOM2, x * 8, 0, y * 8, CHECKPOINT2);
				}
			}
			else if (MapTemp[x][y] > 0) {
				temp = min(MapTemp[x + 1][y], 1) + min(MapTemp[x - 1][y], 1) + min(MapTemp[x][y + 1], 1) + min(MapTemp[x][y - 1], 1);

				switch (temp) {
				case 1:
					if (MapRoomID[ROOM1] < MaxRooms && MapName[x][y] == ROOMEMPTY) {
						if (MapRoom[ROOM1][MapRoomID[ROOM1]] != ROOMEMPTY) {
							MapName[x][y] = MapRoom[ROOM1][MapRoomID[ROOM1]];
						}
					}

					//we do not increment roomsIndex here because we need to edit the room at this index
					rooms[roomsIndex] = CreateRoom(rts, bb, rnd_state, zone, ROOM1, x * 8, 0, y * 8, MapName[x][y]);
					if (MapTemp[x][y + 1]) {
						rooms[roomsIndex].angle = 180;
					}
					else if (MapTemp[x][y - 1]) {
						rooms[roomsIndex].angle = 270;
					}
					else if (MapTemp[x + 1][y]) {
						rooms[roomsIndex].angle = 90;
					}
					else {
						rooms[roomsIndex].angle = 0;
					}
					//Increment here since we didnt do it earlier.
					roomsIndex++;
					MapRoomID[ROOM1] = MapRoomID[ROOM1] + 1;

				case 2:
					//we gotta do more not incrementing roomsIndex here so might be wrong.
					if (MapTemp[x - 1][y] > 0 && MapTemp[x + 1][y] > 0) {
						if (MapRoomID[ROOM2] < MaxRooms && MapName[x][y] == ROOMEMPTY) {
							if (MapRoom[ROOM2][MapRoomID[ROOM2]] != ROOMEMPTY) {
								MapName[x][y] = MapRoom[ROOM2][MapRoomID[ROOM2]];
							}
						}
						rooms[roomsIndex] = CreateRoom(rts, bb, rnd_state, zone, ROOM2, x * 8, 0, y * 8, MapName[x][y]);
						if (bb.bbRand(&rnd_state, 0, 2) == 1) {
							rooms[roomsIndex].angle = 90;
						}
						else {
							rooms[roomsIndex].angle = 270;
						}
						roomsIndex++;
						MapRoomID[ROOM2] = MapRoomID[ROOM2] + 1;

					}
					else if (MapTemp[x][y - 1] > 0 && MapTemp[x][y + 1] > 0) {
						if (MapRoomID[ROOM2] < MaxRooms && MapName[x][y] == ROOMEMPTY) {
							if (MapRoom[ROOM2][MapRoomID[ROOM2]] != ROOMEMPTY) {
								MapName[x][y] = MapRoom[ROOM2][MapRoomID[ROOM2]];
							}
						}
						rooms[roomsIndex] = CreateRoom(rts, bb, rnd_state,zone, ROOM2, x * 8, 0, y * 8, MapName[x][y]);
						if (bb.bbRand(&rnd_state, 0, 2) == 1) {
							rooms[roomsIndex].angle = 180;
						}
						else {
							rooms[roomsIndex].angle = 0;
						}
						roomsIndex++;
						MapRoomID[ROOM2] = MapRoomID[ROOM2] + 1;

					}
					else {
						if (MapRoomID[ROOM2C] < MaxRooms && MapName[x][y] == ROOMEMPTY) {
							if (MapRoom[ROOM2C][MapRoomID[ROOM2C]] != ROOMEMPTY) {
								MapName[x][y] = MapRoom[ROOM2C][MapRoomID[ROOM2C]];
							}
						}

						if (MapTemp[x - 1][y] > 0 && MapTemp[x][y + 1] > 0) {
							rooms[roomsIndex] = CreateRoom(rts, bb, rnd_state, zone, ROOM2C, x * 8, 0, y * 8, MapName[x][y]);
							rooms[roomsIndex].angle = 180;
							roomsIndex++;
						}
						else if (MapTemp[x + 1][y] > 0 && MapTemp[x][y + 1] > 0) {
							rooms[roomsIndex] = CreateRoom(rts, bb, rnd_state, zone, ROOM2C, x * 8, 0, y * 8, MapName[x][y]);
							rooms[roomsIndex].angle = 90;
							roomsIndex++;
						}
						else if (MapTemp[x - 1][y] > 0 && MapTemp[x][y - 1] > 0) {
							rooms[roomsIndex] = CreateRoom(rts, bb, rnd_state, zone, ROOM2C, x * 8, 0, y * 8, MapName[x][y]);
							rooms[roomsIndex].angle = 270;
							roomsIndex++;
						}
						else {
							rooms[roomsIndex] = CreateRoom(rts, bb, rnd_state, zone, ROOM2C, x * 8, 0, y * 8, MapName[x][y]);
							roomsIndex++;
						}						
						MapRoomID[ROOM2C] = MapRoomID[ROOM2C] + 1;
					}

				case 3:
					if (MapRoomID[ROOM3] < MaxRooms && MapName[x][y] == ROOMEMPTY) {
						if (MapRoom[ROOM3][MapRoomID[ROOM3]] != ROOMEMPTY) {
							MapName[x][y] = MapRoom[ROOM3][MapRoomID[ROOM3]];
						}
					}
					rooms[roomsIndex] = CreateRoom(rts, bb, rnd_state, zone, ROOM3, x * 8, 0, y * 8, MapName[x][y]);
					if (!MapTemp[x][y - 1]) {
						rooms[roomsIndex].angle = 180;
						roomsIndex++;
					}
					else if (!MapTemp[x - 1][y]) {
						rooms[roomsIndex].angle = 90;
						roomsIndex++;
					}
					else if (!MapTemp[x + 1][y]) {
						rooms[roomsIndex].angle = 270;
						roomsIndex++;
					}
					else {
						roomsIndex++;
					}
					MapRoomID[ROOM3] = MapRoomID[ROOM3] + 1;

				case 4:
					if (MapRoomID[ROOM4] < MaxRooms && MapName[x][y] == ROOMEMPTY) {
						if (MapRoom[ROOM4][MapRoomID[ROOM4]] != ROOMEMPTY) {
							MapName[x][y] = MapRoom[ROOM4][MapRoomID[ROOM4]];
						}
					}
					rooms[roomsIndex++] = CreateRoom(rts, bb, rnd_state, zone, ROOM4, x * 8, 0, y * 8, MapName[x][y]);
					MapRoomID[ROOM4] = MapRoomID[ROOM4] + 1;
				}
			}
		}
	}

	rooms[roomsIndex++] = CreateRoom(rts, bb, rnd_state, 0, ROOM1, (MapWidth - 1) * 8, 500, 8, GATEA);
	MapRoomID[ROOM1] = MapRoomID[ROOM1] + 1;

	rooms[roomsIndex++] = CreateRoom(rts, bb, rnd_state, 0, ROOM1, (MapWidth - 1) * 8, 0, (MapHeight - 1) * 8, POCKETDIMENSION);
	MapRoomID[ROOM1] = MapRoomID[ROOM1] + 1;

	//If introenabled
	//dont need bc intro never on

	rooms[roomsIndex++] = CreateRoom(rts, bb, rnd_state, 0, ROOM1, 8, 800, 0, DIMENSION1499);
	MapRoomID[ROOM1] = MapRoomID[ROOM1] + 1;
	
	//18*18 because that is length of rooms array
	for (i = 0; i < 18*18; i++) {
		//idk if theres a good way of checking if there is nothing
		//at the index of the rooms array.
		//if the id is -1 that means
		//if (rooms[i].rt.id == -1) break;

		PreventRoomOverlap(rooms, i);
	}

	for (y = 0; y <= MapHeight; y++) {
		for (x = 0; x <= MapWidth; x++) {
			MapTemp[x][y] = min(MapTemp[x][y], 1);
		}
	}
	
	/*
	if (threadIdx.x == 0) {
		uint32_t tempCounter = 0;
		for (int32_t i = 0; i < 18 * 18; i++) {				
			Rooms* r = &rooms[i];

			if (r->id == -1) tempCounter++;

			printf("THREAD: %d\n", threadIdx.x);
			printf("NAME: %d %s\n", static_cast<int32_t>(r->rt.name), RoomIDToName(r->rt.name));
			printf("ROOM ID: %d\n", r->id);
			printf("RT ID: %d\n", r->rt.id);
			printf("X: %f\n", r->x / 8);
			printf("Z: %f\n", r->z / 8);
			printf("\n");						
		}		
		printf("Found %d null rooms.\n", tempCounter);
	}		

	for (int32_t i = 0; i < MapWidth; i++) {
		for (int32_t j = 0; j < MapHeight; j++) {
			printf("(%d, %d) NAME: %s\n", i, j, RoomIDToName(MapName[i][j]));
		}
	}
	*/


}
#endif