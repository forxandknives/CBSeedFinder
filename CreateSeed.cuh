
#include "Random.cuh"
#include "Constants.cuh"
#include "Helpers.cuh"
#include "Rooms.cuh"

#include "stdio.h"

#ifndef CREATE_SEED_CUH
#define CREATE_SEED_CUH

__device__ inline void InitNewGame(bbRandom bb, rnd_state rnd_state);
__device__ inline void CreateMap(bbRandom bb, rnd_state rnd_state);

__device__ inline void InitNewGame(bbRandom bb, rnd_state rnd_state) {
	/*float tempFloat = bb.bbRnd(rnd_state, 0, 100);
	printf("tempFloat: %f\n", tempFloat);
	//4 Rand(1,9) for the access code
	for (uint8_t i = 0; i < 4; i++) {
		bb.bbRand(rnd_state, 1, 9);
	}
	tempFloat = bb.bbRnd(rnd_state, 0, 100);
	printf("tempFloat: %f\n", tempFloat);
	*/
	CreateMap(bb, rnd_state);
}

__device__ inline void CreateMap(bbRandom bb, rnd_state rnd_state) {
	int32_t x, y, temp;
	int32_t i, x2, y2;
	int32_t width, height;

	int32_t zone;
	
	char* MapName[MapWidth][MapHeight];	
	int32_t MapRoomID[ROOM4 + 1] = { 0 };
	
	x = floorf(float(MapWidth / 2));
	y = MapHeight - 2;

	for (i = y; i <= MapHeight - 1; i++) {
		MapTemp[x][i] = true;
	}

	int32_t tempheight;
	int32_t yhallways;	

	do {		
		width = bb.bbRand(&rnd_state, 10, 15);

		if ((float)x > float(MapWidth * 0.6)) {
			width = -width;
		}
		else if ((float)x > float(MapWidth * 0.4)) {
			x = x - (width / 2);
		}
		if (x + width > MapWidth - 3) {
			width = MapWidth - 3 - x;
		}
		else if (x + width < 2) {
			width = -x + 2;
		}

		x = min(x, x + width);
		width = abs((int)width);

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
					if (bb.bbRand(&rnd_state, 0, 2) == 1) {
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
	for (y = 0; y < MapHeight; y++) {
		uint32_t zone = GetZone(y);
		for (x = 0; x < MapWidth; x++) {
			if (MapTemp[x][y] > 0) {
				temp = min(MapTemp[x + 1][y], 1) + min(MapTemp[x - 1][y], 1);
				temp = temp + min(MapTemp[x][y + 1], 1) + min(MapTemp[x][y - 1], 1);
				if (MapTemp[x][y] < 255) {
					MapTemp[x][y] = temp;
				}
				switch (MapTemp[x][y]) {
				case 1:
					Room1Amount[zone] = Room1Amount[zone] + 1;
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
				case 3:
					Room3Amount[zone] = Room3Amount[zone] + 1;
				case 4:
					Room4Amount[zone] = Room4Amount[zone] + 1;
				}
			}
		}
	}
	//Force more room1s.
	for (i = 0; i <= 2; i++) {
		temp = -Room1Amount[i] + 5;
		if (temp > 0) {
			for (y = (MapHeight / ZoneAmount) * (2 - i) + 1; y <= ((MapHeight / ZoneAmount) * ((2 - 1) + 1.0)) - 2; y++) {
				for (x = 2; x <= MapWidth - 2; x++) {
					if (MapTemp[x][y] = 0) {
						if (min(MapTemp[x + 1][y], 1) + min(MapTemp[x - 1][y], 1) + min(MapTemp[x][y + 1], 1) + min(MapTemp[x][y - 1], 1) == 1) {
							if (MapTemp[x + 1][y]) {
								x2 = x;
								y2 = y;
							}
							else if (MapTemp[x - 1][y]) {
								x2 = x - 1;
								y2 = y;
							} 
							else if (MapTemp[x][y+1]) {
								x2 = 2;
								y2 = y + 1;
							}
							else if (MapTemp[x][y - 1]) {
								x2 = x;
								y2 = y - 1;
							}
							uint32_t placed = false;
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
								case 3:
									Room3Amount[i] = Room3Amount[i] - 1;
									Room4Amount[i] = Room4Amount[i] + 1;
									placed = true;
								}
								if (placed) {
									MapTemp[x2][y2] = MapTemp[x2][y2] + 1;

									MapTemp[x][y] = 1;
									Room1Amount[i] = Room1Amount[i] + 1;
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

	uint32_t temp2;

	//force more room4s and room2cs
	for (i = 0; i <= 2; i++) {

		switch (i) {
		case 2:
			zone = 2;
			temp2 = MapHeight / 3;
		case 1:
			zone = MapHeight / 3 + 1;
			temp2 = MapHeight * (2.0 / 3.0) - 1;
		case 0:
			zone = MapHeight * (2.0 / 3.0) + 1;
			temp2 = MapHeight - 2;
		}

		if (Room4Amount[i] < 1) {
			temp = 0;

			for (y = zone; y <= temp2; y++) {
				for (x = 2; x <= MapWidth - 2; x++) {
					if (MapTemp[x][y] == 3) {
						//switch (0)
						if (temp == 1) {
							MapTemp[x][y] = 4;
							Room4Amount[i] = Room4Amount[i] + 1;
							Room3Amount[i] = Room3Amount[i] + 1;
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
			temp2 = temp2 + 1;

			for (y = zone; y <= temp2; y++) {
				for (x = 3; x <= MapWidth - 3; x++) {
					if (MapTemp[x][y] == 1) {
						if (true) {
							if (MapTemp[x - 1][y] > 0) {
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
								if ((MapTemp[x - 1][y] + MapTemp[x - 1][y + 2] + MapTemp[x - 1][y + 1]) == 0) {
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

	uint32_t MaxRooms = 55 * MapWidth / 20;
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
	uint32_t min_pos = 1;
	uint32_t max_pos = Room1Amount[0] - 1;

	MapRoom[ROOM1][0] = "start\0";
	SetRoom("roompj", ROOM1, floorf(0.1 * float(Room1Amount[0])), min_pos, max_pos);
	SetRoom("914", ROOM1, floorf(0.3 * float(Room1Amount[0])), min_pos, max_pos);
	SetRoom("room1archive", ROOM1, floorf(0.5 * float(Room1Amount[0])), min_pos, max_pos);
	SetRoom("room205", ROOM1, floorf(0.6 * float(Room1Amount[0])), min_pos, max_pos);

	MapRoom[ROOM2C][0] = "lockroom\0";

	min_pos = 1;
	max_pos = Room2Amount[0] - 1;

	MapRoom[ROOM2][0] = "room2closets\0";
	SetRoom("room2testroom2", ROOM2, floorf(0.1 * float(Room2Amount[0])), min_pos, max_pos);
	SetRoom("room2scps", ROOM2, floorf(0.2 * float(Room2Amount[0])), min_pos, max_pos);
	SetRoom("room2storage", ROOM2, floorf(0.3 * float(Room2Amount[0])), min_pos, max_pos);
	SetRoom("room2gw_b", ROOM2, floorf(0.4 * float(Room2Amount[0])), min_pos, max_pos);
	SetRoom("room2sl", ROOM2, floorf(0.5 * float(Room2Amount[0])), min_pos, max_pos);
	SetRoom("room012", ROOM2, floorf(0.55 * float(Room2Amount[0])), min_pos, max_pos);
	SetRoom("room2scps2", ROOM2, floorf(0.6 * float(Room2Amount[0])), min_pos, max_pos);
	SetRoom("room1123", ROOM2, floorf(0.7 * float(Room2Amount[0])), min_pos, max_pos);
	SetRoom("room2elevator", ROOM2, floorf(0.85 * float(Room2Amount[0])), min_pos, max_pos);

	uint32_t tempIndex = floorf(bb.bbRnd(&rnd_state, 0.2, 0.8) * float(Room3Amount[0]));
	MapRoom[ROOM3][tempIndex] = "room3storage";

	tempIndex = floorf(0.5 * float(Room2CAmount[0]));
	MapRoom[ROOM2C][tempIndex] = "room1162";

	tempIndex = floorf(0.3 * float(Room4Amount[0]));
	MapRoom[ROOM4][tempIndex] = "room4info";

	//zone 2 --------------------------------------------------------------------------------------
	
	min_pos = Room1Amount[0];
	max_pos = Room1Amount[0] + Room1Amount[1] - 1;

	SetRoom("room079", ROOM1, Room1Amount[0] + floorf(0.15 * float(Room1Amount[1])), min_pos, max_pos);
	SetRoom("room106", ROOM1, Room1Amount[0] + floorf(0.3 * float(Room1Amount[1])), min_pos, max_pos);
	SetRoom("008", ROOM1, Room1Amount[0] + floorf(0.4 * float(Room1Amount[1])), min_pos, max_pos);
	SetRoom("room035", ROOM1, Room1Amount[0] + floorf(0.5 * float(Room1Amount[1])), min_pos, max_pos);
	SetRoom("coffin", ROOM1, Room1Amount[0] + floorf(0.7 * float(Room1Amount[1])), min_pos, max_pos);

	min_pos = Room2Amount[0];
	max_pos = Room2Amount[0] + Room2Amount[1] - 1;


	tempIndex = Room2Amount[0] + floorf(0.1 * float(Room2Amount[1]));
	MapRoom[ROOM2][tempIndex] = "room2nuke";
	SetRoom("room2tunnel", ROOM2, Room2Amount[0] + floorf(0.25 * float(Room2Amount[1])), min_pos, max_pos);
	SetRoom("room049", ROOM2, Room2Amount[0] + floorf(0.4 * float(Room2Amount[1])), min_pos, max_pos);
	SetRoom("room2shaft", ROOM2, Room2Amount[0] + floorf(0.6 * float(Room2Amount[1])), min_pos, max_pos);
	SetRoom("testroom", ROOM2, Room2Amount[0] + floorf(0.7 * float(Room2Amount[1])), min_pos, max_pos);
	SetRoom("room2servers", ROOM2, Room2Amount[0] + floorf(0.9 * Room2Amount[1]), min_pos, max_pos);

	tempIndex = Room3Amount[0] + floorf(0.3 * float(Room3Amount[1]));
	MapRoom[ROOM3][tempIndex] = "room513";
	tempIndex = Room3Amount[0] + floorf(0.6 * float(Room3Amount[1]));
	MapRoom[ROOM3][tempIndex] = "room966";

	tempIndex = Room2CAmount[0] + floorf(0.5 * float(Room2CAmount[1]));
	MapRoom[ROOM2C][tempIndex] = "room2cpit";
	tempIndex = Room2CAmount[0] + floorf(0.5 * float(Room2CAmount[1]));
	MapRoom[ROOM2C][tempIndex] = "room2cpit";

}
#endif