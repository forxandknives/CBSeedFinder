
#include "Random.cuh"
#include "Constants.cuh"
#include "Helpers.cuh"

#include "stdio.h"

#ifndef CREATE_SEED_CUH
#define CREATE_SEED_CUH

__device__ inline void InitNewGame(bbRandom bb, rnd_state rnd_state);
__device__ inline void CreateMap(bbRandom bb, rnd_state rnd_state);

__device__ inline void InitNewGame(bbRandom bb, rnd_state rnd_state) {
	//4 Rand(1,9) for the access code
	for (uint8_t i = 0; i < 4; i++) {
		bb.bbRand(&rnd_state, 1, 9);
	}

	CreateMap(bb, rnd_state);
}

__device__ inline void CreateMap(bbRandom bb, rnd_state rnd_state) {
	uint32_t x, y, temp;
	uint32_t i, x2, y2;
	uint32_t width, height;

	uint32_t zone;
	
	char* MapName[MapWidth][MapHeight];	
	uint32_t MapRoomID[ROOM4 + 1] = { 0 };
	
	x = floorf(MapWidth / 2);
	y = MapHeight - 2;

	for (i = y; i <= MapHeight; i++) {
		MapTemp[x][i] = true;
	}

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
		width = abs((int)width);

		for (i = x; i <= x + width; i++) {
			MapTemp[min(i, MapWidth)][y] = true;
		}

		height = bb.bbRand(&rnd_state, 3, 4);
		if (y - height < 1) {
			height = y - 1;
		}

		uint32_t yhallways = bb.bbRand(&rnd_state, 4, 5);

		if (GetZone(y - height) != GetZone(y - height + 1)) {
			height = height + 1;
		}

		for (i = 1; i <= yhallways; i++) {
			x2 = max(min(bb.bbRand(&rnd_state, x, x + width - 1), MapWidth - 2), 2);
			while (MapTemp[x2][y - 1] || MapTemp[x2 - 1][y - 1] || MapTemp[x2 + 1][y - 1]) {
				x2 += 1;
			}

			uint32_t tempheight;

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

	} while (y < 2);

}



#endif