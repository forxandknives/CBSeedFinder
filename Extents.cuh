
#ifndef EXTENTS_CUH
#define EXTENTS_CUH

#include "cuda_runtime.h"

__host__ void PopulateRoomExtents(float* e);

__host__ void PopulateRoomExtents(float* e) {
    //HERE WE GO
    //This is gonna be some real piRATe software bullshit right here
    // 
    //We are going to populate a float array of length 48,960 into shared memory. (god help us)
    //This array is 195,840 bytes big. (god help us once again)
    //We have 90 rooms, with 8 angles, and 17*4 extents for each angle.
    //Each room template has an index value to tell you
    //where that templates extents start in this array.
    //They have an index since the room templates are not sequential in this array,
    //some rooms get skipped for having disableOverlapCheck enabled.
    //The layout goes like this.
    //
    //Each room gets 8 angles, and 17*4 extents, which means blocks of 544 floats per room
    //Each angle gets 68 floats for the extents
    //The first 17 values are the minX extents, followed by maxX, then minZ, then maxZ
    //
    //We can get the extents we want from the index value of the roomtemplate.
    //Multiply the index value by (68 times 0 through 7) to get the correct angle.
    //Then multiply by 0 through 3 depending on which type of extent you want. (min/max/x/z)
    //Then take the x and z values of the room and divide by 8.
    //Subtract one from those values, then add them to the index.
    //
    //If lockroom is index 0, if we wanted angle 180 we do (index + (68 * 2)) to get index 136.
    //Depending on which extent we want we do (index + (17 * 0-3))
    //Then we can get the final extent by adding the modified x/z value to the index.


    //Add index to room templates
    //Populate array.
}



#endif