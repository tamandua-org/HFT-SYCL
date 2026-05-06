#pragma once

#include "consts.hpp"

struct Asset {
    float ticks[N_TICKS];
};


struct FlatPairInfo {
    float meanI, meanJ, varJ, covIJ;
    float meanSpread, varSpread, beta;
    int   position;
    int   i, j;  // índices originales
};