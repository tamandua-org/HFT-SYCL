#pragma once
#include "consts.hpp"
#include <cstdint>


struct PairInfo {
    float meanI = 0.0f, meanJ = 0.0f;
    float varJ  = 0.0f, covIJ = 0.0f;
    float beta  = 0.0f;
    float meanSpread = 0.0f, varSpread = 0.0f;
    int8_t position = HOLD;
};

// SoA "plana" pensada para vivir en registros del hilo durante el kernel.
// uint16_t en i,j permite hasta 65 535 stocks.
struct FlatPairInfo {
    float    meanI, meanJ, varJ, covIJ;
    float    meanSpread, varSpread, beta;
    uint16_t i, j;
    int8_t   position;
};