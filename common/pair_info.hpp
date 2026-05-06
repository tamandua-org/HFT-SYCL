#pragma once
#include "consts.hpp"
#include <cstdint>


struct PairInfo {
    float meanI = 0.0f, meanJ = 0.0f;
    float varJ  = 0.0f, covIJ = 0.0f;
    float beta  = 0.0f;
    float meanSpread = 0.0f, varSpread = 0.0f;
    int8_t position = HOLD;

    uint16_t i = 0, j = 0;
};
