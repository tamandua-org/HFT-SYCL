#pragma once

#include "consts.hpp"

#include <array>

constexpr int8_t HOLD = 0;
constexpr int8_t BUY = 1;
constexpr int8_t SELL = -1;

struct PairInfo
{
    // float meanSpread = 0.0f;
    // float varSpread = 0.0f;
    //std::array<float, WINDOW_SIZE> window;
    //float sumSq, meanI, meanJ, varJ, covIJ, beta;
    //int8_t position = HOLD;

    float meanI    = 0.0f;
    float meanJ    = 0.0f;
    float varJ     = 0.0f;
    float covIJ    = 0.0f;
    float beta     = 0.0f;
    float meanSpread = 0.0f;
    float varSpread  = 0.0f;
    int8_t   position  = HOLD;
    uint16_t tickCount = 0;
    bool warmedup = false;

};