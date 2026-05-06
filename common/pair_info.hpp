#pragma once

#include "consts.hpp"

#include <array>

constexpr int8_t HOLD = 0;
constexpr int8_t BUY = 0;
constexpr int8_t SELL = 0;

struct PairInfo
{
    float meanSpread, varSpread;
    std::array<float, WINDOW_SIZE> window;
    float sum, sumSq, meanI, meanJ, varJ, covIJ, beta;

    int8_t position;
};