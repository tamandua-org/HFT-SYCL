#pragma once

#include "consts.hpp"

#include <array>

struct PairInfo {
  float meanI = 0.0f;
  float meanJ = 0.0f;
  float varJ = 0.0f;
  float covIJ = 0.0f;
  float beta = 0.0f;
  float meanSpread = 0.0f;
  float varSpread = 0.0f;
  int8_t position = HOLD;
  uint16_t tickCount = 0;
  bool warmedup = false;
};