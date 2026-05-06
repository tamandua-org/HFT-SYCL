#pragma once

#include "tick.hpp"
#include <array>
#include <vector>
#include <cstdint>

std::array<uint8_t, N_STOCKS> runTicks(const std::vector<Tick>& ticks);
