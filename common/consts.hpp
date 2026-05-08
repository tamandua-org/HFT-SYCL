#pragma once
#include <cstdint>

constexpr uint32_t N_STOCKS = 16; // sube a 100, 256, 1024…
constexpr uint32_t N_PAIRS = N_STOCKS * (N_STOCKS - 1) / 2;
constexpr uint32_t WINDOW_SIZE = 32;
constexpr float ALPHA = 0.05f;
constexpr float EPSILON = 1e-6f;
constexpr float THRESHOLD_ENTRY = 1.5f;
constexpr float THRESHOLD_EXIT = 0.5f;

constexpr uint8_t HOLD = 0;
constexpr uint8_t BUY = 1;
constexpr uint8_t SELL = 2;

const int WARMUP_ITERATIONS = 20;
const int ITERATIONS = 200;