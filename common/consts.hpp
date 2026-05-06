#include <stdint.h>

constexpr uint32_t N_STOCKS = 16;
constexpr uint32_t WINDOW_SIZE = 32;
constexpr uint32_t N_TICKS = 256;

constexpr float ALPHA = 0.05f;
constexpr float EPSILON = 1e-6f;
constexpr float THRESHOLD_ENTRY = 2.0f;
constexpr float THRESHOLD_EXIT  = 0.5f;