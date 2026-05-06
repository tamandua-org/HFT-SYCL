#include "consts.hpp"
#include "pair_info.hpp"
#include "tick.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <vector>

namespace common {

void runTicks(const std::vector<Tick> &ticks) {
  std::array<std::array<PairInfo, N_STOCKS>, N_STOCKS> pairs;

  for (const Tick &t : ticks) {
    const auto &prices = t.prices;

    // calculo del tick
    for (int i = 0; i < N_STOCKS; i++) {
      for (int j = i + 1; j < N_STOCKS; j++) {
        PairInfo &pair = pairs[i][j];

        float x_i = prices[i];
        float x_j = prices[j];

        float delta_i = x_i - pair.meanI;
        float delta_j = x_j - pair.meanJ;

        pair.meanI += ALPHA * delta_i;
        pair.meanJ += ALPHA * delta_j;

        pair.varJ = (1.0f - ALPHA) * pair.varJ + ALPHA * delta_j * delta_j;
        pair.covIJ = (1.0f - ALPHA) * pair.covIJ + ALPHA * delta_i * delta_j;

        float varJ_safe = std::max(pair.varJ, EPSILON);
        pair.beta = pair.covIJ / varJ_safe;

        float spread = x_i - pair.beta * x_j;

        float delta_s = spread - pair.meanSpread;

        pair.meanSpread += ALPHA * delta_s;
        pair.varSpread =
            (1 - ALPHA) * pair.varSpread + ALPHA * delta_s * delta_s;

        float stddev = std::sqrt(std::max(pair.varSpread, EPSILON));

        float z = delta_s / stddev;


        if (pair.position == HOLD) {
          if (z > THRESHOLD_ENTRY)
            pair.position = SELL;
          else if (z < -THRESHOLD_ENTRY)
            pair.position = BUY;
        } else if (pair.position == BUY) {
          if (z > -THRESHOLD_EXIT)
            pair.position = HOLD;
        } else if (pair.position == SELL) {
          if (z < THRESHOLD_EXIT)
            pair.position = HOLD;
        }

      }
    }
  }

}

} // namespace common
