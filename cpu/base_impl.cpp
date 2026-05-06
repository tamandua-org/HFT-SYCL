#include "consts.hpp"
#include "pair_info.hpp"
#include "tick.hpp"
#include "run.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <vector>

void choosePosition(PairInfo &pair, const float z) {
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

void constructFinalPositionFromAllInputs(
    const std::array<std::array<PairInfo, N_STOCKS>, N_STOCKS> &pairs,
    std::array<uint8_t, N_STOCKS> &positions) {
  for (int i = 0; i < N_STOCKS; i++) {
    int sumPositions[3];
    auto &stockPositions = pairs[i];

    for (int j = 0; j < N_STOCKS; j++) {
      sumPositions[stockPositions[j].position]++;
    }

    if (sumPositions[BUY] > N_STOCKS / 2)
      positions[i] = BUY;
    else if (sumPositions[SELL] > N_STOCKS / 2)
      positions[i] = SELL;
    else
      positions[i] = HOLD;
  }
}

std::array<uint8_t, N_STOCKS> runTicks(const std::vector<Tick> &ticks) {
  std::array<std::array<PairInfo, N_STOCKS>, N_STOCKS> pairs;
  std::array<uint8_t, N_STOCKS> positions;
  uint32_t tickCount = 0;
  for (const Tick &t : ticks) {
    const auto &prices = t.prices;

    // calculo del tick
    for (int i = 0; i < N_STOCKS; i++) {
      for (int j = i + 1; j < N_STOCKS; j++) {
        PairInfo &pair = pairs[i][j];

        float x_i = prices[i];
        float x_j = prices[j];

        if (tickCount <= WINDOW_SIZE) { //warmup
          float n = tickCount;

          // Media Welford
          float oldMeanI = pair.meanI;
          float oldMeanJ = pair.meanJ;
          pair.meanI += (x_i - pair.meanI) / n;
          pair.meanJ += (x_j - pair.meanJ) / n;

          // Varianza y covarianza de Welford exactas
          // usa (x - oldMean) * (x - newMean) — fórmula online exacta
          pair.varJ += (x_j - oldMeanJ) * (x_j - pair.meanJ);
          pair.covIJ += (x_i - oldMeanI) * (x_j - pair.meanJ);

          if (tickCount < WINDOW_SIZE)
            continue;

          // Normalizar antes de salir del warmup
          pair.varJ /= n;
          pair.covIJ /= n;
        } else { // post warmup
          float delta_i = x_i - pair.meanI;
          float delta_j = x_j - pair.meanJ;
          pair.meanI += ALPHA * delta_i;
          pair.meanJ += ALPHA * delta_j;

          pair.varJ = (1.0f - ALPHA) * pair.varJ + ALPHA * delta_j * delta_j;
          pair.covIJ = (1.0f - ALPHA) * pair.covIJ + ALPHA * delta_i * delta_j;
        }

        float varJ_safe = std::max(pair.varJ, EPSILON);
        pair.beta = pair.covIJ / varJ_safe;

        float spread = x_i - pair.beta * x_j;

        float delta_s = spread - pair.meanSpread;

        pair.meanSpread += ALPHA * delta_s;
        pair.varSpread =
            (1 - ALPHA) * pair.varSpread + ALPHA * delta_s * delta_s;

        float stddev = std::sqrt(std::max(pair.varSpread, EPSILON));

        float z = delta_s / stddev;

        choosePosition(pair, z);
      }
    }

    // segundo paso: En base a las relaciones entre los N stocks, tomamos para
    // cada uno la decision de que posicion tomamos
    constructFinalPositionFromAllInputs(pairs, positions);

    tickCount++;
  }

  return positions;
}

