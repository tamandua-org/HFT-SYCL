#include "consts.hpp"
#include "pair_info.hpp"
#include "pipes.hpp"
#include <sycl/sycl.hpp>

// Position state machine — same logic as base_impl choosePosition.
[[sycl::reqd_work_group_size(1, 1, 1)]]
inline void choosePosition_device(int8_t &position, const float z) {
  if (position == static_cast<int8_t>(HOLD)) {
    if (z > THRESHOLD_ENTRY)
      position = static_cast<int8_t>(SELL);
    else if (z < -THRESHOLD_ENTRY)
      position = static_cast<int8_t>(BUY);
  } else if (position == static_cast<int8_t>(BUY)) {
    if (z > -THRESHOLD_EXIT)
      position = static_cast<int8_t>(HOLD);
  } else if (position == static_cast<int8_t>(SELL)) {
    if (z < THRESHOLD_EXIT)
      position = static_cast<int8_t>(HOLD);
  }
}

class ZScoreKernel;

inline sycl::event submitZScoreKernel(sycl::queue &q) {
  return q.submit([&](sycl::handler &h) {
    h.single_task<ZScoreKernel>([=]() {
      [[intel::fpga_memory]]
      PairInfo pairState[N_STOCKS][N_STOCKS];

#pragma unroll
      for (int i = 0; i < static_cast<int>(N_STOCKS); i++)
#pragma unroll
        for (int j = 0; j < static_cast<int>(N_STOCKS); j++)
          pairState[i][j] = PairInfo{};

      [[intel::fpga_register]] uint32_t tickCount = 1;

      while (true) {

        const TickPayload pkt = TickInPipe::read();

        for (int i = 0; i < static_cast<int>(N_STOCKS); i++) {

          [[intel::initiation_interval(1)]]
          for (int j = i + 1; j < static_cast<int>(N_STOCKS); j++) {

            PairInfo &pair = pairState[i][j];
            const float x_i = pkt.prices[i];
            const float x_j = pkt.prices[j];

            if (tickCount <= WINDOW_SIZE) {
              const float n = static_cast<float>(tickCount);
              const float oldMeanI = pair.meanI;
              const float oldMeanJ = pair.meanJ;

              pair.meanI += (x_i - pair.meanI) / n;
              pair.meanJ += (x_j - pair.meanJ) / n;

              pair.varJ += (x_j - oldMeanJ) * (x_j - pair.meanJ);
              pair.covIJ += (x_i - oldMeanI) * (x_j - pair.meanJ);

              if (tickCount < WINDOW_SIZE)
                continue;

              pair.varJ /= n;
              pair.covIJ /= n;
            } else {
              const float delta_i = x_i - pair.meanI;
              const float delta_j = x_j - pair.meanJ;

              pair.meanI += ALPHA * delta_i;
              pair.meanJ += ALPHA * delta_j;

              pair.varJ =
                  (1.0f - ALPHA) * pair.varJ + ALPHA * delta_j * delta_j;
              pair.covIJ =
                  (1.0f - ALPHA) * pair.covIJ + ALPHA * delta_i * delta_j;
            }

            const float varJ_safe = sycl::fmax(pair.varJ, EPSILON);
            pair.beta = pair.covIJ / varJ_safe;

            const float spread = x_i - pair.beta * x_j;
            const float delta_s = spread - pair.meanSpread;

            pair.meanSpread += ALPHA * delta_s;
            pair.varSpread =
                (1.0f - ALPHA) * pair.varSpread + ALPHA * delta_s * delta_s;

            const float stddev =
                sycl::sqrt(sycl::fmax(pair.varSpread, EPSILON));
            const float z = delta_s / stddev;

            choosePosition_device(pair.position, z);
          }
        }

        // Pack upper triangle into flat payload for VotingKernel.
        PairSignalPayload out{};
#pragma unroll
        for (int i = 0; i < static_cast<int>(N_STOCKS); i++)
#pragma unroll
          for (int j = i + 1; j < static_cast<int>(N_STOCKS); j++)
            out.positions[PAIR_IDX(i, j)] = pairState[i][j].position;

        out.last = pkt.last;
        PairSignalPipe::write(out);

        if (pkt.last)
          break;
        tickCount++;
      }
    });
  });
}