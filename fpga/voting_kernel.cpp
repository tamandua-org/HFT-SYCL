#include "consts.hpp"
#include "pipes.hpp"
#include <sycl/sycl.hpp>

class VotingKernel;

inline sycl::event submitVotingKernel(sycl::queue &q, uint8_t *d_positions) {
  return q.submit([&](sycl::handler &h) {
    h.single_task<VotingKernel>([=]() {
      while (true) {

        const PairSignalPayload pkt = PairSignalPipe::read();

        // Reconstruct local mirror of upper+lower triangle from flat payload.
        // Lower triangle is filled via reverse-symmetry (BUY<->SELL).
        [[intel::fpga_register]] int8_t pos[N_STOCKS][N_STOCKS];

#pragma unroll
        for (int i = 0; i < static_cast<int>(N_STOCKS); i++) {
#pragma unroll
          for (int j = i + 1; j < static_cast<int>(N_STOCKS); j++) {
            const int8_t p = pkt.positions[PAIR_IDX(i, j)];
            pos[i][j] = p;

            if (p == static_cast<int8_t>(BUY))
              pos[j][i] = static_cast<int8_t>(SELL);
            else if (p == static_cast<int8_t>(SELL))
              pos[j][i] = static_cast<int8_t>(BUY);
            else
              pos[j][i] = static_cast<int8_t>(HOLD);
          }
        }

        const int threshold = static_cast<int>(N_STOCKS) / 4;

#pragma unroll
        for (int i = 0; i < static_cast<int>(N_STOCKS); i++) {
          int sumBuy = 0;
          int sumSell = 0;

#pragma unroll
          for (int j = 0; j < static_cast<int>(N_STOCKS); j++) {
            if (j == i)
              continue;
            if (pos[i][j] == static_cast<int8_t>(BUY))
              sumBuy++;
            else if (pos[i][j] == static_cast<int8_t>(SELL))
              sumSell++;
          }

          if (sumBuy > threshold)
            d_positions[i] = BUY;
          else if (sumSell > threshold)
            d_positions[i] = SELL;
          else
            d_positions[i] = HOLD;
        }

        if (pkt.last)
          break;
      }
    });
  });
}