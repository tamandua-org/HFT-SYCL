#pragma once

#include "../common/consts.hpp"
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/ext/intel/experimental/pipes.hpp>

#include <sycl/sycl.hpp>

struct TickPayload {
  float prices[N_STOCKS];
  bool last;
};

// Flat upper-triangle positions. Index via PAIR_IDX(i,j).
struct PairSignalPayload {
  int8_t positions[N_PAIRS];
  bool last;
};

// idx(i,j), j > i
constexpr int PAIR_IDX(int i, int j) {
  return i * N_STOCKS - i * (i + 1) / 2 + (j - i - 1);
}


class PairSignalPipeID;
class TickInPipeID;

using PairSignalPipe =
    sycl::ext::intel::pipe<PairSignalPipeID, PairSignalPayload, 2>;

// using TickInPipe = sycl::ext::intel::pipe<class TickInPipeID, TickPayload, 4>;
using TickInPipeProperties = 
  decltype(sycl::ext::oneapi::experimental::properties(
           sycl::ext::intel::experimental::bits_per_symbol<sizeof(TickPayload)>));

// a host pipe with alias 
using TickInPipe = sycl::ext::intel::experimental::pipe<TickInPipeID, TickPayload, 16, TickInPipeProperties>;