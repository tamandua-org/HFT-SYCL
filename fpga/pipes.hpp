#pragma once
// ============================================================
// fpga/pipes.hpp
//
// Pipe type definitions for the pairs-trading FPGA kernel.
//
// ── What are SYCL pipes? ──────────────────────────────────────
// Pipes are typed FIFO channels between a host and a kernel,
// or between two kernels.  They map to on-chip FIFO primitives
// in the FPGA fabric (no DDR round-trip).
//
// Host writes with:   PipeName::write(queue, value);   (blocking)
// Kernel reads with:  PipeName::read();                (blocking)
//
// The template parameters are:
//   <TagClass, DataType, FIFOdepth>
//
// ── Why pipes instead of buffers here? ───────────────────────
// The tick loop is sequential: tick t must finish before tick
// t+1 starts.  With buffers we'd need a round-trip copy per
// tick.  With pipes the host and kernel run concurrently:
// host writes tick t+1 while kernel still processes tick t.
// ============================================================

#include "../common/consts.hpp"
#include <sycl/ext/intel/fpga_extensions.hpp>

// ── TickPayload ───────────────────────────────────────────────
// Wraps one Tick (N_STOCKS floats) plus a termination flag.
// Using a dedicated struct instead of passing a Tick directly
// lets us add the 'last' flag without changing common/tick.hpp.
struct TickPayload {
    float prices[N_STOCKS]; // mid-prices for all stocks at time t
    bool  last;             // true on the final tick → kernel exits
};

// ── Host → Kernel: one TickPayload per tick ───────────────────
// Depth=4: allows the host to stay 4 ticks ahead of the kernel,
// hiding the pipe-write latency at the start of each tick batch.
using TickInPipe = sycl::ext::intel::pipe<
    class TickInPipeTag,  // unique tag — avoids symbol collisions
    TickPayload,
    4                     // FIFO depth (entries)
>;