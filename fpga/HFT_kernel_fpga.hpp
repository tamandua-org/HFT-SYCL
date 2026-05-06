#pragma once
// ============================================================
// fpga/hft_kernel_fpga.hpp
//
// FPGA implementation of the statistical pairs-trading engine
// defined in base_impl.cpp, ported to SYCL / Intel oneAPI.
//
// ── Algorithm (mirrors base_impl.cpp line by line) ────────────
//
//  For every pair (i,j) with i < j  (upper triangle, N_PAIRS total):
//
//  Warmup  (tick ≤ WINDOW_SIZE=32):
//    • Welford online mean, variance, covariance
//    • No beta/spread/z until the window is full
//
//  Post-warmup (tick > WINDOW_SIZE):
//    • EWMA with α=ALPHA=0.05 for mean, varJ, covIJ
//    • beta     = covIJ / varJ
//    • spread   = price_i − beta * price_j
//    • EWMA for meanSpread, varSpread
//    • z-score  = (spread − meanSpread) / sqrt(varSpread)
//    • choosePosition(z, current_position)
//
//  After all pairs:
//    • constructFinalPositionFromAllInputs: voting across all
//      pairs of each stock → final BUY / SELL / HOLD per stock
//
// ── Key FPGA decisions ────────────────────────────────────────
//
//  A. single_task kernel
//     The outer tick loop is sequential by nature: tick t's
//     output is tick t+1's input state.  single_task = one
//     pipelined hardware datapath, one tick per II clock cycles.
//
//  B. [[intel::fpga_memory("M20K")]] on pairState[N_STOCKS][N_STOCKS]
//     120 PairInfo structs × ~32 bytes = ~3.8 KB → fits in 2 M20K
//     blocks.  Single-cycle read+write latency within the kernel.
//     Without this attribute the compiler may infer LUT-RAM, which
//     is fine for small arrays but M20K is explicit and predictable.
//
//  C. [[intel::initiation_interval(1)]] on the inner j-loop
//     Each pair (i,j) is data-independent of every other pair within
//     the same tick (they share prices[] but don't write to each other
//     during the update phase).  II=1 means the compiler schedules a
//     new pair iteration every clock cycle → throughput = 1 pair/cycle.
//
//  D. #pragma unroll on the voting loops in constructFinal...
//     N_STOCKS=16 is small; full unroll creates 16 parallel adder trees
//     that sum votes in one clock cycle.
//
//  E. No std:: in device code
//     std::max(float,float) → sycl::fmax(float,float)
//     std::sqrt(float)      → sycl::sqrt(float)
//     Both map to single-cycle pipelined FP units on the FPGA.
//
//  F. continue inside a pipelined loop
//     The 'continue' inside the warmup branch (tick < WINDOW_SIZE)
//     is supported by Vitis/oneAPI HLS as a conditional skip.
//     The compiler predicates the downstream computation rather
//     than breaking the pipeline.
//
// ── Compilation ───────────────────────────────────────────────
//   FPGA emulator (no hardware):
//     icpx -fsycl -fintelfpga -DTARGET_FPGA -DFPGA_EMULATOR \
//          -std=c++20 main.cpp fpga_impl.cpp -o hft_fpga_emu
//
//   FPGA hardware (requires Quartus + BSP):
//     icpx -fsycl -fintelfpga -DTARGET_FPGA -DFPGA_HARDWARE \
//          -Xshardware -Xstarget=Agilex7 -std=c++20 \
//          main.cpp fpga_impl.cpp -o hft_fpga
// ============================================================

#include "../common/consts.hpp"
#include "../common/pair_info.hpp"
#include "../common/tick.hpp"
#include "pipes.hpp"

#include <array>
#include <iostream>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>
#include <vector>

// ── Forward-declare kernel name ────────────────────────────────
// Placing the name in global scope reduces symbol-mangling in the
// Quartus / optimization reports, making them easier to read.
class PairsTradingKernel;

// ─────────────────────────────────────────────────────────────
// choosePosition_device
// Device-side version of choosePosition() in base_impl.cpp.
// Identical logic; 'inline' so it folds into the pipeline body.
// ─────────────────────────────────────────────────────────────
[[sycl::reqd_work_group_size(1, 1, 1)]]  // hint: single-thread kernel
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

// ─────────────────────────────────────────────────────────────
// makeFPGAQueue
// Returns a SYCL queue targeting the right FPGA device.
// Controlled by -DFPGA_EMULATOR / -DFPGA_SIMULATOR / -DFPGA_HARDWARE.
// Pattern recommended by Intel oneAPI documentation:
//   "Use a preprocessor define to choose between selectors so
//    switching targets requires only a command-line flag."
// ─────────────────────────────────────────────────────────────
inline sycl::queue makeFPGAQueue() {
#if defined(FPGA_SIMULATOR)
    auto selector = sycl::ext::intel::fpga_simulator_selector_v;
#elif defined(FPGA_HARDWARE)
    auto selector = sycl::ext::intel::fpga_selector_v;
#else   // default: emulator
    auto selector = sycl::ext::intel::fpga_emulator_selector_v;
#endif

    return sycl::queue{
        selector,
        [](sycl::exception_list el) {
            for (auto &e : el) {
                try { std::rethrow_exception(e); }
                catch (sycl::exception &ex) {
                    std::cerr << "[FPGA async error] " << ex.what() << "\n";
                }
            }
        }
    };
}

// ─────────────────────────────────────────────────────────────
// runTicks_FPGA
//
// Drop-in replacement for runTicks() in base_impl.cpp.
// Same inputs, same output, same semantics.
// ─────────────────────────────────────────────────────────────
inline std::array<uint8_t, N_STOCKS>
runTicks_FPGA(const std::vector<Tick> &ticks) {

    sycl::queue q = makeFPGAQueue();
    std::cout << "[FPGA] Device: "
              << q.get_device().get_info<sycl::info::device::name>()
              << "\n";

    const int nTicks = static_cast<int>(ticks.size());

    // ── USM shared buffer for the final position array ─────────
    // Shared (not device) so the host can read it directly after
    // the kernel finishes — no explicit memcpy needed.
    uint8_t *d_positions = sycl::malloc_shared<uint8_t>(N_STOCKS, q);
    for (int i = 0; i < N_STOCKS; i++) d_positions[i] = HOLD;

    // ═══════════════════════════════════════════════════════════
    // Kernel submission
    // ═══════════════════════════════════════════════════════════
    auto kernel_evt = q.submit([&](sycl::handler &h) {
        h.single_task<PairsTradingKernel>([=]() {

            // ── On-chip pair state ─────────────────────────────
            // Full N×N array (upper triangle used, lower triangle
            // filled by constructFinalPositionFromAllInputs).
            // N=16 → 256 × sizeof(PairInfo) ≈ 7 KB → fits in M20K.
            //
            // We declare it as [N_STOCKS][N_STOCKS] rather than a
            // flat N_PAIRS array to mirror base_impl.cpp's pairs[i][j]
            // indexing exactly, avoiding off-by-one errors.
            [[intel::fpga_memory("M20K")]]
            PairInfo pairState[N_STOCKS][N_STOCKS];

            // Zero-initialise: compile-time-known size, #pragma unroll
            // makes this a one-cycle initialisation block in hardware.
            #pragma unroll
            for (int i = 0; i < static_cast<int>(N_STOCKS); i++) {
                #pragma unroll
                for (int j = 0; j < static_cast<int>(N_STOCKS); j++) {
                    pairState[i][j] = PairInfo{};
                }
            }

            // Tick counter — flip-flop register, increments every tick.
            [[intel::fpga_register]] uint32_t tickCount = 1;

            // ── Main tick loop ─────────────────────────────────
            // Blocks on TickInPipe::read() each iteration.
            // Exits when TickPayload.last == true.
            while (true) {

                // Read the next tick from the host pipe.
                const TickPayload pkt = TickInPipe::read();

                // ──────────────────────────────────────────────
                // Step 1: Update all upper-triangle pairs
                //
                // Each (i,j) iteration is independent within a tick
                // → [[intel::initiation_interval(1)]] on the j-loop
                //   tells the compiler to start a new pair every cycle.
                //
                // The i-loop is NOT unrolled here: for N=16, unrolling
                // both loops would instantiate 120 independent compute
                // chains, blowing up area.  Sequential i with pipelined
                // j gives throughput = N_STOCKS pairs per N_STOCKS cycles
                // ≈ 1 pair/cycle on average.
                // ──────────────────────────────────────────────
                for (int i = 0; i < static_cast<int>(N_STOCKS); i++) {

                    [[intel::initiation_interval(1)]]
                    for (int j = i + 1; j < static_cast<int>(N_STOCKS); j++) {

                        PairInfo &pair = pairState[i][j];

                        const float x_i = pkt.prices[i];
                        const float x_j = pkt.prices[j];

                        // ── Warmup: Welford online stats ───────
                        // Mirrors base_impl.cpp tick ≤ WINDOW_SIZE branch.
                        if (tickCount <= WINDOW_SIZE) {
                            const float n        = static_cast<float>(tickCount);
                            const float oldMeanI = pair.meanI;
                            const float oldMeanJ = pair.meanJ;

                            pair.meanI += (x_i - pair.meanI) / n;
                            pair.meanJ += (x_j - pair.meanJ) / n;

                            // Welford exact online formula:
                            //   M2 += (x − old_mean)(x − new_mean)
                            pair.varJ  += (x_j - oldMeanJ) * (x_j - pair.meanJ);
                            pair.covIJ += (x_i - oldMeanI) * (x_j - pair.meanJ);

                            // Don't compute beta/z until window full.
                            // 'continue' is supported in FPGA pipelines:
                            // the compiler predicates the remaining work.
                            if (tickCount < WINDOW_SIZE)
                                continue;

                            // Last warmup tick: normalise accumulators.
                            pair.varJ  /= n;
                            pair.covIJ /= n;
                            // Fall through to beta/spread computation below.
                        } else {
                            // ── Post-warmup: EWMA stats ────────
                            // Mirrors base_impl.cpp else branch exactly.
                            const float delta_i = x_i - pair.meanI;
                            const float delta_j = x_j - pair.meanJ;

                            pair.meanI += ALPHA * delta_i;
                            pair.meanJ += ALPHA * delta_j;

                            pair.varJ  = (1.0f - ALPHA) * pair.varJ
                                         + ALPHA * delta_j * delta_j;
                            pair.covIJ = (1.0f - ALPHA) * pair.covIJ
                                         + ALPHA * delta_i * delta_j;
                        }

                        // ── Beta, spread, z-score ──────────────
                        // sycl::fmax = device-side std::max for floats.
                        // sycl::sqrt = device-side std::sqrt.
                        // Both map to pipelined FP units — no extra cost.
                        const float varJ_safe = sycl::fmax(pair.varJ, EPSILON);
                        pair.beta = pair.covIJ / varJ_safe;

                        const float spread  = x_i - pair.beta * x_j;
                        const float delta_s = spread - pair.meanSpread;

                        pair.meanSpread += ALPHA * delta_s;
                        pair.varSpread   = (1.0f - ALPHA) * pair.varSpread
                                           + ALPHA * delta_s * delta_s;

                        const float stddev = sycl::sqrt(
                            sycl::fmax(pair.varSpread, EPSILON));

                        const float z = delta_s / stddev;

                        // Update this pair's position signal.
                        choosePosition_device(pair.position, z);

                    } // j-loop (pipelined, II=1)
                } // i-loop

                // ──────────────────────────────────────────────
                // Step 2: constructFinalPositionFromAllInputs
                //
                // Mirrors base_impl.cpp constructFinalPositionFromAllInputs()
                // exactly.  #pragma unroll on both inner loops creates
                // N_STOCKS parallel vote-accumulation adder trees that
                // resolve in one clock cycle.
                // ──────────────────────────────────────────────

                // Local position results — held in flip-flop registers.
                [[intel::fpga_register]] uint8_t localPos[N_STOCKS];

                #pragma unroll
                for (int i = 0; i < static_cast<int>(N_STOCKS); i++) {

                    // Vote counters for this stock.
                    // Declared inside the loop so they're independent
                    // registers per unrolled iteration.
                    int sumBuy  = 0;
                    int sumSell = 0;

                    // ── Upper triangle: set reverse symmetry ───
                    // pairs[i][j] → pairs[j][i] mirror logic from
                    // constructFinalPositionFromAllInputs in base_impl.cpp
                    #pragma unroll
                    for (int j = i + 1; j < static_cast<int>(N_STOCKS); j++) {
                        const int8_t pos = pairState[i][j].position;

                        if (pos == static_cast<int8_t>(BUY)) {
                            pairState[j][i].position = static_cast<int8_t>(SELL);
                            sumBuy++;
                        } else if (pos == static_cast<int8_t>(SELL)) {
                            pairState[j][i].position = static_cast<int8_t>(BUY);
                            sumSell++;
                        } else {
                            pairState[j][i].position = static_cast<int8_t>(HOLD);
                        }
                    }

                    // ── Lower triangle: read mirrored values ───
                    #pragma unroll
                    for (int j = 0; j < i; j++) {
                        const int8_t pos = pairState[j][i].position;
                        if      (pos == static_cast<int8_t>(BUY))  sumBuy++;
                        else if (pos == static_cast<int8_t>(SELL)) sumSell++;
                    }

                    // ── Voting threshold ───────────────────────
                    // Same condition as base_impl.cpp:
                    //   if (sumPositions[BUY] > N_STOCKS / 4) ...
                    const int threshold = static_cast<int>(N_STOCKS) / 4;

                    if      (sumBuy  > threshold) localPos[i] = BUY;
                    else if (sumSell > threshold) localPos[i] = SELL;
                    else                          localPos[i] = HOLD;
                }

                // ── Write positions to USM shared memory ───────
                // Always write every tick.  The host reads d_positions
                // after kernel_evt.wait(), so the last tick's values
                // are what matters — but intermediate writes are free.
                #pragma unroll
                for (int i = 0; i < static_cast<int>(N_STOCKS); i++) {
                    d_positions[i] = localPos[i];
                }

                // ── Termination ────────────────────────────────
                if (pkt.last) break;

                tickCount++;

            } // while (streaming tick loop)

        }); // single_task lambda
    }); // q.submit

    // ═══════════════════════════════════════════════════════════
    // Host feed loop
    // Runs concurrently with the kernel (kernel blocks on pipe reads).
    // Pushes one TickPayload per tick into TickInPipe.
    // ═══════════════════════════════════════════════════════════
    for (int t = 0; t < nTicks; ++t) {
        TickPayload pkt{};

        // Copy prices from Tick struct into the pipe payload.
        for (int s = 0; s < static_cast<int>(N_STOCKS); s++)
            pkt.prices[s] = ticks[t].prices[s];

        // Mark the last tick so the kernel knows when to exit.
        pkt.last = (t == nTicks - 1);

        // Blocking write — if the FIFO is full (depth=4), this waits.
        // In practice the kernel processes ticks fast enough that the
        // pipe never backs up beyond 1-2 entries.
        TickInPipe::write(q, pkt);
    }

    // ── Wait for kernel to complete ─────────────────────────────
    kernel_evt.wait();

    // ── Copy result out of USM and free ─────────────────────────
    std::array<uint8_t, N_STOCKS> result{};
    for (int i = 0; i < static_cast<int>(N_STOCKS); i++)
        result[i] = d_positions[i];

    sycl::free(d_positions, q);
    return result;
}