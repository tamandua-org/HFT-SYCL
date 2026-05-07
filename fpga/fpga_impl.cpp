
#include "consts.hpp"
#include "tick.hpp"
#include "pipes.hpp"
#include "zscore_kernel.cpp"
#include "voting_kernel.cpp"

#include <array>
#include <iostream>
#include <vector>
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/ext/intel/experimental/pipes.hpp>

inline sycl::queue makeFPGAQueue() {
#if defined(FPGA_SIMULATOR)
    auto selector = sycl::ext::intel::fpga_simulator_selector_v;
#elif defined(FPGA_HARDWARE)
    auto selector = sycl::ext::intel::fpga_selector_v;
#else
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

std::array<uint8_t, N_STOCKS> runTicks(const std::vector<Tick> &ticks) {
    std::cout << "selector prechoose";
    sycl::queue q = makeFPGAQueue();
    std::cout << "selector chosen";

    uint8_t *d_positions = sycl::malloc_shared<uint8_t>(N_STOCKS, q);
    for (int i = 0; i < static_cast<int>(N_STOCKS); i++)
        d_positions[i] = HOLD;

    // Both kernels can start here since they are blocked until pipes receive inputs
    sycl::event zscore_evt = submitZScoreKernel(q);
    sycl::event voting_evt = submitVotingKernel(q, d_positions);

    const size_t nTicks = ticks.size();
    for (int t = 0; t < nTicks; ++t) {
        TickPayload tickI{};
        for (int s = 0; s < N_STOCKS; s++) //esto puede optimizarse con un memcpy
            tickI.prices[s] = ticks[t].prices[s];
        tickI.last = (t == nTicks - 1);
        TickInPipe::write(q, tickI); //simulate the tick arriving to the fpga by sending 1 by 1
    }

    zscore_evt.wait();
    voting_evt.wait();

    std::array<uint8_t, N_STOCKS> result{};
    for (int i = 0; i < static_cast<int>(N_STOCKS); i++)
        result[i] = d_positions[i];

    sycl::free(d_positions, q);
    return result;
}