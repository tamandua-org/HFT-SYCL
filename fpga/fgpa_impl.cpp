// ============================================================
// fpga/fpga_impl.cpp
//
// Provides the runTicks() symbol that main.cpp calls.
// When TARGET_FPGA is defined at compile time, this file is
// compiled instead of base_impl.cpp.
//
// Why a separate .cpp instead of putting everything in the .hpp?
// -  The SYCL kernel (single_task) must be compiled by icpx
//    with -fsycl -fintelfpga flags.
// -  main.cpp is compiled with the same flags when TARGET_FPGA
//    is active, so a single-translation-unit build works fine.
// -  For larger projects you can split host and device code
//    into separate TUs using -fsycl-link; this file is the
//    natural seam for that split.
// ============================================================

#include "hft_kernel_fpga.hpp"  // pulls in everything

// runTicks — same signature as base_impl.cpp so main.cpp is unchanged.
std::array<uint8_t, N_STOCKS> runTicks(const std::vector<Tick> &ticks) {
    return runTicks_FPGA(ticks);
}