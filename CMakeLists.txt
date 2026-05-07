cmake_minimum_required(VERSION 3.20)
project(HFT_Project CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Architecture Options
option(USE_SYCL "Enable SYCL backend" OFF)
option(USE_FPGA "Enable FPGA backend" OFF)

# FPGA Specific Configuration
set(FPGA_MODE "emulator" CACHE STRING "Target FPGA build mode (emulator, simulation, report, hardware)")
set_property(CACHE FPGA_MODE PROPERTY STRINGS emulator simulation report hardware)

add_subdirectory(common)

# Target resolution based on options
if(USE_FPGA)
    add_subdirectory(fpga)
    add_executable(hft_fpga common/main.cpp)
    target_link_libraries(hft_fpga PRIVATE common_lib fpga_impl)
elseif(USE_SYCL)
    add_subdirectory(heterogeneous)
    add_executable(hft_sycl common/main.cpp)
    target_link_libraries(hft_sycl PRIVATE common_lib sycl_impl)
else()
    add_subdirectory(cpu)
    add_executable(hft_cpu common/main.cpp)
    target_link_libraries(hft_cpu PRIVATE common_lib cpu_impl)
endif()