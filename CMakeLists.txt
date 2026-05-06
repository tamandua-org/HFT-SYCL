cmake_minimum_required(VERSION 3.20)
project(HFT_Project CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(USE_SYCL "Enable SYCL backend" OFF)
option(USE_FPGA "Enable FPGA backend" OFF)

add_subdirectory(common)

# CPU baseline
if(NOT USE_SYCL AND NOT USE_FPGA)
    add_subdirectory(cpu)

    add_executable(hft_cpu common/main.cpp)
    target_link_libraries(hft_cpu PRIVATE common_lib cpu_impl)
endif()

# SYCL (CPU/GPU)
if(USE_SYCL AND NOT USE_FPGA)
    add_subdirectory(heterogeneous)

    add_executable(hft_sycl common/main.cpp)
    target_link_libraries(hft_sycl PRIVATE common_lib sycl_impl)
endif()

# FPGA
if(USE_FPGA)
    add_subdirectory(fpga)

    add_executable(hft_fpga common/main.cpp)
    target_link_libraries(hft_fpga PRIVATE common_lib fpga_impl)
endif()