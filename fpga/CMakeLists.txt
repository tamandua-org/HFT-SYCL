add_library(fpga_impl 
    fpga_impl.cpp
    voting_kernel.cpp
    zscore_kernel.cpp
    pipes.hpp
)

target_include_directories(fpga_impl PUBLIC 
    ${PROJECT_SOURCE_DIR}/common
    ${CMAKE_CURRENT_SOURCE_DIR}
)
# Base FPGA flags
set(FPGA_FLAGS -fsycl -fintelfpga)

# Mode-specific flags
if(FPGA_MODE STREQUAL "emulator")
    message(STATUS "FPGA mode: EMULATOR")
    list(APPEND FPGA_FLAGS -DFPGA_EMULATOR) 

elseif(FPGA_MODE STREQUAL "simulation")
    message(STATUS "FPGA mode: SIMULATION")
    # Added -Xssimulation here, which is usually required for cycle-accurate RTL simulation in Intel oneAPI
    list(APPEND FPGA_FLAGS -DFPGA_SIMULATOR -Xssimulation)

elseif(FPGA_MODE STREQUAL "report")
    message(STATUS "FPGA mode: REPORT")
    list(APPEND FPGA_FLAGS -Xshardware -fsycl-link=early) 

elseif(FPGA_MODE STREQUAL "hardware")
    message(STATUS "FPGA mode: HARDWARE")
    list(APPEND FPGA_FLAGS -Xshardware) 

else()
    message(FATAL_ERROR "Unknown FPGA_MODE: ${FPGA_MODE}")
endif()

# Use PUBLIC so the root executable (hft_fpga) inherits these flags for the linking phase
target_compile_options(fpga_impl PUBLIC ${FPGA_FLAGS})
target_link_options(fpga_impl PUBLIC ${FPGA_FLAGS})