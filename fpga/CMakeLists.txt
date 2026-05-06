add_library(fpga_impl
    fpga_impl.cpp
)

target_include_directories(fpga_impl PUBLIC
    ${PROJECT_SOURCE_DIR}/common
)

# Choose ONE of these modes:
# emulator / report / hardware

set(FPGA_FLAGS "-fsycl -fintelfpga")

target_compile_options(fpga_impl PRIVATE
    ${FPGA_FLAGS}
)

target_link_options(fpga_impl PRIVATE
    ${FPGA_FLAGS}
)