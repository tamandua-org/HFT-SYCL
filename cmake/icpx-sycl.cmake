set(CMAKE_C_COMPILER icx)
set(CMAKE_CXX_COMPILER icpx)

# Optional: enforce FPGA target early
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fintelfpga")