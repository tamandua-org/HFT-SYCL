add_library(sycl_impl
    gpu_impl.cpp
)

target_include_directories(sycl_impl PUBLIC
    ${PROJECT_SOURCE_DIR}/common
)

# SYCL flags (only work with icpx)
target_compile_options(sycl_impl PRIVATE
    -fsycl
    -O2
)

target_link_options(sycl_impl PUBLIC
    -fsycl
)