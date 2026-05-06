add_library(cpu_impl
    base_impl.cpp
)

target_include_directories(cpu_impl PUBLIC
    ${PROJECT_SOURCE_DIR}/common
)

target_compile_options(cpu_impl PRIVATE
    -O0
)