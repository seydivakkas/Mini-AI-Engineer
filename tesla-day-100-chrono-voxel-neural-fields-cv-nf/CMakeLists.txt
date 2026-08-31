cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(tesla_cv_nf LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(Torch REQUIRED)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/cpu
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/cuda
    ${TORCH_INCLUDE_DIRS}
)

set(CUDA_SOURCES
    csrc/cuda/astes_kernel.cu
    csrc/cuda/differentiable_raymarching.cu
)

set(CPU_SOURCES
    csrc/cpu/zero_copy_ring_buffer.cpp
    csrc/cpu/saliency_fast_attribution.cpp
)

add_library(tesla_cv_nf_cuda SHARED ${CUDA_SOURCES} ${CPU_SOURCES})
set_target_properties(tesla_cv_nf_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(tesla_cv_nf_cuda PRIVATE ${TORCH_LIBRARIES})
