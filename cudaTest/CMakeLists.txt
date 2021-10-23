cmake_minimum_required(VERSION 3.5.0)
project(Task)

set(CMAKE_BUILD_TYPE "release")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

set(CMAKE_PREFIX_PATH "/home/sentinel/opencv-4.5.1/")

find_package(OpenCV 4.5.1 REQUIRED)
if (NOT OpenCV_FOUND)
    message("No appropriate opencv package found.")
endif()

find_package(Eigen3 REQUIRED)
if(NOT Eigen3_FOUND)
  message(FATAL_ERROR "Eigen3 not found.")
endif()

set(OpenCV_INCLUDE_DIRS "/home/sentinel/opencv-4.5.1/include/")
set(${OpenCV_LIBS} "/home/sentinel/opencv-4.5.1/lib/")
find_package(PythonLibs)
message(${PYTHON_INCLUDE_DIR})
if (TEST_CUDA)
    message("-- CUDA_TEST Macro is set.")
    add_definitions(-DCUDA_TEST)
endif(TEST_CUDA)

find_package(OpenMP)
if (OPENMP_FOUND)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

find_package(CUDA REQUIRED)
set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -O3 --use_fast_math
    -gencode=arch=compute_61,code=sm_61
    -lcudadevrt -lcudart
)
set(-Wno-deprecated-declarations)
SET(CUDA_SEPARABLE_COMPILATION ON)

include_directories(
    .
    ./include
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    /usr/local/include/
)

link_directories(
    /usr/local/lib
)

cuda_add_executable(Task
    shared_test.cu
)

target_link_libraries(
    Task
    ${OpenCV_LIBS}
    /usr/local/cuda-10.1/targets/x86_64-linux/lib/libcudadevrt.a
)

cuda_add_executable(eigen
    eigenCuda.cu
)

target_link_libraries(
    eigen
    ${OpenCV_LIBS}
    /usr/local/cuda-10.1/targets/x86_64-linux/lib/libcudadevrt.a
)

cuda_add_executable(minval
    atomicMin.cu
)

target_link_libraries(
    minval
    /usr/local/cuda-10.1/targets/x86_64-linux/lib/libcudadevrt.a
)