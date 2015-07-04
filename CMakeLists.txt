cmake_minimum_required(VERSION 2.8)
project(RSBA)

enable_testing()


set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror -std=c++11")
set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -D_DEBUG")

FIND_PACKAGE(OpenMP)
if (OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif (OPENMP_FOUND)

#FIND_PACKAGE(Eigen REQUIRED)
#include_directories(EIGEN3_INCLUDE_DIR)

OPTION(BUILD_TESTING "Enable tests" ON)

set (BASE_DIR "src")
include_directories(${BASE_DIR})

add_subdirectory(${BASE_DIR}/rsba)

