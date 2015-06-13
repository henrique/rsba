cmake_minimum_required(VERSION 2.8)
project(camera360)

enable_testing()


set (CMAKE_CXX_FLAGS "-Wall -Werror -std=c++11")
set (CMAKE_CXX_FLAGS_DEBUG "-ggdb -O0 -D_DEBUG")

FIND_PACKAGE(OpenMP)
if (OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif (OPENMP_FOUND)

set (BASE_DIR "./")
include_directories(${BASE_DIR})


# sfm thrift
set(SFM_THRIFT_SRC
  gen-cpp/sfm_constants.cpp
  gen-cpp/sfm_types.cpp
  gen-cpp/VideoSfM.cpp
  struct/VideoSfMCache.cc
  struct/VideoSfM.cc
)
add_library(sfm_thrift ${SFM_THRIFT_SRC})
target_link_libraries(sfm_thrift
  opencv_core
  thrift
  ssl
  crypto
)



# SfM over Thrift
set(SFM_SRC
  VideoSfMClient.cc
  VideoSfMHandler.cc
  solveRSpnp.cpp
  Sfm2Ply.cc
)
add_library(VideoSfm ${SFM_SRC})
target_link_libraries(VideoSfm
  sfm_thrift
  opencv_calib3d
  opencv_features2d
  opencv_highgui
  opencv_imgproc
  opencv_nonfree
  opencv_video
  opencv_core
  ceres
  spqr
  glog
  pthread
  gomp
  cholmod
  blas
  lapack
)




# mat test
set(MAT_TEST_SRC
  mat_test.cc
)
add_executable(mat_test ${MAT_TEST_SRC})
target_link_libraries(mat_test
  gtest
  gtest_main
  glog
  gflags
  pthread)
add_test(SfMtest mat_test)
  
