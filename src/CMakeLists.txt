cmake_minimum_required(VERSION 2.8)
project(RSBA)

enable_testing()




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




# install
install(TARGETS VideoSfm
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib/static)

install(DIRECTORY . DESTINATION includes/rsba
        FILES_MATCHING PATTERN "*.h")
## TODO HM: select include files

