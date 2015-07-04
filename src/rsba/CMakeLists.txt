cmake_minimum_required(VERSION 2.8)
project(RSBA)

IF (BUILD_TESTING)
  add_subdirectory(test)
ENDIF()




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





# install
install(TARGETS VideoSfm sfm_thrift
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

install(DIRECTORY . DESTINATION include/rsba
        FILES_MATCHING PATTERN "*.h")
## TODO HM: select include files

