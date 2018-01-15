cmake_minimum_required(VERSION 2.8)
project(RSBA)

IF (BUILD_TESTING)
  add_subdirectory(test)
ENDIF()

IF (FORCE_STATIC)
	ADD_LIBRARY(thrift STATIC IMPORTED)
	SET_TARGET_PROPERTIES(thrift PROPERTIES IMPORTED_LOCATION ${LIBDIR}/libthrift.a)
ENDIF()


SET(EIGEN3_INCLUDE_DIR "/usr/include/eigen3" CACHE PATH "Eigen3 path")

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11 -I${EIGEN3_INCLUDE_DIR}")
set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -D_DEBUG")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake_modules")

if (UNIX)
    set(PLATFORM_LIBS "pthread spqr gomp cholmod blas lapack")
endif (UNIX)
if (MSVC)
    set(PLATFORM_LIBS "")
endif (MSVC)

include_directories(${CMAKE_SOURCE_DIR}/..)

FIND_PACKAGE(OpenMP)
if (OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif (OPENMP_FOUND)

find_package( OpenCV REQUIRED )
find_package( Thrift REQUIRED )
find_package( CERES REQUIRED )

include_directories(EIGEN3_INCLUDE_DIR)

# sfm thrift
set(SFM_THRIFT_SRC
  gen-cpp/sfm_constants.cpp
  gen-cpp/sfm_types.cpp
  gen-cpp/VideoSfM.cpp
  struct/VideoSfMCache.cc
  struct/VideoSfM.cc
)
add_library(sfm_thrift ${SFM_THRIFT_SRC})
target_include_directories(sfm_thrift PRIVATE ${OpenCV_INCLUDE_DIRS} ${THRIFT_INCLUDE_DIR})
target_link_libraries(sfm_thrift
  ${OpenCV_LIBS} ${OPENCV_DEBUG_LIBS}
  ${THRIFT_LIBRARIES}
  ${CERES_INCLUDE_DIRS}
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
  ${CERES_LIBS}
  ${PLATFORM_LIBS}
)

# install
install(TARGETS VideoSfm sfm_thrift
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

install(DIRECTORY . DESTINATION include/rsba
        FILES_MATCHING PATTERN "*.h")
## TODO HM: select include files

