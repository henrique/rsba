cmake_minimum_required(VERSION 2.8)
project(RSBA)

enable_testing()

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
