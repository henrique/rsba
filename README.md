### Dependencies

Either install the packages libgflags-dev and libgoogle-glog-dev or install them from source. Google ceres-solver 1.9.0 and Apache Thrift 0.9.2 are also required. Please follow ./.travis.yml for installation instructions.

### Install

    mkdir build; cd build
    cmake -D CMAKE_BUILD_TYPE=Release ..
    make -j4

### Debug

To debug the project you can use cmake to generate the eclipse project file on a folder parallel to the source.

    cmake -G"Eclipse CDT4 - Unix Makefiles" -D CMAKE_BUILD_TYPE=Debug ../rsba

Alternatively, you can also use CMAKE_BUILD_TYPE=RelWithDebInfo for a *much* faster debugging.

### Test

    make test

