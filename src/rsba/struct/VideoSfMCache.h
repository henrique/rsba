#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <string>
#include <vector>
#include <random>
#include <iostream>
#include <openssl/md5.h>

#include "rsba/struct/VideoSfM.h"

namespace vision {
namespace sfm {


class VideoSfMCache {
public:
  VideoSfMCache(const std::string& folder, const std::string& prefix);
  VideoSfMCache(const std::string& folder, const std::string& prefix, const cv::Mat& src);

  bool load(gen::Frame& obj);
  void save(const gen::Frame& obj);


protected:
  void serialize(const gen::Frame& obj);
  void unserialize(gen::Frame& obj);

  char* str2md5(const char* str, int length);

  inline char* str2md5(const std::string& str) {
    return str2md5(str.c_str(), str.length());
  }

  inline bool fexists(const std::string& name) {
    return ( access( name.c_str(), F_OK ) != -1 );
  }

  std::string Mat2str(const cv::Mat& src);

  const std::string folder;
  const std::string prefix;
  const std::string filename;
};


}} //name-spaces

