#include "rsba/struct/VideoSfMCache.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include <iostream>
#include "md5.h"



using namespace ::std;
using namespace ::vision::sfm;


#if __cplusplus < 201103L
#define to_string(...) ""
#endif



VideoSfMCache::VideoSfMCache(const std::string& fullpath)
: folder(fullpath.substr(0, fullpath.find_last_of("\\/")+1)), fullpath(fullpath)
{
}


VideoSfMCache::VideoSfMCache(const std::string& folder, const std::string& fullpath)
: folder(folder), fullpath(folder + fullpath)
{
}

VideoSfMCache::VideoSfMCache(const std::string& folder, const std::string& prefix, const cv::Mat& src)
: folder(folder), fullpath(folder + prefix + hashMat(src))
{
}



string VideoSfMCache::str2md5(const char *str, int length) {
    int n;
    MD5_CTX c;
    unsigned char digest[16];
    char out[33];

    MD5_Init(&c);

    while (length > 0) {
        if (length > 512) {
            MD5_Update(&c, str, 512);
        } else {
            MD5_Update(&c, str, length);
        }
        length -= 512;
        str += 512;
    }

    MD5_Final(digest, &c);

    for (n = 0; n < 16; ++n) {
        snprintf(&(out[n*2]), 16*2, "%02x", (unsigned int)digest[n]);
    }

    return string(out);
}


string VideoSfMCache::hashMat(const cv::Mat& src) {
  int size = src.step[0] * src.rows;
  char* rawPtr = new char[size];
  memcpy(rawPtr, (char*)src.data, size);
  string retval = str2md5(rawPtr, size);
  delete rawPtr;
  return retval;
}
