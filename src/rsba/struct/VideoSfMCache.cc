#include "rsba/struct/VideoSfMCache.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include <iostream>
#include <openssl/md5.h>



using namespace ::std;
using namespace ::vision::sfm;


#if __cplusplus < 201103L
#define to_string(...) ""
#endif



VideoSfMCache::VideoSfMCache(const std::string& fullpath)
: folder(fullpath.substr(0, fullpath.find_last_of("\\/")+1)), filename(fullpath.substr(folder.length()))
{
}


VideoSfMCache::VideoSfMCache(const std::string& folder, const std::string& filename)
: folder(folder), filename(folder + filename)
{
}

VideoSfMCache::VideoSfMCache(const std::string& folder, const std::string& prefix, const cv::Mat& src)
: folder(folder), filename(folder + prefix + str2md5(Mat2str(src)))
{
}



char* VideoSfMCache::str2md5(const char *str, int length) {
    int n;
    MD5_CTX c;
    unsigned char digest[16];
    char *out = (char*)malloc(33);

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

    return out;
}



string VideoSfMCache::Mat2str(const cv::Mat& src) {
  char* rawPtr = new char[src.step[0] * src.rows];
  memcpy(rawPtr, (char*)src.data, src.step[0] * src.rows);
  return (rawPtr);
}
