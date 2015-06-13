#include "VideoSfMCache.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>
#include <fcntl.h>
#include <random>
#include <iostream>
#include <openssl/md5.h>
#include <sys/stat.h>

#include <thrift/transport/TFileTransport.h>
#include <thrift/protocol/TBinaryProtocol.h>


using namespace ::apache::thrift::transport;
using namespace ::apache::thrift::protocol;

using namespace ::std;
using namespace ::vision::sfm;


#if __cplusplus < 201103L
#define to_string(...) ""
#endif





VideoSfMCache::VideoSfMCache(const std::string& folder, const std::string& prefix)
: folder(folder), prefix(prefix), src(src), filename(folder + prefix)
{
}

VideoSfMCache::VideoSfMCache(const std::string& folder, const std::string& prefix, const cv::Mat& src)
: folder(folder), prefix(prefix), src(src), filename(folder + prefix + str2md5(Mat2str(src)))
{
}



bool VideoSfMCache::load(gen::Frame& obj) {
  if (fexists(filename)) {
    unserialize(obj);
    return true;
  }

  return false;
}



void VideoSfMCache::save(const gen::Frame& obj) {
  if (fexists(filename)) {
    unlink(filename.c_str());
    cout << filename << " already exists!!! overriding... " << endl;
  }
  cout << "creating: " << filename << endl;
  mkdir(folder.c_str(), 0777);
  open(filename.c_str(), O_CREAT|O_TRUNC|O_WRONLY, 0666); // create file

  serialize(obj);
}



void VideoSfMCache::serialize(const gen::Frame& obj) {
  boost::shared_ptr<TFileTransport> transport(new TFileTransport(filename));
  boost::shared_ptr<TBinaryProtocol> protocol(new TBinaryProtocol(transport));
  //transport->open();
  obj.write(protocol.get());
  //transport->flush();
}



void VideoSfMCache::unserialize(gen::Frame& obj) {
  boost::shared_ptr<TFileTransport> transport(new TFileTransport(filename));
  boost::shared_ptr<TBinaryProtocol> protocol(new TBinaryProtocol(transport));
  obj.read(protocol.get());
  //transport->flush();
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
