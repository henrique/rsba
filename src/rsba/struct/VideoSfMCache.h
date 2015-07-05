#include "rsba/struct/VideoSfM.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>

#include <thrift/transport/TFileTransport.h>
#include <thrift/protocol/TBinaryProtocol.h>

using ::apache::thrift::transport::TFileTransport;
using ::apache::thrift::protocol::TBinaryProtocol;


namespace vision {
namespace sfm {


class VideoSfMCache {
public:
  VideoSfMCache(const std::string& fullpath);
  VideoSfMCache(const std::string& folder, const std::string& filename);
  VideoSfMCache(const std::string& folder, const std::string& prefix, const cv::Mat& src);


  template <class T>
  bool load(T& obj) {
    if (fexists(filename)) {
      cout << "loading: " << filename << endl;
      unserialize(obj);
      return true;
    }

    return false;
  }


  template <class T>
  void save(const T& obj) {
    if (fexists(filename)) {
      unlink(filename.c_str());
      cout << filename << " already exists!!! overriding... " << endl;
    }
    cout << "creating: " << folder << filename << endl;
    mkdir(folder.c_str(), 0777);
    open(filename.c_str(), O_CREAT|O_TRUNC|O_WRONLY, 0666); // create file

    serialize(obj);
  }


protected:

  template <class T>
  void serialize(const T& obj) {
    boost::shared_ptr<TFileTransport> transport(new TFileTransport(filename));
    boost::shared_ptr<TBinaryProtocol> protocol(new TBinaryProtocol(transport));
    obj.write(protocol.get());
  }

  template <class T>
  void unserialize(T& obj) {
    boost::shared_ptr<TFileTransport> transport(new TFileTransport(filename));
    boost::shared_ptr<TBinaryProtocol> protocol(new TBinaryProtocol(transport));
    obj.read(protocol.get());
  }

  char* str2md5(const char* str, int length);

  inline char* str2md5(const std::string& str) {
    return str2md5(str.c_str(), str.length());
  }

  inline bool fexists(const std::string& name) {
    return ( access( name.c_str(), F_OK ) != -1 );
  }

  std::string Mat2str(const cv::Mat& src);

  const std::string folder;
  const std::string filename;
};


}} //name-spaces

