#include <string>

#include <thrift/transport/TFileTransport.h>
#include <thrift/protocol/TBinaryProtocol.h>

using ::apache::thrift::transport::TFileTransport;
using ::apache::thrift::protocol::TBinaryProtocol;



namespace vision {
namespace sfm {


template <class T>
void serialize(const T& obj, const std::string& filename) {
  boost::shared_ptr<TFileTransport> transport(new TFileTransport(filename));
  boost::shared_ptr<TBinaryProtocol> protocol(new TBinaryProtocol(transport));
  obj.write(protocol.get());
}


template <class T>
void unserialize(T& obj, const std::string& filename) {
  boost::shared_ptr<TFileTransport> transport(new TFileTransport(filename));
  boost::shared_ptr<TBinaryProtocol> protocol(new TBinaryProtocol(transport));
  obj.read(protocol.get());
}


}} //name-spaces


