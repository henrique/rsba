#ifndef _VideoSfMHandler_h
#define _VideoSfMHandler_h

#include "rsba/struct/VideoSfM.h"
#include "rsba/pointers.h"
#include "rsba/SfmOptions.h"
#include "rsba/CeresHandler.h"

#include <vector>

using namespace ::std;
using namespace ::apache::thrift;



namespace vision {


namespace cameras {
class RollingShutterCamera;
}

namespace sfm {

/**
 * Naive implementation of VideoSfM service
 * no error handling!!!
 */
class VideoSfMHandler : virtual public gen::VideoSfMIf {
 public:
  VideoSfMHandler(const SfmOptions& opt);
  VideoSfMHandler(const SfmOptions& opt, const VideoSfMHandler& clone);

  virtual void authenticate(std::string& _return);

  virtual int32_t newSession(const std::string& authToken, const std::vector<double> & camera);
  virtual int32_t cloneSession(const std::string& authToken, const int32_t oldSessionKey);
  virtual int32_t newRsSession(const std::string& authToken, const std::vector<double> & camera, const gen::RollingShutter::type rs, const std::vector<int32_t> & scanlines);

  virtual int32_t newFrame(const std::string& authToken, const int32_t sessionKey, const gen::Frame& frame);
  virtual void getFrame(gen::Frame& _return, const std::string& authToken, const int32_t sessionKey, const int32_t frameKey);

  virtual int32_t newTrack(const std::string& authToken, const int32_t sessionKey, const gen::Track& track);
  virtual void getTracks(std::vector<gen::Track> & _return, const std::string& authToken, const int32_t sessionKey);

  virtual void initialize(const std::string& authToken, const int32_t sessionKey);
  virtual bool fullBA(const std::string& authToken, const int32_t sessionKey, const int32_t maxIter = 20, const bool reproject = false);
  virtual bool windowedBA(const std::string& authToken, const int32_t sessionKey, const int32_t startFrame, const int32_t endFrame, const int32_t maxIter = 20, const bool reproject = false);
  virtual void finalize(const std::string& authToken, const int32_t sessionKey);

  SfmOptions _opt;
  vector<Session> _sessions;

  virtual void solve(const int32_t sessionKey, const int32_t frameKey);
  virtual void printFrame(const int32_t sessionKey, const int32_t frameKey);
  virtual void printFrames(const int32_t sessionKey);
  virtual void cvCorrespondences(const gen::Frame& f, const Session& sess, std::vector<cv::Point3f>& pts, std::vector<cv::Point2f>& obs, bool useValid = true);

  virtual bool reprojectMatches(Session& sess, const size_t frameKey, size_t obsKey);
  virtual void createTracks(Session& sess, const size_t frameKey);
  virtual void evalTracks(Session& sess, const size_t frameKey);

  virtual bool BA(const int32_t sessionKey, const int32_t startFrame, const int32_t endFrame, const SfmOptions& opt, const int32_t maxIter = 20, const bool reproject = false);
  virtual bool BA(const int32_t sessionKey, const int32_t startFrame, const int32_t endFrame) {
    return BA(sessionKey, startFrame, endFrame, _opt);
  };

  virtual bool solveRsPnP(const int32_t sessionKey, const SfmOptions& opt, const int32_t maxIter = 20);
};

}} // name space
#endif
