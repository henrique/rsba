#ifndef _VideoSfMClient_h
#define _VideoSfMClient_h

#include <vector>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/nonfree/features2d.hpp>

#include "rsba/base/pointers.h"
#include "rsba/VideoSfMHandler.h"
#include "rsba/struct/VideoSfM.h"



namespace vision {

struct SfmOptions;


class VideoSfMClient: public sfm::Session
{
public:
  VideoSfMClient(const SfmOptions& opt);
  virtual ~VideoSfMClient();

  virtual void start();
  virtual size_t addFrame(const cv::Mat& inFrame, cv::Mat& outFrame);

  virtual void Position(double pose[3], const size_t frameKey, const double& tau = 0.5);
  virtual void Orientation(double rotation[9], const size_t frameKey, const double& tau = 0.5);

  virtual void initialize();
  virtual void finalize(bool reset = true);
  virtual void fullBA(const size_t iterations = 20);

  static void toGray(const cv::Mat& input, cv::Mat& gray);
  virtual void showObservations(const size_t frameKey, cv::Mat& outFrame, bool showMatches = true);

protected:
  virtual std::vector<cv::DMatch> Match(const cv::Mat& descriptors_1,
                                        const cv::Mat& descriptors_2,
                                        const std::vector<cv::KeyPoint>& keypoints_1,
                                        const std::vector<cv::KeyPoint>& keypoints_2,
                                        bool multiple = false);

  virtual bool parseFrame(const cv::Mat& inFrame, cv::Mat& outFrame, const size_t frameKey);
  virtual void processFrame(const size_t frameKey);


  SfmOptions opt;
  std::string authToken;
  int sessionKey = -1;

  cv::SiftFeatureDetector _siftDetector;
  cv::BFMatcher l2matcher;

  shared_ptr<sfm::gen::VideoSfMIf> handler;

  std::vector<cv::Mat> _descriptors;
  std::vector<std::vector<cv::KeyPoint>> _keypoints;

  cv::Mat _prevImg;
  cv::Mat _nextImg;
  cv::Mat _mask;
};

}; // ::vision

#endif
