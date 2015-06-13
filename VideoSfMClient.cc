#include "VideoSfMClient.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>
#include <random>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <openssl/md5.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/nonfree/features2d.hpp>

#include "mat.h"
#include "Sfm2Ply.h"
#include "struct/VideoSfMCache.h"


using namespace cv;
using namespace std;
using namespace vision::sfm;


#if __cplusplus < 201103L
#define to_string(...) ""
#endif



VideoSfMClient::VideoSfMClient(const SfmOptions& opt)
 : opt(opt),
   _siftDetector(opt.sift.nfeatures, opt.sift.nOctaveLayers, opt.sift.contrastThreshold, opt.sift.edgeThreshold, opt.sift.sigma),
   handler(new sfm::VideoSfMHandler(opt))
{
  if (opt.model.rolling_shutter) {
    scanlines.resize(2);
  }
}



VideoSfMClient::~VideoSfMClient()
{
}



double calc2Ddist(const DMatch& match,
                  const std::vector<cv::KeyPoint>& keypoints_1,
                  const std::vector<cv::KeyPoint>& keypoints_2)
{
  const Point2f& prevPt = keypoints_1[match.queryIdx].pt;
  const Point2f& nextPt = keypoints_2[match.trainIdx].pt;

  // skip points with displacement above the average
  double dx = prevPt.x - nextPt.x;
  double dy = prevPt.y - nextPt.y;
  return sqrt(dx * dx + dy * dy);
}



std::vector<cv::DMatch> VideoSfMClient::Match(const cv::Mat& descriptors_1,
                                              const cv::Mat& descriptors_2,
                                              const std::vector<cv::KeyPoint>& keypoints_1,
                                              const std::vector<cv::KeyPoint>& keypoints_2,
                                              bool multiple)
{
  const static float ratio = 0.80; //TODO make parameter

  std::vector< std::vector<cv::DMatch> > matches;
  cv::BFMatcher matcher;
  matcher.knnMatch(descriptors_1, descriptors_2, matches, (multiple ? 5 : 2)); // Find nearest matches

  /// filter matches
  size_t count = matches.size();
  vector<double> L2D2(count);

  for (size_t mi = 0; mi < count; mi++) {
    L2D2[mi] = calc2Ddist(matches[mi][0], keypoints_1, keypoints_2);
  }

  // calculate mean
  double sum = std::accumulate(L2D2.begin(), L2D2.end(), 0.0);
  double meanL2D2 = sum / count;

//  // standard deviation
//  double accum = 0.0;
//  std::for_each(L2D2.begin(), L2D2.end(), [&](const double d) {
//      accum += (d - meanL2D2) * (d - meanL2D2);
//  });
//  double stdevL2D2 = sqrt(accum / (count-1));

  double thresholdL2D2 = meanL2D2 + meanL2D2;

  vector<cv::DMatch> good_matches;
  if (multiple) { // allow multiple matches per query descriptor
    for (const std::vector<cv::DMatch>& ms : matches) {
      if (calc2Ddist(ms[0], keypoints_1, keypoints_2) < thresholdL2D2)
        good_matches.push_back(ms[0]);

      for (size_t i = 1; i < ms.size(); i++) {
        if (ms[0].distance > ratio * ms[1].distance) {
          if (calc2Ddist(ms[i], keypoints_1, keypoints_2) < thresholdL2D2)
            good_matches.push_back(ms[i]);
        }
      }
    }
  } else {
    for (const std::vector<cv::DMatch>& ms : matches) {
      if (ms[0].distance < ratio * ms[1].distance) {
        if (calc2Ddist(ms[0], keypoints_1, keypoints_2) < thresholdL2D2)
          good_matches.push_back(ms[0]);
      }
    }
  }

  return good_matches;
}



size_t VideoSfMClient::addFrame(const cv::Mat& inFrame, cv::Mat& outFrame)
{
  size_t frameKey = frames.size();
  frames.push_back(sfm::Frame());

  parseFrame(inFrame, outFrame, frameKey);
  processFrame(frameKey);

  if (opt.debug.showTracks or opt.debug.showLastMatches > 0) {
    showObservations(frameKey, outFrame);
  }

  return frameKey;
}



bool VideoSfMClient::parseFrame(const cv::Mat& inFrame, cv::Mat& outFrame, const size_t frameKey)
{
  inFrame.copyTo(outFrame);
  toGray(inFrame, _nextImg);

  gen::Frame& f = frames[frameKey];

  std::stringstream prefix;
  prefix.setf(std::ios_base::right);
  prefix << setw(0)
         << inFrame.cols << "x" << inFrame.rows << ":"
         << opt.sift.nfeatures << ":"
         << opt.sift.nOctaveLayers << ":"
         << opt.sift.sigma << ":"
         << opt.sift.edgeThreshold << ":"
         << opt.sift.contrastThreshold << ":"
         << opt.tracks.maxFramesToMatch << ":"
         << setfill('0') << setw(3) << frameKey << ":";
  VideoSfMCache cache("cache/", prefix.str(), inFrame);

  if ( !opt.tracks.cacheMatches or !cache.load(f) )
  {
    if (frameKey != _descriptors.size()) {
      throw runtime_error("Invalid descriptors or outdated caching");
    }
    _descriptors.push_back(cv::Mat()); // new descriptors
    _keypoints.push_back(std::vector<cv::KeyPoint>()); // new keypoints


    //if(_activeTrackingAlgorithm == TrackingAlgorithmSIFT)
    {
      //TODO make parameter
      //DynamicAdaptedFeatureDetector detector(new StarAdjuster(), opt.sift.nfeatures*0.5, opt.sift.nfeatures, 5);
      //PyramidAdaptedFeatureDetector detector(new GFTTDetector(opt.sift.nfeatures), 1);
      //detector.detect(_nextImg, _keypoints[0]);
      _siftDetector(_nextImg, _mask, _keypoints[frameKey], _descriptors[frameKey]);
      cout << _keypoints[frameKey].size() << " features found" << endl;
      f.obs = convertCV(_keypoints[frameKey], _descriptors[frameKey], inFrame);
      f.__isset.obs = true;
      //TODO load/cache descriptors

      for (size_t i = 1; i < _descriptors.size() and i <= frameKey and i <= opt.tracks.maxFramesToMatch; i++) {
        cout << "matching frame " << frameKey-i << endl;
        vector<DMatch> matches = Match(_descriptors[frameKey], _descriptors[frameKey-i], _keypoints[frameKey], _keypoints[frameKey-i]);
        cout << matches.size() << " matches found" << endl;
        convertCV(f.obs, matches, frameKey-i);
      }
    }

    if (opt.tracks.cacheMatches) cache.save(f);
  } else {
    _keypoints.resize(frameKey+1);
    _descriptors.resize(frameKey+1);
    toCV(f.obs, _keypoints[frameKey], _descriptors[frameKey]);
  }

  return true;
}



void VideoSfMClient::start()
{
  handler->authenticate(authToken);

  if (sessionKey < 0) {
    if (opt.model.rolling_shutter) {
      sessionKey = handler->newRsSession(authToken, cam, rs, scanlines);
    } else {
      sessionKey = handler->newSession(authToken, cam);
    }
  }
}



void VideoSfMClient::processFrame(const size_t frameKey)
{
  cout << "processing..." << endl;

  if (frameKey != (size_t)handler->newFrame(authToken, sessionKey, frames[frameKey])) abort();
  /// frameKeys must be kept synchronized

  // get track info
  handler->getFrame(frames[frameKey], authToken, sessionKey, frameKey);

  if (opt.ceres.baIterationsOnNewFrame > 0 and frameKey+1 >= opt.tracks.minReprojections) { // improve solution
    if (opt.ceres.baWindowOnNewFrame and frameKey >= opt.ceres.baWindowOnNewFrame) {
      handler->windowedBA(authToken, sessionKey, (1 + frameKey - opt.ceres.baWindowOnNewFrame), frameKey, opt.ceres.baIterationsOnNewFrame, true);
    } else {
      handler->windowedBA(authToken, sessionKey, 0, frameKey, opt.ceres.baIterationsOnNewFrame, true);
    }
  }

  if (opt.debug.showTracks)
    handler->getTracks(tracks, authToken, sessionKey);
}



void VideoSfMClient::showObservations(const size_t frameKey, Mat& outFrame, bool showMatches)
{
  const gen::Frame& f = frames[frameKey];
  const double* camera = f.__isset.cam ? f.cam.data() : cam.data();

  Point2f pt, rpt;
  for (const gen::Observation& obs : f.obs) {

    Scalar bgr(0, (obs.__isset.track ? 255 : 0), // green only for used 3d points
              (obs.__isset.track ? 0 : 255), 0); // mark invalid observations in red
    pt.x = obs.x;
    pt.y = obs.y;
    circle(outFrame, pt, 2, bgr, 2);

    if (showMatches) {
      for (const gen::ObservationRef& ref : obs.matches) {
        CHECK_LT((uint)ref.frame, frameKey);

        //if (obs.__isset.track and !ref.valid) continue;
        if ((frameKey - ref.frame) > opt.debug.showLastMatches) continue;

        double l = 3.0 / (2 + frameKey - ref.frame);
        Scalar bgr((204 * (1.25 - l)),
                   (obs.__isset.track ? (255 * l) : 140), // green only for used 3d points
                   (ref.valid ? 0 : 255), 0); // mark invalid matches with orange

        const gen::Observation& obs2 = frames[ref.frame].obs[ref.obs];
        rpt.x = obs2.x;
        rpt.y = obs2.y;
        circle(outFrame, rpt, 2, bgr, 1);
        line(outFrame, pt, rpt, bgr, 1);
      }
    }

    if (obs.__isset.track and tracks.size() > (size_t)obs.track) {
      const gen::Track& t = getTrack(obs.track);
      if ( ! t.__isset.pt) continue;

      double proj[2];
      double o[] = { obs.x, obs.y };
      double pose[NUM_POSE_PARAMS];
      getPose(*this, f, opt, o, pose);

      if (w2i(camera, pose, t.pt.data(), proj)) {
        minus2(proj, o, o);
        double sqrdError = (o[0]*o[0] + o[1]*o[1]);
        bool good = (sqrdError <= opt.tracks.sqrdThreshold);

        Scalar bgr((t.valid ? 255 : 20), 20, (good ? 20 : 255), 0); // mark bad reprojections in purple
        rpt.x = proj[0];
        rpt.y = proj[1];
        circle(outFrame, rpt, 1, bgr, 2);
        line(outFrame, pt, rpt, bgr, 1);
      } else {
        std::cerr << "wrong reprojection of " << cv::Vec3d(t.pt.data()) << endl;
      }
    }
  }
}



void VideoSfMClient::Orientation(double rotation[9], const size_t frameKey, const double& tau)
{
  CHECK_LT(frameKey, frames.size());
  CHECK_GE(tau, 0.0);
  CHECK_LE(tau, 1.0);
  gen::Frame& f = frames[frameKey];

  switch (f.poses.size()) {
    case 0: throw(runtime_error("empty frame"));

    case 1:
      ceres::AngleAxisToRotationMatrix(f.poses[0].data(), rotation);
      break;

    case 2:
      double inter[NUM_POSE_PARAMS];
      interpolate(f.poses[0].data(), f.poses[1].data(), tau, inter, opt.model.interpolateRotation);
      ceres::AngleAxisToRotationMatrix(inter, rotation);
      break;

    default:
      size_t i = round((f.poses.size() - 1) * tau);
      ceres::AngleAxisToRotationMatrix(f.poses[i].data(), rotation);
  }
}



void VideoSfMClient::Position(double pose[3], const size_t frameKey, const double& tau)
{
  CHECK_LT(frameKey, frames.size());
  CHECK_GE(tau, 0.0);
  CHECK_LE(tau, 1.0);
  gen::Frame& f = frames[frameKey];

  switch (f.poses.size()) {
    case 0: throw(runtime_error("empty frame"));

    case 1:
      assign3(f.poses[0].data()+3, pose);
      break;

    case 2:
      double inter[NUM_POSE_PARAMS];
      interpolate(f.poses[0].data(), f.poses[1].data(), tau, inter, false);
      assign3(inter+3, pose);
      break;

    default:
      size_t i = round((f.poses.size() - 1) * tau);
      assign3(f.poses[i].data()+3, pose);
  }
}



// initialize frame poses
void VideoSfMClient::initialize()
{
  // create session
  handler->initialize(authToken, sessionKey);

  // update frames
  for (size_t fi = 0; fi < frames.size(); fi++) {
    gen::Frame& f = frames[fi];
    handler->getFrame(f, authToken, sessionKey, fi);
  }
};



// Run final BA, get frames back from server and release session
void VideoSfMClient::finalize(bool reset)
{
  bool rtn = false;

  if (opt.ceres.baFinalIterations > 0) {
    for (uint i = 0; i < opt.ceres.baFinalRuns; i++) {
      rtn |= handler->fullBA(authToken, sessionKey, opt.ceres.baFinalIterations, true);
    }
  } else {
    rtn = true;
  }

  // check if at least one run was successful
  if ( ! rtn) throw(runtime_error("BA failure"));

  // update frames
  for (size_t fi = 0; fi < frames.size(); fi++) {
    gen::Frame& f = frames[fi];
    handler->getFrame(f, authToken, sessionKey, fi);
  }

  //if (opt.debug.showTracks)
  handler->getTracks(tracks, authToken, sessionKey);

  // release session
  if (reset) handler->finalize(authToken, sessionKey);
};




void VideoSfMClient::fullBA(const size_t iterations)
{
  if ( ! handler->fullBA(authToken, sessionKey, iterations, true)) {
    throw(runtime_error("BA failure"));
  }

  // update frames
  for (size_t fi = 0; fi < frames.size(); fi++) {
    gen::Frame& f = frames[fi];
    handler->getFrame(f, authToken, sessionKey, fi);
  }
};




void VideoSfMClient::toGray(const Mat& input, Mat& gray)
{
  const int numChannes = input.channels();

  if (numChannes == 4) {
    cvtColor(input, gray, CV_BGRA2GRAY);
  } else if (numChannes == 3) {
    cvtColor(input, gray, CV_BGR2GRAY);
  } else if (numChannes == 1) {
    gray = input;
  }
}

