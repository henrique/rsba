#ifndef _VideoSfM_h
#define _VideoSfM_h

#include <vector>
#include <glog/logging.h>
#include <opencv2/core/core.hpp>
#include <opencv2/nonfree/features2d.hpp>

#include "rsba/gen-cpp/VideoSfM.h"
#include "rsba/SfmOptions.h"
#include "rsba/mat.h"

using namespace ::std;
using namespace ::apache::thrift;
using namespace ::vision;



namespace vision { namespace sfm {

// Override raw thrift objects

class Frame : public gen::Frame {};



class Track : public gen::Track {
public:
  inline Track(const double p[NUM_POINT_PARAMS], const string& rgb) : Track(p) {
    __isset.color = true;
    color = rgb;
  };
  inline Track(const double p[NUM_POINT_PARAMS], const unsigned char rgb[3]) : Track(p) {
    __isset.color = true;
    color.resize(3);
    color[0] = (char)rgb[0];
    color[1] = (char)rgb[1];
    color[2] = (char)rgb[2];
  };
  inline Track(const double p[NUM_POINT_PARAMS]) {
    __isset.pt = true;
    pt.resize(NUM_POINT_PARAMS);
    assign3(p, pt.data());
  };
  inline Track(double p0, double p1, double p2) {
    __isset.pt = true;
    pt.resize(NUM_POINT_PARAMS);
    pt[0] = p0;
    pt[1] = p1;
    pt[2] = p2;
  };
};



class Observation : public gen::Observation {
public:
  inline Observation(const double x0, const double y0) {
    x = x0;
    y = y0;
  };

  inline Observation(const double x0, const double y0, const unsigned char rgb[3])
   : Observation(rgb) {
    x = x0;
    y = y0;
  };

  inline Observation(const double x0, const double y0, const string& rgb)
  : Observation(rgb) {
   x = x0;
   y = y0;
 };

  inline Observation(const unsigned char rgb[3]) {
    __isset.color = true;
    color.resize(3);
    color[0] = (char)rgb[0];
    color[1] = (char)rgb[1];
    color[2] = (char)rgb[2];
  };

  inline Observation(const string& rgb) {
    __isset.color = true;
    color = rgb;
  };
};



class ObservationRef : public gen::ObservationRef {
public:
  inline ObservationRef(int32_t f, int32_t o) : gen::ObservationRef() {
    __isset.frame = __isset.obs = true;
    frame = f;
    obs = o;
  };
};



class Session : public gen::Session {
public:
  inline Session() {
    cam.resize(NUM_CAM_PARAMS, 0);
  };

  inline gen::Track& getTrack(const size_t trackKey) {
    return tracks[trackKey];
  };

  inline const gen::Track& getTrack(const size_t trackKey) const {
    return tracks[trackKey];
  };

  inline gen::Track& newTrack(const Track& track, int& trackKey) {
    trackKey = tracks.size();
    tracks.push_back(track);
    return getTrack(trackKey);
  };

  inline gen::Track& newTrack(const Track& track) {
    int trackKey;
    return newTrack(track, trackKey);
  };
};



// conversion functions:

Observation convertCV(const cv::KeyPoint& kp, const unsigned char color[3]);
std::vector<gen::Observation> convertCV(const std::vector<cv::KeyPoint>& kps, const cv::Mat desc, const cv::Mat& inFrame);
void convertCV(std::vector<gen::Observation>& obs, const std::vector<cv::DMatch>& mts, const size_t frameKey);

inline std::vector<double> convertCV(const cv::Matx34d& P) {
  cout << P << endl;
  std::vector<double> pose(NUM_POSE_PARAMS);
  cv::Vec3d tvec;
  assign3(P.col(3).val, tvec.val); // translation
  ceres::RotationMatrixToAngleAxis(P.t().val, pose.data()); // rotation
  double rInv[3];
  invert3(pose.data(), rInv);
  ceres::AngleAxisRotatePoint(rInv, (-tvec).val, pose.data()+3);
  return pose;
};

void toCV(const std::vector<gen::Observation>& obs, std::vector<cv::KeyPoint>& pts, cv::Mat& desc);

void getPose(const gen::Session& sess, const gen::Frame& f, const SfmOptions& opt, const double obs[2], double pose[NUM_POSE_PARAMS]);
double* getPose(const gen::Session& sess, gen::Frame& f, const SfmOptions& opt, const double obs[2]);
bool reproject(const gen::Session& sess, const gen::Frame& f, const SfmOptions& opt, const double pt[3], double obs[2]);
bool validate(const gen::Session& sess, const gen::Frame& f, const SfmOptions& opt, const double pt[3], const double obs[2]);

inline bool inTrack(const gen::Track t, const size_t frameKey) {
  for (const gen::ObservationRef& ref : t.obs) {
    if ((size_t)ref.frame == frameKey) return true;
  }
  return false;
}



inline cv::Matx34d cvP(const gen::Session& sess, const gen::Frame& f, const SfmOptions& opt, const double obs[2]) {
  double pose[NUM_POSE_PARAMS];
  reproject(sess, f, opt, obs, pose);
  invert3(pose, pose); // invert now to transpose back after
  cv::Matx43d P;
  ceres::AngleAxisToRotationMatrix(pose, P.val);
  assign3(pose+3, P.val+9);
  return P.t();
}



inline cv::Mat_<double> cvK(const double cam[NUM_CAM_PARAMS]) {
  cv::Mat_<double> K = cv::Mat_<double>::eye(3,3);
  K(0,0) = cam[CAM_FOCAL_X];
  K(1,1) = cam[CAM_FOCAL_Y];
  K(0,2) = cam[CAM_CENTER_X];
  K(1,2) = cam[CAM_CENTER_Y];
  //cout << "K: " << K << endl;
  return K;
}



/// distortion coefficients
inline cv::Mat_<double> cvDist(const double cam[NUM_CAM_PARAMS]) {
  cv::Mat_<double> dist = cv::Mat::zeros(1, 5, CV_64F);
  dist(0,0) = cam[CAM_DIST_K1];
  dist(0,1) = cam[CAM_DIST_K2];
  dist(0,2) = cam[CAM_DIST_P1];
  dist(0,3) = cam[CAM_DIST_P2];
  dist(0,4) = cam[CAM_DIST_K3];
  return dist;
}



/// openCV to VideoSfM camera intrinsics
inline std::vector<double> sfmCam(const cv::Mat_<double>& K, const cv::Mat_<double>& dist) {
  std::vector<double> cam(NUM_CAM_PARAMS, 0);
  cam[CAM_FOCAL_X]  = K(0,0);
  cam[CAM_FOCAL_Y]  = K(1,1);
  cam[CAM_CENTER_X] = K(0,2);
  cam[CAM_CENTER_Y] = K(1,2);
  cam[CAM_DIST_K1] = dist(0,0);
  cam[CAM_DIST_K2] = dist(0,1);
  cam[CAM_DIST_P1] = dist(0,2);
  cam[CAM_DIST_P2] = dist(0,3);
  cam[CAM_DIST_K3] = dist(0,4);
  return cam;
}



}} // name space
#endif
