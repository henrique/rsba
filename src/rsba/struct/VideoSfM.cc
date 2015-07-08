#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/nonfree/features2d.hpp>

#include "rsba/struct/VideoSfM.h"

using namespace ::std;
using namespace ::apache::thrift;
using namespace ::vision;

#define FEATURE_TYPE CV_32F
#define FEATURE_SIZE 128
const static size_t desc_size = FEATURE_SIZE * sizeof(float); //HACK size of 32F used in SIFT


namespace vision { namespace sfm {

// convertion functions:

Observation convertCV(const cv::KeyPoint& kp, const unsigned char color[3]) {
  // openCV BGR => RGB
  uchar rgb[3] = { color[2], color[1], color[0] };
  Observation o(kp.pt.x, kp.pt.y, rgb);
  o.__isset.color = true;
  return o;
};


std::vector<gen::Observation> convertCV(const std::vector<cv::KeyPoint>& kps, const cv::Mat desc, const cv::Mat& inFrame) {
  std::vector<gen::Observation> obs;
  for (uint i = 0; i < kps.size(); i++) {
//cout<<desc.rows<<":"<<desc.cols<<"::"<<desc.type()<<"::"<<sizeof(desc.ptr<float>(i))<<":"<<(desc.ptr<float>(i))<<endl;
    const cv::KeyPoint& kp = kps[i];
    Observation o(convertCV(kp, inFrame.at<cv::Vec3b>(kp.pt).val));

    char descriptor[desc_size];
    memcpy(descriptor, desc.ptr(i), desc_size);
    o.descriptor.assign(descriptor, descriptor+desc_size);
    o.__isset.descriptor = true;

    obs.push_back(o);
  }

  return obs;
};


void convertCV(std::vector<gen::Observation>& obs, const std::vector<cv::DMatch>& ms, const size_t frameKey) {
  for (const cv::DMatch& m : ms) {
    gen::Observation& o = obs[m.queryIdx];
    o.matches.push_back(ObservationRef(frameKey, m.trainIdx));
    o.__isset.matches = true;
  }
};


void toCV(const std::vector<gen::Observation>& obs, std::vector<cv::KeyPoint>& pts, cv::Mat& desc) {
  desc.create(obs.size(), FEATURE_SIZE, FEATURE_TYPE);

  for (uint i = 0; i < obs.size(); i++) {
    const gen::Observation& o = obs[i];
    pts.push_back(cv::KeyPoint(o.x, o.y, 0)); //TODO check keypoint sizes

    if (o.__isset.descriptor) {
      memcpy(desc.ptr(i), o.descriptor.c_str(), o.descriptor.capacity());
      char descriptor[desc_size];
      memcpy(descriptor, desc.ptr(i), desc_size);
//cout << descriptor << endl;
    }
  }
};



double* getPose(const gen::Session& sess, gen::Frame& f, const SfmOptions& opt, const double obs[2]) {
  switch (f.poses.size()) {
    case 0: throw runtime_error("empty frame");

    case 1: return f.poses[0].data();

    case 2: throw runtime_error("not possible to get a pose reference on linear RS");

    default:
      double line;
      if (sess.rs == gen::RollingShutter::HORIZONTAL) {
        line = obs[0];
      } else {
        line = obs[1];
      }

      if (line < 0) {
        line = 0;
      } else if (line > f.poses.size()-1) {
        line = f.poses.size()-1;
      }

      return f.poses[round(line)].data();
  }
};



void getPose(const gen::Session& sess, const gen::Frame& f, const SfmOptions& opt, const double obs[2], double pose[NUM_POSE_PARAMS]){
  switch (f.poses.size()) {
    case 0: throw runtime_error("empty frame");

    case 1:
      assign<6>(f.poses[0].data(), pose);
      break;

    case 2:
      interpolate_rs(f.poses[0].data(), f.poses[1].data(),
          (SHUTTER)(sess.rs), sess.scanlines.data(), obs, pose,
          opt.model.interpolateRotation);
      break;

    default:
      double line;
      if (sess.rs == gen::RollingShutter::HORIZONTAL) {
        line = obs[0];
      } else {
        line = obs[1];
      }

      if (line < 0) {
        line = 0;
      } else if (line > f.poses.size()-1) {
        line = f.poses.size()-1;
      }

      assign<6>(f.poses[round(line)].data(), pose);
  }
};



// Reproject a 3D point onto an image frame
// The rolling shutter time is unknown but can be iteratively approximated.
bool reproject(const gen::Session& sess, const gen::Frame& f, const SfmOptions& opt, const double pt[3], double obs[2]) {
  double pose[NUM_POSE_PARAMS];
  const double* cam = f.__isset.cam ? f.cam.data() : sess.cam.data();
  double proj0[2], proj[2] = { cam[CAM_CENTER_X], cam[CAM_CENTER_Y] };

  size_t limit = 50;
  do {
    if (--limit < 1) return false;
    assign<2>(proj, proj0);
    getPose(sess, f, opt, proj, pose);
    if (!w2i(cam, pose, pt, proj, true)) return false;
    minus2(proj0, proj, proj0);
  } while (f.poses.size() > 1 and proj0[0] * proj0[0] + proj0[1] * proj0[1] > 1e-6);

  assign<2>(proj, obs);
  return ::vision::validate(cam, pose, obs, pt, opt.tracks.sqrdThreshold);
};



bool validate(const gen::Session& sess, const gen::Frame& f, const SfmOptions& opt, const double pt[3], const double obs[2]) {
  const double* cam = f.__isset.cam ? f.cam.data() : sess.cam.data();
  double pose[NUM_POSE_PARAMS];
  getPose(sess, f, opt, obs, pose);

  double dist[3];
  minus3(pose+3, pt, dist);

  return norm3(dist) >= opt.tracks.minDistanceToCamera
      and ::vision::validate(cam, pose, obs, pt, opt.tracks.sqrdThreshold);
};


}} // name space
