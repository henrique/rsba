#ifndef _RsBundleAdjustment_h
#define _RsBundleAdjustment_h

#include <cmath>
#include <cstdio>
#include <iostream>

#include "rsba/mat.h"
#include "rsba/SfmOptions.h"
#include "rsba/struct/VideoSfM.h"
#include "rsba/video_bundler_free.h"

namespace vision { namespace sfm {

struct RsBundleAdjustment: public ReprojectionError {
  RsBundleAdjustment(const Session& sess,
                     const SfmOptions& opt,
                     const double* const observed) // 2D
      : ReprojectionError(sess.cam.data(), observed),
        sess(sess),
        opt(opt) {
  };


  template <typename T>
  bool operator()(const T* const pose0,
                  const T* const pose1,
                  const T* const point,
                  T* residuals) const {
    T pose[6];
    T obs[2] = { T(observed_x), T(observed_x) };
    interpolate_rs(pose0, pose1, (SHUTTER)sess.rs, sess.scanlines.data(), obs,
        pose, opt.model.interpolateRotation);
    return ReprojectionError::operator()(pose, point, residuals);
  }


  template <typename T>
  bool operator()(const T* const camera, //focal length + 2nd and 4th distortion
                  const T* const pose0,
                  const T* const pose1,
                  const T* const point,
                  T* residuals) const {
    T pose[6];
    T obs[2] = { T(observed_x), T(observed_x) };
    interpolate_rs(pose0, pose1, (SHUTTER)sess.rs, sess.scanlines.data(), obs,
        pose, opt.model.interpolateRotation);
    return ReprojectionError::operator()(camera, pose, point, residuals);
  }


  // Factory to hide the construction of the CostFunction object from the client code.
  static ceres::CostFunction* Create(
      const Session& sess,
      const SfmOptions& opt,
      const double* const observed)
  {
    return (new ceres::AutoDiffCostFunction<RsBundleAdjustment,
        NUM_RESIDUALS,
        NUM_POSE_PARAMS, // Initial frame pose
        NUM_POSE_PARAMS, // Final frame pose
        NUM_POINT_PARAMS>(
                new RsBundleAdjustment(sess, opt, observed)));
  }


  // Factory to hide the construction of the CostFunction object from the client code.
  static ceres::CostFunction* CreateWithCam(
      const Session& sess,
      const SfmOptions& opt,
      const double* const observed)
  {
    return (new ceres::AutoDiffCostFunction<RsBundleAdjustment,
        NUM_RESIDUALS,
        NUM_CAM_PARAMS,
        NUM_POSE_PARAMS, // Initial frame pose
        NUM_POSE_PARAMS, // Final frame pose
        NUM_POINT_PARAMS>(
                new RsBundleAdjustment(sess, opt, observed)));
  }

  const Session& sess;
  const SfmOptions& opt;
};

}} // namespace vision sfm


#endif
