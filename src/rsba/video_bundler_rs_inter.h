#ifndef _VideoBundlerRSinter_h
#define _VideoBundlerRSinter_h

#include <cmath>
#include <cstdio>
#include <iostream>

#include "rsba/mat.h"
#include "rsba/tracking.h"
#include "rsba/video_bundler_free.h"

namespace vision {

struct RsReprojectionError: public ReprojectionError {
  RsReprojectionError(const double camera[NUM_CAM_PARAMS],
                      const framePtr& f,
                      const double* const observed) // 2D
      : ReprojectionError(camera, observed),
        f(f) {
  };


  template <typename T>
  bool operator()(const T* const pose0,
                  const T* const pose1,
                  const T* const point,
                  T* residuals) const {
    T pose[6];
    T obs[2] = { T(observed_x), T(observed_x) };
    interpolate_rs(pose0, pose1, f->shutter, f->scanlines, obs, pose);
    return ReprojectionError::operator()(pose, point, residuals);
  }


  // Factory to hide the construction of the CostFunction object from the client code.
  static ceres::CostFunction* Create(
      const double* const camera_params,
      const framePtr& f,
      const double* const observed)
  {
    return (new ceres::AutoDiffCostFunction<RsReprojectionError,
        NUM_RESIDUALS,
        NUM_POSE_PARAMS, // Initial frame pose
        NUM_POSE_PARAMS, // Final frame pose
        NUM_POINT_PARAMS>(
                new RsReprojectionError(camera_params, f, observed)));
  }

  const framePtr f;
};




struct RsConstVeloPrior {
  static const unsigned short NUM_RESIDUALS = 12; // constant translation
  const double scale;

  RsConstVeloPrior(double scale) : scale(scale)
  { };


  template <typename T>
  bool operator()(const T* const interFrameRatio,
                  const T* const pose0,
                  const T* const end0,
                  const T* const pose1,
                  const T* const end1,
                  T* residuals) const {
    // get residuals for pose0
    minus6(end1, pose1, residuals); // delta = P-1 - P-2
    scalar<6>(residuals, *interFrameRatio, residuals); // scale
    plus6(end1, residuals, residuals); // P' = P-1 + delta
    minus6(pose0, residuals, residuals); // r = P - P'

    // get residuals for end0
    if (*interFrameRatio > T(_EPS)) {
      minus6(pose0, end1, residuals+6); // delta = P-1 - P-2
      scalar<6>(residuals+6, T(1) / *interFrameRatio, residuals+6); // scale
    } else {
      minus6(end1, pose1, residuals+6); // delta = P-1 - P-2
    }
    plus6(pose0, residuals+6, residuals+6); // P' = P-1 + delta
    minus6(end0, residuals+6, residuals+6); // r = P - P'

    scalar<NUM_RESIDUALS>(residuals, T(scale), residuals);

    // down-scale rotation
    scalar3(residuals, T(0.01), residuals);
    scalar3(residuals+6, T(0.01), residuals+6);

    return (*interFrameRatio >= T(0));
  }


  // Factory to hide the construction of the CostFunction object from the client code.
  static ceres::CostFunction* Create(double scale)
  {
    return (new ceres::AutoDiffCostFunction<RsConstVeloPrior,
        NUM_RESIDUALS,
        1,               // intra-frame ratio
        NUM_POSE_PARAMS, // current frame first pose
        NUM_POSE_PARAMS, // current frame last pose
        NUM_POSE_PARAMS, // frame -1 first pose
        NUM_POSE_PARAMS>( // frame -2 last pose
                new RsConstVeloPrior(scale)));
  }
};




struct RsConstAccelerationPrior {
  static const unsigned short NUM_RESIDUALS = 12; // constant translation
  const double scale;

  RsConstAccelerationPrior(double scale) : scale(scale)
  { };


  template <typename T>
  bool operator()(const T* const interFrameRatio,
                  const T* const pose0,
                  const T* const end0,
                  const T* const pose1,
                  const T* const end1,
                  T* residuals) const {
    T v_1t[NUM_POSE_PARAMS];

    // get residuals for pose0
    minus6(pose0, end1, residuals); // v*t = (P - P-1) * 1
    minus6(end1, pose1, v_1t); // v-1 = (P-1 - P-2)
    scalar<6>(v_1t, *interFrameRatio, v_1t); // scale => v-1*t
    minus6(residuals, v_1t, residuals); // a*t² = (v*t - v-1*t)
    scalar<6>(residuals, T(0.5), residuals); // scale => 1/2 * a*t²
    plus6(v_1t, residuals, residuals); // P' = P-1 + v_1t + (1/2 * a*t²)
    plus6(end1, residuals, residuals);
    minus6(pose0, residuals, residuals); // r = P - P'

    // get residuals for end0
    // t = 1
    minus6(end0, pose0, residuals+6); // v = (P - P-1)
    minus6(pose0, end1, v_1t); // v-1 = (P-1 - P-2) / d
    scalar<6>(v_1t, T(1) / *interFrameRatio, v_1t); // scale
    minus6(residuals+6, v_1t, residuals+6); // a*t² = (v - v-1)
    scalar<6>(residuals+6, T(0.5), residuals+6); // scale => 1/2 * a*t²
    plus6(v_1t, residuals+6, residuals+6); // P' = P-1 + v_1t + (1/2 * a*t²)
    plus6(pose0, residuals+6, residuals+6);
    minus6(end0, residuals+6, residuals+6); // r = P - P'

    scalar<NUM_RESIDUALS>(residuals, T(scale), residuals);

    // down-scale rotation
    scalar3(residuals, T(0.01), residuals);
    scalar3(residuals+6, T(0.01), residuals+6);

    return *interFrameRatio >= T(_EPS);
  }


  // Factory to hide the construction of the CostFunction object from the client code.
  static ceres::CostFunction* Create(double scale)
  {
    return (new ceres::AutoDiffCostFunction<RsConstAccelerationPrior,
        NUM_RESIDUALS,
        1,               // intra-frame ratio
        NUM_POSE_PARAMS, // current frame first pose
        NUM_POSE_PARAMS, // current frame last pose
        NUM_POSE_PARAMS, // frame -1 first pose
        NUM_POSE_PARAMS>( // frame -2 last pose
                new RsConstAccelerationPrior(scale)));
  }
};

} // namespace vision


#endif
