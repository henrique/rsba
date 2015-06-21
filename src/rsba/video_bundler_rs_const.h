#ifndef _VideoBundlerRSinter_h
#define _VideoBundlerRSinter_h

#include <cmath>
#include <cstdio>
#include <iostream>

#include "rsba/mat/cam.h"
#include "rsba/tracking.h"
#include "rsba/video_bundler_free.h"

namespace vision {

struct ConstRsReprojectionError: public ReprojectionError {
  ConstRsReprojectionError(const double camera[NUM_CAM_PARAMS],
                      const framePtr& f,
                      const double* const observed) // 2D
      : ReprojectionError(camera, observed),
        f(f) {
  };


  template <typename T>
  bool operator()(const T* const end0,
                  const T* const end1,
                  const T& d,
                  const T* const point,
                  T* residuals) const {
    if (d < T(eps)) return false;

    T pose[6];
    T obs[2] = { T(observed_x), T(observed_x) };
    interpolate(end0, end1, d, obs, pose);
    return ReprojectionError::operator()(pose, point, residuals);
  }

  template <typename T>
  void interpolate(const T end0[6],
                   const T end1[6],
                   const T& d,
                   const T obs[2],
                   T inter[6]) {
    T t = abs(f->scanlines[1] - f->scanlines[0]) / d; // total time between frames
    T tau;
    if (f->shutter == VERTICAL) { // vertical rs
      tau = (obs[1] - T(f->scanlines[0])) / t;
    } else { //if (shutter == HORIZONTAL) { // horizontal rs
      tau = (obs[0] - T(f->scanlines[0])) / t;
    }
    vision::interpolate(end0, end1, tau, inter);
  };


  // Factory to hide the construction of the CostFunction object from the client code.
  static ceres::CostFunction* Create(
      const double* const camera_params,
      const int* const rs_params,
      const double* const observed)
  {
    return (new ceres::AutoDiffCostFunction<ConstRsReprojectionError,
        NUM_RESIDUALS,
        NUM_POSE_PARAMS, // Initial frame pose
        NUM_POSE_PARAMS, // Final frame pose
        NUM_POINT_PARAMS>(
                new ConstRsReprojectionError(camera_params, rs_params, observed)));
  }

  const framePtr f;
};


} // namespace vision


#endif
