#ifndef _VideoBundlerFree_h
#define _VideoBundlerFree_h

#include <cmath>
#include <cstdio>
#include <iostream>

#include <ceres/ceres.h>
#include <ceres/rotation.h>



namespace vision {

// Templated pinhole camera model for used with Ceres.
// The principal point is not modeled (i.e. it is assumed be at the image center)
struct ReprojectionError {
  static const unsigned short NUM_RESIDUALS = 2;

  ReprojectionError() : observed_x(0), observed_y(0) {};
  ReprojectionError(const double observed[2]) // 2D
      : observed_x(observed[0]), observed_y(observed[1]) {};


  ReprojectionError(const double camera[NUM_CAM_PARAMS],
                    const double observed[2]) // 2D
      : ReprojectionError(observed) {
    memcpy(camera_params, camera, sizeof(camera_params));
  };


  template <typename T>
  bool operator()(const T* const pose,
                  const T* const point,
                  T* residuals) const {
    T camera[NUM_CAM_PARAMS];
    for (short i = 0; i < NUM_CAM_PARAMS; i++)
      camera[i] = T(camera_params[i]);

    return operator()(camera, pose, point, residuals);
  }


  template <typename T>
  bool operator()(const T* const camera, //focal length + 2nd and 4th distortion
                  const T* const pose,
                  const T* const point,
                  T* residuals) const
  {
    T proj[2]; //reprojection
    if ( ! w2i(camera, pose, point, proj, true)) {
      return false;
    }

    // The error is the difference between the reprojection and observed position.
    residuals[0] = (proj[0] - T(observed_x));
    residuals[1] = (proj[1] - T(observed_y));

    T threshold(5);
    if (abs(residuals[0]) < threshold and abs(residuals[1]) < threshold) {
      return true;
    } else { //TODO check matches
      return true;
    }
  }



  // Factory to hide the construction of the CostFunction object from the client code.
  static ceres::CostFunction* Create(
      const double* const observed)
  {
    return (new ceres::AutoDiffCostFunction<ReprojectionError,
        NUM_RESIDUALS,
        NUM_CAM_PARAMS,
        NUM_POSE_PARAMS,
        NUM_POINT_PARAMS>(
                new ReprojectionError(observed)));
  }

  // Factory to hide the construction of the CostFunction object from the client code.
  static ceres::CostFunction* Create(
      const double* const camera_params,
      const double* const observed)
  {
    return (new ceres::AutoDiffCostFunction<ReprojectionError,
        NUM_RESIDUALS,
        NUM_POSE_PARAMS,
        NUM_POINT_PARAMS>(
                new ReprojectionError(camera_params, observed)));
  }

  // 2D observation
  double observed_x;
  double observed_y;

  //focal length + distortion
  double camera_params[NUM_CAM_PARAMS];

  //TODO use references or pointers to save memory
};


}  // namespace vision

#endif
