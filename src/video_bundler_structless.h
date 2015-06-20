#ifndef _VideoBundlerStructLess_h
#define _VideoBundlerStructLess_h

#include <cmath>
#include <cstdio>
#include <iostream>
#include <thread>
#include <string.h>
#include <glog/logging.h>
#include <ceres/ceres.h>
#include <ceres/rotation.h>


#include "mat.h"



namespace vision {

// Templated pinhole camera model for used with Ceres.
// The principal point is not modeled (i.e. it is assumed be at the image center)
struct RayDistance {
  static const unsigned short NUM_RESIDUALS = 3;

  RayDistance(const double* const observed,
              const double* const observed2) // 2D
      : observed_x(observed[0]),
        observed_y(observed[1]),
        observed2_x(observed2[0]),
        observed2_y(observed2[1]) {}

  RayDistance(const double camera[NUM_CAM_PARAMS],
              const double* const observed,
              const double* const observed2) // 2D
      : observed_x(observed[0]),
        observed_y(observed[1]),
        observed2_x(observed2[0]),
        observed2_y(observed2[1]) {
    memcpy(camera_params, camera, sizeof(camera_params));
  }

  RayDistance(const double camera[NUM_CAM_PARAMS],
              const double pose2[NUM_POSE_PARAMS],
              const double* const observed,
              const double* const observed2) // 2D
      : observed_x(observed[0]),
        observed_y(observed[1]),
        observed2_x(observed2[0]),
        observed2_y(observed2[1]) {
    memcpy(camera_params, camera, sizeof(camera_params));
    memcpy(prev_pose, pose2, sizeof(prev_pose));
  }

  template <typename T>
  bool operator()(const T* const pose,
                  T* residuals) const {
    T camera[NUM_CAM_PARAMS];
    memcpy(camera, camera_params, sizeof(camera));

    T pose2[NUM_POSE_PARAMS] = {
        T(prev_pose[0]), T(prev_pose[1]), T(prev_pose[2]),
        T(prev_pose[3]), T(prev_pose[4]), T(prev_pose[5])
    };

    return operator()(camera, pose, pose2, residuals);
  }

  template <typename T>
  bool operator()(const T* const pose,
                  const T* const pose2,
                  T* residuals) const {
    T camera[NUM_CAM_PARAMS];
    memcpy(camera, camera_params, sizeof(camera));

    return operator()(camera, pose, pose2, residuals);
  }

  template <typename T>
  bool operator()(const T* const camera, //focal length + 2nd and 4th distortion
                  const T* const pose,
                  const T* const pose2,
                  T* residuals) const
  {
    T obs[2] = { T(observed_x), T(observed_y) };
    T obs2[2] = { T(observed2_x), T(observed2_y) };

    rayDist(camera, pose, obs, camera, pose2, obs2, residuals);

    return true;
  }


  // Factory to hide the construction of the CostFunction object from the client code.
  static ceres::CostFunction* Create(
      const double* const observed,
      const double* const observed2)
  {
    return (new ceres::AutoDiffCostFunction<RayDistance,
        NUM_RESIDUALS,
        NUM_CAM_PARAMS,
        NUM_POSE_PARAMS,
        NUM_POSE_PARAMS>( // second pose
                new RayDistance(observed, observed2)));
  }

  // Factory to hide the construction of the CostFunction object from the client code.
  static ceres::CostFunction* Create(
      const double* const camera_params,
      const double* const observed,
      const double* const observed2)
  {
    return (new ceres::AutoDiffCostFunction<RayDistance,
        NUM_RESIDUALS,
        NUM_POSE_PARAMS,
        NUM_POSE_PARAMS>( // second pose
                new RayDistance(camera_params, observed, observed2)));
  }

  // Factory to hide the construction of the CostFunction object from the client code.
  static ceres::CostFunction* Create(
      const double* const camera_params,
      const double* const prev_pose,  // previous pose
      const double* const observed,
      const double* const observed2)
  {
    return (new ceres::AutoDiffCostFunction<RayDistance,
        NUM_RESIDUALS,
        NUM_POSE_PARAMS>(
                new RayDistance(camera_params, prev_pose, observed, observed2)));
  }

  // 2D observation
  double observed_x;
  double observed_y;

  // second 2D observation
  double observed2_x;
  double observed2_y;

  // focal length + distortion
  double camera_params[NUM_CAM_PARAMS];

  // previous pose
  double prev_pose[NUM_POSE_PARAMS];

  //TODO use references or pointers to save memory
};

}  // namespace vision

#endif
