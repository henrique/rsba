#ifndef _VideoBundler_h
#define _VideoBundler_h

#include <cmath>
#include <cstdio>
#include <iostream>

#include "rsba/ceres/ceres.h"
#include "rsba/ceres/rotation.h"
#include "rsba/Eigen/Core"


#include "rsba/mat/cam.h"


namespace vision {

// Templated pinhole camera model for used with Ceres.
// The principal point is not modeled (i.e. it is assumed be at the image center)
struct ReprojectionKineError {
  static const unsigned short NUM_RESIDUALS = 5;

  ReprojectionKineError(const double* const observed) // 2D
      : observed_x(observed[0]), observed_y(observed[1]) {}

  template <typename T>
  bool operator()(const T* const camera, //focal length + 2nd and 4th distortion
                  const T* const pose,   // t
                  const T* const pose_1, // t-1
                  const T* const pose_2, // t-2
                  const T* const point,
                  T* residuals) const {
    typedef Eigen::Matrix<T, 3, 1> Vector3;
    typedef Eigen::Map<const Vector3> Vector3Ref;

    // pose[0,1,2] are the angle-axis rotation.
    T p[3];
    ceres::AngleAxisRotatePoint(pose, point, p);

    // pose[3,4,5] are the translation.
    p[0] += pose[3];
    p[1] += pose[4];
    p[2] += pose[5];

    //if (p[2] < T(0)) return false;

    // Compute the center of distortion. The sign change comes from
    // the camera model that Noah Snavely's Bundler assumes, whereby
    // the camera coordinate system has a negative z axis.
    T xp = - p[0] / p[2];
    T yp = - p[1] / p[2];

    // Apply second and fourth order radial distortion.
    const T& l1 = camera[1];
    const T& l2 = camera[2];
    T r2 = xp*xp + yp*yp;
    T distortion = T(1.0) + r2  * (l1 + l2  * r2);

    // Compute final projected point position.
    const T& focal = camera[0];
    T predicted_x = focal * distortion * xp;
    T predicted_y = focal * distortion * yp;

    // Constant velocity: (p0 - p1)/dt0 == (p1 - p2)/dt1
    Vector3Ref pos0(pose  +3, 3);
    Vector3Ref pos1(pose_1+3, 3);
    Vector3Ref pos2(pose_2+3, 3);
    Vector3 acc = (pos0 - pos1)/T(delta_t0) - (pos1 - pos2)/T(delta_t1);
    T gain = acc.norm() + T(1); // +1 to avoid zeros

    // The error is the difference between the predicted and observed position.
    residuals[0] = gain * (predicted_x - T(observed_x));
    residuals[1] = gain * (predicted_y - T(observed_y));

    return true;
  }



  // Factory to hide the construction of the CostFunction object from the client code.
  static ceres::CostFunction* Create(
      const double* const observed)
  {
    return (new ceres::AutoDiffCostFunction<ReprojectionKineError,
        NUM_RESIDUALS,
        NUM_CAM_PARAMS,
        NUM_POSE_PARAMS, // t
        NUM_POSE_PARAMS, // t-1
        NUM_POSE_PARAMS, // t-2
        NUM_POINT_PARAMS>(
                new ReprojectionKineError(observed)));
  }

  // 2D observation
  double observed_x;
  double observed_y;

  // time-steps between current and last camera poses
  const unsigned delta_t0 = 10;

  // time-steps between last and second-last camera poses
  const unsigned delta_t1 = 10;
};



}  // namespace vision

#endif
