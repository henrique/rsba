#ifndef __mat_h__
#define __mat_h__


#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include "rsba/mat/eigenTypes.h"
#include "rsba/mat/core.h"


using namespace std;



namespace vision {

// The camera is parameterized using 3 parameters: 1 for focal length and 2 for radial distortion.
// The camera pose is parameterized with 3 for rotation and 3 for translation
#define NUM_POINT_PARAMS 3
#define NUM_POSE_PARAMS 6


enum cam_params {
  CAM_FOCAL_X=0,
  CAM_FOCAL_Y=1,
  CAM_DIST_K1=2,
  CAM_DIST_K2=3,
  CAM_DIST_P1=4,
  CAM_DIST_P2=5,
  CAM_DIST_K3=6,
  CAM_CENTER_X=7,
  CAM_CENTER_Y=8,
  NUM_CAM_PARAMS=9
};


enum SHUTTER {
  GLOBAL=0,
  HORIZONTAL=1, // rolling shutter with horizontal scan-lines
  VERTICAL=2, // rolling shutter with vertical scan-lines
};





// Distort 2D pixel
template <typename T>
inline void distort(const T cam[NUM_CAM_PARAMS], //focal length + distortion
                    const T img[2],
                    T projection[2])
{
  // center of distortion.
  const T& xp = img[0];
  const T& yp = img[1];

  // Apply radial distortion.
  T distortion;
  const T& k1 = cam[2];
  const T& k2 = cam[3];
  const T& p1 = cam[4];
  const T& p2 = cam[5];
  const T& k3 = cam[6];

  T r2 = xp*xp + yp*yp;
  distortion = T(1.) + r2 * (k1 + r2 * (k2 + (r2 * k3)));

  T t2(2.);
  T xy(xp * yp);
  projection[0] = (distortion * xp) + (t2 * p1 * xy + p2 * (r2 + t2 * xp * xp));
  projection[1] = (distortion * yp) + (p1 * (r2 + t2 * yp * yp) + t2 * p2 * xy);
}



// Undistort 2D pixel
template <typename T>
inline bool undistort(const T cam[NUM_CAM_PARAMS], //focal length + distortion
                      const T img[2],
                      T projection[2])
{
  typedef Eigen::Matrix<T, 2, 1> Vector2;
  typedef Eigen::Map<const Vector2> Vector2Ref;

  const Vector2Ref p_n(img);
  Vector2 err, p_d, p_u = p_n;
  T dist, norm = p_n.norm();

  const uint8_t max_iter = 200;
  const T pixel_error = norm * 0.001 / cam[CAM_FOCAL_X]; // max pixel error
  bool valid = false;

  uint8_t num_it = 0;
  for (; num_it < max_iter; num_it++) {
    distort(cam, p_u.data(), p_d.data());
    err = p_d - p_n;
    dist = err.norm();
    if (dist > norm) {
      break; // too far out, not going to improve
    }
    p_u = p_u - err;
    if (dist < pixel_error) {
      valid = true;
      break;
    }
  }

  projection[0] = p_u[0];
  projection[1] = p_u[1];

  return valid;
}



template <typename T>
inline void c2w(const T pose[NUM_POSE_PARAMS], const T pt[3], T p[3]) {
  // To world frame
  T Pinv[3];
  invert3(pose, Pinv);
  ceres::AngleAxisRotatePoint(Pinv, pt, p);

  p[0] += pose[3];
  p[1] += pose[4];
  p[2] += pose[5];
}


// get direction to 3D point in the camera frame of reference
template <typename T>
inline bool c2direction(const T pose[NUM_POSE_PARAMS], const T pt[3], T d[3]) {
  // rotate to world frame
  T Pinv[3];
  invert3(pose, Pinv);
  ceres::AngleAxisRotatePoint(Pinv, pt, d);

  return normalize3(d, d);
}


// get direction to 3D point in the world frame of reference
template <typename T>
inline bool direction(const T pose[NUM_POSE_PARAMS], const T pt[3], T d[3]) {
  // pose[3,4,5] are the translation.
  d[0] = pt[0] - pose[3];
  d[1] = pt[1] - pose[4];
  d[2] = pt[2] - pose[5];

  return normalize3(d, d);
}


// get direction to 2D observation
template <typename T>
bool direction(const T cam[NUM_CAM_PARAMS],
               const T pose[NUM_POSE_PARAMS],
               const T& x0, const T& y0,
               T d[3],
               bool validate = true) // check whether the undistortion was successful
{
  // To camera frame: assume 0 mean x,y; i.e. already centered to camera
  const T Eps(_EPS);
  if (cam[CAM_FOCAL_X] < Eps) return false;
  if (cam[CAM_FOCAL_Y] < Eps) return false;

  d[0] = (x0 - cam[CAM_CENTER_X])/cam[CAM_FOCAL_X];
  d[1] = (y0 - cam[CAM_CENTER_Y])/cam[CAM_FOCAL_Y];
  d[2] = T(1);

  if (!undistort(cam, d, d) and validate) {
    return false;
  } //else keep original point

  // To world frame
  return c2direction(pose, d, d);
}


// get direction to 2D observation
template <typename T>
inline bool direction(const T cam[NUM_CAM_PARAMS], const T pose[NUM_POSE_PARAMS], const T obs[2], T d[3], bool validate = true) {
  return direction(cam, pose, obs[0], obs[1], d, validate);
}



// Link: http://tog.acm.org/resources/RTNews/html/rtnv11n1.html#art3
template <typename T>
bool triangulate(const T center1[3], const T d1[3],
                 const T center2[3], const T d2[3],
                 T p[3]) {
  typedef Eigen::Matrix<T, 3, 3> Matrix3;
  typedef Eigen::Matrix<T, 3, 1> Vector3;
  typedef Eigen::Map<const Vector3> Vector3Ref;

//  Vector3 r1(pose1), r2(pose2);
  const Vector3Ref c1(center1), c2(center2);
  const Vector3Ref a(d1), b(d2);

  // Setup coefficient matrix
  const Matrix3 A = (Matrix3::Identity() - a * a.transpose())
    + (Matrix3::Identity() - b * b.transpose());

  if (A.determinant() < T(_EPS)) {
    cout << A.determinant() << endl;
    abort(); //TODO remove
    return false;
  }

  // Setup solution vector
  const Vector3 y = (Matrix3::Identity() - a * a.transpose()) * c1
    + (Matrix3::Identity() - b * b.transpose()) * c2;

  //cout << "A: " << endl << A << endl;
  //cout << "y: " << endl << y << endl;

  // Solve: A * x = y
  const Vector3 x = A.inverse() * y;
  //Eigen::JacobiSVD<Matrix3> svd;
  //svd.compute(A, Eigen::ComputeFullV | Eigen::ComputeFullU);
  //Vector3 x = svd.solve(y);

  //cout << "x: " << endl << x << endl;
  //cout << "A * X: " << endl << A * x << endl;

  p[0] = x(0);
  p[1] = x(1);
  p[2] = x(2);

  return true;
}


// Link: http://tog.acm.org/resources/RTNews/html/rtnv11n1.html#art3
template <typename T>
bool triangulate(const T cam1[NUM_CAM_PARAMS], const T pose1[NUM_POSE_PARAMS], const T obs1[2],
                 const T cam2[NUM_CAM_PARAMS], const T pose2[NUM_POSE_PARAMS], const T obs2[2],
                 T p[3]) {
  T d1[3], d2[3];
  if (!direction(cam1, pose1, obs1, d1)) return false;
  if (!direction(cam2, pose2, obs2, d2)) return false;

  return triangulate(pose1+3, d1, pose2+3, d2, p);
}



/// spherical linear interpolation
/// tau must be between 0 and 1
template <typename T>
void slerp(const T r0[3], const T r1[3], const T& tau, T inter[3])
{
//    T dot = r0[0]*r1[0] + r0[1]*r1[1] + r0[2]*r1[2];
//    T angle = acos(CLAMP(dot, T(-1.0 + _EPS), T(1.0 - _EPS)));
//
//    T sin1t = sin((1.0 - tau) * angle);
//    T sint  = sin(tau * angle);
//    T sina  = sin(angle);
//    CHECK_GT(sina, T(_EPS));
//
//    inter[0] = (sin1t / sina) * r0[0] + (sint / sina) * r1[0];
//    inter[1] = (sin1t / sina) * r0[1] + (sint / sina) * r1[1];
//    inter[2] = (sin1t / sina) * r0[2] + (sint / sina) * r1[2];

  //linear interpolation: works better for small angles
  inter[0] = r0[0] + (r1[0] - r0[0]) * tau;
  inter[1] = r0[1] + (r1[1] - r0[1]) * tau;
  inter[2] = r0[2] + (r1[2] - r0[2]) * tau;

//  using namespace Eigen;
//
//  T n0 = norm3(r0);
//  T n1 = norm3(r1);
//
//  Vector3Ref<T> interRef(inter);
//
//  Vector<T,3> r0ref(r0);
//  if (n0 < _EPS) r0ref /= n0;
//
//  Vector<T,3> r1ref(r1);
//  if (n1 < _EPS) r1ref /= n1;
//
//  AngleAxis<T> a0(n0, r0ref);
//  AngleAxis<T> a1(n1, r1ref);
//  Quaternion<T> q0(a0), q1(a1), qr(q0.slerp(tau, q1));
//  AngleAxis<T> rst(qr);
//  interRef = rst.axis()*rst.angle();
}



// spherical and linear interpolation for camera poses
template <typename T>
void interpolate(const T pose0[NUM_POSE_PARAMS],
                 const T pose1[NUM_POSE_PARAMS],
                 const T& tau,
                 T inter[NUM_POSE_PARAMS],
                 const bool useSlerp = true) // also interpolate rotation?
{
  // rotation
  if (useSlerp) {
    slerp(pose0, pose1, tau, inter);
  } else {
    assign3(pose0, inter);
  }

  // translation
  inter[3] = pose0[3] + (pose1[3] - pose0[3]) * tau;
  inter[4] = pose0[4] + (pose1[4] - pose0[4]) * tau;
  inter[5] = pose0[5] + (pose1[5] - pose0[5]) * tau;
}


#include <signal.h>
template <typename T>
void interpolate_rs(const T pose0[6], const T pose1[6],
                    const SHUTTER shutter, const int scanlines[2], const T obs[2],
                    T inter[6],
                    const bool useSlerp = true) // also interpolate rotation?
{
  if (shutter == GLOBAL) {
    assign<6>(pose0, inter);
  } else {
    T tau;
    if (shutter == VERTICAL) { // vertical rs
      tau = (obs[1] - T(scanlines[0])) / T(scanlines[1] - scanlines[0]);
      //cout<<obs[0]<<":"<<obs[1]<<" V"<<tau<<endl;
    } else { //if (shutter == HORIZONTAL) { // horizontal rs
      tau = (obs[0] - T(scanlines[0])) / T(scanlines[1] - scanlines[0]);
      //cout<<obs[0]<<":"<<obs[1]<<" H"<<tau<<endl;
    }

#ifndef NDEBUG
    if (tau<T(0) or tau>T(1)) {
      if (sizeof(tau) == 8 and obs[0] != obs[1]) {
        cerr << "Tau out of bounds! " << obs[0] << ":" << obs[1] << " t" << tau << endl;
      }
    }
#endif

    if (tau < T(0))
      tau = T(0);

    if (tau > T(1))
      tau = T(1);

    interpolate(pose0, pose1, tau, inter, useSlerp);
  }
};



// transform from world to camera frame
template <typename T>
inline void w2c(const T pose[NUM_POSE_PARAMS],
                const T point[3],
                T pt[3])
{
  // pose[3,4,5] are the translation.
  pt[0] = point[0] - pose[3];
  pt[1] = point[1] - pose[4];
  pt[2] = point[2] - pose[5];

  // pose[0,1,2] are the angle-axis rotation.
  ceres::AngleAxisRotatePoint(pose, pt, pt);
}



// reproject 3D point to the 2D image frame
template <typename T>
inline bool c2i(const T cam[NUM_CAM_PARAMS], //focal length + distortion
                const T pt[3],
                T projection[2])
{
  if (pt[2] < T(_EPS) and pt[2] > T(-_EPS)) { // non-zero
      return false;
  }

  // dehomogenize
  T img[2] = {
      (pt[0] / pt[2]),
      (pt[1] / pt[2])
  };

  // Apply radial distortion.
  distort(cam, img, projection);

  // Compute final projected point position.
  projection[0] *= cam[CAM_FOCAL_X]; // focal length
  projection[1] *= cam[CAM_FOCAL_Y];
  projection[0] += cam[CAM_CENTER_X]; // camera centers
  projection[1] += cam[CAM_CENTER_Y];
  return true;
}



// reproject 3D point to the 2D image frame
template <typename T>
inline bool w2i(const T cam[NUM_CAM_PARAMS], //focal length + distortion
                const T pose[NUM_POSE_PARAMS],
                const T point[3],
                T projection[2],
                bool validate = true) // check if projection is valid
{
  T pt[3];
  w2c(pose, point, pt);

  if (pt[2] < T(1e-8)) { // in front of the camera
    if (validate) {
      return false;
    } else if (pt[2] < T(_EPS) and pt[2] > T(-_EPS)) { // numerically zero
      pt[2] = T(_EPS);
    }
  }

  return c2i(cam, pt, projection);
}



// Calculates the squared reprojection error
template <typename T>
inline bool reprojection_error(const T cam[NUM_CAM_PARAMS], //focal length + distortion
                const T pose[NUM_POSE_PARAMS],
                const T point2[2],
                const T point3[3],
                T& sqrdError)
{
  double proj[2];
  if (w2i(cam, pose, point3, proj)) {
    minus2(proj, point2, proj);
    sqrdError = (proj[0] * proj[0] + proj[1] * proj[1]);
    return true;
  }

  return false;
}



// Check if reprojection is within squared threshold
template <typename T>
inline bool validate(const T cam[NUM_CAM_PARAMS], //focal length + distortion
                const T pose[NUM_POSE_PARAMS],
                const T point2[2],
                const T point3[3],
                const T& sqrdThreshold)
{
  double sqrdError;
  if (reprojection_error(cam, pose, point2, point3, sqrdError)) {
    if (sqrdError < sqrdThreshold) return true;
  }

  return false;
}


// calculate the distance from the ray origins
// to the closest point on the opposite ray
template <typename T>
bool ray_intersect(const T p2[3], const T d1[3], const T d2[3],
             T dist[3])
{
  // CrossProduct vector points to the shortest path between the two rays
  T c[3];
  ceres::CrossProduct(d1, d2, c);

  // Norm²
  T n2 = norm3(c);

  if (n2 < T(3 * _EPS)) { // parallel lines;
    return false;
  }
  else
  {
    // normalize
    scalar3(c, T(1)/n2, c);

    // The error is shortest distance between the rays
    // p2 + k*d2 == l*d1 + b*c
    T m[9] = { d1[0], d2[0], c[0],
               d1[1], d2[1], c[1],
               d1[2], d2[2], c[2] };

    T minv[9];
    if (!inv33(m, minv)) {
      return false; //shouldn't happen due to the cross product
    }

    dist[0] =  ceres::DotProduct(minv+0, p2); //  l
    dist[1] = -ceres::DotProduct(minv+3, p2); // -k
    dist[2] =  ceres::DotProduct(minv+6, p2); //  b //NOT USED!
  }

  return true;
}



template <typename T>
inline bool ray_intersect(const T pose[NUM_POSE_PARAMS], const T d1[3],
             const T pose2[NUM_POSE_PARAMS], const T d2[3],
             T dist[3])
{
  // Calculate displacement between the two camera poses
  // pose[3,4,5] are the translation
  // and the first pose is moved to the origin, i.e. p1 = { 0, 0, 0 }
  T p2[3] = {
      pose2[3] - pose[3],
      pose2[4] - pose[4],
      pose2[5] - pose[5]
  };

  return ray_intersect(p2, d1, d2, dist);
}



template <typename T>
bool rayDist(const T cam[NUM_CAM_PARAMS], const T pose[NUM_POSE_PARAMS], const T obs[2],
             const T cam2[NUM_CAM_PARAMS], const T pose2[NUM_POSE_PARAMS], const T obs2[2],
             T dist[3])
{
  // calculate ray direction:
  // pose[0,1,2] are the angle-axis rotation.
  T d1[3], d2[3];
  if (!direction(cam, pose, obs, d1)) return false;
  if (!direction(cam2, pose2, obs2, d2)) return false;

  // Calculate displacement between the two camera poses
  // pose[3,4,5] are the translation
  // and the first pose is moved to the origin, i.e. p1 = { 0, 0, 0 }
  T p2[3] = {
      pose2[3] - pose[3],
      pose2[4] - pose[4],
      pose2[5] - pose[5]
  };

  T len[3]; // distances to ray origin
  if ( ! ray_intersect(p2, d1, d2, len))
  {
    // parallel lines; use distance between poses
    dist[0] = p2[0];
    dist[1] = p2[1];
    dist[2] = p2[2];
  }
  else
  {
    T& l = len[0];
    T& k = len[1];
    T& b = len[2];
    scalar3(p2, b, dist);

    //if (l > T(0) and k > T(0)) // in front of the camera
    {
      T s1[3] = { l*d1[0], l*d1[1], l*d1[2] };
      T s2[3] = { p2[0] + k*d2[0], p2[1] + k*d2[1], p2[2] + k*d2[2] };

      dist[0] = s1[0] - s2[0];
      dist[1] = s1[1] - s2[1];
      dist[2] = s1[2] - s2[2];
    }
//    else // penalized for wrong direction
//    {
//      const T p(999);
//      dist[0] = p * d2[0] - p * d1[0];
//      dist[1] = p * d2[1] - p * d1[1];
//      dist[2] = p * d2[2] - p * d1[2];
//      //return false;
//    }
  }

  return true;
}

} // namespace

#endif /* __mat_h__ */
