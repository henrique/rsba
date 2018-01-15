#ifndef __mat_core_h__
#define __mat_core_h__


#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <limits>
#include "rsba/mat/eigenTypes.h"


using namespace std;

#define _EPS std::numeric_limits<double>::epsilon()
#define CLAMP(f,min,max) ((f)<(min)?(min):(f)>(max)?(max):(f))



namespace vision {


template <typename T>
inline T det33(const T M[9]) {
  const ceres::MatrixAdapter<const T, 1, 3> m(M);

  T determinant = m(0,0)*(m(1,1)*m(2,2) - m(2,1)*m(1,2))
                - m(0,1)*(m(1,0)*m(2,2) - m(2,0)*m(1,2))
                + m(0,2)*(m(1,0)*m(2,1) - m(2,0)*m(1,1));
  return determinant;
}


template <typename T>
inline bool inv33(const T M[9], T Minv[9]) {
  const ceres::MatrixAdapter<const T, 1, 3> m(M);
  ceres::MatrixAdapter<T, 1, 3> minv(Minv);

  T d = det33(M);
  if (d < T(_EPS) and d > T(-_EPS)) return false;

  minv(0, 0) = (m(1, 1) * m(2, 2) - m(2, 1) * m(1, 2)) / d;
  minv(0, 1) = (m(0, 2) * m(2, 1) - m(0, 1) * m(2, 2)) / d;
  minv(0, 2) = (m(0, 1) * m(1, 2) - m(0, 2) * m(1, 1)) / d;
  minv(1, 0) = (m(1, 2) * m(2, 0) - m(1, 0) * m(2, 2)) / d;
  minv(1, 1) = (m(0, 0) * m(2, 2) - m(0, 2) * m(2, 0)) / d;
  minv(1, 2) = (m(1, 0) * m(0, 2) - m(0, 0) * m(1, 2)) / d;
  minv(2, 0) = (m(1, 0) * m(2, 1) - m(2, 0) * m(1, 1)) / d;
  minv(2, 1) = (m(2, 0) * m(0, 1) - m(0, 0) * m(2, 1)) / d;
  minv(2, 2) = (m(0, 0) * m(1, 1) - m(1, 0) * m(0, 1)) / d;
  return true;
}


template <typename T>
inline void assign3(const T v[3], T rtn[3]) {
  rtn[0] = v[0];
  rtn[1] = v[1];
  rtn[2] = v[2];
}


template <size_t N, typename T>
inline void assign(const T v[N], T rtn[N]) {
  for (size_t n = 0; n < N; n++)
    rtn[n] = v[n];
}


template <typename T>
inline void scalar3(const T v[3], const T s, T rtn[3]) {
  rtn[0] = v[0] * s;
  rtn[1] = v[1] * s;
  rtn[2] = v[2] * s;
}


template <size_t N, typename T>
inline void scalar(const T v[N], const T s, T rtn[N]) {
  for (size_t n = 0; n < N; n++)
    rtn[n] = v[n] * s;
}


template <typename T>
inline void plus2(const T a[2], const T b[2], T rtn[2]) {
  rtn[0] = a[0] + b[0];
  rtn[1] = a[1] + b[1];
}


template <typename T>
inline void plus3(const T a[3], const T c, T rtn[3]) {
  rtn[0] = a[0] + c;
  rtn[1] = a[1] + c;
  rtn[2] = a[2] + c;
}


template <typename T>
inline void plus3(const T a[3], const T b[3], T rtn[3]) {
  rtn[0] = a[0] + b[0];
  rtn[1] = a[1] + b[1];
  rtn[2] = a[2] + b[2];
}


template <typename T>
inline void plus6(const T a[6], const T b[6], T rtn[6]) {
  rtn[0] = a[0] + b[0];
  rtn[1] = a[1] + b[1];
  rtn[2] = a[2] + b[2];
  rtn[3] = a[3] + b[3];
  rtn[4] = a[4] + b[4];
  rtn[5] = a[5] + b[5];
}


template <typename T>
inline void minus2(const T a[2], const T b[2], T rtn[2]) {
  rtn[0] = a[0] - b[0];
  rtn[1] = a[1] - b[1];
}


template <typename T>
inline void minus3(const T a[3], const T b[3], T rtn[3]) {
  rtn[0] = a[0] - b[0];
  rtn[1] = a[1] - b[1];
  rtn[2] = a[2] - b[2];
}


template <typename T>
inline void minus6(const T a[6], const T b[6], T rtn[6]) {
  rtn[0] = a[0] - b[0];
  rtn[1] = a[1] - b[1];
  rtn[2] = a[2] - b[2];
  rtn[3] = a[3] - b[3];
  rtn[4] = a[4] - b[4];
  rtn[5] = a[5] - b[5];
}


// invert an angle-axis rotation vector
template <typename T>
inline void invert3(const T v[3], T rtn[3]) {
  rtn[0] = -v[0];
  rtn[1] = -v[1];
  rtn[2] = -v[2];
}


template <typename T>
inline T norm(const T& v0, const T& v1) {
  return sqrt(v0 * v0 + v1 * v1);
}

template <typename T>
inline T norm(const T& v0, const T& v1, const T& v2) {
  return sqrt(v0 * v0 + v1 * v1 + v2 * v2);
}

// norm 2 in R3
template <typename T>
inline T norm3(const T v[3]) {
  return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

// norm 2 in R3
template <typename T>
inline bool normalize3(const T v[3], T rtn[3]) {
  T n = norm3(v);
  if (n < T(_EPS)) return false;

  scalar3(v, T(1.0/n), rtn);
  return true;
}


// return distance between two 2D vectors
template <typename T>
inline T dist2(const T a[2], const T b[2]) {
  return norm(a[0] - b[0], a[1] - b[1]);
}


// return distance between two 3D vectors
template <typename T>
inline T dist3(const T a[3], const T b[3]) {
  return norm(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}


} // namespace

#endif /* __mat_core_h__ */
