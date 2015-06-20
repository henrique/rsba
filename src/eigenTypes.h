#ifndef __eigenTypes_h__
#define __eigenTypes_h__

#include "Eigen/Core"

template <class T, size_t N>
using Vector = Eigen::Matrix<T, N, 1>;

template <class T, size_t N>
using VectorRef = Eigen::Map< Eigen::Matrix<T, N, 1> >;

template <size_t N>
using VectorDRef = Eigen::Map< Eigen::Matrix<double, N, 1> >;

template <class T>
using Vector2 = Vector<T, 2>;

template <class T>
using Vector2Ref = Eigen::Map< Vector<T, 2> >;

using Vector2dRef = Vector2Ref<double>;

template <class T>
using Vector3 = Vector<T, 3>;

template <class T>
using Vector3Ref = Eigen::Map< Vector<T, 3> >;

template <class T>
using Vector3CRef = Eigen::Map< const Vector<T, 3> >;

using Vector3dRef = Vector3Ref<double>;

#endif
