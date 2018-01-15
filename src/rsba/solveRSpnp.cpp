// Author: Henrique Mendonça <henrique@apache.org>
#include "rsba/solveRSpnp.h"
#include "rsba/mat/cam.h"
#include "rsba/VideoSfmBaRs.h"
#include "rsba/struct/VideoSfM.h"

#include <ceres/rotation.h>
#include <opencv2/core/core_c.h>
#if (defined(CV_VERSION_EPOCH) && CV_VERSION_EPOCH == 2)
#include <opencv2/core/internal.hpp>
#else
#define __OPENCV_BUILD
#include <opencv2/cvconfig.h>
#include <opencv2/core/utility.hpp>
#define BlockedRange Range
#endif
#include <iostream>
//#include <boost/timer/timer.hpp>

using namespace ::cv;
using namespace ::vision;
using namespace ::vision::sfm;


template <typename TF>
struct RsBA: public ReprojectionError {
  TF point[NUM_POINT_PARAMS];
  SHUTTER rs;
  const int* const scanlines;

  RsBA(const double camera[NUM_CAM_PARAMS],
       const TF* const observed, // 2D
       const TF* const point3d,
       const SHUTTER _rs,
       const int* const _scanlines) // 3D
  : scanlines(_scanlines)
  {
    observed_x = (observed[0]); observed_y = (observed[1]);
    memcpy(camera_params, camera, sizeof(camera_params));
    memcpy(point, point3d, sizeof(point));
    rs = _rs;
  };


  template <typename T>
  bool operator()(const T* const pose0,
                  T* residuals) const {
    return operator()(pose0, pose0, residuals);
  };


  template <typename T>
  bool operator()(const T* const pose0,
                  const T* const pose1,
                  T* residuals) const {
    T pose[6];
    T obs[2] = { T(observed_x), T(observed_x) };
    interpolate_rs(pose0, pose1, rs, scanlines, obs, pose);

    T camera[NUM_CAM_PARAMS];
    for (short i = 0; i < NUM_CAM_PARAMS; i++)
      camera[i] = T(camera_params[i]);

    T proj[2]; //reprojection
    T p[NUM_POINT_PARAMS] = { T(point[0]), T(point[1]), T(point[2]) };
    if ( ! w2i(camera, pose, p, proj, false)) {
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
      const double camera[NUM_CAM_PARAMS],
      const TF* const observed,
      const TF* const point3d,
      const SHUTTER rs,
      const int* const scanlines)
  {
    return (new ceres::AutoDiffCostFunction<RsBA,
        NUM_RESIDUALS
        ,NUM_POSE_PARAMS // Initial frame pose
        ,NUM_POSE_PARAMS
        >( // Final frame pose
                new RsBA(camera, observed, point3d, rs, scanlines)));
  }
};


bool vision::solveRsPnP(InputArray _opoints, InputArray _ipoints,
                  InputArray _cameraMatrix, InputArray _distCoeffs,
                  OutputArray _rvec, OutputArray _tvec,
                  OutputArray _rvec2, OutputArray _tvec2,
                  const SHUTTER shutter, const int scanlines[2],
                  bool useExtrinsicGuess, int flags)
{
  //boost::timer::auto_cpu_timer btimer;

  _rvec.create(3, 1, CV_64F);
  _tvec.create(3, 1, CV_64F);
  _rvec2.create(3, 1, CV_64F);
  _tvec2.create(3, 1, CV_64F);
  cv::Mat rvec = _rvec.getMat(), tvec = _tvec.getMat();
  cv::Mat rvec2 = _rvec2.getMat(), tvec2 = _tvec2.getMat();

  if (cv::norm(rvec, NORM_L1) + cv::norm(tvec, NORM_L1) +
      cv::norm(rvec2, NORM_L1) + cv::norm(tvec2, NORM_L1) == 0) {
    if (solvePnP(_opoints, _ipoints, _cameraMatrix, _distCoeffs, rvec, tvec, useExtrinsicGuess, flags))
    { // GS Init
      rvec.copyTo(rvec2);
      tvec.copyTo(tvec2);
      std::cout << "GS PnP Init: " << rvec << tvec << endl;
    }
  }

  {
    ceres::Problem problem;
    Mat opoints = _opoints.getMat(), ipoints = _ipoints.getMat();

    vector<double> pose(NUM_POSE_PARAMS), pose2(NUM_POSE_PARAMS);
    { // Init poses
      double rInv[3];
      assign3(rvec.at<Vec3d>(0, 0).val, pose.data());
      invert3(pose.data(), rInv);
      ceres::AngleAxisRotatePoint(rInv, (-tvec.at<Vec3d>(0, 0)).val, pose.data()+3);

      assign3(rvec2.at<Vec3d>(0, 0).val, pose2.data());
      invert3(pose2.data(), rInv);
      ceres::AngleAxisRotatePoint(rInv, (-tvec2.at<Vec3d>(0, 0)).val, pose2.data()+3);
    }

    std::vector<double> cam = sfmCam(_cameraMatrix.getMat(), _distCoeffs.getMat());
    for (int i = 0; i < _opoints.size().width; i++) {
      ceres::CostFunction* costFunction = RsBA<float>::Create(cam.data(),
                                                              ipoints.at<Vec2f>(0, i).val,
                                                              opoints.at<Vec3f>(0, i).val,
                                                              shutter, scanlines);
      problem.AddResidualBlock(costFunction, NULL, pose.data(), pose2.data());
    }

    // Make Ceres automatically detect the bundle structure. Note that the
    // standard solver, SPARSE_NORMAL_CHOLESKY, also works fine but it is slower
    // for standard bundle adjustment problems.
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    //options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 10;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    if (summary.IsSolutionUsable())
    {
//      std::cout << summary.BriefReport() << endl;

      assign3(pose.data(), rvec.at<Vec3d>(0, 0).val);
      assign3(pose.data()+3, tvec.at<Vec3d>(0, 0).val);
      // invert translation
      ceres::AngleAxisRotatePoint(pose.data(), tvec.at<Vec3d>(0, 0).val, tvec.at<Vec3d>(0, 0).val);
      tvec *= -1;
//      std::cout << rvec << endl;
//      std::cout << tvec << endl;


      assign3(pose2.data(), rvec2.at<Vec3d>(0, 0).val);
      assign3(pose2.data()+3, tvec2.at<Vec3d>(0, 0).val);
      // invert translation
      ceres::AngleAxisRotatePoint(pose2.data(), tvec2.at<Vec3d>(0, 0).val, tvec2.at<Vec3d>(0, 0).val);
      tvec2 *= -1;
//      std::cout << rvec2 << endl;
//      std::cout << tvec2 << endl;


//      solvePnP(_opoints, _ipoints, _cameraMatrix, _distCoeffs, orvec, otvec, useExtrinsicGuess, flags);
//      std::cout << orvec << endl;
//      std::cout << otvec << endl;

      return true;
    }
  }

  return false;
}

namespace vision
{
    namespace pnpransac
    {
        struct CameraParameters
        {
            void init(Mat _intrinsics, Mat _distCoeffs, const SHUTTER _shutter, const int* _scanlines)
            {
                _intrinsics.copyTo(intrinsics);
                _distCoeffs.copyTo(distortion);
                shutter = _shutter;
                scanlines = _scanlines;
            }

            Mat intrinsics;
            Mat distortion;
            SHUTTER shutter;
            const int* scanlines;
        };

        struct Parameters
        {
            int iterationsCount;
            float reprojectionError;
            int minInliersCount;
            bool useExtrinsicGuess;
            int flags;
            CameraParameters camera;
            int min_points_count;
        };


        static vector<Point2f> project3dPoints(const Mat& opoints, const Mat& ipoints, const Parameters& params,
                                           const Mat& rvec,  const Mat& tvec,
                                           const Mat& rvec2, const Mat& tvec2)
        {
          vector<Point2f> projected_points(opoints.cols);

          for (int i = 0; i < opoints.cols; i++) {
            Point3f op(opoints.at<Vec3f>(0, i));
            Point2f ip(ipoints.at<Vec2f>(0, i));

            vector<double> pose(6, 0);
            assign3(rvec.at<Vec3d>(0, 0).val, pose.data());
            double rInv[3];
            invert3(pose.data(), rInv);
            ceres::AngleAxisRotatePoint(rInv, (-tvec.at<Vec3d>(0, 0)).val, pose.data()+3);

            vector<double> pose2(6, 0);
            assign3(rvec2.at<Vec3d>(0, 0).val, pose2.data());
            invert3(pose2.data(), rInv);
            ceres::AngleAxisRotatePoint(rInv, (-tvec2.at<Vec3d>(0, 0)).val, pose2.data()+3);

            std::vector<double> cam = sfmCam(params.camera.intrinsics, params.camera.distortion);
            double point[] = { op.x, op.y, op.z };
            double obs[] = { ip.x, ip.y };
            double poseInter[NUM_POSE_PARAMS];
            interpolate_rs(pose.data(), pose2.data(), params.camera.shutter, params.camera.scanlines, obs, poseInter);


            double proj[2]; //reprojection
            if ( ! w2i(cam.data(), poseInter, point, proj, false)) {
              std::abort();
            }
            projected_points[i].x = proj[0];
            projected_points[i].y = proj[1];
          }

          return projected_points;
        }

        static void pnpTask(const vector<char>& pointsMask, const Mat& objectPoints, const Mat& imagePoints,
                     const Parameters& params, vector<int>& inliers,
                     Mat& rvec, Mat& tvec, Mat& rvec2, Mat& tvec2,
                     const Mat& rvecInit, const Mat& tvecInit, const Mat& rvecInit2, const Mat& tvecInit2,
                     Mutex& resultsMutex)
        {
            Mat modelObjectPoints(1, params.min_points_count, CV_32FC3), modelImagePoints(1, params.min_points_count, CV_32FC2);
            for (int i = 0, colIndex = 0; i < (int)pointsMask.size(); i++)
            {
                if (pointsMask[i])
                {
                    Mat colModelImagePoints = modelImagePoints(Rect(colIndex, 0, 1, 1));
                    imagePoints.col(i).copyTo(colModelImagePoints);
                    Mat colModelObjectPoints = modelObjectPoints(Rect(colIndex, 0, 1, 1));
                    objectPoints.col(i).copyTo(colModelObjectPoints);
                    colIndex = colIndex+1;
                }
            }

            //filter same 3d points, hang in solveRsPnP
            double eps = 1e-10;
            int num_same_points = 0;
            for (int i = 0; i < params.min_points_count; i++)
                for (int j = i + 1; j < params.min_points_count; j++)
                {
                    if (norm(modelObjectPoints.at<Vec3f>(0, i) - modelObjectPoints.at<Vec3f>(0, j)) < eps)
                        num_same_points++;
                }
            if (num_same_points > 0)
                return;

            Mat localRvec, localTvec;
            Mat localRvec2, localTvec2;
            rvecInit.copyTo(localRvec);
            tvecInit.copyTo(localTvec);
            rvecInit2.copyTo(localRvec2);
            tvecInit2.copyTo(localTvec2);

            vector<int> localInliers;
            vision::solveRsPnP(modelObjectPoints, modelImagePoints,
                params.camera.intrinsics, params.camera.distortion,
                localRvec, localTvec,
                localRvec2, localTvec2,
                params.camera.shutter, params.camera.scanlines,
                params.useExtrinsicGuess, params.flags);


            vector<Point2f> projected_points = project3dPoints(objectPoints, imagePoints, params, localRvec, localTvec, localRvec2, localTvec2);
            for (int i = 0; i < objectPoints.cols; i++) {
                Point2f p(imagePoints.at<Vec2f>(0, i));
                if ((norm(p - projected_points[i]) < params.reprojectionError)) {
                    localInliers.push_back(i);
                }
            }

            if (localInliers.size() > inliers.size())
            {
              cout << localInliers.size() << "/" << objectPoints.cols << endl;
              resultsMutex.lock();

              inliers.clear();
              inliers.resize(localInliers.size());
              memcpy(&inliers[0], &localInliers[0], sizeof(int) * localInliers.size());
              localRvec.copyTo(rvec);
              localTvec.copyTo(tvec);
              localRvec2.copyTo(rvec2);
              localTvec2.copyTo(tvec2);

              resultsMutex.unlock();
            }
        }

        class PnPSolver
        {
        public:
            void operator()( const BlockedRange& r ) const
            {
                vector<char> pointsMask(objectPoints.cols, 0);
                memset(&pointsMask[0], 1, parameters.min_points_count );
#if (defined(CV_VERSION_EPOCH) && CV_VERSION_EPOCH == 2)
                for( int i=r.begin(); i!=r.end(); ++i )
#else
                for( int i=r.start; i!=r.end; ++i )
#endif
                {
                    generateVar(pointsMask);
                    pnpTask(pointsMask, objectPoints, imagePoints, parameters,
                            inliers, rvec, tvec, rvec2, tvec2,
                            initRvec, initTvec, initRvec2, initTvec2, syncMutex);

                    if ((int)inliers.size() >= parameters.minInliersCount)
                    {
#ifdef HAVE_TBB
                        tbb::task::self().cancel_group_execution();
#else
                        break;
#endif
                    }
                }
            }

            PnPSolver(const Mat& _objectPoints, const Mat& _imagePoints, const Parameters& _parameters,
                Mat& _rvec, Mat& _tvec, Mat& _rvec2,
                Mat& _tvec2, vector<int>& _inliers):
            objectPoints(_objectPoints), imagePoints(_imagePoints), parameters(_parameters),
            rvec(_rvec), tvec(_tvec), rvec2(_rvec2), tvec2(_tvec2), inliers(_inliers)
            {
              rvec.copyTo(initRvec);
              tvec.copyTo(initTvec);
              rvec2.copyTo(initRvec2);
              tvec2.copyTo(initTvec2);

              generator.state = theRNG().state; //to control it somehow...
            }

        private:
            PnPSolver& operator=(const PnPSolver&);

            const Mat& objectPoints;
            const Mat& imagePoints;
            const Parameters& parameters;
            Mat &rvec, &tvec;
            Mat &rvec2, &tvec2;
            vector<int>& inliers;
            Mat initRvec, initTvec;
            Mat initRvec2, initTvec2;

            static RNG generator;
            static Mutex syncMutex;

            void generateVar(vector<char>& mask) const
            {
                int size = (int)mask.size();
                for (int i = 0; i < size; i++)
                {
                    int i1 = generator.uniform(0, size);
                    int i2 = generator.uniform(0, size);
                    char curr = mask[i1];
                    mask[i1] = mask[i2];
                    mask[i2] = curr;
                }
            }
        };

        Mutex PnPSolver::syncMutex;
        RNG PnPSolver::generator;

    }
}



void vision::solveRsPnPRansac(InputArray _opoints, InputArray _ipoints,
                        InputArray _cameraMatrix, InputArray _distCoeffs,
                        OutputArray _rvec, OutputArray _tvec,
                        OutputArray _rvec2, OutputArray _tvec2,
                        const SHUTTER shutter, const int scanlines[2],
                        bool useExtrinsicGuess, int iterationsCount,
                        float reprojectionError, int minInliersCount,
                        OutputArray _inliers, int flags, int min_points_count)
{
  //boost::timer::auto_cpu_timer btimer;

    Mat opoints = _opoints.getMat(), ipoints = _ipoints.getMat();
    Mat cameraMatrix = _cameraMatrix.getMat(), distCoeffs = _distCoeffs.getMat();

    CV_Assert(opoints.isContinuous());
    CV_Assert(opoints.depth() == CV_32F);
    CV_Assert((opoints.rows == 1 && opoints.channels() == 3) || opoints.cols*opoints.channels() == 3);
    CV_Assert(ipoints.isContinuous());
    CV_Assert(ipoints.depth() == CV_32F);
    CV_Assert((ipoints.rows == 1 && ipoints.channels() == 2) || ipoints.cols*ipoints.channels() == 2);

    _rvec.create(3, 1, CV_64FC1);
    _tvec.create(3, 1, CV_64FC1);
    _rvec2.create(3, 1, CV_64FC1);
    _tvec2.create(3, 1, CV_64FC1);
    Mat rvec = _rvec.getMat();
    Mat tvec = _tvec.getMat();
    Mat rvec2 = _rvec2.getMat();
    Mat tvec2 = _tvec2.getMat();

    if (cv::norm(rvec, NORM_L1) + cv::norm(tvec, NORM_L1) +
        cv::norm(rvec2, NORM_L1) + cv::norm(tvec2, NORM_L1) == 0) {
      cv::Mat gs_inliers;
      solvePnPRansac(opoints, ipoints, cameraMatrix, distCoeffs, rvec, tvec,
                     useExtrinsicGuess, iterationsCount, reprojectionError*2,
                     minInliersCount, gs_inliers, flags);

      if (gs_inliers.rows > 4)
      { // GS Init
        rvec.copyTo(rvec2);
        tvec.copyTo(tvec2);
        std::cout << "GS PnP Init: " << rvec << tvec << endl;
      }
    }

    Mat objectPoints = opoints.reshape(3, 1), imagePoints = ipoints.reshape(2, 1);

    if (minInliersCount <= 0)
        minInliersCount = objectPoints.cols;
    pnpransac::Parameters params;
    params.iterationsCount = iterationsCount;
    params.minInliersCount = minInliersCount;
    params.reprojectionError = reprojectionError;
    params.useExtrinsicGuess = useExtrinsicGuess;
    params.camera.init(cameraMatrix, distCoeffs, shutter, scanlines);
    params.flags = flags;
    params.min_points_count = min_points_count;

    vector<int> localInliers;
    Mat localRvec, localTvec;
    Mat localRvec2, localTvec2;
    rvec.copyTo(localRvec);
    tvec.copyTo(localTvec);
    rvec2.copyTo(localRvec2);
    tvec2.copyTo(localTvec2);

    if (objectPoints.cols >= params.min_points_count)
    {
#if (defined(CV_VERSION_EPOCH) && CV_VERSION_EPOCH == 2)
        cv::parallel_for(BlockedRange(0,iterationsCount),
#else
        cv::parallel_for_(Range(0,iterationsCount),
#endif
            pnpransac::PnPSolver(objectPoints, imagePoints, params,
                localRvec, localTvec, localRvec2, localTvec2, localInliers));
    }

    if (localInliers.size() >= (size_t)params.min_points_count)
    {
        if (flags != CV_P3P)
        {
            int i, pointsCount = (int)localInliers.size();
            Mat inlierObjectPoints(1, pointsCount, CV_32FC3), inlierImagePoints(1, pointsCount, CV_32FC2);
            for (i = 0; i < pointsCount; i++)
            {
                int index = localInliers[i];
                Mat colInlierImagePoints = inlierImagePoints(Rect(i, 0, 1, 1));
                imagePoints.col(index).copyTo(colInlierImagePoints);
                Mat colInlierObjectPoints = inlierObjectPoints(Rect(i, 0, 1, 1));
                objectPoints.col(index).copyTo(colInlierObjectPoints);
            }
            vision::solveRsPnP(inlierObjectPoints, inlierImagePoints,
                params.camera.intrinsics, params.camera.distortion,
                localRvec, localTvec,
                localRvec2, localTvec2,
                shutter, scanlines,
                true, flags);
        }
        localRvec.copyTo(rvec);
        localTvec.copyTo(tvec);
        localRvec2.copyTo(rvec2);
        localTvec2.copyTo(tvec2);
        if (_inliers.needed())
            Mat(localInliers).copyTo(_inliers);
    }
    else
    {
      tvec.setTo(Scalar(0));
      tvec2.setTo(Scalar(0));
      Mat R = Mat::eye(3, 3, CV_64F);
      Rodrigues(R, rvec);
      Rodrigues(R, rvec2);
      if ( _inliers.needed() ) _inliers.release();
    }
    return;
}

