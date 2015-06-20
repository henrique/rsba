#ifndef __CeresHandler_h__
#define __CeresHandler_h__

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Core>
#include <cstddef>

#include "rsba/mat.h"
#include "rsba/struct/VideoSfM.h"
#include "rsba/video_bundler_free.h"
#include "rsba/video_bundler_life.h"
#include "rsba/video_bundler_structless.h"
#include "rsba/video_bundler_rs_inter.h"
#include "rsba/VideoSfmBaRs.h"


using namespace ::std;


namespace vision { namespace sfm {


struct SphericalPlus {
  template <typename T>
  //bool operator()(const T x[NUM_POSE_PARAMS], const T delta[NUM_POSE_PARAMS], T x_plus_delta[NUM_POSE_PARAMS]) const {
  bool operator()(const T* x, const T* delta, T* x_plus_delta) const {
    plus3(x, delta, x_plus_delta);
    plus3(x+3, delta+3, x_plus_delta+3);
    scalar3(x_plus_delta+3, T(1)/norm3(x_plus_delta+3), x_plus_delta+3);
    return true;
  }
};


struct SphericalPrior {
  static const unsigned short NUM_RESIDUALS = 2;
  SphericalPrior(){};

  template <typename T>
  bool operator()(const T* const pose, T* residuals) const {
    residuals[0] = ( pose[0] * pose[0] + pose[1] * pose[1] + pose[2] * pose[2] );
    residuals[1] = T(1e20) * ( T(1) - abs(pose[3]) - abs(pose[4]) - abs(pose[5]) );
    return residuals[0] < T(1); // rotation limit
  }

  // Factory to hide the construction of the CostFunction object from the client code.
  static ceres::CostFunction* Create() {
    return (new ceres::AutoDiffCostFunction<SphericalPrior, NUM_RESIDUALS, NUM_POSE_PARAMS>(new SphericalPrior()));
  }
};


/// Assume a good initial guess
struct GoodPosePrior {
  static const unsigned short NUM_RESIDUALS = 6;
  double rotation, position;
  GoodPosePrior(double rotation, double position) : rotation(rotation), position(position) {};

  template <typename T>
  bool operator()(const T* const pose0, const T* const pose, T* residuals) const {
    minus6(pose0, pose, residuals);
    scalar3(residuals, T(rotation), residuals);
    scalar3(residuals+3, T(position), residuals+3);
    return residuals[0] < T(1); // rotation limit
  }

  // Factory to hide the construction of the CostFunction object from the client code.
  static ceres::CostFunction* Create(double rotation, double position) {
    return (new ceres::AutoDiffCostFunction<GoodPosePrior, NUM_RESIDUALS, NUM_POSE_PARAMS, NUM_POSE_PARAMS>
            (new GoodPosePrior(rotation, position)));
  }
};


class CeresHandler {
 public:
  ceres::Problem problem;
  SfmOptions opt;
  ceres::LossFunction* lossFunction = nullptr;
  size_t startFrame;



  CeresHandler(const SfmOptions& opt, size_t startFrame = 0) : problem(), opt(opt), startFrame(startFrame)
  {
    if (opt.ceres.huberLoss > 0) {
      lossFunction = new ceres::HuberLoss(opt.ceres.huberLoss);
    }
  };



  void Add(const size_t frameKey, sfm::Session& sess, bool uninitialized = false) {
    gen::Frame& f = sess.frames[frameKey];

    int formerParamNum = problem.NumParameterBlocks();

    if (!f.__isset.poses) {
      /// initialize camera frame
      if (frameKey > 0) {
        gen::Frame& f_1 = sess.frames[frameKey - 1];

        // use last pose as reference
        f.poses = f_1.poses;

        if (frameKey > 1) { // extrapolate linear velocity
          gen::Frame& f_2 = sess.frames[frameKey - 2];

          for (uint pi = 0; pi < f.poses.size(); pi++) {
            minus6(f_1.poses[pi].data(), f_2.poses[pi].data(), f.poses[pi].data()); // delta
            plus6(f_1.poses[pi].data(), f.poses[pi].data(), f.poses[pi].data()); // extrapolation
          }
        } else {
          bool originFrame = true;
          for (auto& pose: f_1.poses)
            if (originFrame)
              for (auto& p: pose)
                if (p != 0) originFrame = false;

          for (auto& pose: f.poses) {
            pose[3] += 1e-4;
            pose[4] += 1e-4;
            pose[5] += 1e-4;
          }

          if (frameKey == 1 and originFrame) {
            ceres::CostFunction* costFunction = SphericalPrior::Create();
            problem.AddResidualBlock(costFunction, nullptr, f.poses[0].data());
          }
        }
      } else { /// frameKey == 0, init with zeros
        if (opt.model.rolling_shutter) {
          f.poses.resize(2);
          f.poses[0].resize(NUM_POSE_PARAMS, 0);
          f.poses[1].resize(NUM_POSE_PARAMS, 0);
        } else { // global shutter
          f.poses.resize(1);
          f.poses[0].resize(NUM_POSE_PARAMS, 0);
        }
      }
      uninitialized = true;
      f.__isset.poses = true;
    }

    // add priors
    if (frameKey >= opt.ceres.fixFirstNCameras) {
      if (frameKey > 0 and (opt.ceres.constFrameVelocity != 0 or opt.ceres.constFrameAcceleration != 0)) {
        gen::Frame& f_1 = sess.frames[frameKey - 1];

        if (f.poses.size() == 2 and f_1.poses.size() == 2) {
          if (opt.ceres.constFrameAcceleration != 0) {
            ceres::CostFunction* costFunction = RsConstAccelerationPrior::Create(opt.ceres.constFrameAcceleration);
            problem.AddResidualBlock(costFunction,
                                     lossFunction,
                                     &opt.ceres.interFrameRatio,
                                     f.poses[0].data(),
                                     f.poses[1].data(),
                                     f_1.poses[0].data(),
                                     f_1.poses[1].data());
            problem.SetParameterLowerBound(&opt.ceres.interFrameRatio, 0, _EPS);
          }
          else if (opt.ceres.constFrameVelocity != 0) {
            ceres::CostFunction* costFunction = RsConstVeloPrior::Create(opt.ceres.constFrameVelocity);
            problem.AddResidualBlock(costFunction,
                                     lossFunction,
                                     &opt.ceres.interFrameRatio,
                                     f.poses[0].data(),
                                     f.poses[1].data(),
                                     f_1.poses[0].data(),
                                     f_1.poses[1].data());
            problem.SetParameterLowerBound(&opt.ceres.interFrameRatio, 0, 0.0);
          }

          if (opt.ceres.interFrameRatio != 1) { //TODO add flag to trust interFrameRatio
            problem.SetParameterBlockConstant(&opt.ceres.interFrameRatio);
          }

          if (frameKey-1 < opt.ceres.fixFirstNCameras) { // fix last cameras; current frame is already fixed below
            for (auto& pose: f_1.poses) {
              problem.SetParameterBlockConstant(pose.data());
            }
            if (f_1.__isset.cam) problem.SetParameterBlockConstant(f_1.cam.data());
          }
        }
      }

      if (opt.ceres.trustPriorCamRotation or opt.ceres.trustPriorCamPosition) {

        if (f.__isset.priorPoses and f.priorPoses.size() > 0) {

          if ( !f.__isset.poses or f.poses.size() != f.priorPoses.size()) {
            f.poses = f.priorPoses;
          }

          for (size_t i = 0; i < f.poses.size(); i++) {
            ceres::CostFunction* costFunction = GoodPosePrior::Create(opt.ceres.trustPriorCamRotation, opt.ceres.trustPriorCamPosition);
            problem.AddResidualBlock(costFunction,
                                     nullptr,
                                     f.priorPoses[i].data(),
                                     f.poses[i].data());
          }
        }
      }
    }

    // AddResidualBlock
    for (gen::Observation& o: f.obs) {
      double obs[2] = { o.x, o.y };

      if (opt.model.use3Dpoints)
      {
        gen::Track* t = nullptr;

        if (o.__isset.track) {
          t = &sess.getTrack(o.track);
          if (!t->__isset.pt or (opt.ceres.useOnlyValidMatches and !t->valid)) t = nullptr;
        }

        if (!t and (uninitialized or !opt.ceres.useOnlyValidMatches)) { // also add bad reprojections
          for (gen::ObservationRef& ref: o.matches) {
            const gen::Frame& f2 = sess.frames[ref.frame];
            const gen::Observation& o2 = f2.obs[ref.obs];
            if (o2.__isset.track)
            {
              t = &sess.getTrack(o2.track);

              if (!t->__isset.pt
                  or (opt.ceres.useOnlyValidMatches and !t->valid)
                  or !validate(sess, f, opt, t->pt.data(), obs)) {
                t = nullptr;
              }
              else break; // found!
            }
          }
        }

        if (t and (t->valid or !opt.ceres.useOnlyValidMatches)) {
          if (opt.ceres.revalidateReprojections) {
            if ( ! validate(sess, f, opt, t->pt.data(), obs)) {
              continue; // skip observation
            }
          }

          if (f.poses.size() == 2) {
            if (opt.model.constVelocity) {
              abort(); //TODO constVelocity
            } else {
              if (opt.model.calibrated) {
                ceres::CostFunction* costFunction = RsBundleAdjustment::Create(sess, opt, obs);
                problem.AddResidualBlock(costFunction,
                                         lossFunction,
                                         f.poses[0].data(),
                                         f.poses[1].data(),
                                         t->pt.data());
              } else {
                ceres::CostFunction* costFunction = RsBundleAdjustment::CreateWithCam(sess, opt, obs);
                problem.AddResidualBlock(costFunction,
                                         lossFunction,
                                         f.__isset.cam ? f.cam.data() : sess.cam.data(), // focal length + distortion coefficients
                                         f.poses[0].data(),
                                         f.poses[1].data(),
                                         t->pt.data());
              }
            }
          } else {
            if (opt.model.calibrated) {
              ceres::CostFunction* costFunction = ReprojectionError::Create(f.__isset.cam ? f.cam.data() : sess.cam.data(), obs);
              problem.AddResidualBlock(costFunction,
                                       lossFunction,
                                       getPose(sess, f, opt, obs),
                                       t->pt.data());
            } else {
              ceres::CostFunction* costFunction = ReprojectionError::Create(obs);
              problem.AddResidualBlock(costFunction,
                                       lossFunction,
                                       f.__isset.cam ? f.cam.data() : sess.cam.data(), // focal length + distortion coefficients
                                       getPose(sess, f, opt, obs),
                                       t->pt.data());
            }

            if (frameKey < opt.ceres.fixFirstNCameras) { // fix first cameras
              problem.SetParameterBlockConstant(getPose(sess, f, opt, obs));
              if ( !opt.model.calibrated and f.__isset.cam) problem.SetParameterBlockConstant(f.cam.data());
            }
          }

          bool fixedOldTrack = false;
          if (startFrame > 0) {
            for (gen::ObservationRef& ref: t->obs) {
              if ((size_t)ref.frame < startFrame) {
                fixedOldTrack = true;
                break;
              }
            }
          }

          if (fixedOldTrack or opt.ceres.const3d) {
            problem.SetParameterBlockConstant(t->pt.data());
          }
        }
      }
      else // use feature rays
      {
        for (gen::ObservationRef& ref: o.matches) {
          CHECK((size_t)ref.frame != frameKey);

          gen::Frame& f2 = sess.frames[ref.frame];
          gen::Observation& o2 = f2.obs[ref.obs];
          double obs2[2] = { o2.x, o2.y };

          if (opt.model.calibrated) {
            ceres::CostFunction* costFunction = LifeTriangulation::Create(f.__isset.cam ? f.cam.data() : sess.cam.data(), obs, obs2);
            problem.AddResidualBlock(costFunction,
                                     lossFunction,
                                     f.poses[0].data(),
                                     f2.poses[0].data());
          } else {
            ceres::CostFunction* costFunction = LifeTriangulation::Create(obs, obs2);
            problem.AddResidualBlock(costFunction,
                                     lossFunction,
                                     f.__isset.cam ? f.cam.data() : sess.cam.data(), // focal length + Undistortion coefficients
                                     f.poses[0].data(),
                                     f2.poses[0].data());
          }

          if ((size_t)ref.frame < opt.ceres.fixFirstNCameras) { // fix first cameras
            problem.SetParameterBlockConstant(f2.poses[0].data());
            //if ( !opt.model.calibrated and f2.__isset.cam) problem.SetParameterBlockConstant(f2.cam.data());
          }
        }
      }
    }

    if (problem.NumParameterBlocks() > formerParamNum) {
//      if (frameKey == 1 and !opt.model.use3Dpoints) {
//        ceres::LocalParameterization* parameterization = new ceres::AutoDiffLocalParameterization<SphericalPlus, NUM_POSE_PARAMS, NUM_POSE_PARAMS>();
//        for (auto& pose: f.poses)
//          problem.SetParameterization(pose.data(), parameterization);
//      }

      if (frameKey < opt.ceres.fixFirstNCameras) { // fix first cameras
        if (f.poses.size() == 2) {
          problem.SetParameterBlockConstant(f.poses[0].data());
          problem.SetParameterBlockConstant(f.poses[1].data());
        } // else already constant
        if ( !opt.model.calibrated and f.__isset.cam) problem.SetParameterBlockConstant(f.cam.data());
      }
      // fix first and last camera position to constrain scale changes
      else if (opt.ceres.fixScale and (frameKey == 0 or frameKey == sess.frames.size()-1) ) {
        vector<int> constant(3);
        constant[0] = 3;
        constant[1] = 4;
        constant[2] = 5;
        ceres::LocalParameterization* parameterization = new ceres::SubsetParameterization(NUM_POSE_PARAMS, constant);

        if (frameKey == 0)
          problem.SetParameterization(f.poses[0].data(), parameterization); // fix first position
        else problem.SetParameterization(f.poses.back().data(), parameterization); // fix last position
      }
      else if (opt.ceres.fixRotation) {
        // fix rotation parameters
        vector<int> constant(3);
        constant[0] = 0;
        constant[1] = 1;
        constant[2] = 2;
        ceres::LocalParameterization* parameterization = new ceres::SubsetParameterization(NUM_POSE_PARAMS, constant);

        for (auto& pose: f.poses)
          problem.SetParameterization(pose.data(), parameterization);
      }
      else if (opt.ceres.fixPosition) {
        // fix rotation parameters
        vector<int> constant(3);
        constant[0] = 3;
        constant[1] = 4;
        constant[2] = 5;
        ceres::LocalParameterization* parameterization = new ceres::SubsetParameterization(NUM_POSE_PARAMS, constant);

        for (auto& pose: f.poses)
          problem.SetParameterization(pose.data(), parameterization);
      }

#ifndef NDEBUG
      double cost = 0.0;
      problem.Evaluate(ceres::Problem::EvaluateOptions(), &cost, NULL, NULL, NULL);
      cout << "Partial evaluation: " << cost << " for " << problem.NumResidualBlocks() << " residual blocks" << endl;
#endif
    }
  };



  ceres::Solver::Summary solve(ceres::Solver::Options* options = nullptr) {

    ceres::Solver::Options tmp;

    // Make Ceres automatically detect the bundle structure. Note that the
    // standard solver, SPARSE_NORMAL_CHOLESKY, also works fine but it is slower
    // for standard bundle adjustment problems.
    if (options == nullptr) {
      options = &tmp;
      options->linear_solver_type = ceres::SPARSE_SCHUR;
      options->minimizer_progress_to_stdout = true;
      options->max_num_iterations = 50;
    }

#ifdef NDEBUG
#if __cplusplus >= 201103L
    unsigned nThreads = std::thread::hardware_concurrency();
    if (nThreads > 0) {
      options->num_linear_solver_threads = options->num_threads = nThreads;
    }
#endif
#endif


    ceres::Solver::Summary summary;
    ceres::Solve(*options, &problem, &summary);

    if (opt.ceres.constFrameVelocity != 0 or opt.ceres.constFrameAcceleration != 0) {
      cout << "interFrameRatio: " << opt.ceres.interFrameRatio << endl;
    }

    return summary;
  }
};

}} //vision::sfm

#endif /* header */
