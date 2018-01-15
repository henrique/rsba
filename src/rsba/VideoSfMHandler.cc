// Author: Henrique Mendonça <henrique@apache.org>
#include "rsba/VideoSfMHandler.h"

#include <vector>

#include "rsba/mat/cam.h"
#include "rsba/Sfm2Ply.h"
#include "rsba/struct/VideoSfM.h"
#include "rsba/solveRSpnp.h"

#include <opencv2/calib3d/calib3d.hpp>


using namespace ::cv;
using namespace ::std;
using namespace ::apache::thrift;
using namespace ::vision;
using namespace ::vision::cameras;


#ifndef _WIN32
 #if __cplusplus < 201103L
 #define to_string(...) ""
 #endif
#endif


namespace vision { namespace sfm {



VideoSfMHandler::VideoSfMHandler(const SfmOptions& cOpt)
: _opt(cOpt) { }


VideoSfMHandler::VideoSfMHandler(const SfmOptions& cOpt, const VideoSfMHandler& clone)
: _opt(cOpt), _sessions(clone._sessions) { }



void VideoSfMHandler::authenticate(std::string& _return) {
  _return = "key"; //TODO authenticate
}



int32_t VideoSfMHandler::newSession(const std::string& authToken, const std::vector<double> & camera) {
  Session sess;
  sess.cam = camera;
  _sessions.push_back(sess);

  const size_t sessionKey = _sessions.size() - 1;
  printFrames(sessionKey);
  return sessionKey;
}



int32_t VideoSfMHandler::cloneSession(const std::string& authToken, const int32_t oldSessionKey) {
  _sessions.push_back(_sessions[oldSessionKey]);
  return _sessions.size() - 1;
}



int32_t VideoSfMHandler::newRsSession(const std::string& authToken, const std::vector<double> & camera, const gen::RollingShutter::type rs, const std::vector<int32_t> & scanlines) {
  Session sess;
  sess.cam = camera;
  sess.rs = rs;
  sess.scanlines = scanlines;
  sess.__isset.rs = sess.__isset.scanlines = true;
  _sessions.push_back(sess);

  const size_t sessionKey = _sessions.size() - 1;
  printFrames(sessionKey);
  return sessionKey;
}



// add a frame and return its key
// will generate new tracks if matches are available
int32_t VideoSfMHandler::newFrame(const std::string& authToken, const int32_t sessionKey, const gen::Frame& frame) {
  CHECK_GE(sessionKey, 0);
  CHECK_LT(sessionKey, _sessions.size());

  Session& sess = _sessions[sessionKey];
  const size_t frameKey = sess.frames.size();
  sess.frames.push_back(frame);

  solve(sessionKey, frameKey);
  printFrame(sessionKey, frameKey);

  const string file = "videoSfM" + to_string(frameKey) + ".ply";
  writePly("raw_" + file, sess, _opt, true, true);

  return frameKey;
}



// get a frame
void VideoSfMHandler::getFrame(gen::Frame& _return, const std::string& authToken, const int32_t sessionKey, const int32_t frameKey) {
  CHECK_GE(sessionKey, 0);
  CHECK_LT(sessionKey, _sessions.size());

  CHECK_GE(frameKey, 0);
  CHECK_LT(frameKey, _sessions[sessionKey].frames.size());

  _return = _sessions[sessionKey].frames[frameKey];
}



// add a track and return its key
int32_t VideoSfMHandler::newTrack(const std::string& authToken, const int32_t sessionKey, const gen::Track& track) {
  CHECK_GE(sessionKey, 0);
  CHECK_LT(sessionKey, _sessions.size());

  Session& sess = _sessions[sessionKey];
  size_t key = sess.tracks.size();
  sess.tracks.push_back(track);
  return key;
}



// get all tracks
void VideoSfMHandler::getTracks(std::vector<gen::Track> & _return, const std::string& authToken, const int32_t sessionKey) {
  CHECK_GE(sessionKey, 0);
  CHECK_LT(sessionKey, _sessions.size());

  _return = _sessions[sessionKey].tracks;
}



// initialize frame poses
void VideoSfMHandler::initialize(const std::string& authToken, const int32_t sessionKey) {
  CHECK_GE(sessionKey, 0);
  CHECK_LT(sessionKey, _sessions.size());

  Session& sess = _sessions[sessionKey];

  if (BA(sessionKey, 0, sess.frames.size()-1, _opt, 20, true)) {
    writePly("init_videoSfM" + to_string(sess.frames.size() - 1) + ".ply", sess, _opt, true, true);
  }
}



// run bundle adjustment on all frames
bool VideoSfMHandler::fullBA(const std::string& authToken, const int32_t sessionKey, const int32_t maxIter, const bool reproject) {
  std::cout << __FUNCTION__ << endl;
  CHECK_GE(sessionKey, 0);
  CHECK_LT(sessionKey, _sessions.size());

  printFrames(sessionKey);
  Session& sess = _sessions[sessionKey];

  SfmOptions opt = _opt;

  if (opt.ceres.useOnlyValidMatches) {
    size_t ntracks = 0;
    for (const gen::Track& t : sess.tracks) {
      if (t.valid) ntracks++;
    }
    if (ntracks < 100) {
      opt.ceres.useOnlyValidMatches = false;
      std::cout << "!!! Not enough valid matches: " << ntracks << " !!!" << endl;
    }
  }

  if (BA(sessionKey, 0, sess.frames.size()-1, opt, maxIter, reproject)) {
    writePly("videoSfM" + to_string(sess.frames.size() - 1) + ".ply", sess, opt, true, true);
    return true;
  }

  return false;
}



// run bundle adjustment on requested frames fixing the remaining poses
bool VideoSfMHandler::windowedBA(const std::string& authToken, const int32_t sessionKey, const int32_t startFrame, const int32_t endFrame, const int32_t maxIter, const bool reproject)
{
  std::cout << __FUNCTION__ << endl;
  CHECK_GE(sessionKey, 0);
  CHECK_LT(sessionKey, _sessions.size());

  Session& sess = _sessions[sessionKey];
  CHECK_LT(endFrame, sess.frames.size());

  SfmOptions opt = _opt;
  opt.ceres.fixScale = false;

  if (opt.ceres.useOnlyValidMatches) {
    size_t ntracks = 0;
    for (const gen::Track& t : sess.tracks) {
      if (t.valid) ntracks++;  //TODO check tracks within window?
    }
    if (ntracks < 100) {
      opt.ceres.useOnlyValidMatches = false;
      std::cout << "!!! Not enough valid matches: " << ntracks << " !!!" << endl;
    }
  }

  if (BA(sessionKey, startFrame, endFrame, opt, maxIter, reproject)) {
    writePly("videoSfM" + to_string(sess.frames.size() - 1) + ".ply", sess, opt, true, true);
    return true;
  }

  return false;
}



// finalize session
void VideoSfMHandler::finalize(const std::string& authToken, const int32_t sessionKey)
{
  CHECK_GE(sessionKey, 0);
  CHECK_LT(sessionKey, _sessions.size());

  _sessions[sessionKey] = Session(); // delete session
}



/// look for a valid track
//TODO use "reproject" ?
bool VideoSfMHandler::reprojectMatches(Session& sess, const size_t frameKey, const size_t obsKey)
{
  double pose[NUM_POSE_PARAMS];
  gen::Frame& f = sess.frames[frameKey];
  gen::Observation& o = f.obs[obsKey];
  const double* cam = f.__isset.cam ? f.cam.data() : sess.cam.data();

  if (o.__isset.matches)
  {
    double obs[2] = { o.x, o.y };
    for (gen::ObservationRef& ref : o.matches)
    {
      gen::Frame& f2 = sess.frames[ref.frame];
      gen::Observation& o2 = f2.obs[ref.obs];

      if (o2.__isset.track)
      {
        gen::Track& t = sess.getTrack(o2.track);
        if (_opt.tracks.maxReprojections > 0 && t.obs.size() >= _opt.tracks.maxReprojections) {
          continue;
        }

        double dist[3];
        minus3(pose+3, t.pt.data(), dist);
        if (norm3(dist) < _opt.tracks.minDistanceToCamera) {
          continue;
        }

        if (!inTrack(t, frameKey)) { // only one observation per frame within track
          getPose(sess, f, _opt, obs, pose);

          if (vision::validate(cam, pose, obs, t.pt.data(), _opt.tracks.sqrdThreshold)) {
            // add valid observation to track
            o.track = o2.track;
            o.__isset.track = true;
            ObservationRef r2(frameKey, obsKey);
            r2.valid = true;
            t.obs.push_back(r2);
            if (!_opt.ceres.const3d && t.obs.size() >= _opt.tracks.minReprojections) t.valid = true;

            return true;
          }
        }
      }
    }
  }

  return false;
}



void VideoSfMHandler::createTracks(Session& sess, const size_t frameKey)
{
  if (_opt.tracks.synthetic) {
    evalTracks(sess, frameKey);
    return;
  }

  gen::Frame& f = sess.frames[frameKey];
  const double* cam = f.__isset.cam ? f.cam.data() : sess.cam.data();

  /// Create tracks and initialize 3D points
  for (size_t obsKey = 0; obsKey < f.obs.size(); obsKey++)
  {
    gen::Observation& o = f.obs[obsKey];

    if (o.__isset.matches)
    {
      // first look for a valid track on visual matches
      reprojectMatches(sess, frameKey, obsKey);

      // create a new track if no valid track was found
      if ( !_opt.ceres.const3d && !o.__isset.track )
      {
        for (gen::ObservationRef& ref : o.matches)
        {
          CHECK(ref.frame != (int)frameKey);
          gen::Frame& f2 = sess.frames[ref.frame];
          CHECK(f2.__isset.poses);
          gen::Observation& o2 = f2.obs[ref.obs];
          const double* cam2 = f2.__isset.cam ? f2.cam.data() : sess.cam.data();

          if (!o2.__isset.track)
          { // also not in a track yet
            //triangulate, reproject and validate
            double obs[2] =
            { o.x, o.y };
            double obs2[2] =
            { o2.x, o2.y };
            double pose[NUM_POSE_PARAMS], pose2[NUM_POSE_PARAMS];

            getPose(sess, f, _opt, obs, pose);
            getPose(sess, f2, _opt, obs2, pose2);

            // check if the cameras are in different positions, obsKey.e. triangulation is possible
            if (memcmp(pose, pose2, sizeof(pose)) != 0)
            {
              double pt[3];
              if (triangulate(cam, pose, obs, cam2, pose2, obs2, pt))
              {
                double dist[3];
                minus3(pose+3, pt, dist);
                if (norm3(dist) < _opt.tracks.minDistanceToCamera) {
                  continue;
                }
                minus3(pose2+3, pt, dist);
                if (norm3(dist) < _opt.tracks.minDistanceToCamera) {
                  continue;
                }

                if (vision::validate(cam, pose, obs, pt, _opt.tracks.sqrdThreshold))
                {
                  if (vision::validate(cam2, pose2, obs2, pt, _opt.tracks.sqrdThreshold))
                  {
                    gen::Track& t = sess.newTrack(Track(pt, o.color), o.track);
                    o2.track = o.track; //new index

                    // store keypoint references in track
                    ref.valid = true;
                    t.obs.push_back(ref);
                    ObservationRef r2(frameKey, obsKey);
                    r2.valid = true;
                    t.obs.push_back(r2);
                    t.valid = (t.obs.size() >= _opt.tracks.minReprojections);

                    o.__isset.track = o2.__isset.track = true;

                    // look for a valid reprojections on other matches (if given)
                    reprojectMatches(sess, frameKey, obsKey);
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  //evalTracks(sess, frameKey);
}



/// validate all tracks
void VideoSfMHandler::evalTracks(Session& sess, const size_t frameKey)
{
  uint countObs = 0, countTrack = 0;
  gen::Frame& f = sess.frames[frameKey];

  for (size_t oi = 0; oi < f.obs.size(); oi++) {
    gen::Observation& o = f.obs[oi];

    if (o.__isset.track) {
      double obs[2] = { o.x, o.y };
      gen::Track& t = sess.getTrack(o.track);

      if (validate(sess, f, _opt, t.pt.data(), obs)) {
        o.__isset.track = false;
        countObs++;

        for (size_t i = 0; i < t.obs.size(); i++) {
          if (t.obs[i].frame == (int)frameKey && t.obs[i].obs == (int)oi) {
            t.obs.erase(t.obs.begin()+i);
            if (t.valid && t.obs.size() < _opt.tracks.minReprojections) {
              t.valid = false;
              countTrack++;
            }
            break;
          }
        }
      }
    }
  }

  if (countTrack || countObs) {
    cout << countObs << " bad reprojections and " << countTrack << " bad tracks removed" << endl;
  }
}



// initialize frame and search for good tracks
void VideoSfMHandler::solve(const int32_t sessionKey, const int32_t frameKey)
{
  Session& sess = _sessions[sessionKey];
  gen::Frame& f = sess.frames[frameKey];

  if (f.obs.size() > 0)
  {
    //TODO remove PnP from solve()
    if (!f.__isset.poses) {
      solveRsPnP(sessionKey, _opt);
    }

    if ((!f.__isset.poses || _opt.ceres.pnpNewFrame) && (uint)frameKey >= _opt.tracks.minReprojections) {
      ceres::Solver::Options cOpt;
      cOpt.linear_solver_type = ceres::SPARSE_SCHUR;
      cOpt.minimizer_progress_to_stdout = true;
      cOpt.max_num_iterations = _opt.ceres.baIterationsOnNewFrame;

      // alternative options for PnP
      SfmOptions opt = _opt;
      opt.ceres.const3d = true;

      if (opt.ceres.useOnlyValidMatches) {
        size_t ntracks = 0;
        for (gen::Observation& o: f.obs) {
          for (gen::ObservationRef& ref: o.matches) {
            gen::Frame& f2 = sess.frames[ref.frame];
            gen::Observation& o2 = f2.obs[ref.obs];
            if (o2.__isset.track) {
              const gen::Track& t = sess.getTrack(o2.track);
              if (t.valid) {
                ntracks++;
                break;
              }
            }
          }
        }
        std::cout << ntracks << " valid tracks found in " << f.obs.size() << " matches" << endl;
        if (ntracks < 100) {
          opt.ceres.useOnlyValidMatches = false;
          std::clog << "!!! Not enough valid matches !!!" << endl;
        }
      }

//      { // first draft without a loss function
//        opt.ceres.huberLoss = 0;
//        CeresHandler solver2(opt);
//        solver2.Add(frameKey, sess);
//        ceres::Solver::Summary summary2 = solver2.solve(&cOpt);
//        std::cout << summary2.FullReport() << "\n:\n";
//
//        printFrame(sessionKey, frameKey);
//        writePly("pnp_videoSfM" + to_string(frameKey) + ".ply", sess, opt, true, true);
//      }

      { // improve solution with default loss function
        opt.ceres.huberLoss = _opt.ceres.huberLoss;
        opt.ceres.fixScale = false;
        CeresHandler solver2(opt);
        solver2.Add(frameKey, sess, true);
        //TODO refactoring: remove linear estimation from Add() and use BA() here
        writePly("est_videoSfM" + to_string(frameKey) + ".ply", sess, opt, true, true);
        printFrame(sessionKey, frameKey);

        std::cout << "Direct PnP" << endl;
        ceres::Solver::Summary summary2 = solver2.solve(&cOpt);
        std::cout << summary2.FullReport() << "\n:\n";

        printFrame(sessionKey, frameKey);
        writePly("pnp2_videoSfM" + to_string(frameKey) + ".ply", sess, opt, true, true);
      }
    }

    if (f.__isset.poses) {
      size_t tracks = sess.tracks.size();
      createTracks(sess, frameKey);
      std::cout << sess.tracks.size() - tracks << " new tracks created" << endl;
    }

//    if (_opt.model.use3Dpoints) {
//      CeresHandler solver2(_opt);
//      ceres::Solver::Summary summary = solver2.solve();
//      std::cout << summary.BriefReport() << "\n:\n";
//
//      printFrames(sessionKey);
//#ifndef NDEBUG
//      writePly(file, sess, _opt, true, true);
//#endif
//      std::cout << "\n\n";
//    }
  }
}



// print a frame
void VideoSfMHandler::printFrame(const int32_t sessionKey, const int32_t frameKey)
{
  const Session& sess = _sessions[sessionKey];
  const gen::Frame& f = sess.frames[frameKey];

  size_t ntracks = 0;
  for (const gen::Observation& o : f.obs) {
    if (o.__isset.track && sess.getTrack(o.track).valid) ntracks++;
  }
  cout << "frame " << frameKey << ":  " << ntracks << " valid tracks";

  if (f.poses.size() >= 2) {
    double diff[NUM_CAM_PARAMS];
    minus6(f.poses.front().data(), f.poses.back().data(), diff);
    cout << ";  angular disp: " << norm3(diff) << "rad"
         << ",  3d disp: " << norm3(diff+3) << "m"
         << ",  velocity: " << norm3(diff+3)*1000*60*60/(72*1000) << "km/h assuming 72ms exposure.";
  }
  cout << endl;

  if (f.__isset.cam) {
    cout << "camera parameters:  ";
    for (const double& c : f.cam) printf("%.5g  ", c);
    cout << endl;
  }

  if (f.poses.size() > 4) {
    { const std::vector<double>& p = f.poses.front();
      printf("   %13.8g %13.8g %13.8g : %13.8g %13.8g %13.8g \n",
                     p[0], p[1], p[2],  p[3], p[4], p[5]);
    }
    cout << "   ...  " << f.poses.size() << " poses" << endl;
    { const std::vector<double>& p = f.poses.back();
      printf("   %13.8g %13.8g %13.8g : %13.8g %13.8g %13.8g \n",
                     p[0], p[1], p[2],  p[3], p[4], p[5]);
    }
  } else {
    for (const std::vector<double>& p : f.poses) {
      printf("   %13.8g %13.8g %13.8g : %13.8g %13.8g %13.8g \n",
                     p[0], p[1], p[2],  p[3], p[4], p[5]);
    }
  }
}



// print all frames
void VideoSfMHandler::printFrames(const int32_t sessionKey)
{
  const Session& sess = _sessions[sessionKey];

  cout << "camera parameters:  ";
  for (const double& c : sess.cam) printf("%.5g  ", c);
  cout << endl;

  for (unsigned fi = 0; fi < sess.frames.size(); fi++) {
    printFrame(sessionKey, fi);
  }
}



// run bundle adjustment on requested frames fixing the remaining poses
bool VideoSfMHandler::BA(const int32_t sessionKey, const int32_t startFrame, const int32_t endFrame, const SfmOptions& opt, const int32_t maxIter, const bool reproject)
{
  Session& sess = _sessions[sessionKey];

  // Perform final adjustment
  ceres::Solver::Options cOpt;
  cOpt.linear_solver_type = ceres::SPARSE_SCHUR;
  cOpt.minimizer_progress_to_stdout = true;
  cOpt.max_num_iterations = maxIter;
  cOpt.min_linear_solver_iterations = 3;
  //cOpt.use_nonmonotonic_steps = true;

  CeresHandler cs(opt, startFrame);
  for (int32_t fi = startFrame; fi <= endFrame; fi++) {
    printFrame(sessionKey, fi);
    cs.Add(fi, sess);
  }

  ceres::Solver::Summary summary = cs.solve(&cOpt);
  std::cout << summary.FullReport() << endl;
  if ( ! summary.IsSolutionUsable()) {
    std::cerr << summary.message << endl;
  }


  for (int32_t fi = startFrame; fi <= endFrame; fi++) {
    if (reproject) createTracks(sess, fi);

    if (opt.debug.calcCovariances) {
      const gen::Frame& f = sess.frames[fi];

      ceres::Covariance::Options options;
      ceres::Covariance covariance(options);
      vector< pair<const double*, const double*> > covariance_blocks;
      covariance_blocks.push_back(make_pair(f.poses[0].data(), f.poses[0].data()));
      covariance_blocks.push_back(make_pair(f.poses[0].data(), f.poses[1].data()));
      covariance_blocks.push_back(make_pair(f.poses[1].data(), f.poses[1].data()));

      if (covariance.Compute(covariance_blocks, &cs.problem)) {
        Eigen::Matrix<double, NUM_POSE_PARAMS, NUM_POSE_PARAMS> pp, pe, ee;
        covariance.GetCovarianceBlock(f.poses[0].data(), f.poses[0].data(), pp.data());
        cout << "pp:" << endl << pp << endl;
        covariance.GetCovarianceBlock(f.poses[0].data(), f.poses[1].data(), pe.data());
        cout << "pe:" << endl << pe << endl;
        covariance.GetCovarianceBlock(f.poses[1].data(), f.poses[1].data(), ee.data());
        cout << "ee:" << endl << ee << endl;
      }
    }
  }

  printFrames(sessionKey);

  // standard error: sqrt(sum(e^2)/N), with the final_cost being sum(e^2)/2 and num_residual_blocks N/2
  cout << "average reprojection error: "
       << sqrt( summary.final_cost / summary.num_residual_blocks_reduced ) << endl;

  return summary.IsSolutionUsable();
}



void VideoSfMHandler::cvCorrespondences(const gen::Frame& f, const Session& sess, std::vector<Point3f>& pts, std::vector<Point2f>& obs, bool useOnlyValid)
{
  for (const gen::Observation& o : f.obs) {
    if (o.__isset.track) {
      const gen::Track& t = sess.getTrack(o.track);

      if ( (t.valid || !useOnlyValid) && t.__isset.pt) {
        pts.push_back(Point3f(t.pt[0], t.pt[1], t.pt[2]));
        obs.push_back(Point2f(o.x, o.y));
        continue; // next observation
      }
    }

    for (const gen::ObservationRef& ref : o.matches) {
      const gen::Frame& f2 = sess.frames[ref.frame];
      const gen::Observation& o2 = f2.obs[ref.obs];

      if (o2.__isset.track) {
        const gen::Track& t = sess.getTrack(o2.track);

        if ( (t.valid || !useOnlyValid) && t.__isset.pt) {
          pts.push_back(Point3f(t.pt[0], t.pt[1], t.pt[2]));
          obs.push_back(Point2f(o.x, o.y));
          break; // next observation
        }
      }
    }
  }
}



// run bundle adjustment on requested frames fixing the remaining poses
bool VideoSfMHandler::solveRsPnP(const int32_t sessionKey, const SfmOptions& opt, const int32_t maxIter)
{
  Session& sess = _sessions[sessionKey];
  size_t frameKey = sess.frames.size() - 1; // get last frame
  gen::Frame& f = sess.frames[frameKey];

  const double* cam = f.__isset.cam ? f.cam.data() : sess.cam.data();
  cv::Mat K(cvK(cam));
  cv::Mat distcoeff(cvDist(cam));

  std::vector<Point3f> pts;
  std::vector<Point2f> obs;
  cvCorrespondences(f, sess, pts, obs);
  if (pts.size() <= 4) {
    std::cerr << "Using all tracks! Not enough valid tracks!" << endl;
    cvCorrespondences(f, sess, pts, obs, false);
  }

  if (pts.size() > 4) {
    std::cout << "solve PnP RANSAC with " << pts.size() << " tracks" << endl;
    cv::Mat inliers;

    cv::Vec3d rvec(_EPS, _EPS, _EPS), // setting EPS avoids automatic GS PnP
              tvec(_EPS, _EPS, _EPS);
    std::cout << rvec << tvec << endl;

    if (f.__isset.priorPoses) {
      f.poses = f.priorPoses;
    }
    else if (_opt.mod_init.reuseLastPose && frameKey > 0) {
      vector<double> pose;
      if (f.poses.size() > 1) {
        pose = f.poses[0];
      } else {
        pose = sess.frames[frameKey-1].poses.back(); // initialize with last pose
      }
      rvec = cv::Vec3d(pose.data());
      tvec = cv::Vec3d(pose.data()+3);

      // invert translation
      ceres::AngleAxisRotatePoint(pose.data(), tvec.val, tvec.val);
      tvec *= -1;
      std::cout << rvec << tvec << endl;
    }

    if (_opt.mod_init.solveGsPnP) {
      std::cout << "solveGsPnP" << endl;
      solvePnPRansac(pts, obs, K, distcoeff,
          rvec, tvec, _opt.mod_init.reuseLastPose, 500,
          sqrt(_opt.tracks.sqrdThreshold)*2,
          pts.size()*.7, inliers, SOLVEPNP_ITERATIVE);

      if (inliers.rows <= 4) {
        solvePnPRansac(pts, obs, K, distcoeff,
            rvec, tvec, _opt.mod_init.reuseLastPose, 500,
            sqrt(_opt.tracks.sqrdThreshold)*2,
            pts.size()*.7, inliers, CV_EPNP);
      }
      if (inliers.rows <= 4) {
        solvePnPRansac(pts, obs, K, distcoeff,
            rvec, tvec, false, 500,
            sqrt(_opt.tracks.sqrdThreshold)*2,
            pts.size()*.7, inliers, CV_P3P);
      }
      std::cout << rvec << tvec << inliers.rows << endl;
    }

    cv::Vec3d rvec2(rvec.val), tvec2(tvec.val);

    if (_opt.model.rolling_shutter &&  _opt.mod_init.solveRsPnP) {
      std::cout << "solveRsPnP" << endl;
      solveRsPnPRansac(pts, obs, K, distcoeff,
          rvec, tvec, rvec2, tvec2,
          (SHUTTER)sess.rs, sess.scanlines.data(), true, 1000,
          sqrt(_opt.tracks.sqrdThreshold),
          pts.size()*.7, inliers, SOLVEPNP_ITERATIVE, _opt.mod_init.minPnPfeatures);
      std::cout << rvec << tvec << endl;
      std::cout << rvec2 << tvec2 << endl;
    }

    if (inliers.rows > 4 || ( !_opt.mod_init.solveGsPnP && !_opt.mod_init.solveRsPnP ) ) {
      std::cout << inliers.rows << " PnP inliers found" << endl;

      vector<double> pose(NUM_POSE_PARAMS);
      assign3(rvec.val, pose.data());
      double rInv[3];
      invert3(pose.data(), rInv);
      ceres::AngleAxisRotatePoint(rInv, (-tvec).val, pose.data()+3);

      f.poses.resize(_opt.model.rolling_shutter ? 2 : 1);
      for (vector<double>& p : f.poses) p = pose;
      f.__isset.poses = true;

      if (f.poses.size() == 2)
      {
        assign3(rvec2.val, f.poses[1].data());
        double rInv[3];
        invert3(f.poses[1].data(), rInv);
        ceres::AngleAxisRotatePoint(rInv, (-tvec2).val, f.poses[1].data()+3);
      }

      if (_opt.mod_init.refinePnP) { // refine estimate
        std::cout << "refinePnP" << endl;
        size_t tracks = sess.tracks.size();
        createTracks(sess, frameKey);
        std::cout << sess.tracks.size() - tracks << " new tracks created" << endl;

        SfmOptions opt(_opt);
        opt.ceres.const3d = true;
        opt.model.calibrated = true;
        opt.ceres.fixScale = false;
        BA(sessionKey, frameKey, frameKey, opt);

        cv::Vec3d rvec(f.poses[0].data()), tvec(f.poses[0].data()+3);
        ceres::AngleAxisRotatePoint(f.poses[0].data(), tvec.val, tvec.val);
        tvec *= -1;
        std::cout << rvec << tvec << endl;

        if (f.poses.size() == 2) {
          cv::Vec3d rvec2(f.poses[1].data()), tvec2(f.poses[1].data()+3);
          ceres::AngleAxisRotatePoint(f.poses[1].data(), tvec2.val, tvec2.val);
          tvec2 *= -1;
          std::cout << rvec2 << tvec2 << endl;
        }
      }

      return true;
    } else {
      std::clog << "!!! PnP failure on frame " << frameKey << endl;
    }
  } else {
    std::clog << "!!! Not enough valid matches for PnP on frame " << frameKey << endl;
  }


  return false;
}



}} // name space

