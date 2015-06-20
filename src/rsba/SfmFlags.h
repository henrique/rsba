#ifndef _SfmFlags_h
#define _SfmFlags_h


#include "rsba/SfmOptions.h"
#include <google/gflags.h>


DEFINE_uint64(sift_nfeatures, 0, "SIFT max number of features per frame");
DEFINE_uint64(sift_nOctaveLayers, 3, "SIFT number of octave layers");
DEFINE_double(sift_contrastThreshold, 0.04, "SIFT contrast threshold");
DEFINE_double(sift_edgeThreshold, 6.5, "SIFT edge threshold");
DEFINE_double(sift_sigma, 1.6, "SIFT sigma");

DEFINE_bool(cacheMatches, true, "save matches under ./cache");
DEFINE_double(pixel_threshold, 3.0, "Reprojection threshold in pixels");
DEFINE_uint64(min_reprojections, 3, "Minimal valid reprojections on a track");
DEFINE_uint64(max_reprojections, 0, "Maximal valid reprojections on a track");
DEFINE_uint64(maxFramesToMatch, 5, "Maximal number of past frames to match");
DEFINE_uint64(fixFirstNCameras, 0, "Number of initial frames to maintain fixed during BA");
DEFINE_bool(fixRotation, false, "fix rotation parameters");
DEFINE_bool(fixPosition, false, "fix 3D position parameters");
DEFINE_bool(fixScale, true, "fix first and last camera position to constrain scale changes");

DEFINE_double(ba_huberLoss, 0.0, "BA Huber loss function parameterization");
DEFINE_uint64(baIterationsOnNewFrame, 0, "BA iterations after every new frame");
DEFINE_uint64(baWindowOnNewFrame, 0, "BA window size after every new frame");
DEFINE_uint64(baFinalIterations, 0, "overall BA iterations");
DEFINE_uint64(baFinalRuns, 1, "overall BA runs (including track creation)");
DEFINE_bool(pnpNewFrame, false, "least squares Point and Perspective BA after every new frame");
DEFINE_double(constFrameVelocity, 0, "scale of constant velocity prior");
DEFINE_double(constFrameAcceleration, 0, "scale of constant acceleration prior");
DEFINE_bool(useOnlyValidMatches, true, "only use valid tracks on BA");

DEFINE_bool(calibrated, true, "assume calibrated cameras");
DEFINE_bool(use3Dpoints, true, "triangulate 3D points or use light rays directly");
DEFINE_bool(rolling_shutter, true, "use rolling shutter model");
DEFINE_bool(interpolateRotation, true, "also interpolate rotation in RS model");
DEFINE_bool(gs_init, false, "initialize poses as global shutter");
DEFINE_bool(init_3d, false, "initialize 3D points from ground truth");
DEFINE_bool(const3d, false, "fix 3D points");
DEFINE_uint64(loadDatasetPoses, 0, "load first 'N' camera poses from the dataset if available");

DEFINE_bool(reuseLastPose, true, "initialize next frame with last known pose");
DEFINE_bool(solveGsPnP, false, "use the traditional PnP RANSAC to find next pose");
DEFINE_bool(solveRsPnP, true, "use the rolling shutter PnP RANSAC to find next pose");
DEFINE_uint64(minPnPfeatures, 6, "Minimal PnP parameterization");
DEFINE_bool(refinePnP, false, "BA frame after PnP to refine estimate; not necessary with solveRsPnP");

DEFINE_bool(writePly, false, "output PLY files for every step on the pipeline");
DEFINE_bool(showTracks, false, "output images with their reprojections");
DEFINE_uint64(showLastMatches, 0, "output images with their matches to a given number of frames back");
DEFINE_bool(calcCovariances, false, "output BA covariances");

DEFINE_double(observation_noise, 0, "adds white noise to 2D observations");
DEFINE_double(rotation_noise, 0, "adds white noise to camera poses");
DEFINE_double(translation_noise, 0, "adds white noise to camera poses");
DEFINE_double(outlier_noise, 0, "adds uniformly distributed noise");


namespace vision {

struct SfmFlags {

  static SfmOptions Parse(int* argc, char*** argv)
  {
    google::ParseCommandLineFlags(argc, argv, true);

    SfmOptions opt;
    opt.model.calibrated = FLAGS_calibrated;
    opt.model.use3Dpoints = FLAGS_use3Dpoints;
    opt.model.rolling_shutter = FLAGS_rolling_shutter;
    opt.model.interpolateRotation = FLAGS_interpolateRotation;
    opt.mod_init.reuseLastPose = FLAGS_reuseLastPose;
    opt.mod_init.solveGsPnP = FLAGS_solveGsPnP;
    opt.mod_init.solveRsPnP = FLAGS_solveRsPnP;
    opt.mod_init.minPnPfeatures = FLAGS_minPnPfeatures;
    opt.mod_init.refinePnP = FLAGS_refinePnP;
    opt.tracks.cacheMatches = FLAGS_cacheMatches;
    opt.tracks.sqrdThreshold = FLAGS_pixel_threshold * FLAGS_pixel_threshold;
    opt.tracks.minReprojections = FLAGS_min_reprojections;
    opt.tracks.maxReprojections = FLAGS_max_reprojections;
    opt.tracks.maxFramesToMatch = FLAGS_maxFramesToMatch;
    opt.init.gs_init = FLAGS_gs_init;
    opt.init.init_3d = FLAGS_init_3d;
    opt.init.observation_noise = FLAGS_observation_noise;
    opt.init.rotation_noise = FLAGS_rotation_noise;
    opt.init.translation_noise = FLAGS_translation_noise;
    opt.init.outlier_noise = FLAGS_outlier_noise;
    opt.init.loadDatasetPoses = FLAGS_loadDatasetPoses;
    opt.sift.nfeatures = FLAGS_sift_nfeatures;
    opt.sift.nOctaveLayers = FLAGS_sift_nOctaveLayers;
    opt.sift.contrastThreshold = FLAGS_sift_contrastThreshold;
    opt.sift.edgeThreshold = FLAGS_sift_edgeThreshold;
    opt.sift.sigma = FLAGS_sift_sigma;
    opt.ceres.const3d = FLAGS_const3d;
    opt.ceres.huberLoss = FLAGS_ba_huberLoss;
    opt.ceres.fixFirstNCameras = FLAGS_fixFirstNCameras;
    opt.ceres.fixRotation = FLAGS_fixRotation;
    opt.ceres.fixPosition = FLAGS_fixPosition;
    opt.ceres.fixScale = FLAGS_fixScale;
    opt.ceres.baIterationsOnNewFrame = FLAGS_baIterationsOnNewFrame;
    opt.ceres.baWindowOnNewFrame = FLAGS_baWindowOnNewFrame;
    opt.ceres.baFinalIterations = FLAGS_baFinalIterations;
    opt.ceres.baFinalRuns = FLAGS_baFinalRuns;
    opt.ceres.pnpNewFrame = FLAGS_pnpNewFrame;
    opt.ceres.constFrameVelocity = FLAGS_constFrameVelocity;
    opt.ceres.constFrameAcceleration = FLAGS_constFrameAcceleration;
    opt.ceres.useOnlyValidMatches = FLAGS_useOnlyValidMatches;
    opt.debug.writePly = FLAGS_writePly;
    opt.debug.showTracks = FLAGS_showTracks;
    opt.debug.showLastMatches = FLAGS_showLastMatches;
    opt.debug.calcCovariances = FLAGS_calcCovariances;

    return opt;
  };

};


}; // namespace

#endif
