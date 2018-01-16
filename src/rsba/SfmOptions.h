#ifndef _SfmOptions_h
#define _SfmOptions_h


#include <ceres/ceres.h>


namespace vision {

struct SfmOptions {

  struct Init {
    bool init_3d = false;
    bool gs_init = false; // initialize with global shutter data
    double observation_noise = 0.0; // in pixels
    double rotation_noise = 0.0; // in radians
    double translation_noise = 0.0; // in meters
    double outlier_noise = 0.0; // ratio of outliers
    uint loadDatasetPoses = 0; // load first 'N' camera poses from the dataset if available
  } init;

  struct Model {
    bool rolling_shutter = true;
    bool fullDoF = false; // full degree of freedoms, i.e. one pose per scanline
    bool interpolateRotation = true;
    bool use3Dpoints = true;
    bool calibrated = true; // fix or optimize camera intrinsics
    bool constVelocity = false;
  } model;

  struct ModelInit {
    bool reuseLastPose = true; // initialize next frame with last known pose
    bool solveGsPnP = false; // use the traditional PnP RANSAC to find next pose
    bool solveRsPnP = true; // use the rolling shutter PnP RANSAC to find next pose
    uint minPnPfeatures = 6; // minimal RS PnP RANSAC parameterization
    bool refinePnP = false; // BA frame after PnP to refine estimate; not necessary with solveRsPnP
  } mod_init;

  struct Tracks {
    bool synthetic = false; // using synthetic tracks, i.e. no matches
    bool cacheMatches = false; // save matches under ./cache
    double sqrdThreshold = 16.0;
    uint minReprojections = 3;
    uint maxReprojections = 10;
    uint maxFramesToMatch = 5;
    uint minDistanceToCamera = 0;
    uint minTracksPerFrame = 500; //TODO
  } tracks;

  struct Features2D {
     std::string name;
  } features2d;

  struct SIFT {
    int nfeatures = 0;
    int nOctaveLayers = 3;
    double contrastThreshold = 0.04;
    double edgeThreshold = 10;
    double sigma = 1.6;
  } sift;

  struct SURF {
    double hessianThreshold = 400;
  } surf;

  struct Ceres {
    bool useOnlyValidMatches = true;
    double huberLoss = 0.0; // default loss function if > 0

    bool const3d = false; // fix 3D points
    uint fixFirstNCameras = 0;
    bool fixScale = false; // fix first and last camera position to constrain scale changes
    bool fixRotation = false; // fix cameras' rotation parameters
    bool fixPosition = false; // fix cameras' 3D position parameters


    double constFrameVelocity = 0; // scale of constant velocity prior
    double constFrameAcceleration = 0; // scale of constant acceleration prior
    double interFrameRatio = 1;  // proportion of time between frame exposures (used on constant motion priors)

    double trustPriorCamPosition = 0; // degree of trust on camera pose "a priori" (before BA)
    double trustPriorCamRotation = 0; // degree of trust on camera pose "a priori" (before BA)

    bool pnpNewFrame = false; // least squares Point and Perspective BA after every new frame
    uint baIterationsOnNewFrame = 0; // BA iterations after every new frame
    uint baWindowOnNewFrame = 0; // BA window size after every new frame, 0 for fullBA

    uint baFinalIterations = 0; // overall BA iterations
    uint baFinalRuns = 1; // overall BA runs (including track creation)

    bool revalidateReprojections = false; // validate reprojections before adding to cost
  } ceres;

  struct Debug {
    bool writePly = false;
    uint showLastMatches = 0; // number of frames back to show
    bool showTracks = false;
    bool calcCovariances = false;
  } debug;

};


}; // namespace

#endif
