// Author: Henrique Mendonça <henrique@apache.org>
#include <opencv2/calib3d/calib3d.hpp>
#include "rsba/mat/cam.h"

namespace vision
{

#if (defined(CV_VERSION_EPOCH) && CV_VERSION_EPOCH == 2)
#define SOLVEPNP_ITERATIVE cv::ITERATIVE
#else
#define SOLVEPNP_ITERATIVE cv::SOLVEPNP_ITERATIVE
#endif

bool solveRsPnP(
    cv::InputArray _opoints,
    cv::InputArray _ipoints,
    cv::InputArray _cameraMatrix,
    cv::InputArray _distCoeffs,
    cv::OutputArray _rvec,
    cv::OutputArray _tvec,
    cv::OutputArray _rvec2,
    cv::OutputArray _tvec2,
    const SHUTTER shutter,
    const int scanlines[2],
    bool useExtrinsicGuess = false,
    int flags = SOLVEPNP_ITERATIVE);


void solveRsPnPRansac(
    cv::InputArray _opoints,
    cv::InputArray _ipoints,
    cv::InputArray _cameraMatrix,
    cv::InputArray _distCoeffs,
    cv::OutputArray _rvec,
    cv::OutputArray _tvec,
    cv::OutputArray _rvec2,
    cv::OutputArray _tvec2,
    const SHUTTER shutter,
    const int scanlines[2],
    bool useExtrinsicGuess = false,
    int iterationsCount = 100,
    float reprojectionError = 8.0,
    int minInliersCount = 100,
    cv::OutputArray inliers = cv::noArray(),
    int flags = SOLVEPNP_ITERATIVE,
    int min_points_count = 6);

}


