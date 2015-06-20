#ifndef _Sfm2Ply_h
#define _Sfm2Ply_h

#include "rsba/struct/VideoSfM.h"
#include "rsba/SfmOptions.h"


using namespace std;

namespace vision {
namespace sfm {

void writeCam(const double pose[6], unsigned cam_index, unsigned num_cameras, FILE* f);

void writePly(const string& file, const sfm::Session& sess, const SfmOptions& opt,
    bool write_cameras = true, bool write_points = true);

}} //namespace vision::sfm

#endif
