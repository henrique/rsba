#include "rsba/Sfm2Ply.h"

#include "rsba/mat/cam.h"
#include "rsba/struct/VideoSfM.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <iostream>
#include <sys/stat.h>
#ifdef _WIN32
#include <direct.h>
#endif

using namespace std;

namespace vision {
namespace sfm {

static const char* trg_dir = "results/";
static const uint maxViewArea = 300;


void writeCam(const double pose[6], unsigned cam_index, unsigned num_cameras, FILE* f)
{
    double cam_inv[3] = {
        - pose[0],
        - pose[1],
        - pose[2]
    };

    // print camera centers from blue to green
    fprintf(f, "%0.16e %0.16e %0.16e 0 %u %u\n", pose[3], pose[4], pose[5],
        (unsigned) ((255 * (1.0 + cam_index) / num_cameras)),
        (unsigned) ((255 - 255 * (0.1 + cam_index) / num_cameras)) );

    // print camera frame
    double p_cam[5][3] = {
      { 0.00, 0.00, 1.0 },
//      { 2.00, 2.00, 2.0 },
//      { 2.00, -2.0, 2.0 },
//      { -2.0, 2.00, 2.0 },
//      { -2.0, -2.0, 2.0 }
    };

    for (short k = 0; k < 1; k++) {
      double p[3];
      ceres::AngleAxisRotatePoint(cam_inv, p_cam[k], p);
      p[0] += pose[3];
      p[1] += pose[4];
      p[2] += pose[5];

      fprintf(f, "%0.16e %0.16e %0.16e 255 255 0\n", p[0], p[1], p[2]);
    }
}



bool inView(const sfm::Session& sess, const gen::Track& t)
{
  bool inView = true;
  if (sess.frames.size() > 0 && sess.frames[0].poses.size() > 0)
  {
    double diff[3];
    minus3(t.pt.data(), sess.frames[0].poses[0].data() + 3, diff);
    inView = norm3(diff) < maxViewArea;
  }
  return inView;
}



void writePly(const string& file,
              const sfm::Session& sess,
              const SfmOptions& opt,
              bool write_cameras,
              bool write_points)
{
  if ( ! opt.debug.writePly) return;

#if defined(_WIN32)
  _mkdir(trg_dir);
#else
  mkdir(trg_dir, 0777);
#endif
  string path = (trg_dir + file);

  FILE* f = fopen(path.c_str(), "w");
  if (f == NULL) {
    printf("Error opening file for writing\n");
    assert(false);
  }
  else {
    printf("Writing file: %s\n", path.c_str());
  }

  static char ply_header[] =
      "ply\nformat ascii 1.0\n"
      "element face 0\n"
      "property list uchar int vertex_indices\n"
      "element vertex %d\n"
      "property float x\n"
      "property float y\n"
      "property float z\n"
      "property uchar diffuse_red\n"
      "property uchar diffuse_green\n"
      "property uchar diffuse_blue\n"
      "end_header\n";

  unsigned num_cameras = write_cameras ? sess.frames.size() : 0;
  unsigned num_points_out = 7; // origin and axis representation

  for (const gen::Frame& frame: sess.frames) {
    if (frame.poses.size() == 2) {
      num_points_out += (2 * 11);
    } else {
      num_points_out += 2 * frame.poses.size();
    }
  }

  bool useOnlyValid = false;
  size_t numInvalid = 0, numValid = 0;
  for (const gen::Track& t: sess.tracks) {
    if (t.valid && inView(sess, t)) {
      num_points_out++;
      numValid++;
      useOnlyValid = true;
    } else {
      numInvalid++;
    }
  }

  /* Print the ply header */
  fprintf(f, ply_header, num_points_out);

  /* origin */
  fprintf(f, "0.00 0.00 0.00 255 0 0\n");
  fprintf(f, "1.00 0.00 0.00 255 0 0\n");
  fprintf(f, "-1.0 0.00 0.00 128 0 0\n");
  fprintf(f, "0.00 1.00 0.00 0 255 0\n");
  fprintf(f, "0.00 -1.0 0.00 0 128 0\n");
  fprintf(f, "0.00 0.00 1.00 0 0 255\n");
  fprintf(f, "0.00 0.00 -1.0 0 0 128\n");


  if (write_cameras) {
    for (unsigned fi = 0; fi < sess.frames.size(); fi++) {
      const gen::Frame& frame = sess.frames[fi];

      if (frame.poses.size() == 2) {
        double pose[6];

        for (double i = 0.0; i <= 1; i += 0.1) {
          interpolate(frame.poses[0].data(), frame.poses[1].data(), i,
              pose, opt.model.interpolateRotation);
          writeCam(pose, fi, num_cameras, f);
        }
      } else {
        for (const std::vector<double>& pose : frame.poses) {
          writeCam(pose.data(), fi, num_cameras, f);
        }
      }
    }
  }


  if (write_points) {
    for (const gen::Track& t : sess.tracks)
    {
      if ( useOnlyValid && (!t.valid || !inView(sess, t)) ) continue;

      uchar rgb[3] = { (uchar)t.color[0], (uchar)t.color[1], (uchar)t.color[2] };

      // redisch numInvalid vertexes
      if ( ! t.valid) {
        rgb[0] = rgb[0]/2 + 127;
        rgb[1] /= 2;
        rgb[2] /= 2;
      }

      /* Output the vertex */
      fprintf(f, "%0.16e %0.16e %0.16e %u %u %u\n",
          t.pt[0],
          t.pt[1],
          t.pt[2],
          rgb[0], rgb[1], rgb[2]);
    }
    printf("Total: %lu inliers and %lu outliers\n", numValid, numInvalid);
  }

  fclose(f);
  printf("Done\n");
}

}} //namespace vision::sfm
