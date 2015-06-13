// Author: Henrique Mendonça <henrique@apache.org>
//

#include <math.h>
#include <string>
#include <stdlib.h>
#include <gtest/gtest.h>
#include <Eigen/Geometry>

#include "base/asserts.h"
#include "base/shUtils.h"
#include "mat.h"
#include "eigenTypes.h"

using namespace std;
using namespace vision;
using namespace Eigen;

static const double pi = M_PI;
static const double pi2 = M_PI_2;


TEST(SfM, Norm) {
  double x[3] = { -1e-100, 2.3, 1e100 };
  CHECK_NEAR(norm(x[0], x[1], x[2]), norm3(x), _EPS);

  // normalize
  scalar3(x, (1.0/norm3(x)), x);
  CHECK_NEAR(norm(x[0], x[1], x[2]), 1.0, _EPS);
}


// left hand rotation, i.e. clockwise
TEST(SfM, Rotation) {
  double r[3] = { 0, pi, 0 };
  double ri[3] = { 0, -pi, 0 };
  double r2[3] = { 0, -pi2, 0 };
  double r3[3] = { pi2, 0, 0 };

  double p[3] = { 0, 0, -10 };
  double p2[3] = { 0, 0, 10 };
  double p3[3] = { 10, 0, 0 };
  double p4[3] = { 0, 10, 0 };

  double test[3];
  //pi
  ceres::AngleAxisRotatePoint(r, p, test);
  CHECK_LE(dist3(p2, test), 1e-9);

  double Pinv[3];
  invert3(r, Pinv);
  ceres::AngleAxisRotatePoint(Pinv, test, test);
  CHECK_LE(dist3(p, test), 1e-9);

  //-pi
  ceres::AngleAxisRotatePoint(ri, p, test);
  CHECK_LE(dist3(p2, test), 1e-9);

  //-pi/2
  ceres::AngleAxisRotatePoint(r2, p, test);
  CHECK_LE(dist3(p3, test), 1e-9);

  //pi/2
  ceres::AngleAxisRotatePoint(r3, p, test);
  CHECK_LE(dist3(p4, test), 1e-9);

  //same with eigen
  Vector3dRef pRef(p);
  AngleAxisd r3e(pi2, Vector3d::UnitX()); //r3
  Vector3d rst = r3e.toRotationMatrix() * pRef;
  CHECK_LE(dist3(p4, rst.data()), 1e-9);
}


TEST(SfM, SLERP) {
  const short num = 16;
  double r[num][3] = {
      { 0, 0.5, 0 },
      { 0, 1, 0 },
      { 1, 0, 0 },
      { 0, 0, 1 },
      { 0, 1, 1 },
      { 1, 1, 1 },
      { 0, -1, 0 },
      { -1, 0, 0 },
      { -1, 0, -1 },
      { 0, pi2, 0 },
      { pi2, 0, 0 },
      { 0, 1-pi2, 0 },
      { 0, -1, pi2 },
      { 0, -1, 1-pi2 },
      { 0, 0, _EPS },
      { 1, -1, _EPS },
  };

  for (short i = 0; i < num; i++) {
    for (short j = 1; j < num; j++) {
      double inter[3];

      slerp(r[i], r[j], 1.0, inter);
      CHECK_LE(dist3(inter, r[j]), 1e-6);

      slerp(r[i], r[j], 0.0, inter);
      CHECK_LE(dist3(inter, r[i]), 1e-6);

      double r05[3], r15[3], r20[3], in[500][3];

      for (double x = 0; x < 500; x++)
        slerp(r[i], r[j], x/100, in[(int)x]);

      slerp(r[i], r[j], 0.5, r05);
      slerp(r[i], r[j], 1.5, r15);
      slerp(r[i], r[j], 2.0, r20);


      slerp(r[i], r05, 2.0, inter);

      slerp(r[i], r20, 0.5, inter);
      CHECK_LE(dist3(inter, r[j]), 1e-6);

      slerp(r05, r15, 0.5, inter);
      CHECK_LE(dist3(inter, r[j]), 1e-6);

      if (i != j) { //TODO
        for (unsigned frac = 2; frac < 500; frac = frac*2 - 1) {
          for (double n = 1; n < frac; n++) {
            double tau = n / frac;
            slerp(r[i], r[j], tau, inter);

            AngleAxisd ri0(norm3(r[i]), Vector3d(r[i])/norm3(r[i]));
            AngleAxisd rj0(norm3(r[j]), Vector3d(r[j])/norm3(r[j]));
            Quaterniond qi(ri0), qj(rj0);
            AngleAxisd rst(qi.slerp(tau, qj));
            Vector3d rst3 = rst.axis()*rst.angle();

            CHECK_LE(dist3(inter, rst3.data()), 0.2);
          }
        }
      }
    }
  }
}


TEST(SfM, Distortion) {
  double cam[4][NUM_CAM_PARAMS] = {
      { 0.1, 0.1, 0, 0, 0, 0, 0 },
      { 100, 100, 0.01, 0, 0, 0, 0 },
      { 500, 500, -0.03, 0, 0, 0, 0 },
      { 500, 500, -0.1, 0.02, 0, 0, 0 },
  };

  double img_d[2], img_u[2];
  double img[3][2] = {
      { 0.10, 0.10 },
      { 0.21, 0.19 },
      { 1.10, 0.50 }
  };

  for (short j = 0; j < 3; j++) {
    for (short k = 0; k < 4; k++) {
      distort(cam[k], img[j], img_d);
      CHECK(undistort(cam[k], img_d, img_u));
      CHECK_LE(dist2(img[j], img_u), 1e-6);
    }
  }
}


TEST(SfM, reprojection) {
  double poseRef[6] = { 0, 0, 0, 20, 20, 0 };
  double pose[11][6] = {
      { 0, 0, 0, 0, 0, 0 },
      { 0, 0, 0, 1, 1, 1 },
      { 0, 0, pi2, 20, 20, 20 },
      { 0, pi2, pi2, -2, 20, 20 },
      { pi2, pi2, pi2, -2, -2, 20 },
      { -1, -1, -1, -2, -2, -2 },
      { -pi2, -1, -1, -20, -2, -2 },
      { 0.5, -pi2, -1, -2, -20, -2 },
      { 0.5, 0.5, -pi2, 0.2, -2, -20 },
      { _EPS, _EPS, _EPS, _EPS, _EPS, _EPS },
      { -_EPS, -_EPS, -_EPS, -_EPS, -_EPS, -_EPS },
  };

  double pt[18][3] = {
      { 10, 10, 10 },
      { 100, 0, 1 },
      { 0, 100, 1 },
      { 0, 0, 100 },
      { -100, 0, 1 },
      { 0, -100, 1 },
      { 0, 0, -100 },
      { 0, 0, -1 },
      { 0, 0, 0 },
      { 1, 1, 1 },
      { -1, -1, -1 },
      { 0.1, 0.1, 0.1 },
      { 100, 100, 100 },
      { -100, -100, -100 },
      { -0.39, 1.25, 2014 },
      { _EPS, _EPS, _EPS },
      { _EPS, _EPS, -_EPS },
      { -_EPS, -_EPS, -_EPS },
  };

  double cam[6][NUM_CAM_PARAMS] = {
      { 0.1, 0.1, 0, 0, 0, 0, 0, 0, 0 },
      { 100, 100, 0, 0, 0, 0, 0, 0, 0 },
      { 500, 500, 0, 0, 0, 0, 0, 640, 480 },
      { 100, 100, _EPS, 0, 0, 0, 0, 0, 0 },
      { 500, 500, -_EPS, -_EPS, 0, 0, 0, 0, 0 },
      { 860, 860, 0.001, 0, 0, 0, 0, 100, 200 },
  };

  for (short i = 0; i < 11; i++) {
    for (short j = 0; j < 18; j++) {
      for (short k = 0; k < 6; k++) {
        double c3[3], w3[3];
        w2c(pose[i], pt[j], c3);
        c2w(pose[i], c3, w3);
        CHECK_LE(dist3(pt[j], w3), 1e-6);

        double d1[3],d2[3], d3[3];
        if (direction(pose[i], pt[j], d1) and c2direction(pose[i], c3, d2)) {
          CHECK_LE(dist3(d1, d2), 1e-6);

          double img[2];
          if (w2i(cam[k], pose[i], pt[j], img) and direction(cam[k], pose[i], img, d3)) {
            CHECK_LE(dist3(d1, d3), 1e-1); // very imprecise!!

            double cRef[3], dRef[3], tri[3];
            w2c(poseRef, pt[j], cRef);
            if (c2direction(poseRef, cRef, dRef)) {
              double p2[3], len[2];
              minus3(poseRef+3, pose[i]+3, p2);
              CHECK(ray_intersect(p2, d1, dRef, len)); //never parallel

              double pd1[3], pd2[3];
              scalar3(d1, len[0], pd1);
              plus3(pd1, pose[i]+3, pd1);
              CHECK_LE(dist3(pt[j], pd1), 1e-9);

              scalar3(dRef, len[1], pd2);
              plus3(pd2, poseRef+3, pd2);
              CHECK_LE(dist3(pt[j], pd2), 1e-9);

              CHECK_GE(len[0], _EPS); // behind the camera

              CHECK(triangulate(pose[i]+3, d1,
                              poseRef+3, dRef,
                              tri));

              CHECK_NEAR(dist3(pose[i]+3, tri), len[0], 1e-5); // imprecise!
              CHECK_NEAR(dist3(poseRef+3, tri), len[1], 1e-5); // imprecise!

              double imgRef[2];
              if (w2i(cam[k], poseRef, tri, imgRef)) {
                CHECK_LE(dist3(pt[j], tri), 1e-5); // imprecise!

                const static double threshold = 1.0; // no outliers
                CHECK(validate(cam[k], pose[i], img, pt[j], threshold));
                CHECK(validate(cam[k], poseRef, imgRef, pt[j], threshold));

                CHECK(ray_intersect(p2, d1, dRef, len)); //never parallel

                CHECK_GE(len[1], _EPS); // behind the camera

                double dist[3];
                if(rayDist(cam[k], pose[i], img,
                        cam[k], poseRef, imgRef,
                        dist))
                  CHECK_LE(norm3(dist), dist3(pose[i]+3, poseRef) + 1e-1); // rays intersect, imprecise!
              }
//              else
//              {
//                w2i(cam[k], poseRef, tri, imgRef, false);
//
//                double dist[3];
//                if (rayDist(cam[k], pose[i], img,
//                        cam[k], poseRef, imgRef,
//                        dist)) {
//                  CHECK_GE(norm3(dist), _EPS); // rays don't intersect
//                }
//              }
            }
          }
        }
      }
    }
  }
}
