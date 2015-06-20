#ifndef _VisionTracking_h
#define _VisionTracking_h

#include <map>
#include <set>
#include <vector>
#include <iostream>
#include <assert.h>

#include "rsba/pointers.h"
#include "rsba/mat.h"



namespace vision {

  struct frame;
  struct match;
  struct observation;

  typedef shared_ptr<frame>        framePtr;
  typedef shared_ptr<match>        matchPtr;
  typedef shared_ptr<observation>  obsPtr;
  typedef weak_ptr<observation>    obsWeakPtr;

  struct obsCompare {
    bool operator()(const observation& a, const observation& b);
    bool operator()(const observation* a, const observation* b);
    bool operator()(const obsPtr& a, const obsPtr& b);
  };

  struct matchCompare {
    bool operator()(const matchPtr& a, const matchPtr& b) {
      return a.get() < b.get();
    }
    bool operator()(const match& a, const match& b) {
      return &a < &b;
    }
  };

  typedef std::set< match, matchCompare > matchSet;
  typedef std::set< obsPtr, obsCompare >  obsSet;


  struct match {
    match(const unsigned char c[3]) {
      color[0] = c[0];
      color[1] = c[1];
      color[2] = c[2];
    };

    mutable bool set = false;
    mutable bool valid = false;
    unsigned char color[3];
    mutable double pt[3] = {0}; //3D point
    mutable std::set<observation*, obsCompare> _obs;

    void Triangulate() const;
    bool Validate() const;
    bool AddObs(observation* obs) const;
  };

  struct frame {
    frame(unsigned num, double* const cam) : num(num), cam(cam) {};

    const obsPtr& Add(const double& x, const double& y);

    const unsigned num;
    bool set = false;
    SHUTTER shutter = GLOBAL;

    double pose[6] = {0}; //initial/default camera pose
    double end[6] = {0}; //end camera pose (rs only)

    int scanlines[2] = {0}; //initial and final rolling shutter image scan-line coordinates

    std::set<obsPtr, obsCompare> _obs;

    double* const cam; // camera parameters
  };

  struct observation {
    observation(const double& x, const double& y, frame* _frame)
    : _frame(_frame),
      _match(nullptr) {
      pt[0] = x;
      pt[1] = y;
    };

    observation( const observation& other )
    : _frame(other._frame),
      _match(other._match) {
      pt[0] = other.pt[0];
      pt[1] = other.pt[1];
    };

    virtual ~observation() {
      //TODO remove itself from match
    };

    bool valid = false;
    double pt[2];

    frame* const _frame;
    const match* _match;
  };
};


#endif
