
namespace cpp vision.sfm.gen



enum RollingShutter {
  HORIZONTAL=1 // rolling shutter with horizontal scan-lines
  VERTICAL=2   // rolling shutter with vertical scan-lines
}



struct Observation {
  1: double x
  2: double y
  3: optional binary descriptor
  4: optional binary color
  5: optional list<ObservationRef> matches

  // track index in current session
  6: optional i32 track
}


struct Track {
  1: list<ObservationRef> obs
  2: optional list<double> pt
  3: optional binary color
  4: bool valid = false
}


struct ObservationRef {
  // frame index in current session
  1: i32 frame

  // observation index in current frame
  2: i32 obs

  3: bool valid = false
}


struct Frame {
  1: list<Observation> obs

  // Multiple poses for rolling shutters
  2: optional list< list<double> > poses
  
  // Camera intrinsics
  3: optional list<double> cam

  // Multiple poses for rolling shutters
  4: optional list< list<double> > priorPoses
}


/*
 * Each session holds its image frames and the feature-matches between them
 */
struct Session {
  // Default camera intrinsics
  1: list<double> cam
  
  2: list<Frame> frames
  3: list<Track> tracks
  4: optional RollingShutter rs

  // idicates first and last scanlines in rolling shutters
  5: optional list<i32> scanlines
  
  6: i32 width
  7: i32 height
}


service VideoSfM {
  // return an authToken
  binary authenticate(/* TODO */)

  // start a session and return its key
  i32 newSession(1: binary authToken, 2: list<double> camera)

  // start a new session by cloning an existing one and return its key
  i32 cloneSession(1: binary authToken, 2: i32 oldSessionKey)

  // start a RollingShutter session and return its key
  i32 newRsSession(1: binary authToken, 2: list<double> camera, 3: RollingShutter rs, 4: list<i32> scanlines)

  // add a frame and return its key
  // will generate new tracks if matches are available
  i32 newFrame(1: binary authToken, 2: i32 sessionKey, 3: Frame frame)

  // get a frame
  Frame getFrame(1: binary authToken, 2: i32 sessionKey, 3: i32 frameKey)

  // add a track and return its key
  i32 newTrack(1: binary authToken, 2: i32 sessionKey, 3: Track track)

  // get all tracks
  list<Track> getTracks(1: binary authToken, 2: i32 sessionKey)

  // initialize frame poses
  void initialize(1: binary authToken, 2: i32 sessionKey)

  // run bundle adjustment on all frames
  bool fullBA(1: binary authToken, 2: i32 sessionKey, 3: i32 maxIter, 4: bool reproject = false)

  // run bundle adjustment on requested frames fixing the remaining poses
  bool windowedBA(1: binary authToken, 2: i32 sessionKey, 3: i32 startFrame, 4: i32 endFrame, 5: i32 maxIter, 6: bool reproject = false)

  // finalize session
  void finalize(1: binary authToken, 2: i32 sessionKey)
}
