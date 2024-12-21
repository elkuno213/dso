/**
 * This file is part of DSO.
 *
 * Copyright 2016 Technical University of Munich and Intel.
 * Developed by Jakob Engel <engelj at in dot tum dot de>,
 * for more information see <http://vision.in.tum.de/dso>.
 * If you use this code, please cite the respective publications as
 * listed on the above website.
 *
 * DSO is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * DSO is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with DSO. If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#define MAX_ACTIVE_FRAMES 100

#include <math.h>
#include <deque>
#include <fstream>
#include <iostream>
#include "vector"

#include "FullSystem/HessianBlocks.h"
#include "FullSystem/PixelSelector2.h"
#include "FullSystem/Residuals.h"
#include "OptimizationBackend/EnergyFunctional.h"
#include "util/FrameShell.h"
#include "util/IndexThreadReduce.h"
#include "util/NumType.h"
#include "util/globalCalib.h"

namespace dso {

namespace IOWrap {
class Output3DWrapper;
} // namespace IOWrap

class PixelSelector;
class PCSyntheticPoint;
class CoarseTracker;
struct FrameHessian;
struct PointHessian;
class CoarseInitializer;
struct ImmaturePointTemporaryResidual;
class ImageAndExposure;
class CoarseDistanceMap;
class EnergyFunctional;

// Delete the i-th element from a vector and move the last element to its place.
template <typename T>
inline void deleteOut(std::vector<T*>& v, const int i) {
  delete v[i];
  v[i] = v.back();
  v.pop_back();
}
// Delete the element from a vector and move the last element to its place.
template <typename T>
inline void deleteOutPt(std::vector<T*>& v, const T* e) {
  delete e;

  for (unsigned int i = 0; i < v.size(); i++) {
    if (v[i] == e) {
      v[i] = v.back();
      v.pop_back();
    }
  }
}
// Delete the i-th element from a vector and increment all following elements by
// one position.
template <typename T>
inline void deleteOutOrder(std::vector<T*>& v, const int i) {
  delete v[i];
  for (unsigned int k = i + 1; k < v.size(); k++) {
    v[k - 1] = v[k];
  }
  v.pop_back();
}
// Delete the element from a vector and increment all following elements by one
// position.
template <typename T>
inline void deleteOutOrder(std::vector<T*>& v, const T* e) {
  int i = -1;
  for (unsigned int k = 0; k < v.size(); k++) {
    if (v[k] == e) {
      i = k;
      break;
    }
  }
  assert(i != -1);

  for (unsigned int k = i + 1; k < v.size(); k++) {
    v[k - 1] = v[k];
  }
  v.pop_back();

  delete e;
}
// Check if a matrix contains NaN values.
inline bool eigenTestNan(const MatXX& m, std::string msg) {
  bool foundNan = false;
  for (int y = 0; y < m.rows(); y++) {
    for (int x = 0; x < m.cols(); x++) {
      if (!std::isfinite((double)m(y, x))) {
        foundNan = true;
      }
    }
  }

  if (foundNan) {
    printf("NAN in %s:\n", msg.c_str());
    std::cout << m << "\n\n";
  }

  return foundNan;
}

class FullSystem {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  FullSystem();
  virtual ~FullSystem();

  // Add a new frame and creates point & residual structs.
  void addActiveFrame(ImageAndExposure* image, int id);

  // Marginalize a frame. Drop/Marginalize points & residuals.
  void marginalizeFrame(FrameHessian* frame);

  void blockUntilMappingIsFinished();

  float optimize(int mnumOptIts);

  void printResult(std::string filename);

  void debugPlot(std::string name);

  void printFrameLifetimes();

  std::vector<IOWrap::Output3DWrapper*> output_3d_wrappers_;

  bool is_lost_;
  bool is_initialization_failed_;
  bool is_initialized_; // True if the full system is initialized.
  bool linear_operation_;

  void setGammaFunction(float* B_inv);

  void setOriginalCalib(const VecXf& calib, int width, int height);

private:
  CalibHessian Hcalib;

  // Optimize single point.
  int optimizePoint(PointHessian* point, int minObs, bool flagOOB);
  PointHessian* optimizeImmaturePoint(
    ImmaturePoint* point,
    int minObs,
    ImmaturePointTemporaryResidual* residuals
  );

  double linAllPointSinle(PointHessian* point, float outlierTHSlack, bool plot);

  // Main pipeline functions.
  Vec4 trackNewCoarse(FrameHessian* hessian);
  void traceNewCoarse(FrameHessian* hessian);
  void activatePoints();
  void activatePointsMT();
  void activatePointsOldFirst();
  void flagPointsForRemoval();
  void makeNewTraces(FrameHessian* newFrame, float* gtDepth);
  void initializeFromInitializer(FrameHessian* newFrame);
  void flagFramesForMarginalization(FrameHessian* newFH);

  void removeOutliers();

  void setPrecalcValues();

  // Solve the system (Eventually migrate to EnergyFunctional).
  void solveSystem(int iteration, double lambda);
  Vec3 linearizeAll(bool fixLinearization);
  bool doStepFromBackup(
    float stepfacC,
    float stepfacT,
    float stepfacR,
    float stepfacA,
    float stepfacD
  );
  void backupState(bool backupLastStep);
  void loadSateBackup();
  double calcLEnergy();
  double calcMEnergy();
  void linearizeAll_Reductor(
    bool fixLinearization,
    std::vector<PointFrameResidual*>* toRemove,
    int min,
    int max,
    Vec10* stats,
    int tid
  );
  void activatePointsMT_Reductor(
    std::vector<PointHessian*>* optimized,
    std::vector<ImmaturePoint*>* toOptimize,
    int min,
    int max,
    Vec10* stats,
    int tid
  );
  void applyRes_Reductor(bool copyJacobians, int min, int max, Vec10* stats, int tid);

  void printOptRes(
    const Vec3& res,
    double resL,
    double resM,
    double resPrior,
    double LExact,
    float a,
    float b
  );

  void debugPlotTracking();

  std::vector<VecX> getNullspaces(
    std::vector<VecX>& nullspaces_pose,
    std::vector<VecX>& nullspaces_scale,
    std::vector<VecX>& nullspaces_affA,
    std::vector<VecX>& nullspaces_affB
  );

  void setNewFrameEnergyTH();

  void printLogLine();
  void printEvalLine();
  void printEigenValLine();
  std::ofstream* calib_logger_;
  std::ofstream* nums_logger_;
  std::ofstream* errorsLog;
  std::ofstream* eigen_all_logger_;
  std::ofstream* eigen_p_logger_;
  std::ofstream* eigen_a_logger_;
  std::ofstream* diagonal_logger_;
  std::ofstream* variances_logger_;
  std::ofstream* nullspaces_logger_;

  std::ofstream* coarse_tracking_logger_;

  // Statistics.
  long int stats_last_num_opt_iters_;
  long int stats_num_dropped_pts_;
  long int stats_num_activated_pts_;
  long int stats_num_created_pts;
  long int stats_num_force_dropped_res_bwd_;
  long int stats_num_force_dropped_res_fwd_;
  long int stats_num_marg_res_fwd_;
  long int stats_num_marg_res_bwd_;
  float statistics_lastFineTrackRMSE;

  // Variables changed by tracker thread, protected by trackMutex.
  boost::mutex tracking_mutex_;
  std::vector<FrameShell*> frames_; // History of all frames.
  CoarseInitializer* coarse_initializer_;
  Vec5 last_coarse_rmse_;

  // Variables changed by mapping thread, protected by mapMutex.
  boost::mutex mapping_mutex_;
  std::vector<FrameShell*> keyframes_;

  EnergyFunctional* ef_;
  IndexThreadReduce<Vec10> treadReduce;

  float* selection_map_;
  PixelSelector* pixel_selector_;
  CoarseDistanceMap* coarse_distance_map_;

  std::vector<FrameHessian*> hessian_frames_; // ONLY changed in marginalizeFrame and addFrame.
  std::vector<PointFrameResidual*> activeResiduals;
  float curr_min_activation_dist_;

  std::vector<float> allResVec;

  // Variables for tracker exchange, protected by [coarseTrackerSwapMutex].
  boost::mutex coarse_tracker_swap_mutex_; // If tracker sees that there is a new reference, tracker
                                           // locks [coarseTrackerSwapMutex] and swaps the two.
  CoarseTracker* coarse_tracker_for_new_keyframe_; // Set as reference. protected by [coarseTrackerSwapMutex].
  CoarseTracker* coarse_tracker_; // Always used to track new frames and protected by [trackMutex].

  float min_id_jet_vis_tracker_, max_id_jet_vis_tracker_;
  float min_id_jet_vis_, max_id_jet_vis_;

  // Mutex for camToWorld's in shells (these are always in a good configuration).
  boost::mutex frame_pose_mutex_;

  // Tracking always uses the newest KF as reference.
  void makeKeyFrame(FrameHessian* fh);
  void makeNonKeyFrame(FrameHessian* fh);
  void deliverTrackedFrame(FrameHessian* fh, bool needKF);
  void mappingLoop();

  // Tracking / mapping synchronization. All protected by [trackMapSyncMutex].
  boost::mutex tracking_mapping_sync_mutex_;
  boost::condition_variable tracked_frame_signal_;
  boost::condition_variable mapped_frame_signal_;
  std::deque<FrameHessian*> unmapped_tracked_frames_;
  int new_keyframe_id_to_make_later_; // Otherwise, a new keyframe id is stored to
                                      // make it later in case of non linearize operation.
  boost::thread mapping_thread_;
  bool is_mapping_running_;
  bool need_to_catchup_mapping_;

  int last_ref_frame_id_;
};

} // namespace dso
