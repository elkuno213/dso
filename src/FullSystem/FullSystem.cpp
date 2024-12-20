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

#include <algorithm>
#include <cmath>
#include <cstdio>

#include <Eigen/Eigenvalues>
#include <Eigen/LU>
#include <Eigen/SVD>

#include "FullSystem/CoarseInitializer.h"
#include "FullSystem/CoarseTracker.h"
#include "FullSystem/FullSystem.h"
#include "FullSystem/ImmaturePoint.h"
#include "FullSystem/PixelSelector.h"
#include "FullSystem/PixelSelector2.h"
#include "FullSystem/ResidualProjections.h"
#include "IOWrapper/ImageDisplay.h"
#include "IOWrapper/Output3DWrapper.h"
#include "OptimizationBackend/EnergyFunctional.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"
#include "util/ImageAndExposure.h"
#include "util/globalCalib.h"
#include "util/globalFuncs.h"

namespace dso {

int FrameHessian::instanceCounter = 0;
int PointHessian::instanceCounter = 0;
int CalibHessian::instanceCounter = 0;

FullSystem::FullSystem() {
  int status = 0;
  if (setting_logStuff) {
    // Remove and create new log directories.
    status += system("rm -rf logs");
    status += system("mkdir logs");
    status += system("rm -rf mats");
    status += system("mkdir mats");

    // Initialize loggers.
    const auto mode = std::ios::trunc | std::ios::out;

    calib_logger_ = new std::ofstream();
    calib_logger_->open("logs/calib_log.txt", mode);
    calib_logger_->precision(12);

    nums_logger_ = new std::ofstream();
    nums_logger_->open("logs/nums_log.txt", mode);
    nums_logger_->precision(10);

    coarse_tracking_logger_ = new std::ofstream();
    coarse_tracking_logger_->open("logs/coarse_tracking_log.txt", mode);
    coarse_tracking_logger_->precision(10);

    eigen_all_logger_ = new std::ofstream();
    eigen_all_logger_->open("logs/eigen_all_log.txt", mode);
    eigen_all_logger_->precision(10);

    eigen_p_logger_ = new std::ofstream();
    eigen_p_logger_->open("logs/eigen_p_log.txt", mode);
    eigen_p_logger_->precision(10);

    eigen_a_logger_ = new std::ofstream();
    eigen_a_logger_->open("logs/eigen_a_log.txt", mode);
    eigen_a_logger_->precision(10);

    diagonal_logger_ = new std::ofstream();
    diagonal_logger_->open("logs/diagonal_log.txt", mode);
    diagonal_logger_->precision(10);

    variances_logger_ = new std::ofstream();
    variances_logger_->open("logs/variances_log.txt", mode);
    variances_logger_->precision(10);

    nullspaces_logger_ = new std::ofstream();
    nullspaces_logger_->open("logs/nullspaces_log.txt", mode);
    nullspaces_logger_->precision(10);
  } else {
    nullspaces_logger_ = nullptr;
    variances_logger_  = nullptr;
    diagonal_logger_   = nullptr;
    eigen_a_logger_    = nullptr;
    eigen_p_logger_    = nullptr;
    eigen_all_logger_  = nullptr;
    nums_logger_       = nullptr;
    calib_logger_      = nullptr;
  }

  // Check the return status of the system calls.
  assert(status != 293847);

  // Initialize variables.
  selection_map_ = new float[wG[0] * hG[0]];

  coarse_distance_map_       = new CoarseDistanceMap(wG[0], hG[0]);
  coarse_tracker_            = new CoarseTracker    (wG[0], hG[0]);
  coarse_tracker_for_new_kf_ = new CoarseTracker    (wG[0], hG[0]);
  coarse_initializer_        = new CoarseInitializer(wG[0], hG[0]);
  pixel_selector_            = new PixelSelector    (wG[0], hG[0]);

  stats_last_num_opt_iters_        = 0;
  stats_num_dropped_pts_           = 0;
  stats_num_activated_pts_         = 0;
  stats_num_created_pts            = 0;
  stats_num_force_dropped_res_bwd_ = 0;
  stats_num_force_dropped_res_fwd_ = 0;
  stats_num_marg_res_fwd_          = 0;
  stats_num_marg_res_bwd_          = 0;

  last_coarse_rmse_.setConstant(100.0);

  curr_min_activation_dist_ = 2.0f;
  is_initialized_           = false;

  ef_      = new EnergyFunctional();
  ef_->red = &this->treadReduce;

  is_lost_                  = false;
  is_initialization_failed_ = false;

  new_kf_id_to_make_later_ = -1;

  linear_operation_   = true;
  is_mapping_running_ = true;
  mapping_thread_     = boost::thread(&FullSystem::mappingLoop, this);

  last_ref_frame_id_ = 0;

  min_id_jet_vis_         = -1.0f;
  max_id_jet_vis_         = -1.0f;
  min_id_jet_vis_tracker_ = -1.0f;
  max_id_jet_vis_tracker_ = -1.0f;
}

FullSystem::~FullSystem() {
  // Wait for the mapping thread to finish.
  blockUntilMappingIsFinished();

  // Close and delete loggers.
  if (setting_logStuff) {
    calib_logger_->close();
    delete calib_logger_;
    nums_logger_->close();
    delete nums_logger_;
    coarse_tracking_logger_->close();
    delete coarse_tracking_logger_;
    // errorsLog->close(); delete errorsLog;
    eigen_all_logger_->close();
    delete eigen_all_logger_;
    eigen_p_logger_->close();
    delete eigen_p_logger_;
    eigen_a_logger_->close();
    delete eigen_a_logger_;
    diagonal_logger_->close();
    delete diagonal_logger_;
    variances_logger_->close();
    delete variances_logger_;
    nullspaces_logger_->close();
    delete nullspaces_logger_;
  }

  // Delete all other pointers.
  delete[] selection_map_;

  for (FrameShell* f : frames_) {
    delete f;
  }
  for (FrameHessian* f : unmapped_tracked_frames_) {
    delete f;
  }

  delete coarse_distance_map_;
  delete coarse_tracker_;
  delete coarse_tracker_for_new_kf_;
  delete coarse_initializer_;
  delete pixel_selector_;
  delete ef_;
}

void FullSystem::setOriginalCalib(const VecXf& calib, int width, int height) {
}

void FullSystem::setGammaFunction(float* B_inv) {
  // Return if the gamma function is null.
  if (B_inv == nullptr) {
    return;
  }

  // Copy the gamma function.
  memcpy(Hcalib.Binv, B_inv, sizeof(float) * 256);

  // Calculate the inverse gamma function by linear interpolation.
  for (int i = 1; i < 255; i++) {
    for (int s = 1; s < 255; s++) {
      if (B_inv[s] <= static_cast<float>(i) && B_inv[s + 1] >= static_cast<float>(i)) {
        Hcalib.B[i] = s + (i - B_inv[s]) / (B_inv[s + 1] - B_inv[s]);
        break;
      }
    }
  }
  Hcalib.B[0]   = 0.0f;
  Hcalib.B[255] = 255.0f;
}

void FullSystem::printResult(std::string filename) {
  boost::unique_lock<boost::mutex> tracking_lock(tracking_mutex_);
  boost::unique_lock<boost::mutex> frame_pose_lock(frame_pose_mutex_);

  std::ofstream file;
  file.open(filename.c_str());
  file << std::setprecision(15);

  for (FrameShell* f : frames_) {
    // Skip if the frame's pose is not valid.
    if (!f->poseValid) {
      continue;
    }
    // Skip if only the keyframe poses should be logged and the frame is
    // marginalized.
    if (setting_onlyLogKFPoses && f->marginalizedAt == f->id) {
      continue;
    }
    // Write the frame's pose to the file.
    file << f->timestamp                              << " "
         << f->camToWorld.translation().transpose()   << " "
         << f->camToWorld.so3().unit_quaternion().x() << " "
         << f->camToWorld.so3().unit_quaternion().y() << " "
         << f->camToWorld.so3().unit_quaternion().z() << " "
         << f->camToWorld.so3().unit_quaternion().w() << "\n";
  }
  file.close();
}

Vec4 FullSystem::trackNewCoarse(FrameHessian* hessian) {
  // Make sure the frame history is not empty.
  assert(!frames_.empty());

  // Push the frame to the output wrappers for visualization.
  for (IOWrap::Output3DWrapper* wrapper : output_3d_wrappers_) {
    wrapper->pushLiveFrame(hessian);
  }

  // Get last hessian frame from the last reference one of the coarse tracker.
  FrameHessian* last_hessian = coarse_tracker_->lastRef;

  // Initialize the transformation from the last frame to the current frame.
  AffLight last_affine = AffLight(0.0, 0.0);

  // Construct a list of possible transformations from the last to current hessian frames.
  std::vector<SE3, Eigen::aligned_allocator<SE3>> tries;
  if (frames_.size() == 2) {
    // If there are only two frames in the history, push the identity transformation.
    for (std::size_t i = 0; i < tries.size(); i++) {
      tries.push_back(SE3());
    }
  } else {
    // Otherwise, get the last and previous last frames from the history.

    FrameShell* last_shell      = frames_[frames_.size() - 2];
    FrameShell* prev_last_shell = frames_[frames_.size() - 3];
    SE3 last_shell_to_prev_last_shell;
    SE3 last_hessian_to_last_shell;
    // Lock on global pose consistency.
    {
      boost::unique_lock<boost::mutex> frame_pose_lock(frame_pose_mutex_);
      last_shell_to_prev_last_shell = prev_last_shell->camToWorld.inverse() * last_shell->camToWorld;
      last_hessian_to_last_shell = last_shell->camToWorld.inverse() * last_hessian->shell->camToWorld;
      last_affine = last_shell->aff_g2l;
    }
    SE3 curr_hessian_to_last_shell = last_shell_to_prev_last_shell; // Assumed to be the same as fh_2_slast.

    // Get last delta movement.
    tries.push_back(curr_hessian_to_last_shell.inverse() * last_hessian_to_last_shell); // Assume constant motion.
    tries.push_back(curr_hessian_to_last_shell.inverse() * curr_hessian_to_last_shell.inverse() * last_hessian_to_last_shell); // Assume double motion (frame skipped)
    tries.push_back(SE3::exp(curr_hessian_to_last_shell.log() * 0.5).inverse() * last_hessian_to_last_shell); // Assume half motion.
    tries.push_back(last_hessian_to_last_shell); // Assume zero motion.
    tries.push_back(SE3()); // Assume zero motion FROM KF.

    // Just try a TON of different initializations (all rotations). In the end,
    // if they don't work they will only be tried on the coarsest level, which
    // is super fast anyway. Also, if tracking rails here we loose, so we
    // really want to avoid that.
    // TODO(VuHoi): increment doesn't work with float. Correct it.
    for (float d_rot = 0.02; d_rot < 0.05; d_rot++) {
      const SE3 last_hessian_to_curr_hessian
        = curr_hessian_to_last_shell.inverse() * last_hessian_to_last_shell;
      const Vec3 zero_trans = Vec3(0, 0, 0);

      // Assume constant motion in different orientations.
      tries.push_back(last_hessian_to_curr_hessian * SE3(Sophus::Quaterniond(1,  d_rot,      0,      0), zero_trans));
      tries.push_back(last_hessian_to_curr_hessian * SE3(Sophus::Quaterniond(1,      0,  d_rot,      0), zero_trans));
      tries.push_back(last_hessian_to_curr_hessian * SE3(Sophus::Quaterniond(1,      0,      0,  d_rot), zero_trans));
      tries.push_back(last_hessian_to_curr_hessian * SE3(Sophus::Quaterniond(1, -d_rot,      0,      0), zero_trans));
      tries.push_back(last_hessian_to_curr_hessian * SE3(Sophus::Quaterniond(1,      0, -d_rot,      0), zero_trans));
      tries.push_back(last_hessian_to_curr_hessian * SE3(Sophus::Quaterniond(1,      0,      0, -d_rot), zero_trans));
      tries.push_back(last_hessian_to_curr_hessian * SE3(Sophus::Quaterniond(1,  d_rot,  d_rot,      0), zero_trans));
      tries.push_back(last_hessian_to_curr_hessian * SE3(Sophus::Quaterniond(1,      0,  d_rot,  d_rot), zero_trans));
      tries.push_back(last_hessian_to_curr_hessian * SE3(Sophus::Quaterniond(1,  d_rot,      0,  d_rot), zero_trans));
      tries.push_back(last_hessian_to_curr_hessian * SE3(Sophus::Quaterniond(1, -d_rot,  d_rot,      0), zero_trans));
      tries.push_back(last_hessian_to_curr_hessian * SE3(Sophus::Quaterniond(1,      0, -d_rot,  d_rot), zero_trans));
      tries.push_back(last_hessian_to_curr_hessian * SE3(Sophus::Quaterniond(1, -d_rot,      0,  d_rot), zero_trans));
      tries.push_back(last_hessian_to_curr_hessian * SE3(Sophus::Quaterniond(1,  d_rot, -d_rot,      0), zero_trans));
      tries.push_back(last_hessian_to_curr_hessian * SE3(Sophus::Quaterniond(1,      0,  d_rot, -d_rot), zero_trans));
      tries.push_back(last_hessian_to_curr_hessian * SE3(Sophus::Quaterniond(1,  d_rot,      0, -d_rot), zero_trans));
      tries.push_back(last_hessian_to_curr_hessian * SE3(Sophus::Quaterniond(1, -d_rot, -d_rot,      0), zero_trans));
      tries.push_back(last_hessian_to_curr_hessian * SE3(Sophus::Quaterniond(1,      0, -d_rot, -d_rot), zero_trans));
      tries.push_back(last_hessian_to_curr_hessian * SE3(Sophus::Quaterniond(1, -d_rot,      0, -d_rot), zero_trans));
      tries.push_back(last_hessian_to_curr_hessian * SE3(Sophus::Quaterniond(1, -d_rot, -d_rot, -d_rot), zero_trans));
      tries.push_back(last_hessian_to_curr_hessian * SE3(Sophus::Quaterniond(1, -d_rot, -d_rot,  d_rot), zero_trans));
      tries.push_back(last_hessian_to_curr_hessian * SE3(Sophus::Quaterniond(1, -d_rot,  d_rot, -d_rot), zero_trans));
      tries.push_back(last_hessian_to_curr_hessian * SE3(Sophus::Quaterniond(1, -d_rot,  d_rot,  d_rot), zero_trans));
      tries.push_back(last_hessian_to_curr_hessian * SE3(Sophus::Quaterniond(1,  d_rot, -d_rot, -d_rot), zero_trans));
      tries.push_back(last_hessian_to_curr_hessian * SE3(Sophus::Quaterniond(1,  d_rot, -d_rot,  d_rot), zero_trans));
      tries.push_back(last_hessian_to_curr_hessian * SE3(Sophus::Quaterniond(1,  d_rot,  d_rot, -d_rot), zero_trans));
      tries.push_back(last_hessian_to_curr_hessian * SE3(Sophus::Quaterniond(1,  d_rot,  d_rot,  d_rot), zero_trans));
    }

    if (!last_shell->poseValid || !prev_last_shell->poseValid || !last_hessian->shell->poseValid) {
      tries.clear();
      tries.push_back(SE3());
    }
  }

  // Initialize some variables for attempting to find the best transformation.
  Vec3 flow = Vec3(100.0, 100.0, 100.0);
  SE3 last_hessian_to_curr_hessian = SE3();
  AffLight affine = AffLight(0.0, 0.0);

  // As long as maxResForImmediateAccept is not reached, I'll continue through
  // the options. I'll keep track of the so-far best achieved residual for each
  // level in achievedRes. If on a coarse level, tracking is WORSE than
  // achievedRes, we will not continue to save time.

  // Check all tries of the transformations to get the best one.
  Vec5 rmse = Vec5::Constant(NAN); // Contain residuals of good tries.
  bool at_least_one_good_try = false;
  int iters = 0;
  for (std::size_t i = 0; i < tries.size(); i++) {
    // Attempt tracking with the current try of the transformation.
    AffLight affine_try = last_affine;
    SE3 last_hessian_to_curr_hessian_try = tries[i];
    bool is_tracking_good = coarse_tracker_->trackNewestCoarse(
      hessian,
      last_hessian_to_curr_hessian_try,
      affine_try,
      pyrLevelsUsed - 1,
      rmse
    ); // In each level has to be at least as good as the last try.
    iters++;

    // Print the tracking result.
    if (i != 0) {
      printf(
        "RE-TRACK ATTEMPT %zu with initOption %zu and start-lvl %d (ab %f %f): "
        "%f %f %f %f %f -> %f %f %f %f %f \n",
        i,
        i,
        pyrLevelsUsed - 1,
        affine_try.a,
        affine_try.b,
        rmse[0],
        rmse[1],
        rmse[2],
        rmse[3],
        rmse[4],
        coarse_tracker_->lastResiduals[0],
        coarse_tracker_->lastResiduals[1],
        coarse_tracker_->lastResiduals[2],
        coarse_tracker_->lastResiduals[3],
        coarse_tracker_->lastResiduals[4]
      );
    }

    // If the tracking and the RMSE are good, update the best transformation.
    if (
      is_tracking_good &&
      std::isfinite(static_cast<float>(coarse_tracker_->lastResiduals[0])) &&
      !(coarse_tracker_->lastResiduals[0] >= rmse[0])
    ) {
      // printf("take over. minRes %f -> %f!\n", achievedRes[0], coarseTracker->lastResiduals[0]);
      flow = coarse_tracker_->lastFlowIndicators;
      affine = affine_try;
      last_hessian_to_curr_hessian = last_hessian_to_curr_hessian_try;
      at_least_one_good_try = true;
    }

    // Update the achieved residuals if at least one good transformation was found.
    if (at_least_one_good_try) {
      for (int i = 0; i < 5; i++) {
        // Take over if RMSE is either bigger or NAN.
        if (
          !std::isfinite(static_cast<float>(rmse[i])) ||
          rmse[i] > coarse_tracker_->lastResiduals[i]
        ) {
          rmse[i] = coarse_tracker_->lastResiduals[i];
        }
      }
    }

    // If a good transformation was found and the residual is below the threshold, break.
    if (at_least_one_good_try && rmse[0] < last_coarse_rmse_[0] * setting_reTrackThreshold) {
      break;
    }
  }

  // If no good transformation was found, print an error message and take the
  // last transformation and affine.
  if (!at_least_one_good_try) {
    printf(
      "BIG ERROR! tracking failed entirely. Take predictred pose and hope we "
      "may somehow recover.\n"
    );
    flow = Vec3(0.0, 0.0, 0.0);
    affine = last_affine;
    last_hessian_to_curr_hessian = tries[0];
  }

  // Update the cached coarse RMSE.
  last_coarse_rmse_ = rmse;

  // Update the frame with the best transformation. No lock required, as hessian
  // frame is not used anywhere yet.
  hessian->shell->camToTrackingRef = last_hessian_to_curr_hessian.inverse();
  hessian->shell->trackingRef      = last_hessian->shell;
  hessian->shell->aff_g2l          = affine;
  hessian->shell->camToWorld       = hessian->shell->trackingRef->camToWorld * hessian->shell->camToTrackingRef;

  // Update the first coarse RMSE.
  if (coarse_tracker_->firstCoarseRMSE < 0.0) {
    coarse_tracker_->firstCoarseRMSE = rmse[0];
  }

  if (!setting_debugout_runquiet) {
    printf(
      "Coarse Tracker tracked ab = %f %f (exp %f). Res %f!\n",
      affine.a,
      affine.b,
      hessian->ab_exposure,
      rmse[0]
    );
  }

  // Log the tracking result.
  if (setting_logStuff) {
    (*coarse_tracking_logger_)
      << std::setprecision(16)
      << hessian->shell->id                           << " "
      << hessian->shell->timestamp                    << " "
      << hessian->ab_exposure                         << " "
      << hessian->shell->camToWorld.log().transpose() << " "
      << affine.a                                     << " "
      << affine.b                                     << " "
      << rmse[0]                                      << " "
      << iters                                        << "\n";
  }

  return Vec4(rmse[0], flow[0], flow[1], flow[2]);
}

void FullSystem::traceNewCoarse(FrameHessian* hessian) {
  boost::unique_lock<boost::mutex> lock(mapping_mutex_);

  // Initialize some tracing counters.
  int total = 0, good = 0, oob = 0, out = 0, skip = 0, bad_condition = 0, uninitialized = 0;

  Mat33f K = Mat33f::Identity();
  K(0, 0)  = Hcalib.fxl();
  K(1, 1)  = Hcalib.fyl();
  K(0, 2)  = Hcalib.cxl();
  K(1, 2)  = Hcalib.cyl();

  // Go through all active frames and trace the points.
  for (FrameHessian* host : hessian_frames_) {
    SE3 host_to_new = hessian->PRE_worldToCam * host->PRE_camToWorld;
    Mat33f KRKi     = K * host_to_new.rotationMatrix().cast<float>() * K.inverse();
    Vec3f Kt        = K * host_to_new.translation().cast<float>();

    Vec2f affine = AffLight::fromToVecExposure(
      host->ab_exposure,
      hessian->ab_exposure,
      host->aff_g2l(),
      hessian->aff_g2l()
    ).cast<float>();

    for (ImmaturePoint* pt : host->immaturePoints) {
      pt->traceOn(hessian, KRKi, Kt, affine, &Hcalib, false);

      if (pt->lastTraceStatus == ImmaturePointStatus::IPS_GOOD) {
        good++;
      }
      if (pt->lastTraceStatus == ImmaturePointStatus::IPS_BADCONDITION) {
        bad_condition++;
      }
      if (pt->lastTraceStatus == ImmaturePointStatus::IPS_OOB) {
        oob++;
      }
      if (pt->lastTraceStatus == ImmaturePointStatus::IPS_OUTLIER) {
        out++;
      }
      if (pt->lastTraceStatus == ImmaturePointStatus::IPS_SKIPPED) {
        skip++;
      }
      if (pt->lastTraceStatus == ImmaturePointStatus::IPS_UNINITIALIZED) {
        uninitialized++;
      }
      total++;
    }
  }
  // printf(
  //   "ADD: TRACE: %'d points. %'d (%.0f%%) good. %'d (%.0f%%) skip. % 'd "
  //   "(%.0f%%) badcond. %' d(% .0f % %) oob. % 'd (%.0f%%) out. %' d(% .0f % %) "
  //   "uninit.\n",
  //   trace_total,
  //   trace_good,
  //   100 * trace_good / (float)trace_total,
  //   trace_skip,
  //   100 * trace_skip / (float)trace_total,
  //   trace_badcondition,
  //   100 * trace_badcondition / (float)trace_total,
  //   trace_oob,
  //   100 * trace_oob / (float)trace_total,
  //   trace_out,
  //   100 * trace_out / (float)trace_total,
  //   trace_uninitialized,
  //   100 * trace_uninitialized / (float)trace_total
  // );
}

void FullSystem::activatePointsMT_Reductor(
  std::vector<PointHessian*>* optimized,
  std::vector<ImmaturePoint*>* toOptimize,
  int min,
  int max,
  Vec10* stats,
  int tid
) {
  ImmaturePointTemporaryResidual* tr
    = new ImmaturePointTemporaryResidual[hessian_frames_.size()];
  for (int k = min; k < max; k++) {
    (*optimized)[k] = optimizeImmaturePoint((*toOptimize)[k], 1, tr);
  }
  delete[] tr;
}

void FullSystem::activatePointsMT() {
  // Update the minimum activation distance.
  if (static_cast<float>(ef_->nPoints) < setting_desiredPointDensity * 0.66f) {
    curr_min_activation_dist_ -= 0.8f;
  }
  if (static_cast<float>(ef_->nPoints) < setting_desiredPointDensity * 0.8f) {
    curr_min_activation_dist_ -= 0.5f;
  } else if (static_cast<float>(ef_->nPoints) < setting_desiredPointDensity * 0.9f) {
    curr_min_activation_dist_ -= 0.2f;
  } else if (static_cast<float>(ef_->nPoints) < setting_desiredPointDensity) {
    curr_min_activation_dist_ -= 0.1f;
  }

  if (static_cast<float>(ef_->nPoints) > setting_desiredPointDensity * 1.5f) {
    curr_min_activation_dist_ += 0.8f;
  }
  if (static_cast<float>(ef_->nPoints) > setting_desiredPointDensity * 1.3f) {
    curr_min_activation_dist_ += 0.5f;
  }
  if (static_cast<float>(ef_->nPoints) > setting_desiredPointDensity * 1.15f) {
    curr_min_activation_dist_ += 0.2f;
  }
  if (static_cast<float>(ef_->nPoints) > setting_desiredPointDensity) {
    curr_min_activation_dist_ += 0.1f;
  }

  if (curr_min_activation_dist_ < 0.0f) {
    curr_min_activation_dist_ = 0.0f;
  }
  if (curr_min_activation_dist_ > 4.0f) {
    curr_min_activation_dist_ = 4.0f;
  }

  // Print messages if debug enabled.
  if (!setting_debugout_runquiet) {
    printf(
      "SPARSITY:  MinActDist %f (need %d points, have %d points)!\n",
      curr_min_activation_dist_,
      (int)(setting_desiredPointDensity),
      ef_->nPoints
    );
  }

  // Get the newest frame.
  FrameHessian* newestHs = hessian_frames_.back();

  // Make distance map.
  coarse_distance_map_->makeK(&Hcalib);
  coarse_distance_map_->makeDistanceMap(hessian_frames_, newestHs);

  // coarseTracker->debugPlotDistMap("distMap");

  std::vector<ImmaturePoint*> toOptimize;
  toOptimize.reserve(20000);

  // Go through all active frames and add points to the optimization.
  for (FrameHessian* host : hessian_frames_) {
    // Skip the newest frame.
    if (host == newestHs) {
      continue;
    }

    SE3 fhToNew = newestHs->PRE_worldToCam * host->PRE_camToWorld;
    Mat33f KRKi = (coarse_distance_map_->K[1] * fhToNew.rotationMatrix().cast<float>() * coarse_distance_map_->Ki[0]);
    Vec3f Kt = (coarse_distance_map_->K[1] * fhToNew.translation().cast<float>());

    for (std::size_t i = 0; i < host->immaturePoints.size(); i++) {
      ImmaturePoint* ph       = host->immaturePoints[i];
      ph->idxInImmaturePoints = static_cast<int>(i);

      // Delete points that have never been traced successfully, or that are
      // outlier on the last trace.
      if (!std::isfinite(ph->idepth_max) || ph->lastTraceStatus == IPS_OUTLIER) {
        // immature_invalid_deleted++;

        delete ph;
        host->immaturePoints[i] = nullptr;
        continue;
      }

      // Otherwise, see if we can activate the point.
      bool canActivate = (
        ph->lastTraceStatus == IPS_GOOD         ||
        ph->lastTraceStatus == IPS_SKIPPED      ||
        ph->lastTraceStatus == IPS_BADCONDITION ||
        ph->lastTraceStatus == IPS_OOB
      ) &&  ph->lastTracePixelInterval < 8.0f
        &&  ph->quality > setting_minTraceQuality
        && (ph->idepth_max + ph->idepth_min) > 0.0f;

      // If I cannot activate the point, skip it. Maybe also delete it if it
      // will be out afterwards.
      if (!canActivate) {
        if (ph->host->flaggedForMarginalization || ph->lastTraceStatus == IPS_OOB) {
          // immature_notReady_deleted++;
          delete ph;
          host->immaturePoints[i] = nullptr;
        }
        // immature_notReady_skipped++;
        continue;
      }

      // See if we need to activate point due to distance map.
      Vec3f ptp = KRKi * Vec3f(ph->u, ph->v, 1) + Kt * (0.5f * (ph->idepth_max + ph->idepth_min));
      int u = static_cast<int>(ptp[0] / ptp[2] + 0.5f);
      int v = static_cast<int>(ptp[1] / ptp[2] + 0.5f);

      if ((u > 0 && v > 0 && u < wG[1] && v < hG[1])) {
        float dist = coarse_distance_map_->fwdWarpedIDDistFinal[u + wG[1] * v]
                   + (ptp[0] - floorf((float)(ptp[0])));

        if (dist >= curr_min_activation_dist_ * ph->my_type) {
          coarse_distance_map_->addIntoDistFinal(u, v);
          toOptimize.push_back(ph);
        }
      } else {
        delete ph;
        host->immaturePoints[i] = nullptr;
      }
    }
  }

  // printf("ACTIVATE: %d. (del %d, notReady %d, marg %d, good %d, marg-skip
  // %d)\n", (int)toOptimize.size(), immature_deleted, immature_notReady,
  // immature_needMarg, immature_want, immature_margskip);

  // Optimize the points.
  std::vector<PointHessian*> optimized;
  optimized.resize(toOptimize.size());

  if (multiThreading) {
    treadReduce.reduce(
      boost::bind(
        &FullSystem::activatePointsMT_Reductor,
        this,
        &optimized,
        &toOptimize,
        _1,
        _2,
        _3,
        _4
      ),
      0,
      toOptimize.size(),
      50
    );
  } else {
    activatePointsMT_Reductor(&optimized, &toOptimize, 0, toOptimize.size(), 0, 0);
  }

  // Insert the optimized points into the energy functional and delete the
  // points at the end of tracking and to be marginalized.
  for (std::size_t k = 0; k < toOptimize.size(); k++) {
    PointHessian* newpoint = optimized[k];
    ImmaturePoint* ph      = toOptimize[k];

    // TODO(VuHoi): improve 2nd check: newpoint != (PointHessian*)((long)(-1))
    if (newpoint != nullptr && newpoint != (PointHessian*)((long)(-1))) {
      newpoint->host->immaturePoints[ph->idxInImmaturePoints] = nullptr;
      newpoint->host->pointHessians.push_back(newpoint);
      ef_->insertPoint(newpoint);
      for (PointFrameResidual* r : newpoint->residuals) ef_->insertResidual(r);
      assert(newpoint->efPoint != 0);
      delete ph;
    } else if (newpoint == (PointHessian*)((long)(-1)) || ph->lastTraceStatus == IPS_OOB) {
      delete ph;
      ph->host->immaturePoints[ph->idxInImmaturePoints] = nullptr;
    } else {
      assert(newpoint == nullptr || newpoint == (PointHessian*)((long)(-1)));
    }
  }

  // Remove the points that were not activated.
  for (FrameHessian* host : hessian_frames_) {
    for (std::size_t i = 0; i < host->immaturePoints.size(); i++) {
      if (host->immaturePoints[i] == nullptr) {
        host->immaturePoints[i] = host->immaturePoints.back();
        host->immaturePoints.pop_back();
        i--;
      }
    }
  }
}

void FullSystem::activatePointsOldFirst() {
  assert(false);
}

void FullSystem::flagPointsForRemoval() {
  assert(EFIndicesValid);

  std::vector<FrameHessian*> fhsToKeepPoints;
  std::vector<FrameHessian*> fhsToMargPoints;

  // Go through all active frames and flag the points to be kept and marginalized.
  // if(setting_margPointVisWindow>0)
  {
    for (
      std::size_t i = hessian_frames_.size() - 1;
      i >= 0 && i >= hessian_frames_.size(); // TODO(VuHoi): check this condition.
      i--
    ) {
      if (!hessian_frames_[i]->flaggedForMarginalization) {
        fhsToKeepPoints.push_back(hessian_frames_[i]);
      }
    }

    for (std::size_t i = 0; i < hessian_frames_.size(); i++) {
      if (hessian_frames_[i]->flaggedForMarginalization) {
        fhsToMargPoints.push_back(hessian_frames_[i]);
      }
    }
  }

  // ef->setAdjointsF();
  // ef->setDeltaF(&Hcalib);
  int flag_oob = 0, flag_in = 0, flag_inin = 0, flag_nores = 0;

  // Go through all active frames and update the points to be kept and marginalized.
  for (FrameHessian* host : hessian_frames_) {
    for (std::size_t i = 0; i < host->pointHessians.size(); i++) {
      PointHessian* ph = host->pointHessians[i];
      if (ph == nullptr) {
        continue;
      }

      if (ph->idepth_scaled < 0.0f || ph->residuals.empty()) {
        host->pointHessiansOut.push_back(ph);
        ph->efPoint->stateFlag = EFPointStatus::PS_DROP;
        host->pointHessians[i] = 0;
        flag_nores++;
      } else if (ph->isOOB(fhsToKeepPoints, fhsToMargPoints) || host->flaggedForMarginalization) {
        flag_oob++;
        if (ph->isInlierNew()) {
          flag_in++;
          int ngoodRes = 0;
          for (PointFrameResidual* r : ph->residuals) {
            r->resetOOB();
            r->linearize(&Hcalib);
            r->efResidual->isLinearized = false;
            r->applyRes(true);
            if (r->efResidual->isActive()) {
              r->efResidual->fixLinearizationF(ef_);
              ngoodRes++;
            }
          }
          if (ph->idepth_hessian > setting_minIdepthH_marg) {
            flag_inin++;
            ph->efPoint->stateFlag = EFPointStatus::PS_MARGINALIZE;
            host->pointHessiansMarginalized.push_back(ph);
          } else {
            ph->efPoint->stateFlag = EFPointStatus::PS_DROP;
            host->pointHessiansOut.push_back(ph);
          }

        } else {
          host->pointHessiansOut.push_back(ph);
          ph->efPoint->stateFlag = EFPointStatus::PS_DROP;

          // printf("drop point in frame %d (%d goodRes, %d activeRes)\n",
          // ph->host->idx, ph->numGoodResiduals, (int)ph->residuals.size());
        }

        host->pointHessians[i] = nullptr;
      }
    }

    for (int i = 0; i < static_cast<int>(host->pointHessians.size()); i++) {
      if (host->pointHessians[i] == nullptr) {
        host->pointHessians[i] = host->pointHessians.back();
        host->pointHessians.pop_back();
        i--;
      }
    }
  }
}

void FullSystem::addActiveFrame(ImageAndExposure* image, int id) {
  // Skip if the tracking is lost.
  if (is_lost_) {
    return;
  }

  boost::unique_lock<boost::mutex> lock(tracking_mutex_);

  // Add frame to the history.
  FrameHessian* fh  = new FrameHessian();
  FrameShell* shell = new FrameShell();
  shell->camToWorld = SE3(); // No lock required, as fh is not used anywhere yet.
  shell->aff_g2l    = AffLight(0.0, 0.0);
  shell->marginalizedAt = shell->id = frames_.size();
  shell->timestamp                  = image->timestamp;
  shell->incoming_id                = id;
  fh->shell                         = shell;
  frames_.push_back(shell);

  // Set the image and exposure.
  fh->ab_exposure = image->exposure_time;
  fh->makeImages(image->image, &Hcalib);

  if (!is_initialized_) {
    // Initialize if not.
    if (coarse_initializer_->frameID < 0) {
      // If no frame is set, first frame is set and fh is kept by
      // coarseInitializer.
      coarse_initializer_->setFirst(&Hcalib, fh);
    } else if (coarse_initializer_->trackFrame(fh, output_3d_wrappers_)) {
      // Otherwise, if the frame is tracked, the frame is snapped. So initialize
      // the system.
      initializeFromInitializer(fh);
      lock.unlock();
      deliverTrackedFrame(fh, true);
    } else {
      // Otherwise, the frame's pose is not valid, so delete the frame and
      // return.
      fh->shell->poseValid = false;
      delete fh;
    }
    return;
  } else {
    // Initialized, so let's do front-end operation.

    // Swap the coarse tracker if refence frame is newer.
    if (coarse_tracker_for_new_kf_->refFrameID > coarse_tracker_->refFrameID) {
      boost::unique_lock<boost::mutex> crlock(coarseTrackerSwapMutex);
      CoarseTracker* tmp     = coarse_tracker_;
      coarse_tracker_          = coarse_tracker_for_new_kf_;
      coarse_tracker_for_new_kf_ = tmp;
    }

    // Check if the tracking is lost.
    Vec4 tres = trackNewCoarse(fh);
    if (!std::isfinite(tres[0]) || !std::isfinite(tres[1]) || !std::isfinite(tres[2]) || !std::isfinite(tres[3])) {
      printf("Initial Tracking failed: LOST!\n");
      is_lost_ = true;
      return;
    }

    // Flag if the keyframe is needed.
    bool needToMakeKF = false;
    if (setting_keyframesPerSecond > 0.0f) {
      needToMakeKF
        = frames_.size() == 1
       || (fh->shell->timestamp - allKeyFramesHistory.back()->timestamp) > 0.95f / setting_keyframesPerSecond;
    } else {
      Vec2 refToFh = AffLight::fromToVecExposure(
        coarse_tracker_->lastRef->ab_exposure,
        fh->ab_exposure,
        coarse_tracker_->lastRef_aff_g2l,
        fh->shell->aff_g2l
      );

      // Check brightness.
      needToMakeKF
        = frames_.size() == 1
       || setting_kfGlobalWeight * setting_maxShiftWeightT  * std::sqrt(tres[1]) / (wG[0] + hG[0])
        + setting_kfGlobalWeight * setting_maxShiftWeightR  * std::sqrt(tres[2]) / (wG[0] + hG[0])
        + setting_kfGlobalWeight * setting_maxShiftWeightRT * std::sqrt(tres[3]) / (wG[0] + hG[0])
        + setting_kfGlobalWeight * setting_maxAffineWeight  * std::abs(std::log(refToFh[0])) > 1.0f
       || 2.0 * coarse_tracker_->firstCoarseRMSE < tres[0];
    }

    // Publish the new frame to visualization wrapper.
    for (IOWrap::Output3DWrapper* ow : output_3d_wrappers_) {
      ow->publishCamPose(fh->shell, &Hcalib);
    }

    // Deliver the tracked frame.
    lock.unlock();
    deliverTrackedFrame(fh, needToMakeKF);
    return;
  }
}

void FullSystem::deliverTrackedFrame(FrameHessian* fh, bool needKF) {
  // Linearize operation = sequentialize the tracking and mapping.
  // If enabled, add the tracked frame to visualization wrapper and make
  // keyframe if needed.
  if (linear_operation_) {
    if (goStepByStep && last_ref_frame_id_ != coarse_tracker_->refFrameID) {
      MinimalImageF3 img(wG[0], hG[0], fh->dI);
      IOWrap::displayImage("frameToTrack", &img);
      while (true) {
        char k = IOWrap::waitKey(0);
        if (k == ' ') {
          break;
        }
        handleKey(k);
      }
      last_ref_frame_id_ = coarse_tracker_->refFrameID;
    } else {
      handleKey(IOWrap::waitKey(1));
    }

    if (needKF) {
      makeKeyFrame(fh);
    } else {
      makeNonKeyFrame(fh);
    }
  } else {
    // Otherwise, add the frame to the unmapped tracked queue if needed.
    boost::unique_lock<boost::mutex> lock(trackMapSyncMutex);
    unmapped_tracked_frames_.push_back(fh);
    if (needKF) {
      new_kf_id_to_make_later_ = fh->shell->trackingRef->id;
    }
    trackedFrameSignal.notify_all();

    while (
      coarse_tracker_for_new_kf_->refFrameID == -1 &&
      coarse_tracker_->refFrameID == -1
    ) {
      mappedFrameSignal.wait(lock);
    }

    lock.unlock();
  }
}

void FullSystem::mappingLoop() {
  boost::unique_lock<boost::mutex> lock(trackMapSyncMutex);

  while (is_mapping_running_) {
    // No tracked frame to map, so wait for the signal and return if the mapping
    // is not running.
    while (unmapped_tracked_frames_.empty()) {
      trackedFrameSignal.wait(lock);
      if (!is_mapping_running_) {
        return;
      }
    }

    // Get the oldest tracked frame.
    FrameHessian* fh = unmapped_tracked_frames_.front();
    unmapped_tracked_frames_.pop_front();

    // Make sure to make a keyframe for the very first 2 tracked frames.
    if (allKeyFramesHistory.size() <= 2) {
      lock.unlock();
      makeKeyFrame(fh);
      lock.lock();
      mappedFrameSignal.notify_all();
      continue;
    }

    // If there are more than 3 frames to track, we need to catch up with the mapping.
    if (unmapped_tracked_frames_.size() > 3) {
      needToKetchupMapping = true;
    }

    if (!unmapped_tracked_frames_.empty()) {
      // If there are still other unmapped tracked frames, make the current
      // frame as non-keyframe.
      lock.unlock();
      makeNonKeyFrame(fh);
      lock.lock();

      if (needToKetchupMapping && !unmapped_tracked_frames_.empty()) {
        FrameHessian* fh = unmapped_tracked_frames_.front();
        unmapped_tracked_frames_.pop_front();
        {
          boost::unique_lock<boost::mutex> crlock(frame_pose_mutex_);
          assert(fh->shell->trackingRef != nullptr);
          fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;
          fh->setEvalPT_scaled(fh->shell->camToWorld.inverse(), fh->shell->aff_g2l);
        }
        delete fh;
      }

    } else {
      // Otherwise, make the tracked frame as keyframe if possible, otherwise as non-keyframe.
      if (setting_realTimeMaxKF || new_kf_id_to_make_later_ >= hessian_frames_.back()->shell->id) {
        lock.unlock();
        makeKeyFrame(fh);
        needToKetchupMapping = false;
        lock.lock();
      } else {
        lock.unlock();
        makeNonKeyFrame(fh);
        lock.lock();
      }
    }
    mappedFrameSignal.notify_all();
  }
  printf("MAPPING FINISHED!\n");
}

void FullSystem::blockUntilMappingIsFinished() {
  boost::unique_lock<boost::mutex> lock(trackMapSyncMutex);
  is_mapping_running_ = false;
  trackedFrameSignal.notify_all();
  lock.unlock();

  mapping_thread_.join();
  mapping_thread_.join();

  mapping_thread_.join();

}

void FullSystem::makeNonKeyFrame(FrameHessian* fh) {
  // This function needs to be called by the mapping thread only, so no lock is required.

  // Update the frame's pose, perform tracing and delete the frame.
  {
    boost::unique_lock<boost::mutex> crlock(frame_pose_mutex_);
    assert(fh->shell->trackingRef != nullptr);
    fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;
    fh->setEvalPT_scaled(fh->shell->camToWorld.inverse(), fh->shell->aff_g2l);
  }
  traceNewCoarse(fh);
  delete fh;
}

void FullSystem::makeKeyFrame(FrameHessian* fh) {
  // This function needs to be called by the mapping thread.

  // Update the frame's pose and perform tracing.
  {
    boost::unique_lock<boost::mutex> crlock(frame_pose_mutex_);
    assert(fh->shell->trackingRef != nullptr);
    fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;
    fh->setEvalPT_scaled(fh->shell->camToWorld.inverse(), fh->shell->aff_g2l);
  }
  traceNewCoarse(fh);

  // Lock to make the keyframe.
  boost::unique_lock<boost::mutex> lock(mapping_mutex_);

  // Flag the frames to be marginalized.
  flagFramesForMarginalization(fh);

  // Add the frame to the energy functional.
  fh->idx = hessian_frames_.size();
  hessian_frames_.push_back(fh);
  fh->frameID = allKeyFramesHistory.size();
  allKeyFramesHistory.push_back(fh->shell);
  ef_->insertFrame(fh, &Hcalib);

  setPrecalcValues();

  // Add new residuals for old points.
  int numFwdResAdde = 0;
  for (FrameHessian* fh1 : hessian_frames_) {
    if (fh1 == fh) {
      continue;
    }
    for (PointHessian* ph : fh1->pointHessians) {
      PointFrameResidual* r = new PointFrameResidual(ph, fh1, fh);
      r->setState(ResState::IN);
      ph->residuals.push_back(r);
      ef_->insertResidual(r);
      ph->lastResiduals[1] = ph->lastResiduals[0];
      ph->lastResiduals[0]
        = std::pair<PointFrameResidual*, ResState>(r, ResState::IN);
      numFwdResAdde += 1;
    }
  }

  // Activate points and flag for marginalization.
  activatePointsMT();
  ef_->makeIDX();

  // Optimize all.
  fh->frameEnergyTH = hessian_frames_.back()->frameEnergyTH;
  float rmse        = optimize(setting_maxOptIterations);

  // Check if the initialization failed.
  if (allKeyFramesHistory.size() <= 4) {
    if (allKeyFramesHistory.size() == 2 && rmse > 20.0f * benchmark_initializerSlackFactor) {
      printf("I THINK INITIALIZATINO FAILED! Resetting.\n");
      is_initialization_failed_ = true;
    }
    if (allKeyFramesHistory.size() == 3 && rmse > 13.0f * benchmark_initializerSlackFactor) {
      printf("I THINK INITIALIZATINO FAILED! Resetting.\n");
      is_initialization_failed_ = true;
    }
    if (allKeyFramesHistory.size() == 4 && rmse > 9.0f * benchmark_initializerSlackFactor) {
      printf("I THINK INITIALIZATINO FAILED! Resetting.\n");
      is_initialization_failed_ = true;
    }
  }

  // Return if the tracking is lost.
  if (is_lost_) {
    return;
  }

  // Remove outliers.
  removeOutliers();

  {
    boost::unique_lock<boost::mutex> crlock(coarseTrackerSwapMutex);
    coarse_tracker_for_new_kf_->makeK(&Hcalib);
    coarse_tracker_for_new_kf_->setCoarseTrackingRef(hessian_frames_);

    coarse_tracker_for_new_kf_->debugPlotIDepthMap(
      &min_id_jet_vis_tracker_,
      &max_id_jet_vis_tracker_,
      output_3d_wrappers_
    );
    coarse_tracker_for_new_kf_->debugPlotIDepthMapFloat(output_3d_wrappers_);
  }

  debugPlot("post Optimize");

  // Marginalize points.
  flagPointsForRemoval();
  ef_->dropPointsF();
  getNullspaces(
    ef_->lastNullspaces_pose,
    ef_->lastNullspaces_scale,
    ef_->lastNullspaces_affA,
    ef_->lastNullspaces_affB
  );
  ef_->marginalizePointsF();

  // Add new immature points and residuals.
  makeNewTraces(fh, nullptr);

  // Publish the keyframe to visualization wrapper.
  for (IOWrap::Output3DWrapper* ow : output_3d_wrappers_) {
    ow->publishGraph(ef_->connectivityMap);
    ow->publishKeyframes(hessian_frames_, false, &Hcalib);
  }

  // Marginalize frames.
  for (std::size_t i = 0; i < hessian_frames_.size(); i++) {
    if (hessian_frames_[i]->flaggedForMarginalization) {
      marginalizeFrame(hessian_frames_[i]);
      i = 0;
    }
  }

  printLogLine();
  // printEigenValLine();
}

void FullSystem::initializeFromInitializer(FrameHessian* newFrame) {
  boost::unique_lock<boost::mutex> lock(mapping_mutex_);

  // Add first frame.
  FrameHessian* firstFrame = coarse_initializer_->firstFrame;
  firstFrame->idx          = hessian_frames_.size();
  hessian_frames_.push_back(firstFrame);
  firstFrame->frameID = allKeyFramesHistory.size();
  allKeyFramesHistory.push_back(firstFrame->shell);
  ef_->insertFrame(firstFrame, &Hcalib);
  setPrecalcValues();

  // int numPointsTotal = makePixelStatus(firstFrame->dI, selectionMap, wG[0], hG[0], setting_desiredDensity);
  // int numPointsTotal = pixelSelector->makeMaps(firstFrame->dIp, selectionMap,setting_desiredDensity);

  firstFrame->pointHessians.reserve(wG[0] * hG[0] * 0.2f);
  firstFrame->pointHessiansMarginalized.reserve(wG[0] * hG[0] * 0.2f);
  firstFrame->pointHessiansOut.reserve(wG[0] * hG[0] * 0.2f);

  // Calculate the rescale factor.
  float sumID = 1e-5f, numID = 1e-5f;
  for (int i = 0; i < coarse_initializer_->numPoints[0]; i++) {
    sumID += coarse_initializer_->points[0][i].iR;
    numID++;
  }
  float rescaleFactor = 1.0f / (sumID / numID);

  // Randomly sub-select the points.
  float keepPercentage = setting_desiredPointDensity / coarse_initializer_->numPoints[0];

  if (!setting_debugout_runquiet) {
    printf(
      "Initialization: keep %.1f%% (need %d, have %d)!\n",
      100.0f * keepPercentage,
      static_cast<int>(setting_desiredPointDensity),
      coarse_initializer_->numPoints[0]
    );
  }

  for (int i = 0; i < coarse_initializer_->numPoints[0]; i++) {
    if (rand() / static_cast<float>(RAND_MAX) > keepPercentage) {
      continue;
    }

    Pnt* point        = coarse_initializer_->points[0] + i;
    ImmaturePoint* pt = new ImmaturePoint(
      point->u + 0.5f,
      point->v + 0.5f,
      firstFrame,
      point->my_type,
      &Hcalib
    );

    if (!std::isfinite(pt->energyTH)) {
      delete pt;
      continue;
    }

    pt->idepth_max = pt->idepth_min = 1.0f;
    PointHessian* ph = new PointHessian(pt, &Hcalib);
    delete pt;
    if (!std::isfinite(ph->energyTH)) {
      delete ph;
      continue;
    }

    ph->setIdepthScaled(point->iR * rescaleFactor);
    ph->setIdepthZero(ph->idepth);
    ph->hasDepthPrior = true;
    ph->setPointStatus(PointHessian::ACTIVE);

    firstFrame->pointHessians.push_back(ph);
    ef_->insertPoint(ph);
  }

  SE3 firstToNew = coarse_initializer_->thisToNext;
  firstToNew.translation() /= rescaleFactor;

  // Initialize (no lock required, as we are initializing).
  {
    boost::unique_lock<boost::mutex> crlock(frame_pose_mutex_);
    firstFrame->shell->camToWorld = SE3();
    firstFrame->shell->aff_g2l    = AffLight(0.0, 0.0);
    firstFrame->setEvalPT_scaled(
      firstFrame->shell->camToWorld.inverse(),
      firstFrame->shell->aff_g2l
    );
    firstFrame->shell->trackingRef      = nullptr;
    firstFrame->shell->camToTrackingRef = SE3();

    newFrame->shell->camToWorld = firstToNew.inverse();
    newFrame->shell->aff_g2l    = AffLight(0.0, 0.0);
    newFrame->setEvalPT_scaled(
      newFrame->shell->camToWorld.inverse(),
      newFrame->shell->aff_g2l
    );
    newFrame->shell->trackingRef      = firstFrame->shell;
    newFrame->shell->camToTrackingRef = firstToNew.inverse();
  }

  is_initialized_ = true;
  printf(
    "INITIALIZE FROM INITIALIZER (%d pts)!\n",
    (int)firstFrame->pointHessians.size()
  );
}

void FullSystem::makeNewTraces(FrameHessian* newFrame, float* gtDepth) {
  // Initialize new frame.
  pixel_selector_->allowFast = true;
  // int numPointsTotal       = makePixelStatus(
  //   newFrame->dI,
  //   selectionMap,
  //   wG[0],
  //   hG[0],
  //   setting_desiredDensity
  // );
  int numPointsTotal = pixel_selector_->makeMaps(
    newFrame,
    selection_map_,
    setting_desiredImmatureDensity
  );

  newFrame->pointHessians.reserve(static_cast<std::size_t>(numPointsTotal * 1.2f));
  // fh->pointHessiansInactive.reserve(static_cast<std::size_t>(numPointsTotal * 1.2f));
  newFrame->pointHessiansMarginalized.reserve(static_cast<std::size_t>(numPointsTotal * 1.2f));
  newFrame->pointHessiansOut.reserve(static_cast<std::size_t>(numPointsTotal * 1.2f));

  // Add immature points to the new frame if its energy is finite.
  for (int y = patternPadding + 1; y < hG[0] - patternPadding - 2; y++) {
    for (int x = patternPadding + 1; x < wG[0] - patternPadding - 2; x++) {
      int i = x + y * wG[0];
      if (selection_map_[i] == 0.0f) {
        continue;
      }

      ImmaturePoint* impt
        = new ImmaturePoint(x, y, newFrame, selection_map_[i], &Hcalib);
      if (!std::isfinite(impt->energyTH)) {
        delete impt;
      } else {
        newFrame->immaturePoints.push_back(impt);
      }
    }
  }
  // printf("MADE %d IMMATURE POINTS!\n", (int)newFrame->immaturePoints.size());
}

void FullSystem::setPrecalcValues() {
  for (FrameHessian* fh : hessian_frames_) {
    fh->targetPrecalc.resize(hessian_frames_.size());
    for (std::size_t i = 0; i < hessian_frames_.size(); i++) {
      fh->targetPrecalc[i].set(fh, hessian_frames_[i], &Hcalib);
    }
  }

  ef_->setDeltaF(&Hcalib);
}

void FullSystem::printLogLine() {
  // Return if no keyframes are available.
  if (hessian_frames_.empty()) {
    return;
  }

  // Print the log line if quiet mode is disabled.
  if (!setting_debugout_runquiet) {
    printf(
      "LOG %d: %.3f fine. Res: %d A, %d L, %d M; (%'d / %'d) forceDrop. a=%f, "
      "b=%f. Window %d (%d)\n",
      allKeyFramesHistory.back()->id,
      statistics_lastFineTrackRMSE,
      ef_->resInA,
      ef_->resInL,
      ef_->resInM,
      (int)stats_num_force_dropped_res_fwd_,
      (int)stats_num_force_dropped_res_bwd_,
      allKeyFramesHistory.back()->aff_g2l.a,
      allKeyFramesHistory.back()->aff_g2l.b,
      hessian_frames_.back()->shell->id - hessian_frames_.front()->shell->id,
      (int)hessian_frames_.size()
    );
  }

  // Return if logging is disabled.
  if (!setting_logStuff) {
    return;
  }

  // Log.
  if (nums_logger_ != nullptr) {
    (*nums_logger_)
      << allKeyFramesHistory.back()->id     << " "
      << statistics_lastFineTrackRMSE       << " "
      << (int)stats_num_created_pts   << " "
      << (int)stats_num_activated_pts_ << " "
      << (int)stats_num_dropped_pts_   << " "
      << (int)stats_last_num_opt_iters_      << " "
      << ef_->resInA                         << " "
      << ef_->resInL                         << " "
      << ef_->resInM                         << " "
      << stats_num_marg_res_fwd_           << " "
      << stats_num_marg_res_bwd_           << " "
      << stats_num_force_dropped_res_fwd_   << " "
      << stats_num_force_dropped_res_bwd_   << " "
      << hessian_frames_.back()->aff_g2l().a  << " "
      << hessian_frames_.back()->aff_g2l().b  << " "
      << hessian_frames_.back()->shell->id - hessian_frames_.front()->shell->id << " "
      << (int)hessian_frames_.size() << " "
      << "\n";
    nums_logger_->flush();
  }
}

void FullSystem::printEigenValLine() {
  // Return if logging is disabled.
  if (!setting_logStuff) {
    return;
  }

  if (ef_->lastHS.rows() < 12) {
    return;
  }

  // Initialize the Hessian matrices.
  MatXX Hp = ef_->lastHS.bottomRightCorner(
    ef_->lastHS.cols() - CPARS,
    ef_->lastHS.cols() - CPARS
  );
  MatXX Ha = ef_->lastHS.bottomRightCorner(
    ef_->lastHS.cols() - CPARS,
    ef_->lastHS.cols() - CPARS
  );
  int n = Hp.cols() / 8;
  assert(Hp.cols() % 8 == 0);

  // Sub-select.
  for (int i = 0; i < n; i++) {
    MatXX tmp6                   = Hp.block(i * 8, 0, 6, n * 8);
    Hp.block(i * 6, 0, 6, n * 8) = tmp6;

    MatXX tmp2                   = Ha.block(i * 8 + 6, 0, 2, n * 8);
    Ha.block(i * 2, 0, 2, n * 8) = tmp2;
  }
  for (int i = 0; i < n; i++) {
    MatXX tmp6                   = Hp.block(0, i * 8, n * 8, 6);
    Hp.block(0, i * 6, n * 8, 6) = tmp6;

    MatXX tmp2                   = Ha.block(0, i * 8 + 6, n * 8, 2);
    Ha.block(0, i * 2, n * 8, 2) = tmp2;
  }

  // Calculate the eigenvalues and sort them.
  VecX eigenvaluesAll = ef_->lastHS.eigenvalues().real();
  VecX eigenP         = Hp.topLeftCorner(n * 6, n * 6).eigenvalues().real();
  VecX eigenA         = Ha.topLeftCorner(n * 2, n * 2).eigenvalues().real();
  VecX diagonal       = ef_->lastHS.diagonal();

  std::sort(eigenvaluesAll.data(), eigenvaluesAll.data() + eigenvaluesAll.size());
  std::sort(eigenP.data(), eigenP.data() + eigenP.size());
  std::sort(eigenA.data(), eigenA.data() + eigenA.size());

  // Print.
  int nz = std::max(100, setting_maxFrames * 10);

  if (eigen_all_logger_ != nullptr) {
    VecX ea                        = VecX::Zero(nz);
    ea.head(eigenvaluesAll.size()) = eigenvaluesAll;
    (*eigen_all_logger_)
      << allKeyFramesHistory.back()->id << " "
      << ea.transpose()                 << "\n";
    eigen_all_logger_->flush();
  }
  if (eigen_a_logger_ != nullptr) {
    VecX ea                = VecX::Zero(nz);
    ea.head(eigenA.size()) = eigenA;
    (*eigen_a_logger_)
      << allKeyFramesHistory.back()->id << " "
      << ea.transpose()                 << "\n";
    eigen_a_logger_->flush();
  }
  if (eigen_p_logger_ != nullptr) {
    VecX ea                = VecX::Zero(nz);
    ea.head(eigenP.size()) = eigenP;
    (*eigen_p_logger_)
    << allKeyFramesHistory.back()->id << " "
    << ea.transpose()                 << "\n";
    eigen_p_logger_->flush();
  }

  if (diagonal_logger_ != nullptr) {
    VecX ea                  = VecX::Zero(nz);
    ea.head(diagonal.size()) = diagonal;
    (*diagonal_logger_)
      << allKeyFramesHistory.back()->id << " "
      << ea.transpose()                 << "\n";
    diagonal_logger_->flush();
  }

  if (variances_logger_ != nullptr) {
    VecX ea                  = VecX::Zero(nz);
    ea.head(diagonal.size()) = ef_->lastHS.inverse().diagonal();
    (*variances_logger_)
      << allKeyFramesHistory.back()->id << " "
      << ea.transpose()                 << "\n";
    variances_logger_->flush();
  }

  std::vector<VecX>& nsp = ef_->lastNullspaces_forLogging;
  (*nullspaces_logger_) << allKeyFramesHistory.back()->id << " ";
  for (std::size_t i = 0; i < nsp.size(); i++) {
    (*nullspaces_logger_)
      << nsp[i].dot(ef_->lastHS * nsp[i]) << " "
      << nsp[i].dot(ef_->lastbS)          << " ";
  }
  (*nullspaces_logger_) << "\n";
  nullspaces_logger_->flush();
}

void FullSystem::printFrameLifetimes() {
  // Return if logging is disabled.
  if (!setting_logStuff) {
    return;
  }

  boost::unique_lock<boost::mutex> lock(tracking_mutex_);

  // Log lifetime information of all frames.

  std::ofstream* lg = new std::ofstream();
  lg->open("logs/lifetimeLog.txt", std::ios::trunc | std::ios::out);
  lg->precision(15);

  for (FrameShell* s : frames_) {
    (*lg) << s->id                          << " "
          << s->marginalizedAt              << " "
          << s->statistics_goodResOnThis    << " "
          << s->statistics_outlierResOnThis << " "
          << s->movedByOpt;

    (*lg) << "\n";
  }

  lg->close();
  delete lg;
}

void FullSystem::printEvalLine() {
  // Currently not implemented.
  return;
}

} // namespace dso
