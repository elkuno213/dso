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
  // Initialize the return status and open the log files.
  int retstat = 0;
  if (setting_logStuff) {
    retstat += system("rm -rf logs");
    retstat += system("mkdir logs");

    retstat += system("rm -rf mats");
    retstat += system("mkdir mats");

    calibLog = new std::ofstream();
    calibLog->open("logs/calibLog.txt", std::ios::trunc | std::ios::out);
    calibLog->precision(12);

    numsLog = new std::ofstream();
    numsLog->open("logs/numsLog.txt", std::ios::trunc | std::ios::out);
    numsLog->precision(10);

    coarseTrackingLog = new std::ofstream();
    coarseTrackingLog->open("logs/coarseTrackingLog.txt", std::ios::trunc | std::ios::out);
    coarseTrackingLog->precision(10);

    eigenAllLog = new std::ofstream();
    eigenAllLog->open("logs/eigenAllLog.txt", std::ios::trunc | std::ios::out);
    eigenAllLog->precision(10);

    eigenPLog = new std::ofstream();
    eigenPLog->open("logs/eigenPLog.txt", std::ios::trunc | std::ios::out);
    eigenPLog->precision(10);

    eigenALog = new std::ofstream();
    eigenALog->open("logs/eigenALog.txt", std::ios::trunc | std::ios::out);
    eigenALog->precision(10);

    DiagonalLog = new std::ofstream();
    DiagonalLog->open("logs/diagonal.txt", std::ios::trunc | std::ios::out);
    DiagonalLog->precision(10);

    variancesLog = new std::ofstream();
    variancesLog->open("logs/variancesLog.txt", std::ios::trunc | std::ios::out);
    variancesLog->precision(10);

    nullspacesLog = new std::ofstream();
    nullspacesLog->open("logs/nullspacesLog.txt", std::ios::trunc | std::ios::out);
    nullspacesLog->precision(10);
  } else {
    nullspacesLog = 0;
    variancesLog  = 0;
    DiagonalLog   = 0;
    eigenALog     = 0;
    eigenPLog     = 0;
    eigenAllLog   = 0;
    numsLog       = 0;
    calibLog      = 0;
  }

  assert(retstat != 293847);

  // Initialize variables.
  selectionMap = new float[wG[0] * hG[0]];

  coarseDistanceMap      = new CoarseDistanceMap(wG[0], hG[0]);
  coarseTracker          = new CoarseTracker    (wG[0], hG[0]);
  coarseTracker_forNewKF = new CoarseTracker    (wG[0], hG[0]);
  coarseInitializer      = new CoarseInitializer(wG[0], hG[0]);
  pixelSelector          = new PixelSelector    (wG[0], hG[0]);

  statistics_lastNumOptIts         = 0;
  statistics_numDroppedPoints      = 0;
  statistics_numActivatedPoints    = 0;
  statistics_numCreatedPoints      = 0;
  statistics_numForceDroppedResBwd = 0;
  statistics_numForceDroppedResFwd = 0;
  statistics_numMargResFwd         = 0;
  statistics_numMargResBwd         = 0;

  lastCoarseRMSE.setConstant(100);

  currentMinActDist = 2;
  initialized       = false;

  ef      = new EnergyFunctional();
  ef->red = &this->treadReduce;

  isLost     = false;
  initFailed = false;

  needNewKFAfter = -1;

  linearizeOperation = true;
  runMapping         = true;
  mappingThread      = boost::thread(&FullSystem::mappingLoop, this);
  lastRefStopID      = 0;

  minIdJetVisDebug   = -1;
  maxIdJetVisDebug   = -1;
  minIdJetVisTracker = -1;
  maxIdJetVisTracker = -1;
}

FullSystem::~FullSystem() {
  // Wait for the mapping thread to finish.
  blockUntilMappingIsFinished();

  // Close the log files.
  if (setting_logStuff) {
    calibLog->close();
    delete calibLog;
    numsLog->close();
    delete numsLog;
    coarseTrackingLog->close();
    delete coarseTrackingLog;
    // errorsLog->close(); delete errorsLog;
    eigenAllLog->close();
    delete eigenAllLog;
    eigenPLog->close();
    delete eigenPLog;
    eigenALog->close();
    delete eigenALog;
    DiagonalLog->close();
    delete DiagonalLog;
    variancesLog->close();
    delete variancesLog;
    nullspacesLog->close();
    delete nullspacesLog;
  }

  // Delete all pointers.
  delete[] selectionMap;

  for (FrameShell* s : allFrameHistory) {
    delete s;
  }
  for (FrameHessian* fh : unmappedTrackedFrames) {
    delete fh;
  }

  delete coarseDistanceMap;
  delete coarseTracker;
  delete coarseTracker_forNewKF;
  delete coarseInitializer;
  delete pixelSelector;
  delete ef;
}

void FullSystem::setOriginalCalib(const VecXf& originalCalib, int originalW, int originalH) {
}

void FullSystem::setGammaFunction(float* BInv) {
  // Return if the gamma function is null.
  if (BInv == 0) {
    return;
  }

  // Copy the gamma function.
  memcpy(Hcalib.Binv, BInv, sizeof(float) * 256);

  // Calculate the inverse gamma function by linear interpolation.
  for (int i = 1; i < 255; i++) {
    for (int s = 1; s < 255; s++) {
      if (BInv[s] <= i && BInv[s + 1] >= i) {
        Hcalib.B[i] = s + (i - BInv[s]) / (BInv[s + 1] - BInv[s]);
        break;
      }
    }
  }
  Hcalib.B[0]   = 0;
  Hcalib.B[255] = 255;
}

void FullSystem::printResult(std::string file) {
  boost::unique_lock<boost::mutex> lock(trackMutex);
  boost::unique_lock<boost::mutex> crlock(shellPoseMutex);

  std::ofstream myfile;
  myfile.open(file.c_str());
  myfile << std::setprecision(15);

  for (FrameShell* s : allFrameHistory) {
    // Skip if the frame's pose is not valid.
    if (!s->poseValid) {
      continue;
    }
    // Skip if only the keyframe poses should be logged and the frame is
    // marginalized.
    if (setting_onlyLogKFPoses && s->marginalizedAt == s->id) {
      continue;
    }
    // Write the frame's pose to the file.
    myfile << s->timestamp                              << " "
           << s->camToWorld.translation().transpose()   << " "
           << s->camToWorld.so3().unit_quaternion().x() << " "
           << s->camToWorld.so3().unit_quaternion().y() << " "
           << s->camToWorld.so3().unit_quaternion().z() << " "
           << s->camToWorld.so3().unit_quaternion().w() << "\n";
  }
  myfile.close();
}

Vec4 FullSystem::trackNewCoarse(FrameHessian* fh) {
  // Make sure the frame history is not empty.
  assert(allFrameHistory.size() > 0);

  // Push the frame to the output wrappers for visualization.
  for (IOWrap::Output3DWrapper* ow : outputWrapper) {
    ow->pushLiveFrame(fh);
  }

  // Get Hessians of the last frame.
  FrameHessian* lastF = coarseTracker->lastRef;

  // Initialize the transformation from the last frame to the current frame.
  AffLight aff_last_2_l = AffLight(0, 0);

  // Construct a list of possible transformations from the last frame to the current frame.
  std::vector<SE3, Eigen::aligned_allocator<SE3>> lastF_2_fh_tries;
  if (allFrameHistory.size() == 2) {
    // If there are only two frames in the history, push the identity transformation.
    for (unsigned int i = 0; i < lastF_2_fh_tries.size(); i++) {
      lastF_2_fh_tries.push_back(SE3());
    }
  } else {
    // Otherwise, get the second-to-last and third-to-last frames.

    FrameShell* slast    = allFrameHistory[allFrameHistory.size() - 2];
    FrameShell* sprelast = allFrameHistory[allFrameHistory.size() - 3];
    SE3 slast_2_sprelast;
    SE3 lastF_2_slast;
    // Lock on global pose consistency.
    {
      boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
      slast_2_sprelast = sprelast->camToWorld.inverse() * slast->camToWorld;
      lastF_2_slast    = slast->camToWorld.inverse() * lastF->shell->camToWorld;
      aff_last_2_l     = slast->aff_g2l;
    }
    SE3 fh_2_slast = slast_2_sprelast; // Assumed to be the same as fh_2_slast.

    // Get last delta movement.
    lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast); // Assume constant motion.
    lastF_2_fh_tries.push_back(fh_2_slast.inverse() * fh_2_slast.inverse() * lastF_2_slast); // Assume double motion (frame skipped)
    lastF_2_fh_tries.push_back(SE3::exp(fh_2_slast.log() * 0.5).inverse() * lastF_2_slast); // Assume half motion.
    lastF_2_fh_tries.push_back(lastF_2_slast); // Assume zero motion.
    lastF_2_fh_tries.push_back(SE3()); // Assume zero motion FROM KF.

    // Just try a TON of different initializations (all rotations). In the end,
    // if they don't work they will only be tried on the coarsest level, which
    // is super fast anyway. Also, if tracking rails here we loose, so we
    // really want to avoid that.
    for (float rotDelta = 0.02; rotDelta < 0.05; rotDelta++) {
      // Assume constant motion in different orientations.
      lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,  rotDelta,         0,         0), Vec3(0,0,0)));
      lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,         0,  rotDelta,         0), Vec3(0,0,0)));
      lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,         0,         0,  rotDelta), Vec3(0,0,0)));
      lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, -rotDelta,         0,         0), Vec3(0,0,0)));
      lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,         0, -rotDelta,         0), Vec3(0,0,0)));
      lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,         0,         0, -rotDelta), Vec3(0,0,0)));
      lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,  rotDelta,  rotDelta,         0), Vec3(0,0,0)));
      lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,         0,  rotDelta,  rotDelta), Vec3(0,0,0)));
      lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,  rotDelta,         0,  rotDelta), Vec3(0,0,0)));
      lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, -rotDelta,  rotDelta,         0), Vec3(0,0,0)));
      lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,         0, -rotDelta,  rotDelta), Vec3(0,0,0)));
      lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, -rotDelta,         0,  rotDelta), Vec3(0,0,0)));
      lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,  rotDelta, -rotDelta,         0), Vec3(0,0,0)));
      lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,         0,  rotDelta, -rotDelta), Vec3(0,0,0)));
      lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,  rotDelta,         0, -rotDelta), Vec3(0,0,0)));
      lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, -rotDelta, -rotDelta,         0), Vec3(0,0,0)));
      lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,         0, -rotDelta, -rotDelta), Vec3(0,0,0)));
      lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, -rotDelta,         0, -rotDelta), Vec3(0,0,0)));
      lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, -rotDelta, -rotDelta, -rotDelta), Vec3(0,0,0)));
      lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, -rotDelta, -rotDelta,  rotDelta), Vec3(0,0,0)));
      lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, -rotDelta,  rotDelta, -rotDelta), Vec3(0,0,0)));
      lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, -rotDelta,  rotDelta,  rotDelta), Vec3(0,0,0)));
      lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,  rotDelta, -rotDelta, -rotDelta), Vec3(0,0,0)));
      lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,  rotDelta, -rotDelta,  rotDelta), Vec3(0,0,0)));
      lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,  rotDelta,  rotDelta, -rotDelta), Vec3(0,0,0)));
      lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,  rotDelta,  rotDelta,  rotDelta), Vec3(0,0,0)));
    }

    if (!slast->poseValid || !sprelast->poseValid || !lastF->shell->poseValid) {
      lastF_2_fh_tries.clear();
      lastF_2_fh_tries.push_back(SE3());
    }
  }

  // Initialize some variables for attempting to find the best transformation.
  Vec3 flowVecs    = Vec3(100, 100, 100);
  SE3 lastF_2_fh   = SE3();
  AffLight aff_g2l = AffLight(0, 0);

  // As long as maxResForImmediateAccept is not reached, I'll continue through
  // the options. I'll keep track of the so-far best achieved residual for each
  // level in achievedRes. If on a coarse level, tracking is WORSE than
  // achievedRes, we will not continue to save time.

  // Check all tries of the transformations to get the best one.
  Vec5 achievedRes  = Vec5::Constant(NAN);
  bool haveOneGood  = false;
  int tryIterations = 0;
  for (unsigned int i = 0; i < lastF_2_fh_tries.size(); i++) {
    // Attempt tracking with the current try of the transformation.
    AffLight aff_g2l_this = aff_last_2_l;
    SE3 lastF_2_fh_this   = lastF_2_fh_tries[i];
    bool trackingIsGood   = coarseTracker->trackNewestCoarse(
      fh,
      lastF_2_fh_this,
      aff_g2l_this,
      pyrLevelsUsed - 1,
      achievedRes
    ); // In each level has to be at least as good as the last try.
    tryIterations++;

    // Print the tracking result.
    if (i != 0) {
      printf(
        "RE-TRACK ATTEMPT %d with initOption %d and start-lvl %d (ab %f %f): "
        "%f %f %f %f %f -> %f %f %f %f %f \n",
        i,
        i,
        pyrLevelsUsed - 1,
        aff_g2l_this.a,
        aff_g2l_this.b,
        achievedRes[0],
        achievedRes[1],
        achievedRes[2],
        achievedRes[3],
        achievedRes[4],
        coarseTracker->lastResiduals[0],
        coarseTracker->lastResiduals[1],
        coarseTracker->lastResiduals[2],
        coarseTracker->lastResiduals[3],
        coarseTracker->lastResiduals[4]
      );
    }

    // If the tracking and the residuals are good, update the best transformation.
    if (
      trackingIsGood &&
      std::isfinite((float)coarseTracker->lastResiduals[0]) &&
      !(coarseTracker->lastResiduals[0] >= achievedRes[0])
    ) {
      // printf("take over. minRes %f -> %f!\n", achievedRes[0], coarseTracker->lastResiduals[0]);
      flowVecs    = coarseTracker->lastFlowIndicators;
      aff_g2l     = aff_g2l_this;
      lastF_2_fh  = lastF_2_fh_this;
      haveOneGood = true;
    }

    // Update the achieved residuals if at least one good transformation was found.
    if (haveOneGood) {
      for (int i = 0; i < 5; i++) {
        // Take over if achievedRes is either bigger or NAN.
        if (
          !std::isfinite((float)achievedRes[i]) ||
          achievedRes[i] > coarseTracker->lastResiduals[i]
        ) {
          achievedRes[i] = coarseTracker->lastResiduals[i];
        }
      }
    }

    // If a good transformation was found and the residual is below the threshold, break.
    if (haveOneGood && achievedRes[0] < lastCoarseRMSE[0] * setting_reTrackThreshold) {
      break;
    }
  }

  // If no good transformation was found, print an error message.
  if (!haveOneGood) {
    printf(
      "BIG ERROR! tracking failed entirely. Take predictred pose and hope we "
      "may somehow recover.\n"
    );
    flowVecs   = Vec3(0, 0, 0);
    aff_g2l    = aff_last_2_l;
    lastF_2_fh = lastF_2_fh_tries[0];
  }

  lastCoarseRMSE = achievedRes;

  // Update the frame with the best transformation. No lock required, as fh is
  // not used anywhere yet.
  fh->shell->camToTrackingRef = lastF_2_fh.inverse();
  fh->shell->trackingRef      = lastF->shell;
  fh->shell->aff_g2l          = aff_g2l;
  fh->shell->camToWorld       = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;

  // Update the first coarse RMSE.
  if (coarseTracker->firstCoarseRMSE < 0) {
    coarseTracker->firstCoarseRMSE = achievedRes[0];
  }

  if (!setting_debugout_runquiet) {
    printf(
      "Coarse Tracker tracked ab = %f %f (exp %f). Res %f!\n",
      aff_g2l.a,
      aff_g2l.b,
      fh->ab_exposure,
      achievedRes[0]
    );
  }

  // Log the tracking result.
  if (setting_logStuff) {
    (*coarseTrackingLog) << std::setprecision(16)
                         << fh->shell->id                           << " "
                         << fh->shell->timestamp                    << " "
                         << fh->ab_exposure                         << " "
                         << fh->shell->camToWorld.log().transpose() << " "
                         << aff_g2l.a                               << " "
                         << aff_g2l.b                               << " "
                         << achievedRes[0]                          << " "
                         << tryIterations                           << "\n";
  }

  return Vec4(achievedRes[0], flowVecs[0], flowVecs[1], flowVecs[2]);
}

void FullSystem::traceNewCoarse(FrameHessian* fh) {
  boost::unique_lock<boost::mutex> lock(mapMutex);

  // Initialize some tracing variables.
  int trace_total = 0, trace_good = 0, trace_oob = 0, trace_out = 0,
      trace_skip = 0, trace_badcondition = 0, trace_uninitialized = 0;

  Mat33f K = Mat33f::Identity();
  K(0, 0)  = Hcalib.fxl();
  K(1, 1)  = Hcalib.fyl();
  K(0, 2)  = Hcalib.cxl();
  K(1, 2)  = Hcalib.cyl();

  // Go through all active frames and trace the points.
  for (FrameHessian* host : frameHessians) {
    SE3 hostToNew = fh->PRE_worldToCam * host->PRE_camToWorld;
    Mat33f KRKi   = K * hostToNew.rotationMatrix().cast<float>() * K.inverse();
    Vec3f Kt      = K * hostToNew.translation().cast<float>();

    Vec2f aff = AffLight::fromToVecExposure(
      host->ab_exposure,
      fh->ab_exposure,
      host->aff_g2l(),
      fh->aff_g2l()
    ).cast<float>();

    for (ImmaturePoint* ph : host->immaturePoints) {
      ph->traceOn(fh, KRKi, Kt, aff, &Hcalib, false);

      if (ph->lastTraceStatus == ImmaturePointStatus::IPS_GOOD) {
        trace_good++;
      }
      if (ph->lastTraceStatus == ImmaturePointStatus::IPS_BADCONDITION) {
        trace_badcondition++;
      }
      if (ph->lastTraceStatus == ImmaturePointStatus::IPS_OOB) {
        trace_oob++;
      }
      if (ph->lastTraceStatus == ImmaturePointStatus::IPS_OUTLIER) {
        trace_out++;
      }
      if (ph->lastTraceStatus == ImmaturePointStatus::IPS_SKIPPED) {
        trace_skip++;
      }
      if (ph->lastTraceStatus == ImmaturePointStatus::IPS_UNINITIALIZED) {
        trace_uninitialized++;
      }
      trace_total++;
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
    = new ImmaturePointTemporaryResidual[frameHessians.size()];
  for (int k = min; k < max; k++) {
    (*optimized)[k] = optimizeImmaturePoint((*toOptimize)[k], 1, tr);
  }
  delete[] tr;
}

void FullSystem::activatePointsMT() {
  // Update the minimum activation distance.
  if (ef->nPoints < setting_desiredPointDensity * 0.66) {
    currentMinActDist -= 0.8;
  }
  if (ef->nPoints < setting_desiredPointDensity * 0.8) {
    currentMinActDist -= 0.5;
  } else if (ef->nPoints < setting_desiredPointDensity * 0.9) {
    currentMinActDist -= 0.2;
  } else if (ef->nPoints < setting_desiredPointDensity) {
    currentMinActDist -= 0.1;
  }

  if (ef->nPoints > setting_desiredPointDensity * 1.5) {
    currentMinActDist += 0.8;
  }
  if (ef->nPoints > setting_desiredPointDensity * 1.3) {
    currentMinActDist += 0.5;
  }
  if (ef->nPoints > setting_desiredPointDensity * 1.15) {
    currentMinActDist += 0.2;
  }
  if (ef->nPoints > setting_desiredPointDensity) {
    currentMinActDist += 0.1;
  }

  if (currentMinActDist < 0) {
    currentMinActDist = 0;
  }
  if (currentMinActDist > 4) {
    currentMinActDist = 4;
  }

  // Print messages if debug enabled.
  if (!setting_debugout_runquiet) {
    printf(
      "SPARSITY:  MinActDist %f (need %d points, have %d points)!\n",
      currentMinActDist,
      (int)(setting_desiredPointDensity),
      ef->nPoints
    );
  }

  // Get the newest frame.
  FrameHessian* newestHs = frameHessians.back();

  // Make distance map.
  coarseDistanceMap->makeK(&Hcalib);
  coarseDistanceMap->makeDistanceMap(frameHessians, newestHs);

  // coarseTracker->debugPlotDistMap("distMap");

  std::vector<ImmaturePoint*> toOptimize;
  toOptimize.reserve(20000);

  // Go through all active frames and add points to the optimization.
  for (FrameHessian* host : frameHessians) {
    // Skip the newest frame.
    if (host == newestHs) {
      continue;
    }

    SE3 fhToNew = newestHs->PRE_worldToCam * host->PRE_camToWorld;
    Mat33f KRKi = (coarseDistanceMap->K[1] * fhToNew.rotationMatrix().cast<float>() * coarseDistanceMap->Ki[0]);
    Vec3f Kt = (coarseDistanceMap->K[1] * fhToNew.translation().cast<float>());

    for (unsigned int i = 0; i < host->immaturePoints.size(); i += 1) {
      ImmaturePoint* ph       = host->immaturePoints[i];
      ph->idxInImmaturePoints = i;

      // Delete points that have never been traced successfully, or that are
      // outlier on the last trace.
      if (!std::isfinite(ph->idepth_max) || ph->lastTraceStatus == IPS_OUTLIER) {
        // immature_invalid_deleted++;

        delete ph;
        host->immaturePoints[i] = 0;
        continue;
      }

      // Otherwise, see if we can activate the point.
      bool canActivate = (
        ph->lastTraceStatus == IPS_GOOD         ||
        ph->lastTraceStatus == IPS_SKIPPED      ||
        ph->lastTraceStatus == IPS_BADCONDITION ||
        ph->lastTraceStatus == IPS_OOB
      ) &&  ph->lastTracePixelInterval < 8
        &&  ph->quality > setting_minTraceQuality
        && (ph->idepth_max + ph->idepth_min) > 0;

      // If I cannot activate the point, skip it. Maybe also delete it if it
      // will be out afterwards.
      if (!canActivate) {
        if (ph->host->flaggedForMarginalization || ph->lastTraceStatus == IPS_OOB) {
          // immature_notReady_deleted++;
          delete ph;
          host->immaturePoints[i] = 0;
        }
        // immature_notReady_skipped++;
        continue;
      }

      // See if we need to activate point due to distance map.
      Vec3f ptp = KRKi * Vec3f(ph->u, ph->v, 1) + Kt * (0.5f * (ph->idepth_max + ph->idepth_min));
      int u = ptp[0] / ptp[2] + 0.5f;
      int v = ptp[1] / ptp[2] + 0.5f;

      if ((u > 0 && v > 0 && u < wG[1] && v < hG[1])) {
        float dist = coarseDistanceMap->fwdWarpedIDDistFinal[u + wG[1] * v]
                   + (ptp[0] - floorf((float)(ptp[0])));

        if (dist >= currentMinActDist * ph->my_type) {
          coarseDistanceMap->addIntoDistFinal(u, v);
          toOptimize.push_back(ph);
        }
      } else {
        delete ph;
        host->immaturePoints[i] = 0;
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
  for (unsigned k = 0; k < toOptimize.size(); k++) {
    PointHessian* newpoint = optimized[k];
    ImmaturePoint* ph      = toOptimize[k];

    if (newpoint != 0 && newpoint != (PointHessian*)((long)(-1))) {
      newpoint->host->immaturePoints[ph->idxInImmaturePoints] = 0;
      newpoint->host->pointHessians.push_back(newpoint);
      ef->insertPoint(newpoint);
      for (PointFrameResidual* r : newpoint->residuals) ef->insertResidual(r);
      assert(newpoint->efPoint != 0);
      delete ph;
    } else if (newpoint == (PointHessian*)((long)(-1)) || ph->lastTraceStatus == IPS_OOB) {
      delete ph;
      ph->host->immaturePoints[ph->idxInImmaturePoints] = 0;
    } else {
      assert(newpoint == 0 || newpoint == (PointHessian*)((long)(-1)));
    }
  }

  // Remove the points that were not activated.
  for (FrameHessian* host : frameHessians) {
    for (int i = 0; i < (int)host->immaturePoints.size(); i++) {
      if (host->immaturePoints[i] == 0) {
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
      int i = ((int)frameHessians.size()) - 1;
      i >= 0 && i >= ((int)frameHessians.size());
      i--
    ) {
      if (!frameHessians[i]->flaggedForMarginalization) {
        fhsToKeepPoints.push_back(frameHessians[i]);
      }
    }

    for (int i = 0; i < (int)frameHessians.size(); i++) {
      if (frameHessians[i]->flaggedForMarginalization) {
        fhsToMargPoints.push_back(frameHessians[i]);
      }
    }
  }

  // ef->setAdjointsF();
  // ef->setDeltaF(&Hcalib);
  int flag_oob = 0, flag_in = 0, flag_inin = 0, flag_nores = 0;

  // Go through all active frames and update the points to be kept and marginalized.
  for (FrameHessian* host : frameHessians) {
    for (unsigned int i = 0; i < host->pointHessians.size(); i++) {
      PointHessian* ph = host->pointHessians[i];
      if (ph == 0) {
        continue;
      }

      if (ph->idepth_scaled < 0 || ph->residuals.size() == 0) {
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
              r->efResidual->fixLinearizationF(ef);
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

        host->pointHessians[i] = 0;
      }
    }

    for (int i = 0; i < (int)host->pointHessians.size(); i++) {
      if (host->pointHessians[i] == 0) {
        host->pointHessians[i] = host->pointHessians.back();
        host->pointHessians.pop_back();
        i--;
      }
    }
  }
}

void FullSystem::addActiveFrame(ImageAndExposure* image, int id) {
  // Skip if the tracking is lost.
  if (isLost) {
    return;
  }

  boost::unique_lock<boost::mutex> lock(trackMutex);

  // Add frame to the history.
  FrameHessian* fh  = new FrameHessian();
  FrameShell* shell = new FrameShell();
  shell->camToWorld = SE3(); // No lock required, as fh is not used anywhere yet.
  shell->aff_g2l    = AffLight(0, 0);
  shell->marginalizedAt = shell->id = allFrameHistory.size();
  shell->timestamp                  = image->timestamp;
  shell->incoming_id                = id;
  fh->shell                         = shell;
  allFrameHistory.push_back(shell);

  // Set the image and exposure.
  fh->ab_exposure = image->exposure_time;
  fh->makeImages(image->image, &Hcalib);

  if (!initialized) {
    // Initialize if not.
    if (coarseInitializer->frameID < 0) {
      // If no frame is set, first frame is set and fh is kept by
      // coarseInitializer.
      coarseInitializer->setFirst(&Hcalib, fh);
    } else if (coarseInitializer->trackFrame(fh, outputWrapper)) {
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
    if (coarseTracker_forNewKF->refFrameID > coarseTracker->refFrameID) {
      boost::unique_lock<boost::mutex> crlock(coarseTrackerSwapMutex);
      CoarseTracker* tmp     = coarseTracker;
      coarseTracker          = coarseTracker_forNewKF;
      coarseTracker_forNewKF = tmp;
    }

    // Check if the tracking is lost.
    Vec4 tres = trackNewCoarse(fh);
    if (!std::isfinite((double)tres[0]) || !std::isfinite((double)tres[1]) || !std::isfinite((double)tres[2]) || !std::isfinite((double)tres[3])) {
      printf("Initial Tracking failed: LOST!\n");
      isLost = true;
      return;
    }

    // Flag if the keyframe is needed.
    bool needToMakeKF = false;
    if (setting_keyframesPerSecond > 0) {
      needToMakeKF
        = allFrameHistory.size() == 1
       || (fh->shell->timestamp - allKeyFramesHistory.back()->timestamp) > 0.95f / setting_keyframesPerSecond;
    } else {
      Vec2 refToFh = AffLight::fromToVecExposure(
        coarseTracker->lastRef->ab_exposure,
        fh->ab_exposure,
        coarseTracker->lastRef_aff_g2l,
        fh->shell->aff_g2l
      );

      // Check brightness.
      needToMakeKF
        = allFrameHistory.size() == 1
       || setting_kfGlobalWeight * setting_maxShiftWeightT  * sqrtf((double)tres[1]) / (wG[0] + hG[0])
        + setting_kfGlobalWeight * setting_maxShiftWeightR  * sqrtf((double)tres[2]) / (wG[0] + hG[0])
        + setting_kfGlobalWeight * setting_maxShiftWeightRT * sqrtf((double)tres[3]) / (wG[0] + hG[0])
        + setting_kfGlobalWeight * setting_maxAffineWeight  * fabs(logf((float)refToFh[0])) > 1
       || 2 * coarseTracker->firstCoarseRMSE < tres[0];
    }

    // Publish the new frame to visualization wrapper.
    for (IOWrap::Output3DWrapper* ow : outputWrapper) {
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
  if (linearizeOperation) {
    if (goStepByStep && lastRefStopID != coarseTracker->refFrameID) {
      MinimalImageF3 img(wG[0], hG[0], fh->dI);
      IOWrap::displayImage("frameToTrack", &img);
      while (true) {
        char k = IOWrap::waitKey(0);
        if (k == ' ') {
          break;
        }
        handleKey(k);
      }
      lastRefStopID = coarseTracker->refFrameID;
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
    unmappedTrackedFrames.push_back(fh);
    if (needKF) {
      needNewKFAfter = fh->shell->trackingRef->id;
    }
    trackedFrameSignal.notify_all();

    while (
      coarseTracker_forNewKF->refFrameID == -1 &&
      coarseTracker->refFrameID == -1
    ) {
      mappedFrameSignal.wait(lock);
    }

    lock.unlock();
  }
}

void FullSystem::mappingLoop() {
  boost::unique_lock<boost::mutex> lock(trackMapSyncMutex);

  while (runMapping) {
    // No tracked frame to map, so wait for the signal and return if the mapping
    // is not running.
    while (unmappedTrackedFrames.size() == 0) {
      trackedFrameSignal.wait(lock);
      if (!runMapping) {
        return;
      }
    }

    // Get the oldest tracked frame.
    FrameHessian* fh = unmappedTrackedFrames.front();
    unmappedTrackedFrames.pop_front();

    // Make sure to make a keyframe for the very first 2 tracked frames.
    if (allKeyFramesHistory.size() <= 2) {
      lock.unlock();
      makeKeyFrame(fh);
      lock.lock();
      mappedFrameSignal.notify_all();
      continue;
    }

    // If there are more than 3 frames to track, we need to catch up with the mapping.
    if (unmappedTrackedFrames.size() > 3) {
      needToKetchupMapping = true;
    }

    if (unmappedTrackedFrames.size() > 0) {
      // If there are still other unmapped tracked frames, make the current
      // frame as non-keyframe.
      lock.unlock();
      makeNonKeyFrame(fh);
      lock.lock();

      if (needToKetchupMapping && unmappedTrackedFrames.size() > 0) {
        FrameHessian* fh = unmappedTrackedFrames.front();
        unmappedTrackedFrames.pop_front();
        {
          boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
          assert(fh->shell->trackingRef != 0);
          fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;
          fh->setEvalPT_scaled(fh->shell->camToWorld.inverse(), fh->shell->aff_g2l);
        }
        delete fh;
      }

    } else {
      // Otherwise, make the tracked frame as keyframe if possible, otherwise as non-keyframe.
      if (setting_realTimeMaxKF || needNewKFAfter >= frameHessians.back()->shell->id) {
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
  runMapping = false;
  trackedFrameSignal.notify_all();
  lock.unlock();

  mappingThread.join();
  mappingThread.join();

  mappingThread.join();

}

void FullSystem::makeNonKeyFrame(FrameHessian* fh) {
  // This function needs to be called by the mapping thread only, so no lock is required.

  // Update the frame's pose, perform tracing and delete the frame.
  {
    boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
    assert(fh->shell->trackingRef != 0);
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
    boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
    assert(fh->shell->trackingRef != 0);
    fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;
    fh->setEvalPT_scaled(fh->shell->camToWorld.inverse(), fh->shell->aff_g2l);
  }
  traceNewCoarse(fh);

  // Lock to make the keyframe.
  boost::unique_lock<boost::mutex> lock(mapMutex);

  // Flag the frames to be marginalized.
  flagFramesForMarginalization(fh);

  // Add the frame to the energy functional.
  fh->idx = frameHessians.size();
  frameHessians.push_back(fh);
  fh->frameID = allKeyFramesHistory.size();
  allKeyFramesHistory.push_back(fh->shell);
  ef->insertFrame(fh, &Hcalib);

  setPrecalcValues();

  // Add new residuals for old points.
  int numFwdResAdde = 0;
  for (FrameHessian* fh1 : frameHessians) {
    if (fh1 == fh) {
      continue;
    }
    for (PointHessian* ph : fh1->pointHessians) {
      PointFrameResidual* r = new PointFrameResidual(ph, fh1, fh);
      r->setState(ResState::IN);
      ph->residuals.push_back(r);
      ef->insertResidual(r);
      ph->lastResiduals[1] = ph->lastResiduals[0];
      ph->lastResiduals[0]
        = std::pair<PointFrameResidual*, ResState>(r, ResState::IN);
      numFwdResAdde += 1;
    }
  }

  // Activate points and flag for marginalization.
  activatePointsMT();
  ef->makeIDX();

  // Optimize all.
  fh->frameEnergyTH = frameHessians.back()->frameEnergyTH;
  float rmse        = optimize(setting_maxOptIterations);

  // Check if the initialization failed.
  if (allKeyFramesHistory.size() <= 4) {
    if (allKeyFramesHistory.size() == 2 && rmse > 20 * benchmark_initializerSlackFactor) {
      printf("I THINK INITIALIZATINO FAILED! Resetting.\n");
      initFailed = true;
    }
    if (allKeyFramesHistory.size() == 3 && rmse > 13 * benchmark_initializerSlackFactor) {
      printf("I THINK INITIALIZATINO FAILED! Resetting.\n");
      initFailed = true;
    }
    if (allKeyFramesHistory.size() == 4 && rmse > 9 * benchmark_initializerSlackFactor) {
      printf("I THINK INITIALIZATINO FAILED! Resetting.\n");
      initFailed = true;
    }
  }

  // Return if the tracking is lost.
  if (isLost) {
    return;
  }

  // Remove outliers.
  removeOutliers();

  {
    boost::unique_lock<boost::mutex> crlock(coarseTrackerSwapMutex);
    coarseTracker_forNewKF->makeK(&Hcalib);
    coarseTracker_forNewKF->setCoarseTrackingRef(frameHessians);

    coarseTracker_forNewKF->debugPlotIDepthMap(
      &minIdJetVisTracker,
      &maxIdJetVisTracker,
      outputWrapper
    );
    coarseTracker_forNewKF->debugPlotIDepthMapFloat(outputWrapper);
  }

  debugPlot("post Optimize");

  // Marginalize points.
  flagPointsForRemoval();
  ef->dropPointsF();
  getNullspaces(
    ef->lastNullspaces_pose,
    ef->lastNullspaces_scale,
    ef->lastNullspaces_affA,
    ef->lastNullspaces_affB
  );
  ef->marginalizePointsF();

  // Add new immature points and residuals.
  makeNewTraces(fh, 0);

  // Publish the keyframe to visualization wrapper.
  for (IOWrap::Output3DWrapper* ow : outputWrapper) {
    ow->publishGraph(ef->connectivityMap);
    ow->publishKeyframes(frameHessians, false, &Hcalib);
  }

  // Marginalize frames.
  for (unsigned int i = 0; i < frameHessians.size(); i++) {
    if (frameHessians[i]->flaggedForMarginalization) {
      marginalizeFrame(frameHessians[i]);
      i = 0;
    }
  }

  printLogLine();
  // printEigenValLine();
}

void FullSystem::initializeFromInitializer(FrameHessian* newFrame) {
  boost::unique_lock<boost::mutex> lock(mapMutex);

  // Add first frame.
  FrameHessian* firstFrame = coarseInitializer->firstFrame;
  firstFrame->idx          = frameHessians.size();
  frameHessians.push_back(firstFrame);
  firstFrame->frameID = allKeyFramesHistory.size();
  allKeyFramesHistory.push_back(firstFrame->shell);
  ef->insertFrame(firstFrame, &Hcalib);
  setPrecalcValues();

  // int numPointsTotal = makePixelStatus(firstFrame->dI, selectionMap, wG[0], hG[0], setting_desiredDensity);
  // int numPointsTotal = pixelSelector->makeMaps(firstFrame->dIp, selectionMap,setting_desiredDensity);

  firstFrame->pointHessians.reserve(wG[0] * hG[0] * 0.2f);
  firstFrame->pointHessiansMarginalized.reserve(wG[0] * hG[0] * 0.2f);
  firstFrame->pointHessiansOut.reserve(wG[0] * hG[0] * 0.2f);

  // Calculate the rescale factor.
  float sumID = 1e-5, numID = 1e-5;
  for (int i = 0; i < coarseInitializer->numPoints[0]; i++) {
    sumID += coarseInitializer->points[0][i].iR;
    numID++;
  }
  float rescaleFactor = 1 / (sumID / numID);

  // Randomly sub-select the points.
  float keepPercentage = setting_desiredPointDensity / coarseInitializer->numPoints[0];

  if (!setting_debugout_runquiet) {
    printf(
      "Initialization: keep %.1f%% (need %d, have %d)!\n",
      100 * keepPercentage,
      (int)(setting_desiredPointDensity),
      coarseInitializer->numPoints[0]
    );
  }

  for (int i = 0; i < coarseInitializer->numPoints[0]; i++) {
    if (rand() / (float)RAND_MAX > keepPercentage) {
      continue;
    }

    Pnt* point        = coarseInitializer->points[0] + i;
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

    pt->idepth_max = pt->idepth_min = 1;
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
    ef->insertPoint(ph);
  }

  SE3 firstToNew = coarseInitializer->thisToNext;
  firstToNew.translation() /= rescaleFactor;

  // Initialize (no lock required, as we are initializing).
  {
    boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
    firstFrame->shell->camToWorld = SE3();
    firstFrame->shell->aff_g2l    = AffLight(0, 0);
    firstFrame->setEvalPT_scaled(
      firstFrame->shell->camToWorld.inverse(),
      firstFrame->shell->aff_g2l
    );
    firstFrame->shell->trackingRef      = 0;
    firstFrame->shell->camToTrackingRef = SE3();

    newFrame->shell->camToWorld = firstToNew.inverse();
    newFrame->shell->aff_g2l    = AffLight(0, 0);
    newFrame->setEvalPT_scaled(
      newFrame->shell->camToWorld.inverse(),
      newFrame->shell->aff_g2l
    );
    newFrame->shell->trackingRef      = firstFrame->shell;
    newFrame->shell->camToTrackingRef = firstToNew.inverse();
  }

  initialized = true;
  printf(
    "INITIALIZE FROM INITIALIZER (%d pts)!\n",
    (int)firstFrame->pointHessians.size()
  );
}

void FullSystem::makeNewTraces(FrameHessian* newFrame, float* gtDepth) {
  // Initialize new frame.
  pixelSelector->allowFast = true;
  // int numPointsTotal       = makePixelStatus(
  //   newFrame->dI,
  //   selectionMap,
  //   wG[0],
  //   hG[0],
  //   setting_desiredDensity
  // );
  int numPointsTotal = pixelSelector->makeMaps(
    newFrame,
    selectionMap,
    setting_desiredImmatureDensity
  );

  newFrame->pointHessians.reserve(numPointsTotal * 1.2f);
  // fh->pointHessiansInactive.reserve(numPointsTotal * 1.2f);
  newFrame->pointHessiansMarginalized.reserve(numPointsTotal * 1.2f);
  newFrame->pointHessiansOut.reserve(numPointsTotal * 1.2f);

  // Add immature points to the new frame if its energy is finite.
  for (int y = patternPadding + 1; y < hG[0] - patternPadding - 2; y++) {
    for (int x = patternPadding + 1; x < wG[0] - patternPadding - 2; x++) {
      int i = x + y * wG[0];
      if (selectionMap[i] == 0) {
        continue;
      }

      ImmaturePoint* impt
        = new ImmaturePoint(x, y, newFrame, selectionMap[i], &Hcalib);
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
  for (FrameHessian* fh : frameHessians) {
    fh->targetPrecalc.resize(frameHessians.size());
    for (unsigned int i = 0; i < frameHessians.size(); i++) {
      fh->targetPrecalc[i].set(fh, frameHessians[i], &Hcalib);
    }
  }

  ef->setDeltaF(&Hcalib);
}

void FullSystem::printLogLine() {
  // Return if no keyframes are available.
  if (frameHessians.size() == 0) return;

  // Print the log line if quiet mode is disabled.
  if (!setting_debugout_runquiet) {
    printf(
      "LOG %d: %.3f fine. Res: %d A, %d L, %d M; (%'d / %'d) forceDrop. a=%f, "
      "b=%f. Window %d (%d)\n",
      allKeyFramesHistory.back()->id,
      statistics_lastFineTrackRMSE,
      ef->resInA,
      ef->resInL,
      ef->resInM,
      (int)statistics_numForceDroppedResFwd,
      (int)statistics_numForceDroppedResBwd,
      allKeyFramesHistory.back()->aff_g2l.a,
      allKeyFramesHistory.back()->aff_g2l.b,
      frameHessians.back()->shell->id - frameHessians.front()->shell->id,
      (int)frameHessians.size()
    );
  }

  // Return if logging is disabled.
  if (!setting_logStuff) {
    return;
  }

  // Log.
  if (numsLog != 0) {
    (*numsLog)
      << allKeyFramesHistory.back()->id     << " "
      << statistics_lastFineTrackRMSE       << " "
      << (int)statistics_numCreatedPoints   << " "
      << (int)statistics_numActivatedPoints << " "
      << (int)statistics_numDroppedPoints   << " "
      << (int)statistics_lastNumOptIts      << " "
      << ef->resInA                         << " "
      << ef->resInL                         << " "
      << ef->resInM                         << " "
      << statistics_numMargResFwd           << " "
      << statistics_numMargResBwd           << " "
      << statistics_numForceDroppedResFwd   << " "
      << statistics_numForceDroppedResBwd   << " "
      << frameHessians.back()->aff_g2l().a  << " "
      << frameHessians.back()->aff_g2l().b  << " "
      << frameHessians.back()->shell->id - frameHessians.front()->shell->id << " "
      << (int)frameHessians.size() << " "
      << "\n";
    numsLog->flush();
  }
}

void FullSystem::printEigenValLine() {
  // Return if logging is disabled.
  if (!setting_logStuff) {
    return;
  }

  if (ef->lastHS.rows() < 12) {
    return;
  }

  // Initialize the Hessian matrices.
  MatXX Hp = ef->lastHS.bottomRightCorner(
    ef->lastHS.cols() - CPARS,
    ef->lastHS.cols() - CPARS
  );
  MatXX Ha = ef->lastHS.bottomRightCorner(
    ef->lastHS.cols() - CPARS,
    ef->lastHS.cols() - CPARS
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
  VecX eigenvaluesAll = ef->lastHS.eigenvalues().real();
  VecX eigenP         = Hp.topLeftCorner(n * 6, n * 6).eigenvalues().real();
  VecX eigenA         = Ha.topLeftCorner(n * 2, n * 2).eigenvalues().real();
  VecX diagonal       = ef->lastHS.diagonal();

  std::sort(eigenvaluesAll.data(), eigenvaluesAll.data() + eigenvaluesAll.size());
  std::sort(eigenP.data(), eigenP.data() + eigenP.size());
  std::sort(eigenA.data(), eigenA.data() + eigenA.size());

  // Print.
  int nz = std::max(100, setting_maxFrames * 10);

  if (eigenAllLog != 0) {
    VecX ea                        = VecX::Zero(nz);
    ea.head(eigenvaluesAll.size()) = eigenvaluesAll;
    (*eigenAllLog)
      << allKeyFramesHistory.back()->id << " "
      << ea.transpose()                 << "\n";
    eigenAllLog->flush();
  }
  if (eigenALog != 0) {
    VecX ea                = VecX::Zero(nz);
    ea.head(eigenA.size()) = eigenA;
    (*eigenALog)
      << allKeyFramesHistory.back()->id << " "
      << ea.transpose()                 << "\n";
    eigenALog->flush();
  }
  if (eigenPLog != 0) {
    VecX ea                = VecX::Zero(nz);
    ea.head(eigenP.size()) = eigenP;
    (*eigenPLog)
    << allKeyFramesHistory.back()->id << " "
    << ea.transpose()                 << "\n";
    eigenPLog->flush();
  }

  if (DiagonalLog != 0) {
    VecX ea                  = VecX::Zero(nz);
    ea.head(diagonal.size()) = diagonal;
    (*DiagonalLog)
      << allKeyFramesHistory.back()->id << " "
      << ea.transpose()                 << "\n";
    DiagonalLog->flush();
  }

  if (variancesLog != 0) {
    VecX ea                  = VecX::Zero(nz);
    ea.head(diagonal.size()) = ef->lastHS.inverse().diagonal();
    (*variancesLog)
      << allKeyFramesHistory.back()->id << " "
      << ea.transpose()                 << "\n";
    variancesLog->flush();
  }

  std::vector<VecX>& nsp = ef->lastNullspaces_forLogging;
  (*nullspacesLog) << allKeyFramesHistory.back()->id << " ";
  for (unsigned int i = 0; i < nsp.size(); i++) {
    (*nullspacesLog)
      << nsp[i].dot(ef->lastHS * nsp[i]) << " "
      << nsp[i].dot(ef->lastbS)          << " ";
  }
  (*nullspacesLog) << "\n";
  nullspacesLog->flush();
}

void FullSystem::printFrameLifetimes() {
  // Return if logging is disabled.
  if (!setting_logStuff) {
    return;
  }

  boost::unique_lock<boost::mutex> lock(trackMutex);

  // Log lifetime information of all frames.

  std::ofstream* lg = new std::ofstream();
  lg->open("logs/lifetimeLog.txt", std::ios::trunc | std::ios::out);
  lg->precision(15);

  for (FrameShell* s : allFrameHistory) {
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
