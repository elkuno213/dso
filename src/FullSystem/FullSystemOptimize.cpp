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

#include "FullSystem/FullSystem.h"
#include "FullSystem/ResidualProjections.h"
#include "IOWrapper/ImageDisplay.h"
#include "OptimizationBackend/EnergyFunctional.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"
#include "util/globalCalib.h"
#include "util/globalFuncs.h"

namespace dso {

void FullSystem::linearizeAll_Reductor(
  bool fixLinearization,
  std::vector<PointFrameResidual*>* toRemove,
  int min,
  int max,
  Vec10* stats,
  int tid
) {
  for (int k = min; k < max; k++) {
    PointFrameResidual* r = activeResiduals[k];
    (*stats)[0] += r->linearize(&Hcalib);

    // Fix linearization if requested.
    if (fixLinearization) {
      r->applyRes(true);

      if (r->efResidual->isActive()) {
        // If the residual is actively new, update the point's max relative
        // baseline.
        if (r->isNew) {
          PointHessian* p = r->point;
          // Projected point assuming infinite depth.
          Vec3f ptp_inf = r->host->targetPrecalc[r->target->idx].PRE_KRKiTll * Vec3f(p->u, p->v, 1);
          // Projected point with real depth.
          Vec3f ptp = ptp_inf + r->host->targetPrecalc[r->target->idx].PRE_KtTll * p->idepth_scaled;
          // 0.01 = one pixel.
          float relBS = 0.01f * ((ptp_inf.head<2>() / ptp_inf[2]) - (ptp.head<2>() / ptp[2])).norm();

          if (relBS > p->maxRelBaseline) {
            p->maxRelBaseline = relBS;
          }

          p->numGoodResiduals++;
        }
      } else {
        // Otherwise, remove the residual.
        toRemove[tid].push_back(activeResiduals[k]);
      }
    }
  }
}

void FullSystem::applyRes_Reductor(
  bool copyJacobians,
  int min,
  int max,
  Vec10* stats,
  int tid
) {
  // Apply residuals for all active ones.
  for (int k = min; k < max; k++) {
    activeResiduals[k]->applyRes(true);
  }
}

void FullSystem::setNewFrameEnergyTH() {
  // Collect all residuals and make decision on TH.
  allResVec.clear();
  allResVec.reserve(activeResiduals.size() * 2);
  FrameHessian* newFrame = hessian_frames_.back();

  for (PointFrameResidual* r : activeResiduals) {
    if (r->state_NewEnergyWithOutlier >= 0.0 && r->target == newFrame) {
      allResVec.push_back(r->state_NewEnergyWithOutlier);
    }
  }

  // If there are no residuals, set a default value and return.
  if (allResVec.empty()) {
    newFrame->frameEnergyTH = 12.0f * 12.0f * patternNum;
    return; // It should never happen, but let's make sure.
  }

  // Find the adaptive residual for the frame.
  int nthIdx = setting_frameEnergyTHN * allResVec.size();
  assert(nthIdx < static_cast<int>(allResVec.size()));
  assert(setting_frameEnergyTHN < 1.0f);
  std::nth_element(allResVec.begin(), allResVec.begin() + nthIdx, allResVec.end());
  float nthElement = std::sqrt(allResVec[nthIdx]);

  newFrame->frameEnergyTH = nthElement * setting_frameEnergyTHFacMedian;
  newFrame->frameEnergyTH = 26.0f * setting_frameEnergyTHConstWeight
                          + newFrame->frameEnergyTH * (1.0f - setting_frameEnergyTHConstWeight);
  newFrame->frameEnergyTH = newFrame->frameEnergyTH * newFrame->frameEnergyTH;
  newFrame->frameEnergyTH *= setting_overallEnergyTHWeight * setting_overallEnergyTHWeight;

  // int good=0,bad=0;
  // for(float f : allResVec) if(f<newFrame->frameEnergyTH) good++; else bad++;
  // printf("EnergyTH: mean %f, median %f, result %f (in %d, out %d)! \n",
  //   meanElement, nthElement, sqrtf(newFrame->frameEnergyTH),
  //   good, bad);
}

Vec3 FullSystem::linearizeAll(bool fixLinearization) {
  double lastEnergyP = 0.0;
  double lastEnergyR = 0.0;
  double num         = 0.0;

  std::vector<PointFrameResidual*> toRemove[NUM_THREADS];
  for (int i = 0; i < NUM_THREADS; i++) {
    toRemove[i].clear();
  }

  // Linearize all residuals in both cases: multi-threaded and single-threaded.
  if (multiThreading) {
    treadReduce.reduce(
      boost::bind(
        &FullSystem::linearizeAll_Reductor,
        this,
        fixLinearization,
        toRemove,
        _1,
        _2,
        _3,
        _4
      ),
      0,
      activeResiduals.size(),
      0
    );
    lastEnergyP = treadReduce.stats[0];
  } else {
    Vec10 stats;
    linearizeAll_Reductor(fixLinearization, toRemove, 0, activeResiduals.size(), &stats, 0);
    lastEnergyP = stats[0];
  }

  setNewFrameEnergyTH();

  if (fixLinearization) {
    // Set state for active residuals.
    for (PointFrameResidual* r : activeResiduals) {
      PointHessian* ph = r->point;
      if (ph->lastResiduals[0].first == r) {
        ph->lastResiduals[0].second = r->state_state;
      } else if (ph->lastResiduals[1].first == r) {
        ph->lastResiduals[1].second = r->state_state;
      }
    }

    // Remove residuals that are not active.
    int nResRemoved = 0;
    for (int i = 0; i < NUM_THREADS; i++) {
      for (PointFrameResidual* r : toRemove[i]) {
        PointHessian* ph = r->point;

        if (ph->lastResiduals[0].first == r){
          ph->lastResiduals[0].first = nullptr;
        } else if (ph->lastResiduals[1].first == r) {
          ph->lastResiduals[1].first = nullptr;
        }

        for (std::size_t k = 0; k < ph->residuals.size(); k++) {
          if (ph->residuals[k] == r) {
            ef_->dropResidual(r->efResidual);
            deleteOut<PointFrameResidual>(ph->residuals, static_cast<int>(k));
            nResRemoved++;
            break;
          }
        }
      }
    }
    // printf("FINAL LINEARIZATION: removed %d / %d residuals!\n", nResRemoved,
    // (int)activeResiduals.size());
  }

  return Vec3(lastEnergyP, lastEnergyR, num);
}

// Apply step to linearization point.
bool FullSystem::doStepFromBackup(
  float stepfacC,
  float stepfacT,
  float stepfacR,
  float stepfacA,
  float stepfacD
) {
  // float meanStepC=0,meanStepP=0,meanStepD=0;
  // meanStepC += Hcalib.step.norm();

  Vec10 pstepfac;
  pstepfac.segment<3>(0).setConstant(stepfacT);
  pstepfac.segment<3>(3).setConstant(stepfacR);
  pstepfac.segment<4>(6).setConstant(stepfacA);

  float sumA = 0.0f, sumB = 0.0f, sumT = 0.0f, sumR = 0.0f, sumID = 0.0f, numID = 0.0f;
  float sumNID = 0.0f;

  if (setting_solverMode & SOLVER_MOMENTUM) {
    // Calculate sums for momentum mode.
    Hcalib.setValue(Hcalib.value_backup + Hcalib.step);
    for (FrameHessian* fh : hessian_frames_) {
      Vec10 step = fh->step;
      step.head<6>() += 0.5f * (fh->step_backup.head<6>());

      fh->setState(fh->state_backup + step);
      sumA += step[6] * step[6];
      sumB += step[7] * step[7];
      sumT += step.segment<3>(0).squaredNorm();
      sumR += step.segment<3>(3).squaredNorm();

      for (PointHessian* ph : fh->pointHessians) {
        float step = ph->step + 0.5f * (ph->step_backup);
        ph->setIdepth(ph->idepth_backup + step);
        sumID += step * step;
        sumNID += std::abs(ph->idepth_backup);
        numID++;

        ph->setIdepthZero(ph->idepth_backup + step);
      }
    }
  } else {
    // Calculate sums for non-momentum mode.
    Hcalib.setValue(Hcalib.value_backup + stepfacC * Hcalib.step);
    for (FrameHessian* fh : hessian_frames_) {
      fh->setState(fh->state_backup + pstepfac.cwiseProduct(fh->step));
      sumA += fh->step[6] * fh->step[6];
      sumB += fh->step[7] * fh->step[7];
      sumT += fh->step.segment<3>(0).squaredNorm();
      sumR += fh->step.segment<3>(3).squaredNorm();

      for (PointHessian* ph : fh->pointHessians) {
        ph->setIdepth(ph->idepth_backup + stepfacD * ph->step);
        sumID += ph->step * ph->step;
        sumNID += std::abs(ph->idepth_backup);
        numID++;

        ph->setIdepthZero(ph->idepth_backup + stepfacD * ph->step);
      }
    }
  }

  sumA   /= hessian_frames_.size();
  sumB   /= hessian_frames_.size();
  sumR   /= hessian_frames_.size();
  sumT   /= hessian_frames_.size();
  sumID  /= numID;
  sumNID /= numID;

  // Debug output.
  if (!setting_debugout_runquiet) {
    printf(
      "STEPS: A %.1f; B %.1f; R %.1f; T %.1f. \t",
      std::sqrt(sumA)          / (0.0005f  * setting_thOptIterations),
      std::sqrt(sumB)          / (0.00005f * setting_thOptIterations),
      std::sqrt(sumR)          / (0.00005f * setting_thOptIterations),
      std::sqrt(sumT) * sumNID / (0.00005f * setting_thOptIterations)
    );
  }

  EFDeltaValid = false;
  setPrecalcValues();

  return std::sqrt(sumA)          < 0.0005f  * setting_thOptIterations
      && std::sqrt(sumB)          < 0.00005f * setting_thOptIterations
      && std::sqrt(sumR)          < 0.00005f * setting_thOptIterations
      && std::sqrt(sumT) * sumNID < 0.00005f * setting_thOptIterations;

  // printf("mean steps: %f %f %f!\n",
  //   meanStepC, meanStepP, meanStepD);
}

// Set linearization point.
void FullSystem::backupState(bool backupLastStep) {
  if (setting_solverMode & SOLVER_MOMENTUM) {
    // Backup state and step for momentum mode.
    if (backupLastStep) {
      Hcalib.step_backup  = Hcalib.step;
      Hcalib.value_backup = Hcalib.value;
      for (FrameHessian* fh : hessian_frames_) {
        fh->step_backup  = fh->step;
        fh->state_backup = fh->get_state();
        for (PointHessian* ph : fh->pointHessians) {
          ph->idepth_backup = ph->idepth;
          ph->step_backup   = ph->step;
        }
      }
    } else {
      Hcalib.step_backup.setZero();
      Hcalib.value_backup = Hcalib.value;
      for (FrameHessian* fh : hessian_frames_) {
        fh->step_backup.setZero();
        fh->state_backup = fh->get_state();
        for (PointHessian* ph : fh->pointHessians) {
          ph->idepth_backup = ph->idepth;
          ph->step_backup   = 0.0f;
        }
      }
    }
  } else {
    // Backup state and step for non-momentum mode.
    Hcalib.value_backup = Hcalib.value;
    for (FrameHessian* fh : hessian_frames_) {
      fh->state_backup = fh->get_state();
      for (PointHessian* ph : fh->pointHessians) {
        ph->idepth_backup = ph->idepth;
      }
    }
  }
}

// Set linearization point.
void FullSystem::loadSateBackup() {
  Hcalib.setValue(Hcalib.value_backup);
  for (FrameHessian* fh : hessian_frames_) {
    fh->setState(fh->state_backup);
    for (PointHessian* ph : fh->pointHessians) {
      ph->setIdepth(ph->idepth_backup);
      ph->setIdepthZero(ph->idepth_backup);
    }
  }

  EFDeltaValid = false;
  setPrecalcValues();
}

double FullSystem::calcMEnergy() {
  if (setting_forceAceptStep) {
    return 0.0;
  }
  // calculate (x-x0)^T * [2b + H * (x-x0)] for everything saved in L.
  // ef->makeIDX();
  // ef->setDeltaF(&Hcalib);
  return ef_->calcMEnergyF();
}

void FullSystem::printOptRes(
  const Vec3& res,
  double resL,
  double resM,
  double resPrior,
  double LExact,
  float a,
  float b
) {
  printf(
    "A(%f)=(AV %.3f). Num: A(%'d) + M(%'d); ab %f %f!\n",
    res[0],
    std::sqrt(static_cast<float>(res[0] / (patternNum * ef_->resInA))),
    ef_->resInA,
    ef_->resInM,
    a,
    b
  );
}

float FullSystem::optimize(int mnumOptIts) {
  if (hessian_frames_.size() < 2) {
    return 0.0f;
  }
  if (hessian_frames_.size() < 3) {
    mnumOptIts = 20;
  }
  if (hessian_frames_.size() < 4) {
    mnumOptIts = 15;
  }

  // Get statistics and active residuals.
  activeResiduals.clear();
  int numPoints = 0;
  int numLRes   = 0;
  for (FrameHessian* fh : hessian_frames_) {
    for (PointHessian* ph : fh->pointHessians) {
      for (PointFrameResidual* r : ph->residuals) {
        if (!r->efResidual->isLinearized) {
          activeResiduals.push_back(r);
          r->resetOOB();
        } else {
          numLRes++;
        }
      }
      numPoints++;
    }
  }

  // Debug output.
  if (!setting_debugout_runquiet) {
    printf(
      "OPTIMIZE %d pts, %zu active res, %d lin res!\n",
      ef_->nPoints,
      activeResiduals.size(),
      numLRes
    );
  }

  Vec3 lastEnergy    = linearizeAll(false);
  double lastEnergyL = calcLEnergy();
  double lastEnergyM = calcMEnergy();

  // Apply residuals for both cases: multi-threaded and single-threaded.
  if (multiThreading) {
    treadReduce.reduce(
      boost::bind(&FullSystem::applyRes_Reductor, this, true, _1, _2, _3, _4),
      0,
      activeResiduals.size(),
      50
    );
  } else {
    applyRes_Reductor(true, 0, activeResiduals.size(), 0, 0);
  }

  // Debug output.
  if (!setting_debugout_runquiet) {
    printf("Initial Error       \t");
    printOptRes(
      lastEnergy,
      lastEnergyL,
      lastEnergyM,
      0.0,
      0.0,
      hessian_frames_.back()->aff_g2l().a,
      hessian_frames_.back()->aff_g2l().b
    );
  }

  debugPlotTracking();

  double lambda  = 1e-1;
  float stepsize = 1.0f;
  VecX previousX = VecX::Constant(static_cast<std::size_t>(CPARS + 8 * hessian_frames_.size()), NAN);
  for (int iteration = 0; iteration < mnumOptIts; iteration++) {
    // Solve the system.
    backupState(iteration != 0);
    // solveSystemNew(0);
    solveSystem(iteration, lambda);
    double incDirChange = (1e-20 + previousX.dot(ef_->lastX))
                        / (1e-20 + previousX.norm() * ef_->lastX.norm());
    previousX = ef_->lastX;

    if (std::isfinite(incDirChange) && (setting_solverMode & SOLVER_STEPMOMENTUM)) {
      float newStepsize = std::exp(incDirChange * 1.4f);
      if (incDirChange < 0.0 && stepsize > 1.0f) {
        stepsize = 1.0f;
      }

      stepsize = std::sqrt(std::sqrt(newStepsize * stepsize * stepsize * stepsize));
      if (stepsize > 2.0f) {
        stepsize = 2.0f;
      }
      if (stepsize < 0.25f) {
        stepsize = 0.25f;
      }
    }

    bool canbreak = doStepFromBackup(stepsize, stepsize, stepsize, stepsize, stepsize);

    // Evaluate new energy.
    Vec3 newEnergy    = linearizeAll(false);
    double newEnergyL = calcLEnergy();
    double newEnergyM = calcMEnergy();

    // Debug output.
    if (!setting_debugout_runquiet) {
      printf(
        "%s %d (L %.2f, dir %.2f, ss %.1f): \t",
        (newEnergy[0] + newEnergy[1] + newEnergyL + newEnergyM
         < lastEnergy[0] + lastEnergy[1] + lastEnergyL + lastEnergyM)
          ? "ACCEPT"
          : "REJECT",
        iteration,
        std::log10(lambda),
        incDirChange,
        stepsize
      );
      printOptRes(
        newEnergy,
        newEnergyL,
        newEnergyM,
        0.0,
        0.0,
        hessian_frames_.back()->aff_g2l().a,
        hessian_frames_.back()->aff_g2l().b
      );
    }

    if (setting_forceAceptStep || (newEnergy[0] + newEnergy[1] + newEnergyL + newEnergyM < lastEnergy[0] + lastEnergy[1] + lastEnergyL + lastEnergyM)) {
      // Apply residuals if forced or if the new energy is better.

      if (multiThreading) {
        treadReduce.reduce(
          boost::bind(&FullSystem::applyRes_Reductor, this, true, _1, _2, _3, _4),
          0,
          activeResiduals.size(),
          50
        );
      } else {
        applyRes_Reductor(true, 0, activeResiduals.size(), 0, 0);
      }

      lastEnergy  = newEnergy;
      lastEnergyL = newEnergyL;
      lastEnergyM = newEnergyM;

      lambda *= 0.25;
    } else {
      // Otherwise, restore the state.
      loadSateBackup();
      lastEnergy  = linearizeAll(false);
      lastEnergyL = calcLEnergy();
      lastEnergyM = calcMEnergy();
      lambda *= 1e2;
    }

    // Break if possible.
    if (canbreak && iteration >= setting_minOptIterations) {
      break;
    }
  }

  Vec10 newStateZero         = Vec10::Zero();
  newStateZero.segment<2>(6) = hessian_frames_.back()->get_state().segment<2>(6);

  hessian_frames_.back()->setEvalPT(hessian_frames_.back()->PRE_worldToCam, newStateZero);
  EFDeltaValid    = false;
  EFAdjointsValid = false;
  ef_->setAdjointsF(&Hcalib);
  setPrecalcValues();

  lastEnergy = linearizeAll(true);

  // Check if the tracking failed.
  if (!std::isfinite(lastEnergy[0]) || !std::isfinite(lastEnergy[1]) || !std::isfinite(lastEnergy[2])) {
    printf("KF Tracking failed: LOST!\n");
    is_lost_ = true;
  }

  statistics_lastFineTrackRMSE = std::sqrt(static_cast<float>(lastEnergy[0] / (patternNum * ef_->resInA)));

  // Log the calibration if requested.
  if (calib_logger_ != nullptr) {
    (*calib_logger_) << Hcalib.value_scaled.transpose() << " "
                << hessian_frames_.back()->get_state_scaled().transpose() << " "
                << std::sqrt(static_cast<float>(lastEnergy[0] / (patternNum * ef_->resInA)))
                << " " << ef_->resInM << "\n";
    calib_logger_->flush();
  }

  {
    boost::unique_lock<boost::mutex> crlock(frame_pose_mutex_);
    for (FrameHessian* fh : hessian_frames_) {
      fh->shell->camToWorld = fh->PRE_camToWorld;
      fh->shell->aff_g2l    = fh->aff_g2l();
    }
  }

  debugPlotTracking();

  return std::sqrt(static_cast<float>(lastEnergy[0] / (patternNum * ef_->resInA)));
}

void FullSystem::solveSystem(int iteration, double lambda) {
  ef_->lastNullspaces_forLogging = getNullspaces(
    ef_->lastNullspaces_pose,
    ef_->lastNullspaces_scale,
    ef_->lastNullspaces_affA,
    ef_->lastNullspaces_affB
  );

  ef_->solveSystemF(iteration, lambda, &Hcalib);
}

double FullSystem::calcLEnergy() {
  // Return 0 if forced to accept the step.
  if (setting_forceAceptStep) {
    return 0.0;
  }

  double Ef = ef_->calcLEnergyF_MT();
  return Ef;
}

void FullSystem::removeOutliers() {
  // Remove outliers of all points of all frames which has no residuals.
  int numPointsDropped = 0;
  for (FrameHessian* fh : hessian_frames_) {
    for (unsigned int i = 0; i < fh->pointHessians.size(); i++) {
      PointHessian* ph = fh->pointHessians[i];
      if (ph == nullptr) {
        continue;
      }

      if (ph->residuals.empty()) {
        fh->pointHessiansOut.push_back(ph);
        ph->efPoint->stateFlag = EFPointStatus::PS_DROP;
        fh->pointHessians[i]   = fh->pointHessians.back();
        fh->pointHessians.pop_back();
        i--;
        numPointsDropped++;
      }
    }
  }
  ef_->dropPointsF();
}

std::vector<VecX> FullSystem::getNullspaces(
  std::vector<VecX>& nullspaces_pose,
  std::vector<VecX>& nullspaces_scale,
  std::vector<VecX>& nullspaces_affA,
  std::vector<VecX>& nullspaces_affB
) {
  // Clear the current nullspaces.
  nullspaces_pose.clear();
  nullspaces_scale.clear();
  nullspaces_affA.clear();
  nullspaces_affB.clear();

  int n = static_cast<int>(CPARS + hessian_frames_.size() * 8);
  std::vector<VecX> nullspaces_x0_pre;
  for (int i = 0; i < 6; i++) {
    VecX nullspace_x0(n);
    nullspace_x0.setZero();
    for (FrameHessian* fh : hessian_frames_) {
      nullspace_x0.segment<6>(CPARS + fh->idx * 8) = fh->nullspaces_pose.col(i);
      nullspace_x0.segment<3>(CPARS + fh->idx * 8) *= SCALE_XI_TRANS_INVERSE;
      nullspace_x0.segment<3>(CPARS + fh->idx * 8 + 3) *= SCALE_XI_ROT_INVERSE;
    }
    nullspaces_x0_pre.push_back(nullspace_x0);
    nullspaces_pose.push_back(nullspace_x0);
  }
  for (int i = 0; i < 2; i++) {
    VecX nullspace_x0(n);
    nullspace_x0.setZero();
    for (FrameHessian* fh : hessian_frames_) {
      nullspace_x0.segment<2>(CPARS + fh->idx * 8 + 6) = fh->nullspaces_affine.col(i).head<2>();
      nullspace_x0[CPARS + fh->idx * 8 + 6] *= SCALE_A_INVERSE;
      nullspace_x0[CPARS + fh->idx * 8 + 7] *= SCALE_B_INVERSE;
    }
    nullspaces_x0_pre.push_back(nullspace_x0);
    if (i == 0) {
      nullspaces_affA.push_back(nullspace_x0);
    }
    if (i == 1) {
      nullspaces_affB.push_back(nullspace_x0);
    }
  }

  VecX nullspace_x0(n);
  nullspace_x0.setZero();
  for (FrameHessian* fh : hessian_frames_) {
    nullspace_x0.segment<6>(CPARS + fh->idx * 8) = fh->nullspaces_scale;
    nullspace_x0.segment<3>(CPARS + fh->idx * 8) *= SCALE_XI_TRANS_INVERSE;
    nullspace_x0.segment<3>(CPARS + fh->idx * 8 + 3) *= SCALE_XI_ROT_INVERSE;
  }
  nullspaces_x0_pre.push_back(nullspace_x0);
  nullspaces_scale.push_back(nullspace_x0);

  return nullspaces_x0_pre;
}

} // namespace dso
