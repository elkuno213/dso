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
#include "FullSystem/ImmaturePoint.h"
#include "IOWrapper/ImageDisplay.h"
#include "util/globalCalib.h"
#include "util/globalFuncs.h"

namespace dso {

PointHessian* FullSystem::optimizeImmaturePoint(
  ImmaturePoint* point,
  int minObs,
  ImmaturePointTemporaryResidual* residuals
) {
  // Calculate the number of frames that are not the host frame and update the
  // residuals.
  int nres = 0;
  for (FrameHessian* fh : hessian_frames_) {
    if (fh != point->host) {
      residuals[nres].state_NewEnergy = residuals[nres].state_energy = 0.0;
      residuals[nres].state_NewState = ResState::OUTLIER;
      residuals[nres].state_state    = ResState::IN;
      residuals[nres].target         = fh;
      nres++;
    }
  }
  assert(nres == static_cast<int>(hessian_frames_.size() - 1));

  bool print = false; // rand() % 50 == 0;

  // Calculate the sum of energy of the residuals.
  float lastEnergy    = 0.0f;
  float lastHdd       = 0.0f;
  float lastbd        = 0.0f;
  float currentIdepth = (point->idepth_max + point->idepth_min) * 0.5f;

  for (int i = 0; i < nres; i++) {
    lastEnergy += point->linearizeResidual(
      &Hcalib,
      1000.0f,
      residuals + i,
      lastHdd,
      lastbd,
      currentIdepth
    );
    residuals[i].state_state  = residuals[i].state_NewState;
    residuals[i].state_energy = residuals[i].state_NewEnergy;
  }

  // Check if the point is well-constrained.
  if (!std::isfinite(lastEnergy) || lastHdd < setting_minIdepthH_act) {
    if (print) {
      printf(
        "OptPoint: Not well-constrained (%d res, H=%.1f). E=%f. SKIP!\n",
        nres,
        lastHdd,
        lastEnergy
      );
    }
    return nullptr;
  }

  if (print) {
    printf(
      "Activate point. %d residuals. H=%f. Initial Energy: %f. Initial Id=%f\n",
      nres,
      lastHdd,
      lastEnergy,
      currentIdepth
    );
  }

  // Calculate the smallest sum of energy through a number of iterations.
  float lambda = 0.1f;
  for (int iteration = 0; iteration < setting_GNItsOnPointActivation; iteration++) {
    float H = lastHdd;
    H *= 1.0f + lambda;
    float step      = (1.0f / H) * lastbd;
    float newIdepth = currentIdepth - step;

    float newHdd    = 0.0f;
    float newbd     = 0.0f;
    float newEnergy = 0.0f;
    for (int i = 0; i < nres; i++) {
      newEnergy += point->linearizeResidual(&Hcalib, 1.0f, residuals + i, newHdd, newbd, newIdepth);
    }

    if (!std::isfinite(lastEnergy) || newHdd < setting_minIdepthH_act) {
      if (print) {
        printf(
          "OptPoint: Not well-constrained (%d res, H=%.1f). E=%f. SKIP!\n",
          nres,
          newHdd,
          lastEnergy
        );
      }
      return nullptr;
    }

    if (print) {
      printf(
        "%s %d (L %.2f) %s: %f -> %f (idepth %f)!\n",
        (true || newEnergy < lastEnergy) ? "ACCEPT" : "REJECT",
        iteration,
        std::log10(lambda),
        "",
        lastEnergy,
        newEnergy,
        newIdepth
      );
    }

    if (newEnergy < lastEnergy) {
      currentIdepth = newIdepth;
      lastHdd       = newHdd;
      lastbd        = newbd;
      lastEnergy    = newEnergy;
      for (int i = 0; i < nres; i++) {
        residuals[i].state_state  = residuals[i].state_NewState;
        residuals[i].state_energy = residuals[i].state_NewEnergy;
      }

      lambda *= 0.5f;
    } else {
      lambda *= 5.0f;
    }

    if (std::abs(step) < 0.0001f * currentIdepth) {
      break;
    }
  }

  // Check if the point's depth is finite.
  if (!std::isfinite(currentIdepth)) {
    printf(
      "MAJOR ERROR! point idepth is nan after initialization (%f).\n",
      currentIdepth
    );
    return (PointHessian*)((long)(-1)); // Yeah I'm like 99% sure this is OK on 32bit systems.
  }

  // Count the number of good residuals.
  int numGoodRes = 0;
  for (int i = 0; i < nres; i++) {
    if (residuals[i].state_state == ResState::IN) {
      numGoodRes++;
    }
  }

  // Check if the number of good residuals is below the minimum.
  if (numGoodRes < minObs) {
    if (print) {
      printf("OptPoint: OUTLIER!\n");
    }
    return (PointHessian*)((long)(-1)); // Yeah I'm like 99% sure this is OK on 32bit systems.
  }

  // Create a new point and add the residuals.
  PointHessian* p = new PointHessian(point, &Hcalib);
  if (!std::isfinite(p->energyTH)) {
    delete p;
    return (PointHessian*)((long)(-1));
  }

  p->lastResiduals[0].first  = nullptr;
  p->lastResiduals[0].second = ResState::OOB;
  p->lastResiduals[1].first  = nullptr;
  p->lastResiduals[1].second = ResState::OOB;
  p->setIdepthZero(currentIdepth);
  p->setIdepth(currentIdepth);
  p->setPointStatus(PointHessian::ACTIVE);

  for (int i = 0; i < nres; i++)
    if (residuals[i].state_state == ResState::IN) {
      PointFrameResidual* r = new PointFrameResidual(p, p->host, residuals[i].target);
      r->state_NewEnergy = r->state_energy = 0;
      r->state_NewState                    = ResState::OUTLIER;
      r->setState(ResState::IN);
      p->residuals.push_back(r);

      if (r->target == hessian_frames_.back()) {
        p->lastResiduals[0].first  = r;
        p->lastResiduals[0].second = ResState::IN;
      } else if(r->target == (hessian_frames_.size() < 2 ? nullptr : hessian_frames_[hessian_frames_.size() - 2])) {
        p->lastResiduals[1].first  = r;
        p->lastResiduals[1].second = ResState::IN;
      }
    }

  if (print) {
    printf("point activated!\n");
  }

  stats_num_activated_pts_++;
  return p;
}

} // namespace dso
