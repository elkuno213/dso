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
#include <cstdio>

#include <Eigen/Eigenvalues>
#include <Eigen/LU>
#include <Eigen/SVD>

#include "FullSystem/FullSystem.h"
#include "FullSystem/ImmaturePoint.h"
#include "IOWrapper/ImageDisplay.h"
#include "IOWrapper/ImageRW.h"
#include "util/globalCalib.h"
#include "util/globalFuncs.h"

namespace dso {

void FullSystem::debugPlotTracking() {
  // No display if disabled.
  if (disableAllDisplay) {
    return;
  }
  // No display if render setting is disabled.
  if (!setting_render_plotTrackingFull) {
    return;
  }

  // Calculate the number of pixels.
  int wh = hG[0] * wG[0];

  // Loop through all frames and make images for each frame.
  int idx = 0;
  for (FrameHessian* f : hessian_frames_) {
    std::vector<MinimalImageB3*> images;

    // Make images for all frames. They will be deleted by the FrameHessian's
    // destructor.
    for (FrameHessian* f2 : hessian_frames_) {
      if (f2->debugImage == nullptr) {
        f2->debugImage = new MinimalImageB3(wG[0], hG[0]);
      }
    }

    for (FrameHessian* f2 : hessian_frames_) {
      MinimalImageB3* debugImage = f2->debugImage;
      images.push_back(debugImage);

      Eigen::Vector3f* fd = f2->dI;

      Vec2 affL = AffLight::fromToVecExposure(
        f2->ab_exposure,
        f->ab_exposure,
        f2->aff_g2l(),
        f->aff_g2l()
      );

      for (int i = 0; i < wh; i++) {
        // Transfer brightness.
        float colL = affL[0] * fd[i][0] + affL[1];
        if (colL < 0.0f) {
          colL = 0.0f;
        }
        if (colL > 255.0f) {
          colL = 255.0f;
        }
        debugImage->at(i) = Vec3b(colL, colL, colL);
      }
    }

    for (PointHessian* ph : f->pointHessians) {
      assert(ph->status == PointHessian::ACTIVE);
      if (ph->status == PointHessian::ACTIVE || ph->status == PointHessian::MARGINALIZED) {
        for (PointFrameResidual* r : ph->residuals) {
          r->debugPlot();
        }
        f->debugImage->setPixel9(
          static_cast<int>(ph->u + 0.5f),
          static_cast<int>(ph->v + 0.5f),
          makeRainbow3B(ph->idepth_scaled)
        );
      }
    }

    // Add the images to the visualization wrapper.
    char buf[100];
    snprintf(buf, 100, "IMG %d", idx);
    IOWrap::displayImageStitch(buf, images);
    idx++;
  }

  IOWrap::waitKey(0);
}

void FullSystem::debugPlot(std::string name) {
  // No display if disabled.
  if (disableAllDisplay) {
    return;
  }
  // No display if render setting is disabled.
  if (!setting_render_renderWindowFrames) {
    return;
  }

  std::vector<MinimalImageB3*> images;

  // Calculate the maximum and minimum IDs.
  float minID = 0.0f, maxID = 0.0f;
  if (static_cast<int>(freeDebugParam5 + 0.5f) == 7 || (debugSaveImages && false)) {
    std::vector<float> allID;
    for (std::size_t f = 0; f < hessian_frames_.size(); f++) {
      for (PointHessian* ph : hessian_frames_[f]->pointHessians) {
        if (ph != nullptr) {
          allID.push_back(ph->idepth_scaled);
        }
      }

      for (PointHessian* ph : hessian_frames_[f]->pointHessiansMarginalized) {
        if (ph != nullptr) {
          allID.push_back(ph->idepth_scaled);
        }
      }

      for (PointHessian* ph : hessian_frames_[f]->pointHessiansOut) {
        if (ph != nullptr) {
          allID.push_back(ph->idepth_scaled);
        }
      }
    }
    std::sort(allID.begin(), allID.end());
    std::size_t n = allID.size() - 1;
    minID = allID[static_cast<std::size_t>(n * 0.05)];
    maxID = allID[static_cast<std::size_t>(n * 0.95)];

    // Slowly adapt: change by maximum 10% of old span.
    float maxChange = 0.1f * (max_id_jet_vis_ - min_id_jet_vis_);
    if (max_id_jet_vis_ < 0.0f || min_id_jet_vis_ < 0.0f) {
      maxChange = 1e5f;
    }

    if (minID < min_id_jet_vis_ - maxChange) {
      minID = min_id_jet_vis_ - maxChange;
    }
    if (minID > min_id_jet_vis_ + maxChange) {
      minID = min_id_jet_vis_ + maxChange;
    }

    if (maxID < max_id_jet_vis_ - maxChange) {
      maxID = max_id_jet_vis_ - maxChange;
    }
    if (maxID > max_id_jet_vis_ + maxChange) {
      maxID = max_id_jet_vis_ + maxChange;
    }

    max_id_jet_vis_ = maxID;
    min_id_jet_vis_ = minID;
  }

  // Loop through all frames and make images for each frame.
  int wh = hG[0] * wG[0];
  for (std::size_t f = 0; f < hessian_frames_.size(); f++) {
    MinimalImageB3* img = new MinimalImageB3(wG[0], hG[0]);
    images.push_back(img);
    // float* fd = hessian_frames_[f]->I;
    Eigen::Vector3f* fd = hessian_frames_[f]->dI;

    for (int i = 0; i < wh; i++) {
      int c = static_cast<int>(fd[i][0] * 0.9f);
      if (c > 255) {
        c = 255;
      }
      img->at(i) = Vec3b(c, c, c);
    }

    if (static_cast<int>(freeDebugParam5 + 0.5f) == 0) {
      for (PointHessian* ph : hessian_frames_[f]->pointHessians) {
        if (ph == nullptr) {
          continue;
        }

        img->setPixelCirc(
          static_cast<int>(ph->u + 0.5f),
          static_cast<int>(ph->v + 0.5f),
          makeRainbow3B(ph->idepth_scaled)
        );
      }
      for (PointHessian* ph : hessian_frames_[f]->pointHessiansMarginalized) {
        if (ph == nullptr) {
          continue;
        }
        img->setPixelCirc(
          static_cast<int>(ph->u + 0.5f),
          static_cast<int>(ph->v + 0.5f),
          makeRainbow3B(ph->idepth_scaled)
        );
      }
      for (PointHessian* ph : hessian_frames_[f]->pointHessiansOut) {
        img->setPixelCirc(
          static_cast<int>(ph->u + 0.5f),
          static_cast<int>(ph->v + 0.5f),
          Vec3b(255, 255, 255)
        );
      }
    } else if (static_cast<int>(freeDebugParam5 + 0.5f) == 1) {
      for (PointHessian* ph : hessian_frames_[f]->pointHessians) {
        if (ph == nullptr) {
          continue;
        }
        img->setPixelCirc(
          static_cast<int>(ph->u + 0.5f),
          static_cast<int>(ph->v + 0.5f),
          makeRainbow3B(ph->idepth_scaled)
        );
      }

      for (PointHessian* ph : hessian_frames_[f]->pointHessiansMarginalized) {
        img->setPixelCirc(
          static_cast<int>(ph->u + 0.5f),
          static_cast<int>(ph->v + 0.5f),
          Vec3b(0, 0, 0)
        );
      }

      for (PointHessian* ph : hessian_frames_[f]->pointHessiansOut) {
        img->setPixelCirc(
          static_cast<int>(ph->u + 0.5f),
          static_cast<int>(ph->v + 0.5f),
          Vec3b(255, 255, 255)
        );
      }
    } else if (static_cast<int>(freeDebugParam5 + 0.5f) == 2) {
      //
    } else if (static_cast<int>(freeDebugParam5 + 0.5f) == 3) {
      for (ImmaturePoint* ph : hessian_frames_[f]->immaturePoints) {
        if (ph == nullptr) {
          continue;
        }
        if (ph->lastTraceStatus == ImmaturePointStatus::IPS_GOOD || ph->lastTraceStatus == ImmaturePointStatus::IPS_SKIPPED || ph->lastTraceStatus == ImmaturePointStatus::IPS_BADCONDITION) {
          if (!std::isfinite(ph->idepth_max)) {
            img->setPixelCirc(
              static_cast<int>(ph->u + 0.5f),
              static_cast<int>(ph->v + 0.5f),
              Vec3b(0, 0, 0)
            );
          } else {
            img->setPixelCirc(
              static_cast<int>(ph->u + 0.5f),
              static_cast<int>(ph->v + 0.5f),
              makeRainbow3B((ph->idepth_min + ph->idepth_max) * 0.5f)
            );
          }
        }
      }
    } else if (static_cast<int>(freeDebugParam5 + 0.5f) == 4) {
      for (ImmaturePoint* ph : hessian_frames_[f]->immaturePoints) {
        if (ph == nullptr) {
          continue;
        }

        if (ph->lastTraceStatus == ImmaturePointStatus::IPS_GOOD) {
          img->setPixelCirc(
            static_cast<int>(ph->u + 0.5f),
            static_cast<int>(ph->v + 0.5f),
            Vec3b(0, 255, 0)
          );
        }
        if (ph->lastTraceStatus == ImmaturePointStatus::IPS_OOB) {
          img->setPixelCirc(
            static_cast<int>(ph->u + 0.5f),
            static_cast<int>(ph->v + 0.5f),
            Vec3b(255, 0, 0)
          );
        }
        if (ph->lastTraceStatus == ImmaturePointStatus::IPS_OUTLIER) {
          img->setPixelCirc(
            static_cast<int>(ph->u + 0.5f),
            static_cast<int>(ph->v + 0.5f),
            Vec3b(0, 0, 255)
          );
        }
        if (ph->lastTraceStatus == ImmaturePointStatus::IPS_SKIPPED) {
          img->setPixelCirc(
            static_cast<int>(ph->u + 0.5f),
            static_cast<int>(ph->v + 0.5f),
            Vec3b(255, 255, 0)
          );
        }
        if (ph->lastTraceStatus == ImmaturePointStatus::IPS_BADCONDITION) {
          img->setPixelCirc(
            static_cast<int>(ph->u + 0.5f),
            static_cast<int>(ph->v + 0.5f),
            Vec3b(255, 255, 255)
          );
        }
        if (ph->lastTraceStatus == ImmaturePointStatus::IPS_UNINITIALIZED) {
          img->setPixelCirc(
            static_cast<int>(ph->u + 0.5f),
            static_cast<int>(ph->v + 0.5f),
            Vec3b(0, 0, 0)
          );
        }
      }
    } else if (static_cast<int>(freeDebugParam5 + 0.5f) == 5) {
      for (ImmaturePoint* ph : hessian_frames_[f]->immaturePoints) {
        if (ph == nullptr) {
          continue;
        }

        if (ph->lastTraceStatus == ImmaturePointStatus::IPS_UNINITIALIZED) {
          continue;
        }
        float d = freeDebugParam1 * (sqrtf(ph->quality) - 1);
        if (d < 0.0f) {
          d = 0.0f;
        }
        if (d > 1.0f) {
          d = 1.0f;
        }
        img->setPixelCirc(
          static_cast<int>(ph->u + 0.5f),
          static_cast<int>(ph->v + 0.5f),
          Vec3b(0, d * 255, (1 - d) * 255)
        );
      }

    } else if (static_cast<int>(freeDebugParam5 + 0.5f) == 6) {
      for (PointHessian* ph : hessian_frames_[f]->pointHessians) {
        if (ph == nullptr) {
          continue;
        }
        if (ph->my_type == 0.0f) {
          img->setPixelCirc(
            static_cast<int>(ph->u + 0.5f),
            static_cast<int>(ph->v + 0.5f),
            Vec3b(255, 0, 255)
          );
        }
        if (ph->my_type == 1.0f) {
          img->setPixelCirc(
            static_cast<int>(ph->u + 0.5f),
            static_cast<int>(ph->v + 0.5f),
            Vec3b(255, 0, 0)
          );
        }
        if (ph->my_type == 2.0f) {
          img->setPixelCirc(
            static_cast<int>(ph->u + 0.5f),
            static_cast<int>(ph->v + 0.5f),
            Vec3b(0, 0, 255)
          );
        }
        if (ph->my_type == 3.0f) {
          img->setPixelCirc(
            static_cast<int>(ph->u + 0.5f),
            static_cast<int>(ph->v + 0.5f),
            Vec3b(0, 255, 255)
          );
        }
      }
      for (PointHessian* ph : hessian_frames_[f]->pointHessiansMarginalized) {
        if (ph == nullptr) {
          continue;
        }
        if (ph->my_type == 0.0f) {
          img->setPixelCirc(
            static_cast<int>(ph->u + 0.5f),
            static_cast<int>(ph->v + 0.5f),
            Vec3b(255, 0, 255)
          );
        }
        if (ph->my_type == 1.0f) {
          img->setPixelCirc(
            static_cast<int>(ph->u + 0.5f),
            static_cast<int>(ph->v + 0.5f),
            Vec3b(255, 0, 0)
          );
        }
        if (ph->my_type == 2.0f) {
          img->setPixelCirc(
            static_cast<int>(ph->u + 0.5f),
            static_cast<int>(ph->v + 0.5f),
            Vec3b(0, 0, 255)
          );
        }
        if (ph->my_type == 3.0f) {
          img->setPixelCirc(
            static_cast<int>(ph->u + 0.5f),
            static_cast<int>(ph->v + 0.5f),
            Vec3b(0, 255, 255)
          );
        }
      }
    }
    if (static_cast<int>(freeDebugParam5 + 0.5f) == 7) {
      for (PointHessian* ph : hessian_frames_[f]->pointHessians) {
        img->setPixelCirc(
          static_cast<int>(ph->u + 0.5f),
          static_cast<int>(ph->v + 0.5f),
          makeJet3B((ph->idepth_scaled - minID) / ((maxID - minID)))
        );
      }
      for (PointHessian* ph : hessian_frames_[f]->pointHessiansMarginalized) {
        if (ph == nullptr) {
          continue;
        }
        img->setPixelCirc(
          static_cast<int>(ph->u + 0.5f),
          static_cast<int>(ph->v + 0.5f),
          Vec3b(0, 0, 0)
        );
      }
    }
  }
  // Add the images to the visualization wrapper.
  IOWrap::displayImageStitch(name.c_str(), images);
  IOWrap::waitKey(5);

  // Delete the images.
  for (std::size_t i = 0; i < images.size(); i++) {
    delete images[i];
  }

  // Save the images if debug setting is enabled.
  if ((debugSaveImages && false)) {
    for (std::size_t f = 0; f < hessian_frames_.size(); f++) {
      MinimalImageB3* img = new MinimalImageB3(wG[0], hG[0]);
      Eigen::Vector3f* fd = hessian_frames_[f]->dI;

      for (int i = 0; i < wh; i++) {
        int c = fd[i][0] * 0.9f;
        if (c > 255) {
          c = 255;
        }
        img->at(i) = Vec3b(c, c, c);
      }

      for (PointHessian* ph : hessian_frames_[f]->pointHessians) {
        img->setPixelCirc(
          static_cast<int>(ph->u + 0.5f),
          static_cast<int>(ph->v + 0.5f),
          makeJet3B((ph->idepth_scaled - minID) / ((maxID - minID)))
        );
      }
      for (PointHessian* ph : hessian_frames_[f]->pointHessiansMarginalized) {
        if (ph == nullptr) {
          continue;
        }
        img->setPixelCirc(
          static_cast<int>(ph->u + 0.5f),
          static_cast<int>(ph->v + 0.5f),
          Vec3b(0, 0, 0)
        );
      }

      char buf[1000];
      snprintf(
        buf,
        1000,
        "images_out/kf_%05d_%05d_%02zu.png",
        hessian_frames_.back()->shell->id,
        hessian_frames_.back()->frameID,
        f
      );
      IOWrap::writeImage(buf, img);

      delete img;
    }
  }
}

} // namespace dso
