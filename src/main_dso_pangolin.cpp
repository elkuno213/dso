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

#include <locale.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <thread>

#include "FullSystem/FullSystem.h"
#include "FullSystem/PixelSelector2.h"
#include "IOWrapper/ImageDisplay.h"
#include "IOWrapper/Output3DWrapper.h"
#include "IOWrapper/OutputWrapper/SampleOutputWrapper.h"
#include "IOWrapper/Pangolin/PangolinDSOViewer.h"
#include "OptimizationBackend/MatrixAccumulators.h"
#include "util/DatasetReader.h"
#include "util/NumType.h"
#include "util/globalCalib.h"
#include "util/globalFuncs.h"
#include "util/settings.h"

std::string vignette    = "";
std::string gamma_calib = "";
std::string source      = "";
std::string calib       = "";
double rescale          = 1.0;
bool reverse            = false;
bool disable_ros        = false;
int start               = 0;
int end                 = 100000;
bool prefetch           = false;
float playback_speed    = 0.f; // 0 for linearize (play as fast as possible, while sequentializing
                               // tracking & mapping). otherwise, factor on timestamps.
bool preload            = false;
bool use_sample_output  = false;
int mode                = 0;
bool first_ros_spin     = false;

using namespace dso;

void exit_handler(int s) {
  printf("Caught signal %d\n", s);
  exit(1);
}

void exit_func() {
  struct sigaction sig_int_handler;
  sig_int_handler.sa_handler = exit_handler;
  sigemptyset(&sig_int_handler.sa_mask);
  sig_int_handler.sa_flags = 0;
  sigaction(SIGINT, &sig_int_handler, NULL);

  first_ros_spin = true;
  while (true) {
    pause();
  }
}

void set_default_settings(const int preset) {
  printf("\n=============== PRESET Settings: ===============\n");
  if (preset == 0 || preset == 1) {
    printf(
      "DEFAULT settings:\n"
      "- %s real-time enforcing\n"
      "- 2000 active points\n"
      "- 5-7 active frames\n"
      "- 1-6 LM iteration each KF\n"
      "- original image resolution\n",
      preset == 0 ? "no " : "1x"
    );

    playback_speed                 = (preset == 0 ? 0.f : 1.f);
    preload                        = preset == 1;
    setting_desiredImmatureDensity = 1500;
    setting_desiredPointDensity    = 2000;
    setting_minFrames              = 5;
    setting_maxFrames              = 7;
    setting_maxOptIterations       = 6;
    setting_minOptIterations       = 1;

    setting_logStuff = false;
  }

  if (preset == 2 || preset == 3) {
    printf(
      "FAST settings:\n"
      "- %s real-time enforcing\n"
      "- 800 active points\n"
      "- 4-6 active frames\n"
      "- 1-4 LM iteration each KF\n"
      "- 424 x 320 image resolution\n",
      preset == 0 ? "no " : "5x"
    );

    playback_speed                 = (preset == 2 ? 0.f : 5.f);
    preload                        = preset == 3;
    setting_desiredImmatureDensity = 600;
    setting_desiredPointDensity    = 800;
    setting_minFrames              = 4;
    setting_maxFrames              = 6;
    setting_maxOptIterations       = 4;
    setting_minOptIterations       = 1;

    benchmarkSetting_width  = 424;
    benchmarkSetting_height = 320;

    setting_logStuff = false;
  }

  printf("==============================================\n");
}

void parse_arg(char* arg) {
  int option_int;
  float option_float;
  char buf[1000];

  if (1 == sscanf(arg, "sampleoutput=%d", &option_int)) {
    if (option_int == 1) {
      use_sample_output = true;
      printf("USING SAMPLE OUTPUT WRAPPER!\n");
    }
    return;
  }

  if (1 == sscanf(arg, "quiet=%d", &option_int)) {
    if (option_int == 1) {
      setting_debugout_runquiet = true;
      printf("QUIET MODE, I'll shut up!\n");
    }
    return;
  }

  if (1 == sscanf(arg, "preset=%d", &option_int)) {
    set_default_settings(option_int);
    return;
  }

  if (1 == sscanf(arg, "rec=%d", &option_int)) {
    if (option_int == 0) {
      disableReconfigure = true;
      printf("DISABLE RECONFIGURE!\n");
    }
    return;
  }

  if (1 == sscanf(arg, "noros=%d", &option_int)) {
    if (option_int == 1) {
      disable_ros        = true;
      disableReconfigure = true;
      printf("DISABLE ROS (AND RECONFIGURE)!\n");
    }
    return;
  }

  if (1 == sscanf(arg, "nolog=%d", &option_int)) {
    if (option_int == 1) {
      setting_logStuff = false;
      printf("DISABLE LOGGING!\n");
    }
    return;
  }
  if (1 == sscanf(arg, "reverse=%d", &option_int)) {
    if (option_int == 1) {
      reverse = true;
      printf("REVERSE!\n");
    }
    return;
  }
  if (1 == sscanf(arg, "nogui=%d", &option_int)) {
    if (option_int == 1) {
      disableAllDisplay = true;
      printf("NO GUI!\n");
    }
    return;
  }
  if (1 == sscanf(arg, "nomt=%d", &option_int)) {
    if (option_int == 1) {
      multiThreading = false;
      printf("NO MultiThreading!\n");
    }
    return;
  }
  if (1 == sscanf(arg, "prefetch=%d", &option_int)) {
    if (option_int == 1) {
      prefetch = true;
      printf("PREFETCH!\n");
    }
    return;
  }
  if (1 == sscanf(arg, "start=%d", &option_int)) {
    start = option_int;
    printf("START AT %d!\n", start);
    return;
  }
  if (1 == sscanf(arg, "end=%d", &option_int)) {
    end = option_int;
    printf("END AT %d!\n", start);
    return;
  }

  if (1 == sscanf(arg, "files=%s", buf)) {
    source = buf;
    printf("loading data from %s!\n", source.c_str());
    return;
  }

  if (1 == sscanf(arg, "calib=%s", buf)) {
    calib = buf;
    printf("loading calibration from %s!\n", calib.c_str());
    return;
  }

  if (1 == sscanf(arg, "vignette=%s", buf)) {
    vignette = buf;
    printf("loading vignette from %s!\n", vignette.c_str());
    return;
  }

  if (1 == sscanf(arg, "gamma=%s", buf)) {
    gamma_calib = buf;
    printf("loading gammaCalib from %s!\n", gamma_calib.c_str());
    return;
  }

  if (1 == sscanf(arg, "rescale=%f", &option_float)) {
    rescale = option_float;
    printf("RESCALE %f!\n", rescale);
    return;
  }

  if (1 == sscanf(arg, "speed=%f", &option_float)) {
    playback_speed = option_float;
    printf("PLAYBACK SPEED %f!\n", playback_speed);
    return;
  }

  if (1 == sscanf(arg, "save=%d", &option_int)) {
    if (option_int == 1) {
      debugSaveImages = true;
      if (42 == system("rm -rf images_out"))
        printf(
          "system call returned 42 - what are the odds?. This is only here "
          "to shut up the compiler.\n"
        );
      if (42 == system("mkdir images_out"))
        printf(
          "system call returned 42 - what are the odds?. This is only here "
          "to shut up the compiler.\n"
        );
      if (42 == system("rm -rf images_out"))
        printf(
          "system call returned 42 - what are the odds?. This is only here "
          "to shut up the compiler.\n"
        );
      if (42 == system("mkdir images_out"))
        printf(
          "system call returned 42 - what are the odds?. This is only here "
          "to shut up the compiler.\n"
        );
      printf("SAVE IMAGES!\n");
    }
    return;
  }

  if (1 == sscanf(arg, "mode=%d", &option_int)) {
    mode = option_int;
    if (option_int == 0) {
      printf("PHOTOMETRIC MODE WITH CALIBRATION!\n");
    }
    if (option_int == 1) {
      printf("PHOTOMETRIC MODE WITHOUT CALIBRATION!\n");
      setting_photometricCalibration = 0;
      setting_affineOptModeA = 0; //-1: fix. >=0: optimize (with prior, if > 0).
      setting_affineOptModeB = 0; //-1: fix. >=0: optimize (with prior, if > 0).
    }
    if (option_int == 2) {
      printf("PHOTOMETRIC MODE WITH PERFECT IMAGES!\n");
      setting_photometricCalibration = 0;
      setting_affineOptModeA = -1; //-1: fix. >=0: optimize (with prior, if > 0).
      setting_affineOptModeB = -1; //-1: fix. >=0: optimize (with prior, if > 0).
      setting_minGradHistAdd = 3;
    }
    return;
  }

  printf("could not parse argument \"%s\"!!!!\n", arg);
}

int main(int argc, char** argv) {
  // setlocale(LC_ALL, "");

  // Parse commandline arguments.
  for (int i = 1; i < argc; i++) {
    parse_arg(argv[i]);
  }

  // Initialize the image reader.
  ImageFolderReader* reader
    = new ImageFolderReader(source, calib, gamma_calib, vignette);
  reader->setGlobalCalibration();
  // Check if photometric calibration is available.
  if (setting_photometricCalibration > 0 && reader->getPhotometricGamma() == nullptr) {
    printf(
      "ERROR: dont't have photometric calibation. Need to use commandline "
      "options mode=1 or mode=2 "
    );
    exit(1);
  }
  // Update image ids if reverse is set.
  int id_start = start;
  int id_end   = end;
  int id_incr  = 1;
  if (reverse) {
    printf("REVERSE!!!!");
    id_start = end - 1;
    if (id_start >= reader->getNumImages()) {
      id_start = reader->getNumImages() - 1;
    }
    id_end = start;
    id_incr = -1;
  }

  // Initialize the full system.
  FullSystem* full_system = new FullSystem();
  full_system->setGammaFunction(reader->getPhotometricGamma());
  full_system->linear_operation_ = (std::abs(playback_speed) < 1e-6f);

  // Initialize the output wrapper for the viewer and push it to the full system
  // if display is enabled.
  IOWrap::PangolinDSOViewer* viewer = nullptr;
  if (!disableAllDisplay) {
    viewer = new IOWrap::PangolinDSOViewer(wG[0], hG[0], false);
    full_system->output_3d_wrappers_.push_back(viewer);
  }

  if (use_sample_output) {
    full_system->output_3d_wrappers_.push_back(new IOWrap::SampleOutputWrapper());
  }

  // Initialize the exit thread (hook ctrl+C).
  std::thread exit_thread(exit_func);

  // Start the run thread.
  std::thread run_thread([&]() {
    // Create all ids and times to play at (for forward and reverse playback).
    std::vector<int> ids;
    std::vector<double> stamps;
    for (
      int id = id_start;
      id >= 0 && id < reader->getNumImages() && id_incr * id < id_incr * id_end;
      id += id_incr
    ) {
      ids.push_back(id);
      if (stamps.empty()) {
        stamps.push_back(0.0);
      } else {
        const double curr_stamp = reader->getTimestamp(ids.back()); // last frame
        const double prev_stamp = reader->getTimestamp(*std::prev(ids.end(), 2)); // second-to-last
        stamps.push_back(stamps.back() + std::abs(curr_stamp - prev_stamp) / playback_speed);
      }
    }

    // Preload all images if enabled.
    std::vector<ImageAndExposure*> preloaded_images;
    if (preload) {
      printf("LOADING ALL IMAGES!\n");
      for (const auto& id : ids) {
        preloaded_images.push_back(reader->getImage(id));
      }
    }

    // Initialize the clock for measuring elapsed time.
    struct timeval tv_start;
    gettimeofday(&tv_start, NULL);
    clock_t started      = clock();
    double stamp_initial = 0.0;

    // Loop through all images to play.
    for (std::size_t j = 0; j < ids.size(); j++) {
      // If the system is not initialized, reset the clock.
      if (!full_system->is_initialized_) {
        gettimeofday(&tv_start, NULL);
        started       = clock();
        stamp_initial = stamps[j];
      }

      const int id = ids[j];

      // Load the image.
      ImageAndExposure* img;
      if (preload) {
        img = preloaded_images[j];
      } else {
        img = reader->getImage(id);
      }

      // Sleep or skip frames based on the playback speed and timestamps.
      bool skip_frame = false;
      if (std::abs(playback_speed) > 1e-6f) {
        struct timeval tv_now;
        gettimeofday(&tv_now, NULL);
        const double stamp_curr
          = stamp_initial
          + ((tv_now.tv_sec - tv_start.tv_sec) + (tv_now.tv_usec - tv_start.tv_usec) / 1e6f);

        if (stamp_curr < stamps[j]) {
          usleep(static_cast<int>((stamps[j] - stamp_curr) * 1e6));
        } else if (stamp_curr > stamps[j] + 0.5 + 0.1 * static_cast<double>(j % 2)) {
          printf(
            "SKIPFRAME %zu (play at %f, now it is %f)!\n",
            j,
            stamps[j],
            stamp_curr
          );
          skip_frame = true;
        }
      }

      // Add the active frame to the full system if it is not skipped.
      if (!skip_frame) {
        full_system->addActiveFrame(img, id);
      }

      // Delete the image.
      delete img;

      // Reset the system if it failed or a full reset is requested.
      if (full_system->is_initialization_failed_ || setting_fullResetRequested) {
        if (j < 250 || setting_fullResetRequested) {
          printf("RESETTING!\n");

          std::vector<IOWrap::Output3DWrapper*> wraps = full_system->output_3d_wrappers_;
          delete full_system;

          for (IOWrap::Output3DWrapper* wrap : wraps) {
            wrap->reset();
          }

          full_system = new FullSystem();
          full_system->setGammaFunction(reader->getPhotometricGamma());
          full_system->linear_operation_ = (std::abs(playback_speed) < 1e-6f);

          full_system->output_3d_wrappers_ = wraps;

          setting_fullResetRequested = false;
        }
      }
      // Break if the system is lost.
      if (full_system->is_lost_) {
        printf("LOST!!\n");
        break;
      }
    }

    full_system->blockUntilMappingIsFinished();

    // Stop the clock and print the results.
    clock_t ended = clock();
    struct timeval tv_end;
    gettimeofday(&tv_end, NULL);

    full_system->printResult("result.txt");

    const int num_frames_processed = std::abs(ids.front() - ids.back());
    const double num_seconds_processed
      = std::abs(reader->getTimestamp(ids.front()) - reader->getTimestamp(ids.back()));
    const double ms_taken_single = 1e3 * (ended - started) / static_cast<double>(CLOCKS_PER_SEC);
    const double ms_taken_multi
      = stamp_initial
      + ((tv_end.tv_sec - tv_start.tv_sec) * 1e3 + (tv_end.tv_usec - tv_start.tv_usec) / 1e3);
    printf(
      "\n======================"
      "\n%d Frames (%.1f fps)"
      "\n%.2fms per frame (single core); "
      "\n%.2fms per frame (multi core); "
      "\n%.3fx (single core); "
      "\n%.3fx (multi core); "
      "\n======================\n\n",
      num_frames_processed,
      num_frames_processed / num_seconds_processed,
      ms_taken_single / num_frames_processed,
      ms_taken_multi / (float)num_frames_processed,
      1e3 / (ms_taken_single / num_seconds_processed),
      1e3 / (ms_taken_multi / num_seconds_processed)
    );

    // fullSystem->printFrameLifetimes();

    // Log the frame lifetimes if enabled.
    if (setting_logStuff) {
      std::ofstream file;
      file.open("logs/time.txt", std::ios::trunc | std::ios::out);
      file << 1e3 * (ended - started) / static_cast<double>(CLOCKS_PER_SEC * reader->getNumImages())
            << " "
            << ((tv_end.tv_sec - tv_start.tv_sec) * 1e3 + (tv_end.tv_usec - tv_start.tv_usec) / 1e3) / static_cast<double>(reader->getNumImages())
            << "\n";
      file.flush();
      file.close();
    }
  });

  // Run the viewer if it is enabled.
  if (viewer != nullptr) {
    viewer->run();
  }

  // Join the run thread.
  run_thread.join();
  // Join the viewer thread.
  for (IOWrap::Output3DWrapper* wrap : full_system->output_3d_wrappers_) {
    wrap->join();
    delete wrap;
  }
  // Join the exit thread.
  exit_thread.join();

  // Print the end messages and delete pointers.
  printf("DELETE FULLSYSTEM!\n");
  delete full_system;

  printf("DELETE READER!\n");
  delete reader;

  printf("EXIT NOW!\n");
  return 0;
}
