#pragma once
#include "helper_math.h"
#include "sim_params.h"
#include <cuda_runtime.h>

struct Scene {
  SimParams params = SimParams::defaultParams();
};

class Scenes {
public:
  // Standard
  static Scene Default() {
    Scene scene;
    const int d = 256;
    scene.params.gx = d;
    scene.params.gy = d;
    scene.params.gz = d;
    scene.params.dx = 10000.f / d;
    return scene;
  };

  static Scene SPGridCmp() {
    Scene scene;
    const int d = 1024;
    scene.params.gx = d;
    scene.params.gy = 2 * d;
    scene.params.gz = d;
    scene.params.dx = 10000.f / d;
    scene.params.enable_additional_solids = true;
    return scene;
  };
};
