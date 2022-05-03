#pragma once
#include "data/scenes.h"
#include "data/sim_params.h"
#include "fluid_simulation.h"
#include "utils/cudatextures.h"
#include "utils/sim_utils.h"
#include <cuda_runtime.h>
#include <engine.h>
#include <renderable.h>

class Simulation : public Renderable {
public:
  Simulation(Scene &scene, Engine *engine);
  ~Simulation();
  void handleInput();
  void updateSimulation();
  void renderFluid(const int2 &viewportResolution, const float3 &eyePos,
                   const float3 &eyeDir, const float &fov,
                   const float3 &sunDir);
  void Update();
  void Render(std::shared_ptr<Shader> shaderIn);
  void RenderUi();
  void renderGeneralSettings();
  void renderChannelSettings();
  void renderDensitySettings();
  void renderColorSettings();
  void renderRenderSettings();

private:
  GLuint vertexArrayObject;
  GLuint vertexBuffer;

  TextureResource m_tex_fluid;
  int2 m_prevResolution = make_int2(1, 1);

  Engine *m_engine;
  SimParams &m_params;
  FluidSimulation *m_simulation;

  int m_iteration = 0;

  bool m_paused = true;
  bool m_simulateAdaptive = true;
  size_t m_maxNumBlocks = 1750000;
};