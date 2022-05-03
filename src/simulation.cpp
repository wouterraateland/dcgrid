#include "simulation.h"
#include "dcgrid/fluid_simulation_dcgrid.h"
#include "helper_math.h"
#include "uniformgrid/fluid_simulation_uniform.h"
#include "utils/grid_math.cuh"
#include "utils/sim_utils.h"
#include "utils/timer.h"
#include <camera.h>
#include <float.h>
#include <gl/shader.h>
#include <input.h>
#include <light.h>
#include <params.h>
#include <stb_image_write.h>
#include <string>

Simulation::Simulation(Scene &scene, Engine *engine)
    : m_params(scene.params), m_engine(engine) {
  copySimParamsToDevice(m_params);

  glGenVertexArrays(1, &vertexArrayObject);
  glGenBuffers(1, &vertexBuffer);
  glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
  GLfloat verts[]{-1, -1, 1, -1, 1, 1, -1, -1, -1, 1, 1, 1};
  glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  const int3 size = make_int3(m_params.gx, m_params.gy, m_params.gz);

  m_tex_fluid.init(make_int2(1, 1), 4);

  if (m_simulateAdaptive)
    m_simulation = new FluidSimulationDCGrid(size, m_maxNumBlocks);
  else
    m_simulation = new FluidSimulationUniform(size);
}

Simulation::~Simulation() {
  delete m_simulation;
  m_tex_fluid.destroy();
}

void Simulation::handleInput() {
  if (Input::inst()->KeyPressed(GLFW_KEY_SPACE))
    m_paused = !m_paused;

  if (Input::inst()->KeyPressed(GLFW_KEY_H)) {
    m_simulation->reset();
    m_iteration = 0;
  }

  if (Input::inst()->KeyPressed(GLFW_KEY_N)) {
    m_simulateAdaptive = !m_simulateAdaptive;

    delete m_simulation;
    const int3 size = make_int3(m_params.gx, m_params.gy, m_params.gz);

    if (m_simulateAdaptive)
      m_simulation = new FluidSimulationDCGrid(size, m_maxNumBlocks);
    else
      m_simulation = new FluidSimulationUniform(size);

    m_iteration = 0;
  }

  if (Input::inst()->KeyPressed(GLFW_KEY_1)) {
    m_params.render_channel = RenderChannel::Fluid;
    m_params.render_shadows = true;
  }
  if (Input::inst()->KeyPressed(GLFW_KEY_2)) {
    m_params.render_channel = RenderChannel::Density;
    m_params.render_shadows = false;
  }
  if (Input::inst()->KeyPressed(GLFW_KEY_3)) {
    m_params.render_channel = RenderChannel::Velocity;
    m_params.render_shadows = false;
  }
  if (Input::inst()->KeyPressed(GLFW_KEY_4)) {
    m_params.render_channel = RenderChannel::Fluidity;
    m_params.render_shadows = false;
  }
  if (Input::inst()->KeyPressed(GLFW_KEY_5)) {
    m_params.render_channel = RenderChannel::Resolution;
    m_params.render_shadows = false;
  }

  if (Input::inst()->KeyPressed(GLFW_KEY_T)) {
    m_params.render_solids = !m_params.render_solids;
    copySimParamsToDevice(m_params);
  }
}

void Simulation::updateSimulation() {
  copySimParamsToDevice(m_params);

  if (m_paused)
    return;

  m_iteration++;
  if (m_iteration % 150 == 0)
    m_paused = true;

  // Timer::tic();
  m_simulation->advectVelocity();
  // Timer::tocSync("Advect V");
  m_simulation->adaptTopology();
  // Timer::tocSync("Topology");
  m_simulation->project();
  // m_simulation->projectLocal();
  // Timer::tocSync("Project");
  m_simulation->advectDensity();
  // Timer::tocSync("Advect Q");

  // m_simulation->debugStats();
  // Timer::tocSync("Debug");
}

void Simulation::renderFluid(const int2 &viewportResolution,
                             const float3 &eyePos, const float3 &eyeDir,
                             const float &fov, const float3 &sunDir) {
  // Update resolution on resize
  if (viewportResolution.x != m_prevResolution.x ||
      viewportResolution.y != m_prevResolution.y) {
    m_tex_fluid.destroy();
    m_tex_fluid.init(viewportResolution, 4);
    getLastCudaErrorSync("Resize");
  }
  m_prevResolution = viewportResolution;

  m_params.render_channel = RenderChannel::Resolution;
  m_params.aa_samples = 8.f;
  m_params.render_solids = false;
  m_params.render_shadows = false;
  copySimParamsToDevice(m_params);
  m_simulation->render(m_tex_fluid.mapAsSurface(), viewportResolution, eyePos,
                       eyeDir, fov, sunDir);
  m_tex_fluid.unmap();

  if (!m_paused) {
    GLubyte *pixels = new GLubyte[m_prevResolution.x * m_prevResolution.y * 4];
    glBindTexture(GL_TEXTURE_2D, m_tex_fluid.m_GLTexture);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels);

    char filename[50];
    sprintf(filename, "sequences\\\\resolution_static\\\\%d.png", m_iteration);
    stbi_flip_vertically_on_write(true);
    stbi_write_png(filename, m_prevResolution.x, m_prevResolution.y, 4, pixels,
                   m_prevResolution.x * 4);
    delete[] pixels;
  }

  // Timer::tic();
  m_params.render_channel = RenderChannel::Fluid;
  m_params.aa_samples = 1.f;
  m_params.render_solids = true;
  m_params.render_shadows = true;
  copySimParamsToDevice(m_params);
  m_simulation->render(m_tex_fluid.mapAsSurface(), viewportResolution, eyePos,
                       eyeDir, fov, sunDir);
  m_tex_fluid.unmap();
  // Timer::tocSync("Render");

  if (!m_paused) {
    GLubyte *pixels = new GLubyte[m_prevResolution.x * m_prevResolution.y * 4];
    glBindTexture(GL_TEXTURE_2D, m_tex_fluid.m_GLTexture);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels);

    char filename[50];
    sprintf(filename, "sequences\\\\density_static\\\\%d.png", m_iteration);
    stbi_flip_vertically_on_write(true);
    stbi_write_png(filename, m_prevResolution.x, m_prevResolution.y, 4, pixels,
                   m_prevResolution.x * 4);
    delete[] pixels;
  }
}

void Simulation::Update() {
  handleInput();
  updateSimulation();
}

void Simulation::Render(std::shared_ptr<Shader> shaderIn) {
  glm::ivec2 viewportResolution = Params::inst()->ScreenDim();
  std::weak_ptr<Camera> w_camera = *m_engine->m_active_camera;
  std::weak_ptr<Light> w_sun = m_engine->m_lights[0];

  if (!w_camera.expired() && !w_sun.expired()) {
    std::shared_ptr<Camera> camera = w_camera.lock();
    std::shared_ptr<Light> sun = w_sun.lock();
    const auto cameraParams = camera->CameraParams();

    const int2 resolution =
        make_int2(viewportResolution.x, viewportResolution.y);

    const float3 eyePos =
        make_float3(cameraParams.pos.x, cameraParams.pos.y, cameraParams.pos.z);
    const float3 eyeDir =
        make_float3(cameraParams.dir.x, cameraParams.dir.y, cameraParams.dir.z);
    const float fov = sinf(glm::radians(cameraParams.fov));

    const float3 sunDir = -make_float3(sun->ViewVector().x, sun->ViewVector().y,
                                       sun->ViewVector().z);

    renderFluid(resolution, eyePos, eyeDir, fov, sunDir);
  }

  const GLint loc = shaderIn->GetUniformLocation("uTex");
  shaderIn->SetUniform1i("uTex", loc);
  glActiveTexture(GL_TEXTURE0 + loc);
  glBindTexture(GL_TEXTURE_2D, m_tex_fluid.m_GLTexture);

  glBindVertexArray(vertexArrayObject);
  glEnableVertexAttribArray(0);
  glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, (void *)0);

  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  glDrawArrays(GL_TRIANGLES, 0, 6);

  glDisable(GL_BLEND);

  glDisableVertexAttribArray(0);
  glBindVertexArray(0);
}
