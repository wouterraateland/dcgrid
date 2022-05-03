#include "data/scenes.h"
#include "simulation.h"
#include <camera_orbit.h>
#include <engine.h>
#include <gl/shader.h>
#include <l.h>
#include <light.h>
#include <shaders.h>

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb/stb_image_resize.h>
#include <stb/stb_image_write.h>

static Scene scene = Scenes::SPGridCmp();

void loadScene(Engine *engine) {
  scene.params.rdx = 1.f / scene.params.dx;

  // Light
  std::shared_ptr<Light> directionalLight =
      std::make_shared<Light>(glm::vec3(0, 0, 0), glm::vec3(1), 15.f, true);
  directionalLight->Type(LightType::Directional);
  directionalLight->Pitch(-2.5f);
  directionalLight->Yaw(-2.1f);
  engine->AddRenderable(directionalLight);

  // Camera
  std::shared_ptr<CameraOrbit> camera = std::make_shared<CameraOrbit>();
  const glm::vec3 domain =
      glm::vec3(scene.params.gx, scene.params.gy, scene.params.gz) *
      scene.params.dx;
  camera->m_center = .5f * glm::vec3(domain.x, domain.y, domain.z);
  const float minD = fmaxf(domain.x, domain.y) / sinf(glm::radians(50.f));
  camera->Position(camera->m_center + glm::vec3(0.f, 0.f, minD));
  engine->AddRenderable(camera);

  // Sim
  std::shared_ptr<Simulation> simulation =
      std::make_shared<Simulation>(scene, engine);
  simulation->SetRenderPass(2, "FluidRaymarching");
  engine->AddRenderable(simulation);
  engine->SelectRenderable(simulation);
}

int main(int argc, char *argv[]) {
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  cudaSetDevice(deviceCount - 1);

  // cudaDeviceProp prop;
  // cudaGetDeviceProperties(&prop, deviceCount - 1);
  // printf("Graphics card: %s\n", prop.name);

  L framework = L(1024, 576, "../extern/L");

  std::shared_ptr<Shader> shader =
      std::make_shared<Shader>(std::vector<std::string>{
          "../shaders/fluid.vert", "../shaders/fluid.frag"});

  Shaders::inst()->AddShader("FluidRaymarching", shader);

  Engine *engine = framework.GetEngine();
  engine->CallbackLoadScene(loadScene);
  engine->LoadScene();

  framework.Run();
}
