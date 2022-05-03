#include "simulation.h"
#include <imgui.h>

void Simulation::renderGeneralSettings() {
  ImGui::Checkbox("Additional solids", &m_params.enable_additional_solids);

  if (ImGui::CollapsingHeader("Tweaking", ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::SliderFloat("Timestep", &m_params.dt, 0.f, 60.f, "%.0fs");
    ImGui::SliderFloat("Emission velocity", &m_params.velocity_emission_rate,
                       0.f, 500.f, "%.0fs");
  }
}

void Simulation::renderChannelSettings() {
  if (ImGui::RadioButton("Default",
                         m_params.render_channel == RenderChannel::Fluid))
    m_params.render_channel = RenderChannel::Fluid;
  if (ImGui::RadioButton("Density",
                         m_params.render_channel == RenderChannel::Density))
    m_params.render_channel = RenderChannel::Density;
  if (ImGui::RadioButton("Velocity",
                         m_params.render_channel == RenderChannel::Velocity))
    m_params.render_channel = RenderChannel::Velocity;
  if (ImGui::RadioButton("Resolution",
                         m_params.render_channel == RenderChannel::Resolution))
    m_params.render_channel = RenderChannel::Resolution;
  if (ImGui::RadioButton("Fluidity rate",
                         m_params.render_channel == RenderChannel::Fluidity))
    m_params.render_channel = RenderChannel::Fluidity;
}

void Simulation::renderColorSettings() {
  ImGui::ColorEdit3("Background", &m_params.background_color.x,
                    ImGuiColorEditFlags_NoInputs);

  ImGui::ColorEdit3("Smoke", &m_params.smoke_color.x,
                    ImGuiColorEditFlags_NoInputs);

  ImGui::ColorEdit3("Scene", &m_params.scene_color.x,
                    ImGuiColorEditFlags_NoInputs);
}

void Simulation::renderRenderSettings() {
  ImGui::Checkbox("Render solids", &m_params.render_solids);
  ImGui::Checkbox("Render shadows", &m_params.render_shadows);
  ImGui::Checkbox("Render precise", &m_params.render_precise);
  ImGui::SliderFloat("AA", &m_params.aa_samples, 1.f, 8.f, "%.0f samples");
  if (ImGui::CollapsingHeader("Render channel"))
    renderChannelSettings();
  if (ImGui::CollapsingHeader("Colors"))
    renderColorSettings();
}

void Simulation::RenderUi() {
  if (ImGui::Begin("Settings")) {
    if (ImGui::BeginTabBar("SettingsTabs", ImGuiTabBarFlags_None)) {
      if (ImGui::BeginTabItem("General")) {
        renderGeneralSettings();
        ImGui::EndTabItem();
      }
      if (ImGui::BeginTabItem("Rendering")) {
        renderRenderSettings();
        ImGui::EndTabItem();
      }
      ImGui::EndTabBar();
    }
    ImGui::End();
  }
}
