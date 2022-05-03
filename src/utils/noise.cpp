#include "noise.h"
#define _USE_MATH_DEFINES
#include <cmath>
#include <math.h>

Noise::Noise(int d, int octaves, float frequency, float persistence)
    : m_d(d), m_octaves(octaves), m_frequency(frequency),
      m_persistence(persistence) {}

float Noise::LinearInterp(float a, float b, float x) {
  return a * (1 - x) + b * x;
}

float Noise::CosineInterp(float a, float b, float x) {
  auto ft = x * M_PI;
  auto f = (1 - std::cosf(ft)) / 2.f;
  return a * (1 - f) + b * f;
}

float Noise::Noise1d(int x) {
  int a = 15731;
  int b = 789221;
  int c = 1376312589;

  x = (x << 13) ^ x;
  return (1.0 - ((x * (x * x * a + b) + c) & 0x7fffffff) / float(0x40000000));
}

float Noise::InterpolateNoise1d(float x) {
  auto int_x = int(x);
  auto frac_x = x - int_x;

  auto v1 = SmoothedNoise1d(int_x);
  auto v2 = SmoothedNoise1d(int_x + 1);

  return CosineInterp(v1, v2, frac_x);
}

float Noise::SmoothedNoise1d(float x) {
  auto int_x = int(x);
  auto frac_x = x - int_x;
  auto v1 = Noise1d(int_x);
  auto v2 = Noise1d(int_x + 1);
  return CosineInterp(v1, v2, frac_x);
}

float Noise::PerlinNoise1d(int x) {
  auto total = 0.f;

  for (auto i = 0; i < m_octaves; i++) {
    auto frequency = std::powf(2.f, i) * m_frequency;
    auto amplitude = std::powf(m_persistence, i);
    total += InterpolateNoise1d(x * frequency) * amplitude;
  }

  return total;
}

float Noise::Noise2d(int x, int y) { return Noise1d(x + y * m_d); }

float Noise::SmoothedNoise2d(int x, int y) {
  auto corners = (Noise2d(x - 1, y - 1) + Noise2d(x + 1, y - 1) +
                  Noise2d(x - 1, y + 1) + Noise2d(x + 1, y + 1)) /
                 16;
  auto sides = (Noise2d(x - 1, y) + Noise2d(x + 1, y) + Noise2d(x, y - 1) +
                Noise2d(x, y + 1)) /
               8;
  auto center = Noise2d(x, y) / 4;

  return corners + sides + center;
}

float Noise::InterpolateNoise2d(float x, float y) {
  auto int_x = int(x);
  auto frac_x = x - int_x;

  auto int_y = int(y);
  auto frac_y = y - int_y;

  auto v1 = SmoothedNoise2d(int_x, int_y);
  auto v2 = SmoothedNoise2d(int_x + 1, int_y);
  auto v3 = SmoothedNoise2d(int_x, int_y + 1);
  auto v4 = SmoothedNoise2d(int_x + 1, int_y + 1);

  auto i1 = CosineInterp(v1, v2, frac_x);
  auto i2 = CosineInterp(v3, v4, frac_x);

  return CosineInterp(i1, i2, frac_y);
}

float Noise::PerlinNoise2d(int x, int y) {
  auto total = 0.f;

  for (auto i = 0; i < m_octaves; i++) {
    auto frequency = std::powf(2.f, i) * m_frequency;
    auto amplitude = std::powf(m_persistence, i);
    total += InterpolateNoise2d(x * frequency, y * frequency) * amplitude;
  }

  return total;
}

float Noise::Noise3d(int x, int y, int z) {
  return Noise1d((z * m_d + y) * m_d + x);
}

float Noise::SmoothedNoise3d(int x, int y, int z) {
  auto corners = (Noise3d(x - 1, y - 1, z - 1) + Noise3d(x + 1, y + 1, z + 1) +
                  Noise3d(x + 1, y - 1, z - 1) + Noise3d(x + 1, y - 1, z + 1) +
                  Noise3d(x + 1, y + 1, z - 1) + Noise3d(x - 1, y - 1, z + 1) +
                  Noise3d(x - 1, y + 1, z - 1) + Noise3d(x - 1, y + 1, z + 1)) /
                 (16 * 1);
  auto sides =
      (Noise3d(x - 1, y, z) + Noise3d(x + 1, y, z) + Noise3d(x, y - 1, z) +
       Noise3d(x, y + 1, z) + Noise3d(x, y, z - 1) + Noise3d(x, y, z + 1)) /
      (8 * 1);
  auto center = Noise3d(x, y, z) / (4 * 1);

  return corners + sides + center;
}

float Noise::InterpolateNoise3d(float x, float y, float z) {
  auto int_x = int(x);
  auto frac_x = x - int_x;

  auto int_y = int(y);
  auto frac_y = y - int_y;

  auto int_z = int(z);
  auto frac_z = z - int_z;

  auto v000 = SmoothedNoise3d(int_x, int_y, int_z);
  auto v101 = SmoothedNoise3d(int_x + 1, int_y, int_z + 1);
  auto v001 = SmoothedNoise3d(int_x, int_y, int_z + 1);
  auto v100 = SmoothedNoise3d(int_x + 1, int_y, int_z);
  auto v110 = SmoothedNoise3d(int_x + 1, int_y + 1, int_z);
  auto v010 = SmoothedNoise3d(int_x, int_y + 1, int_z);
  auto v011 = SmoothedNoise3d(int_x, int_y + 1, int_z + 1);
  auto v111 = SmoothedNoise3d(int_x + 1, int_y + 1, int_z + 1);

  auto i00 = CosineInterp(v000, v100, frac_x);
  auto i01 = CosineInterp(v001, v101, frac_x);
  auto i10 = CosineInterp(v010, v110, frac_x);
  auto i11 = CosineInterp(v011, v111, frac_x);

  auto i0 = CosineInterp(i00, i10, frac_y);
  auto i1 = CosineInterp(i01, i11, frac_y);

  return CosineInterp(i0, i1, frac_z);
}

float Noise::PerlinNoise3d(int x, int y, int z, float off_x, float off_y,
                           float off_z) {
  auto total = 0.f;

  for (auto i = 0; i < m_octaves; i++) {
    auto frequency = std::powf(2.f, i) * m_frequency;
    auto amplitude = std::powf(m_persistence, i);
    total += InterpolateNoise3d(x * frequency + off_x * frequency,
                                y * frequency + off_y * frequency,
                                z * frequency + off_z * frequency) *
             amplitude;
  }

  return total;
}