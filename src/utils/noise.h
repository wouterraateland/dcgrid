#pragma once

// http://www.arendpeter.com/Perlin_Noise.html
// https://gist.github.com/dragon0/f70e2637e6d4e64a6ab210faf8a85a50

class Noise {
public:
  Noise(int d, int octaves = 1, float frequency = 1.0f,
        float persistence = 0.5f);
  float PerlinNoise1d(int x);
  float PerlinNoise2d(int x, int y);
  float PerlinNoise3d(int x, int y, int z, float off_x = 0.f, float off_y = 0.f,
                      float off_z = 0.f);

private:
  float LinearInterp(float a, float b, float x);
  float CosineInterp(float a, float b, float x);

  float Noise1d(int x);
  float InterpolateNoise1d(float x);
  float SmoothedNoise1d(float x);

  float Noise2d(int x, int y);
  float SmoothedNoise2d(int x, int y);
  float InterpolateNoise2d(float x, float y);

  float Noise3d(int x, int y, int z);
  float SmoothedNoise3d(int x, int y, int z);
  float InterpolateNoise3d(float x, float y, float z);

private:
  int m_d;
  int m_octaves;
  float m_frequency;
  float m_persistence;
};