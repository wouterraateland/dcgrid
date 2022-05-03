#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

inline __device__ uint8_t hash(int32_t i) {
  static const uint8_t perm[256] = {
      151, 160, 137, 91,  90,  15,  131, 13,  201, 95,  96,  53,  194, 233, 7,
      225, 140, 36,  103, 30,  69,  142, 8,   99,  37,  240, 21,  10,  23,  190,
      6,   148, 247, 120, 234, 75,  0,   26,  197, 62,  94,  252, 219, 203, 117,
      35,  11,  32,  57,  177, 33,  88,  237, 149, 56,  87,  174, 20,  125, 136,
      171, 168, 68,  175, 74,  165, 71,  134, 139, 48,  27,  166, 77,  146, 158,
      231, 83,  111, 229, 122, 60,  211, 133, 230, 220, 105, 92,  41,  55,  46,
      245, 40,  244, 102, 143, 54,  65,  25,  63,  161, 1,   216, 80,  73,  209,
      76,  132, 187, 208, 89,  18,  169, 200, 196, 135, 130, 116, 188, 159, 86,
      164, 100, 109, 198, 173, 186, 3,   64,  52,  217, 226, 250, 124, 123, 5,
      202, 38,  147, 118, 126, 255, 82,  85,  212, 207, 206, 59,  227, 47,  16,
      58,  17,  182, 189, 28,  42,  223, 183, 170, 213, 119, 248, 152, 2,   44,
      154, 163, 70,  221, 153, 101, 155, 167, 43,  172, 9,   129, 22,  39,  253,
      19,  98,  108, 110, 79,  113, 224, 232, 178, 185, 112, 104, 218, 246, 97,
      228, 251, 34,  242, 193, 238, 210, 144, 12,  191, 179, 162, 241, 81,  51,
      145, 235, 249, 14,  239, 107, 49,  192, 214, 31,  181, 199, 106, 157, 184,
      84,  204, 176, 115, 121, 50,  45,  127, 4,   150, 254, 138, 236, 205, 93,
      222, 114, 67,  29,  24,  72,  243, 141, 128, 195, 78,  66,  215, 61,  156,
      180};

  return perm[static_cast<uint8_t>(i)];
}

inline __device__ float grad(int32_t hash, float x, float y, float z) {
  int h = hash & 15;
  float u = h < 8 ? x : y;
  float v = h < 4 ? y : h == 12 || h == 14 ? x : z;
  return ((h & 1) ? -u : u) + ((h & 2) ? -v : v);
}

inline __device__ float noise(float x, float y, float z) {
  float n0, n1, n2, n3;

  static const float F3 = 1.f / 3.f;
  static const float G3 = 1.f / 6.f;

  float s = (x + y + z) * F3;
  int i = floorf(x + s);
  int j = floorf(y + s);
  int k = floorf(z + s);
  float t = (i + j + k) * G3;
  float X0 = i - t;
  float Y0 = j - t;
  float Z0 = k - t;
  float x0 = x - X0;
  float y0 = y - Y0;
  float z0 = z - Z0;

  int i1, j1, k1;
  int i2, j2, k2;
  if (x0 >= y0) {
    if (y0 >= z0) {
      i1 = 1;
      j1 = 0;
      k1 = 0;
      i2 = 1;
      j2 = 1;
      k2 = 0;
    } else if (x0 >= z0) {
      i1 = 1;
      j1 = 0;
      k1 = 0;
      i2 = 1;
      j2 = 0;
      k2 = 1;
    } else {
      i1 = 0;
      j1 = 0;
      k1 = 1;
      i2 = 1;
      j2 = 0;
      k2 = 1;
    }
  } else {
    if (y0 < z0) {
      i1 = 0;
      j1 = 0;
      k1 = 1;
      i2 = 0;
      j2 = 1;
      k2 = 1;
    } else if (x0 < z0) {
      i1 = 0;
      j1 = 1;
      k1 = 0;
      i2 = 0;
      j2 = 1;
      k2 = 1;
    } else {
      i1 = 0;
      j1 = 1;
      k1 = 0;
      i2 = 1;
      j2 = 1;
      k2 = 0;
    }
  }

  float x1 = x0 - i1 + G3;
  float y1 = y0 - j1 + G3;
  float z1 = z0 - k1 + G3;
  float x2 = x0 - i2 + 2.f * G3;
  float y2 = y0 - j2 + 2.f * G3;
  float z2 = z0 - k2 + 2.f * G3;
  float x3 = x0 - 1.f + 3.f * G3;
  float y3 = y0 - 1.f + 3.f * G3;
  float z3 = z0 - 1.f + 3.f * G3;

  int gi0 = hash(i + hash(j + hash(k)));
  int gi1 = hash(i + i1 + hash(j + j1 + hash(k + k1)));
  int gi2 = hash(i + i2 + hash(j + j2 + hash(k + k2)));
  int gi3 = hash(i + 1 + hash(j + 1 + hash(k + 1)));

  float t0 = .6f - x0 * x0 - y0 * y0 - z0 * z0;
  if (t0 < 0.f) {
    n0 = 0.f;
  } else {
    t0 *= t0;
    n0 = t0 * t0 * grad(gi0, x0, y0, z0);
  }
  float t1 = .6f - x1 * x1 - y1 * y1 - z1 * z1;
  if (t1 < 0.f) {
    n1 = 0.f;
  } else {
    t1 *= t1;
    n1 = t1 * t1 * grad(gi1, x1, y1, z1);
  }
  float t2 = .6f - x2 * x2 - y2 * y2 - z2 * z2;
  if (t2 < 0.f) {
    n2 = 0.f;
  } else {
    t2 *= t2;
    n2 = t2 * t2 * grad(gi2, x2, y2, z2);
  }
  float t3 = .6f - x3 * x3 - y3 * y3 - z3 * z3;
  if (t3 < 0.f) {
    n3 = 0.f;
  } else {
    t3 *= t3;
    n3 = t3 * t3 * grad(gi3, x3, y3, z3);
  }

  return 32.f * (n0 + n1 + n2 + n3);
}