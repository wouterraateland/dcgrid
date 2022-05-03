#pragma once
#include "helper_math.h"
#include <cuda_runtime.h>

inline __device__ __host__ void mix(const float3 &srcColor,
                                    const float &srcAlpha, float3 &dstColor,
                                    float &dstAlpha) {
  if (srcAlpha > 1e-6f) {
    const float impact = (1.f - dstAlpha) * srcAlpha;
    dstColor =
        ((dstColor * dstAlpha) + (srcColor * impact)) / (dstAlpha + impact);
    dstAlpha += impact;
  }
}

// Undoes gamma-correction from an RGB-encoded color.
// https://en.wikipedia.org/wiki/SRGB#Specification_of_the_transformation
// https://stackoverflow.com/questions/596216/formula-to-determine-brightness-of-rgb-color
inline __device__ __host__ float sRGB2linearRGB(const float &c) {
  // Send this function a decimal sRGB gamma encoded color value
  // between 0.0 and 1.0, and it returns a linearized value.
  if (c <= .04045f)
    return c / 12.92f;
  else
    return powf((c + .055f) / 1.055f, 2.4f);
}

inline __device__ __host__ float3 sRGB2linearRGB(const float3 &c) {
  return make_float3(sRGB2linearRGB(c.x), sRGB2linearRGB(c.y),
                     sRGB2linearRGB(c.z));
}

// Converts RGB color to CIE 1931 XYZ color space.
// https://www.image-engineering.de/library/technotes/958-how-to-convert-between-srgb-and-ciexyz
inline __device__ __host__ float3 rgb2xyz(const float &r, const float &g,
                                          const float &b) {
  const float x = .4124f * r + .3576f * g + .1805f * b;
  const float y = .2126f * r + .7152f * g + .0722f * b;
  const float z = .0193f * r + .1192f * g + .9505f * b;
  return 1e2f * make_float3(x, y, z);
}

inline __device__ __host__ float3 rgb2xyz(const float3 &c) {
  return rgb2xyz(c.x, c.y, c.z);
}

// Converts RGB color to CIE 1931 XYZ color space.
// https://www.image-engineering.de/library/technotes/958-how-to-convert-between-srgb-and-ciexyz
inline __device__ __host__ float3 xyz2rgb(const float &x, const float &y,
                                          const float &z) {
  const float r = 3.240f * x + -1.537f * y + -0.499f * z;
  const float g = -0.969f * x + 1.876f * y + 0.042f * z;
  const float b = 0.056f * x + -0.204f * y + 1.057f * z;
  return 1e-2f * make_float3(r, g, b);
}

inline __device__ __host__ float3 xyz2rgb(const float3 &c) {
  return xyz2rgb(c.x, c.y, c.z);
}

// X, Y, Z of a "D65" light source.
// "D65" is a standard 6500K Daylight light source.
// https://en.wikipedia.org/wiki/Illuminant_D65
static const float3 D65 = make_float3(95.047f, 100.f, 108.883f);

// Converts CIE 1931 XYZ colors to CIE L*a*b*.
// The conversion formula comes from <http://www.easyrgb.com/en/math.php>.
// https://github.com/cangoektas/xyz-to-lab/blob/master/src/index.js
// The CIE 1931 XYZ color to convert which refers to the D65/2� standard
// illuminant. The color in the CIE L*a*b* color space.
inline __device__ __host__ float3 xyz2lab(float x, float y, float z) {
  x /= D65.x;
  y /= D65.y;
  z /= D65.z;
  x = x > .008856f ? powf(x, 1.f / 3.f) : x * 7.787f + 16.f / 116.f;
  y = y > .008856f ? powf(y, 1.f / 3.f) : y * 7.787f + 16.f / 116.f;
  z = z > .008856f ? powf(z, 1.f / 3.f) : z * 7.787f + 16.f / 116.f;

  const float l = 116.f * y - 16.f;
  const float a = 500.f * (x - y);
  const float b = 200.f * (y - z);
  return make_float3(l, a, b);
}

inline __device__ __host__ float3 xyz2lab(const float3 &c) {
  return xyz2lab(c.x, c.y, c.z);
}

inline __device__ __host__ float3 lab2xyz(const float &l, const float &a,
                                          const float &b) {
  float y = (l + 16.f) / 116.f;
  float x = a / 500.f + y;
  float z = y - b / 200.f;

  x = (powf(x, 3.f) > .008856f) ? powf(x, 3.f) : (x - 16.f / 116.f) / 7.787f;
  y = (powf(y, 3.f) > .008856f) ? powf(y, 3.f) : (y - 16.f / 116.f) / 7.787f;
  z = (powf(z, 3.f) > .008856f) ? powf(z, 3.f) : (z - 16.f / 116.f) / 7.787f;

  x *= D65.x;
  y *= D65.y;
  z *= D65.z;
  return make_float3(x, y, z);
}

inline __device__ __host__ float3 lab2xyz(const float3 &c) {
  return lab2xyz(c.x, c.y, c.z);
}

// Converts a and b of Lab color space to Hue of LCH color space.
// https://stackoverflow.com/questions/53733379/conversion-of-cielab-to-cielchab-not-yielding-correct-result
inline __device__ __host__ float ab2hue(const float &a, const float &b) {
  if (a >= 0.f && b == 0.f)
    return 0.f;
  if (a < 0.f && b == 0.f)
    return 3.1415f;
  if (a == 0.f && b > 0.f)
    return 3.1415f * .5f;
  if (a == 0.f && b < 0.f)
    return 3.1415f * 1.5f;

  float xBias;
  if (a > 0.f && b > 0.f)
    xBias = 0.f;
  else if (a < 0.f)
    xBias = 3.1415f;
  else if (a > 0.f && b < 0.f)
    xBias = 6.2830f;

  return atanf(b / a) + xBias;
}

// Converts Lab color space to Luminance-Chroma-Hue color space.
// http://www.brucelindbloom.com/index.html?Eqn_Lab_to_LCH.html
inline __device__ __host__ float3 lab2lch(const float &l, const float &a,
                                          const float &b) {
  const float c = sqrtf(a * a + b * b);
  const float h = ab2hue(a, b);
  return make_float3(l, c, h);
}

inline __device__ __host__ float3 lab2lch(const float3 &c) {
  return lab2lch(c.x, c.y, c.z);
}

inline __device__ __host__ float3 lch2lab(const float &l, const float &c,
                                          const float &h) {
  const float a = c * cosf(h);
  const float b = c * sinf(h);
  return make_float3(l, a, b);
}

inline __device__ __host__ float3 lch2lab(const float3 &c) {
  return lch2lab(c.x, c.y, c.z);
}

inline __device__ __host__ float3 rgb2hsl(const float &r, const float &g,
                                          const float &b) {
  const float cMin = fminf(r, fminf(g, b));
  const float cMax = fmaxf(r, fmaxf(g, b));

  float h;
  if (cMax == r)
    h = (1.f / 6.f) * (g - b) / (cMax - cMin);
  if (cMax == g)
    h = (1.f / 6.f) * (b - r) / (cMax - cMin) + (1.f / 3.f);
  if (cMax == b)
    h = (1.f / 6.f) * (r - g) / (cMax - cMin) - (1.f / 3.f);
  h -= floorf(h);

  float s;
  if (cMin == cMax)
    s = 0.f;
  else if (cMin + cMax < 1.f)
    s = (cMax - cMin) / (cMax + cMin);
  else
    s = (cMax - cMin) / (2.f - cMax - cMin);

  const float l = .5f * (cMin + cMax);

  return make_float3(h, s, l);
}

inline __device__ __host__ float3 rgb2hsl(const float3 &c) {
  return rgb2hsl(c.x, c.y, c.z);
}

inline __device__ __host__ float hue2rgb(const float &p, const float &q,
                                         float t) {
  t -= floorf(t);
  if (t < 1.f / 6.f)
    return p + (q - p) * 6.f * t;
  if (t < 1.f / 2.f)
    return q;
  if (t < 2.f / 3.f)
    return p + (q - p) * (2.f / 3.f - t) * 6.f;
  return p;
}

inline __device__ __host__ float3 hsl2rgb(const float &h, const float &s,
                                          const float &l) {
  if (s == 0.f)
    return make_float3(l, l, l);

  const float q = l < .5f ? l * (1.f + s) : l + s - l * s;
  const float p = 2.f * l - q;
  return make_float3(hue2rgb(p, q, h + 1.f / 3.f), hue2rgb(p, q, h),
                     hue2rgb(p, q, h - 1.f / 3.f));
}

inline __device__ __host__ float3 hsl2rgb(const float3 &c) {
  return hsl2rgb(c.x, c.y, c.z);
}