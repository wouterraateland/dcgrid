#ifndef CUDAMATH_H
#define CUDAMATH_H

#include <cmath>
#include <cuda_runtime.h>
#include <helper_math.h>

template <class T> class Mat3x3 {

public:
  __host__ __device__ Mat3x3() { clear(); }

  __host__ __device__ Mat3x3(T v1, T v2, T v3) {
    clear();
    element(0, 0) = v1;
    element(1, 1) = v2;
    element(2, 2) = v3;
  }

  __host__ __device__ void make_identity() {
    clear();

    for (int r = 0; r < 3; r++)
      for (int c = 0; c < 3; c++)
        element(r, c) = (r == c) ? (T)1.0f : (T)0.0f;
  }

  __host__ __device__ void clear() {
    for (int r = 0; r < 3; r++)
      for (int c = 0; c < 3; c++)
        element(r, c) = (T)0.0f;
  }

  __host__ __device__ static Mat3x3<T> tensorProduct(float3 a, float3 b) {
    Mat3x3<T> t;

    t.element(0, 0) = a.x * b.x;
    t.element(0, 1) = a.x * b.y;
    t.element(0, 2) = a.x * b.z;

    t.element(1, 0) = a.y * b.x;
    t.element(1, 1) = a.y * b.y;
    t.element(1, 2) = a.y * b.z;

    t.element(2, 0) = a.z * b.x;
    t.element(2, 1) = a.z * b.y;
    t.element(2, 2) = a.z * b.z;

    return t;
  }

  __host__ __device__ static Mat3x3<T> rotationFromAxis(const float3 &axis,
                                                        const float &degRad) {
    Mat3x3<T> t;

    t.make_identity();

    t.element(0, 0) = cos(degRad) + axis.x * axis.x * (1.0f - cos(degRad));
    t.element(1, 0) =
        axis.y * axis.x * (1.0f - cos(degRad)) + axis.z * sin(degRad);
    t.element(2, 0) =
        axis.z * axis.x * (1.0f - cos(degRad)) - axis.y * sin(degRad);

    t.element(0, 1) =
        axis.x * axis.y * (1.0f - cos(degRad)) - axis.z * sin(degRad);
    t.element(1, 1) = cos(degRad) + axis.y * axis.y * (1.0f - cos(degRad));
    t.element(2, 1) =
        axis.z * axis.y * (1.0f - cos(degRad)) + axis.x * sin(degRad);

    t.element(0, 2) =
        axis.x * axis.z * (1.0f - cos(degRad)) + axis.y * sin(degRad);
    t.element(1, 2) =
        axis.y * axis.z * (1.0f - cos(degRad)) - axis.x * sin(degRad);
    t.element(2, 2) = cos(degRad) + axis.z * axis.z * (1.0f - cos(degRad));

    return t;
  }

  __host__ __device__ static Mat3x3<T> scale(float s) {
    Mat3x3<T> t;

    for (int k = 0; k < 3; k++)
      t.element(k, k) = s;

    return t;
  }

  __host__ __device__ Mat3x3<T> transpose() {
    Mat3x3<T> t;

    for (int r = 0; r < 3; r++) {
      for (int c = 0; c < 3; c++) {
        t.element(r, c) = element(c, r);
      }
    }

    return t;
  }

  __host__ __device__ bool isSymmetric() {
    float symmetric = true;

    for (int r = 0; r < 3; r++)
      for (int c = 0; c < 3; c++)
        if (element(r, c) != element(c, r))
          symmetric = false;

    return symmetric;
  }

  __host__ __device__ T determinant() {
    return element(0, 0) *
               (element(1, 1) * element(2, 2) - element(1, 2) * element(2, 1)) -
           element(0, 1) *
               (element(1, 0) * element(2, 2) - element(1, 2) * element(2, 0)) +
           element(0, 2) *
               (element(1, 0) * element(2, 1) - element(1, 1) * element(2, 0));
  }

  __host__ __device__ Mat3x3<T> inverse() {
    Mat3x3<T> t;

    T D = determinant();

    // no inverse
    if (D == (T)0)
      return t;

    t.element(0, 0) =
        element(1, 1) * element(2, 2) - element(1, 2) * element(2, 1);
    t.element(1, 0) =
        element(1, 2) * element(2, 0) - element(1, 0) * element(2, 2);
    t.element(2, 0) =
        element(1, 0) * element(2, 1) - element(1, 1) * element(2, 0);

    t.element(0, 1) =
        element(0, 2) * element(2, 1) - element(0, 1) * element(2, 2);
    t.element(1, 1) =
        element(0, 0) * element(2, 2) - element(0, 2) * element(2, 0);
    t.element(2, 1) =
        element(0, 1) * element(2, 0) - element(0, 0) * element(2, 1);

    t.element(0, 2) =
        element(0, 1) * element(1, 2) - element(0, 2) * element(1, 1);
    t.element(1, 2) =
        element(0, 2) * element(1, 0) - element(0, 0) * element(1, 2);
    t.element(2, 2) =
        element(0, 0) * element(1, 1) - element(0, 1) * element(1, 0);

    return t / D;
  }

  __host__ __device__ float frobeniusNorm() {
    float f = 0.0f;

    for (int r = 0; r < 3; r++) {
      for (int c = 0; c < 3; c++) {
        f += pow(element(r, c), 2.0f);
      }
    }

    return sqrtf(f);
  }

  __host__ __device__ float magnitude() { return frobeniusNorm(); }

  __host__ __device__ float trace() {
    float t = 0.0f;

    for (int i = 0; i < 3; i++) {
      t += element(i, i);
    }

    return t;
  }

  __host__ __device__ T &operator()(int row, int col) {
    return element(row, col);
  }

  __host__ __device__ const T &operator()(int row, int col) const {
    return element(row, col);
  }

  __host__ __device__ T &element(int row, int col) { return _array[row][col]; }

  __host__ __device__ const T &element(int row, int col) const {
    return _array[row][col];
  }

  // operators
  __host__ __device__ bool operator==(const Mat3x3<T> &s) {
    bool equal = true;

    for (int r = 0; r < 3; r++) {
      for (int c = 0; c < 3; c++) {
        if (element(r, c) != s.element(r, c))
          equal = false;
      }
    }

    return equal;
  }

  __host__ __device__ Mat3x3<T> operator*(const T &s) {
    Mat3x3<T> t;

    for (int r = 0; r < 3; r++)
      for (int c = 0; c < 3; c++)
        t.element(r, c) = element(r, c) * s;

    return t;
  }

  __host__ __device__ Mat3x3<T> operator*(const Mat3x3<T> &s) {
    Mat3x3<T> t;

    for (int r = 0; r < 3; r++)
      for (int c = 0; c < 3; c++)
        for (int k = 0; k < 3; k++)
          t.element(r, c) += element(r, k) * s.element(k, c);

    return t;
  }

  __host__ __device__ Mat3x3<T> operator/(const T &s) {
    Mat3x3<T> t;

    for (int r = 0; r < 3; r++)
      for (int c = 0; c < 3; c++)
        t.element(r, c) = element(r, c) / s;

    return t;
  }

  __host__ __device__ Mat3x3<T> operator-(const T &s) {
    Mat3x3<T> t;

    for (int r = 0; r < 3; r++)
      for (int c = 0; c < 3; c++)
        t.element(r, c) = element(r, c) - s;

    return t;
  }

  __host__ __device__ Mat3x3<T> operator-(Mat3x3<T> &t_in) {
    Mat3x3<T> t;

    for (int r = 0; r < 3; r++)
      for (int c = 0; c < 3; c++)
        t.element(r, c) = element(r, c) - t_in.element(r, c);

    return t;
  }

  __host__ __device__ Mat3x3<T> operator+(Mat3x3<T> &t_in) {
    Mat3x3<T> t;

    for (int r = 0; r < 3; r++)
      for (int c = 0; c < 3; c++)
        t.element(r, c) = element(r, c) + t_in.element(r, c);

    return t;
  }

  __host__ __device__ float3 operator*(const float3 &v) {
    float3 r = make_float3(0.0f);

    r.x = element(0, 0) * v.x + element(0, 1) * v.y + element(0, 2) * v.z;
    r.y = element(1, 0) * v.x + element(1, 1) * v.y + element(1, 2) * v.z;
    r.z = element(2, 0) * v.x + element(2, 1) * v.y + element(2, 2) * v.z;

    return r;
  }

  // data
  T _array[3][3];
};

// Polyfill atomic min and max for floats
__device__ __forceinline__ float atomicMinFloat(float *addr, float value) {
  float old;
  old = (value >= 0)
            ? __int_as_float(atomicMin((int *)addr, __float_as_int(value)))
            : __uint_as_float(
                  atomicMax((unsigned int *)addr, __float_as_uint(value)));

  return old;
}

__device__ __forceinline__ float atomicMaxFloat(float *addr, float value) {
  float old;
  old = (value >= 0)
            ? __int_as_float(atomicMax((int *)addr, __float_as_int(value)))
            : __uint_as_float(
                  atomicMin((unsigned int *)addr, __float_as_uint(value)));

  return old;
}

// Polyfill atomic sub for other types
__device__ __forceinline__ size_t atomicSubSize_t(size_t *address, size_t val) {
  size_t old = *address, assumed;
  do {
    assumed = old;
    old = atomicCAS(address, assumed, assumed - val);
  } while (assumed != old);
  return old;
}

__device__ __forceinline__ uint64_t atomicSubUint64_t(uint64_t *address,
                                                      uint64_t val) {
  uint64_t old = *address, assumed;
  do {
    assumed = old;
    old = atomicCAS(address, assumed, assumed - val);
  } while (assumed != old);
  return old;
}

#endif