#include <chrono>
#include <helper_cuda.h>
#include <log.h>
#include <stdio.h>

static auto _start = std::chrono::high_resolution_clock::now();

class Timer {
public:
  static void tic() {
    cudaThreadSynchronize();
    _start = std::chrono::high_resolution_clock::now();
  }

  static void toc(const std::string s) {
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - _start);
    _start = end;
    printf("%s: %gms\n", s.c_str(), ((float)duration.count()) / 1000.f);
    // char msg[100];
    // sprintf(msg, "%s: %gs", s.c_str(), ((float)duration.count()) /
    // 1000000.f); Log::inst()->Add(msg);
  }

  static void tocSync(const std::string s) {
    cudaThreadSynchronize();
    Timer::toc(s);
  }
};