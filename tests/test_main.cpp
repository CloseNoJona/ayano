#include <chrono>
#include <cstring>
#include <iostream>

#include "core/attacks.h"
#include "core/zobrist.h"
#include "test_framework.h"

int main(int argc, char** argv) {
  bool run_slow = false;
  const char* filter = nullptr;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--slow") == 0) {
      run_slow = true;
    } else {
      filter = argv[i];
    }
  }

  enroque::attacks::init();
  enroque::zobrist::init();

  int executed = 0;
  for (const testfw::TestCase& test : testfw::registry()) {
    if (test.slow && !run_slow) continue;
    if (filter && !std::strstr(test.name, filter)) continue;
    const int failures_before = testfw::failure_count();
    const auto start = std::chrono::steady_clock::now();
    test.fn();
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start);
    const bool ok = testfw::failure_count() == failures_before;
    std::cout << (ok ? "[ ok ] " : "[FAIL] ") << test.name << " (" << elapsed.count() << " ms)\n";
    ++executed;
  }

  std::cout << executed << " tests, " << testfw::failure_count() << " fallos\n";
  return testfw::failure_count() == 0 ? 0 : 1;
}
