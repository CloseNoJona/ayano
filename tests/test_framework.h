// Mini framework de tests: sin dependencias externas para que el proyecto
// compile solo con el compilador.
#pragma once

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace testfw {

struct TestCase {
  const char* name;
  void (*fn)();
  bool slow;
};

inline std::vector<TestCase>& registry() {
  static std::vector<TestCase> tests;
  return tests;
}

inline int& failure_count() {
  static int count = 0;
  return count;
}

struct Registrar {
  Registrar(const char* name, void (*fn)(), bool slow) { registry().push_back({name, fn, slow}); }
};

inline void fail(const char* file, int line, const std::string& message) {
  ++failure_count();
  std::cerr << "  FALLO " << file << ":" << line << ": " << message << "\n";
}

}  // namespace testfw

#define TEST(name)                                                     \
  static void name();                                                  \
  static const testfw::Registrar registrar_##name(#name, name, false); \
  static void name()

#define SLOW_TEST(name)                                               \
  static void name();                                                 \
  static const testfw::Registrar registrar_##name(#name, name, true); \
  static void name()

#define CHECK(cond)                                       \
  do {                                                    \
    if (!(cond)) testfw::fail(__FILE__, __LINE__, #cond); \
  } while (0)

#define CHECK_EQ(actual, expected)                                   \
  do {                                                               \
    const auto actual_value = (actual);                              \
    const auto expected_value = (expected);                          \
    if (!(actual_value == expected_value)) {                         \
      std::ostringstream stream;                                     \
      stream << #actual << " = " << actual_value << ", se esperaba " \
             << expected_value;                                      \
      testfw::fail(__FILE__, __LINE__, stream.str());                \
    }                                                                \
  } while (0)
