# Compila el motor y los tests en Linux con g++ o clang++. En Windows: scripts\build.cmd.
#
#   make                 AVX2 + BMI2/PEXT, para CPUs modernas
#   make ARCH=avx2       AVX2 sin PEXT, para AMD Ryzen 1000 a 3000
#   make ARCH=portable   sin instrucciones especiales
#   make test            compila y pasa los tests
#   make STATIC=1        binario estatico, para que funcione en otras distribuciones
#
# Variables: CXX, EXE, EVALFILE (la red que se integra en el ejecutable) y CXXFLAGS, que se
# suman a las opciones de optimizacion. Tras cambiar CXXFLAGS hace falta `make clean`.

CXX ?= g++
EXE ?= ayano
ARCH ?= native
EVALFILE ?= nets/default.nnue
BUILD ?= build/linux-$(notdir $(CXX))-$(ARCH)

ifeq ($(ARCH),native)
  ARCH_FLAGS := -mavx2 -mbmi -mbmi2 -mpopcnt -mlzcnt -DAYANO_USE_PEXT
else ifeq ($(ARCH),avx2)
  ARCH_FLAGS := -mavx2 -mbmi -mbmi2 -mpopcnt -mlzcnt
else ifeq ($(ARCH),portable)
  ARCH_FLAGS :=
else
  $(error ARCH debe ser native, avx2 o portable)
endif

ALL_FLAGS := -std=c++20 -O3 -DNDEBUG -Wall -Wextra -ffp-contract=off -Isrc -Itests -pthread $(ARCH_FLAGS) \
             -DAYANO_HIDDEN=1024 -DAYANO_HAS_EMBEDDED_NET $(CXXFLAGS)
LINK_FLAGS := -pthread $(if $(STATIC),-static) $(LDFLAGS)

ENGINE_SOURCES := $(filter-out src/main.cpp,$(wildcard src/*.cpp src/*/*.cpp))
TEST_SOURCES := $(wildcard tests/*.cpp)
ENGINE_OBJECTS := $(ENGINE_SOURCES:%.cpp=$(BUILD)/%.o) $(BUILD)/embedded_net.o
TEST_OBJECTS := $(TEST_SOURCES:%.cpp=$(BUILD)/%.o)

$(EXE): $(ENGINE_OBJECTS) $(BUILD)/src/main.o
	$(CXX) $^ $(LINK_FLAGS) -o $@

test: $(BUILD)/$(EXE)_tests
	$(BUILD)/$(EXE)_tests

$(BUILD)/$(EXE)_tests: $(ENGINE_OBJECTS) $(TEST_OBJECTS)
	$(CXX) $^ $(LINK_FLAGS) -o $@

$(BUILD)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(ALL_FLAGS) -MMD -MP -c $< -o $@

$(BUILD)/embedded_net.o: $(BUILD)/embedded_net.cpp
	$(CXX) $(ALL_FLAGS) -MMD -MP -c $< -o $@

$(BUILD)/embed_net: tools/embed_net.cpp
	@mkdir -p $(dir $@)
	$(CXX) -std=c++20 -O1 $< -o $@

$(BUILD)/embedded_net.cpp: $(EVALFILE) $(BUILD)/embed_net
	$(BUILD)/embed_net $(EVALFILE) $@

clean:
	rm -rf build/linux-* $(EXE)

.PHONY: test clean
.DELETE_ON_ERROR:

-include $(ENGINE_OBJECTS:.o=.d) $(TEST_OBJECTS:.o=.d)
