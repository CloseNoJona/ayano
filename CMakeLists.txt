cmake_minimum_required(VERSION 3.25)
project(ayano VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# El runtime de C++ va dentro del ejecutable: arranca en equipos sin Visual C++ instalado.
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

option(AYANO_NATIVE "Usar AVX2 + BMI2/PEXT (CPUs modernas)" ON)
option(AYANO_PEXT "Usar PEXT para las piezas deslizantes (lento en AMD Zen 1 y 2)" ON)
option(AYANO_TUNE "Exponer los parametros de busqueda como opciones UCI" OFF)
set(AYANO_PGO "OFF" CACHE STRING "Optimizacion guiada por perfil: OFF, GENERATE o USE")
set(AYANO_HIDDEN 512 CACHE STRING "Neuronas de la capa oculta; la red integrada exige 512")

if(CMAKE_BUILD_TYPE STREQUAL "Release")
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
endif()

add_library(ayano_flags INTERFACE)
target_include_directories(ayano_flags INTERFACE src)

if(MSVC)
  target_compile_options(ayano_flags INTERFACE
    /W4 /permissive- /utf-8 /EHsc /Zc:__cplusplus /wd4324
    $<$<CONFIG:Release>:/O2 /Oi /Ot /Gy /DNDEBUG>)
  if(AYANO_NATIVE)
    target_compile_options(ayano_flags INTERFACE /arch:AVX2)
  endif()
else()
  target_compile_options(ayano_flags INTERFACE -Wall -Wextra -Wpedantic
    $<$<CONFIG:Release>:-O3 -DNDEBUG>)
  if(AYANO_NATIVE)
    target_compile_options(ayano_flags INTERFACE -mavx2 -mbmi -mbmi2 -mpopcnt -mlzcnt)
  endif()
endif()

if(AYANO_NATIVE AND AYANO_PEXT)
  target_compile_definitions(ayano_flags INTERFACE AYANO_USE_PEXT)
endif()

if(AYANO_TUNE)
  target_compile_definitions(ayano_flags INTERFACE AYANO_TUNE)
endif()

target_compile_definitions(ayano_flags INTERFACE AYANO_HIDDEN=${AYANO_HIDDEN})

# La busqueda recursiva guarda una lista de jugadas por nodo: con MAX_PLY niveles
# supera la pila de 1 MB que Windows da por defecto a cada hilo.
if(MSVC)
  target_link_options(ayano_flags INTERFACE /STACK:16777216)
endif()

find_package(Threads REQUIRED)

# --- Biblioteca del motor (compartida por el ejecutable, los tests y las herramientas)
file(GLOB_RECURSE AYANO_SOURCES CONFIGURE_DEPENDS src/*.cpp)
list(REMOVE_ITEM AYANO_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp)
add_library(ayano_lib STATIC ${AYANO_SOURCES})
target_link_libraries(ayano_lib PUBLIC ayano_flags Threads::Threads)

# --- Red NNUE integrada en el ejecutable. Con otro tamano de capa no encajaria, asi que
# esa compilacion no integra ninguna y la red se carga con EvalFile.
set(AYANO_NET ${CMAKE_CURRENT_SOURCE_DIR}/nets/default.nnue)
if(EXISTS ${AYANO_NET} AND AYANO_HIDDEN EQUAL 512)
  set(AYANO_NET_CPP ${CMAKE_CURRENT_BINARY_DIR}/embedded_net.cpp)
  add_custom_command(
    OUTPUT ${AYANO_NET_CPP}
    COMMAND ${CMAKE_COMMAND} -DINPUT=${AYANO_NET} -DOUTPUT=${AYANO_NET_CPP}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed.cmake
    DEPENDS ${AYANO_NET} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed.cmake
    COMMENT "Integrando la red NNUE")
  target_sources(ayano_lib PRIVATE ${AYANO_NET_CPP})
  target_compile_definitions(ayano_lib PRIVATE AYANO_HAS_EMBEDDED_NET)
endif()

# --- Motor UCI
add_executable(ayano src/main.cpp)
target_link_libraries(ayano PRIVATE ayano_lib)
if(MSVC AND AYANO_PGO STREQUAL "GENERATE")
  target_link_options(ayano PRIVATE /GENPROFILE:PGD=${CMAKE_BINARY_DIR}/ayano.pgd)
elseif(MSVC AND AYANO_PGO STREQUAL "USE")
  target_link_options(ayano PRIVATE /USEPROFILE:PGD=${CMAKE_BINARY_DIR}/ayano.pgd)
endif()

# --- Tests
enable_testing()
file(GLOB AYANO_TESTS CONFIGURE_DEPENDS tests/*.cpp)
add_executable(ayano_tests ${AYANO_TESTS})
target_include_directories(ayano_tests PRIVATE tests)
target_link_libraries(ayano_tests PRIVATE ayano_lib)
add_test(NAME unit COMMAND ayano_tests)

# --- Herramientas (partidas, generacion de datos, entrenamiento)
file(GLOB AYANO_TOOLS CONFIGURE_DEPENDS tools/*.cpp)
foreach(tool_src ${AYANO_TOOLS})
  get_filename_component(tool_name ${tool_src} NAME_WE)
  add_executable(${tool_name} ${tool_src})
  target_link_libraries(${tool_name} PRIVATE ayano_lib)
endforeach()
