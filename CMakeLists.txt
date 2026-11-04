cmake_minimum_required(VERSION 3.25)
project(enroque VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(ENROQUE_NATIVE "Usar AVX2 + BMI2/PEXT (CPUs modernas)" ON)
option(ENROQUE_PEXT "Usar PEXT para las piezas deslizantes (lento en AMD Zen 1 y 2)" ON)
option(ENROQUE_TUNE "Exponer los parametros de busqueda como opciones UCI" OFF)
set(ENROQUE_PGO "OFF" CACHE STRING "Optimizacion guiada por perfil: OFF, GENERATE o USE")
set(ENROQUE_HIDDEN 512 CACHE STRING "Neuronas de la capa oculta; la red integrada exige 512")

if(CMAKE_BUILD_TYPE STREQUAL "Release")
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
endif()

add_library(enroque_flags INTERFACE)
target_include_directories(enroque_flags INTERFACE src)

if(MSVC)
  target_compile_options(enroque_flags INTERFACE
    /W4 /permissive- /utf-8 /EHsc /Zc:__cplusplus /wd4324
    $<$<CONFIG:Release>:/O2 /Oi /Ot /Gy /DNDEBUG>)
  if(ENROQUE_NATIVE)
    target_compile_options(enroque_flags INTERFACE /arch:AVX2)
  endif()
else()
  target_compile_options(enroque_flags INTERFACE -Wall -Wextra -Wpedantic
    $<$<CONFIG:Release>:-O3 -DNDEBUG>)
  if(ENROQUE_NATIVE)
    target_compile_options(enroque_flags INTERFACE -mavx2 -mbmi -mbmi2 -mpopcnt -mlzcnt)
  endif()
endif()

if(ENROQUE_NATIVE AND ENROQUE_PEXT)
  target_compile_definitions(enroque_flags INTERFACE ENROQUE_USE_PEXT)
endif()

if(ENROQUE_TUNE)
  target_compile_definitions(enroque_flags INTERFACE ENROQUE_TUNE)
endif()

target_compile_definitions(enroque_flags INTERFACE ENROQUE_HIDDEN=${ENROQUE_HIDDEN})

# La busqueda recursiva guarda una lista de jugadas por nodo: con MAX_PLY niveles
# supera la pila de 1 MB que Windows da por defecto a cada hilo.
if(MSVC)
  target_link_options(enroque_flags INTERFACE /STACK:16777216)
endif()

find_package(Threads REQUIRED)

# --- Biblioteca del motor (compartida por el ejecutable, los tests y las herramientas)
file(GLOB_RECURSE ENROQUE_SOURCES CONFIGURE_DEPENDS src/*.cpp)
list(REMOVE_ITEM ENROQUE_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp)
add_library(enroque_lib STATIC ${ENROQUE_SOURCES})
target_link_libraries(enroque_lib PUBLIC enroque_flags Threads::Threads)

# --- Red NNUE integrada en el ejecutable. Con otro tamano de capa no encajaria, asi que
# esa compilacion no integra ninguna y la red se carga con EvalFile.
set(ENROQUE_NET ${CMAKE_CURRENT_SOURCE_DIR}/nets/default.nnue)
if(EXISTS ${ENROQUE_NET} AND ENROQUE_HIDDEN EQUAL 512)
  set(ENROQUE_NET_CPP ${CMAKE_CURRENT_BINARY_DIR}/embedded_net.cpp)
  add_custom_command(
    OUTPUT ${ENROQUE_NET_CPP}
    COMMAND ${CMAKE_COMMAND} -DINPUT=${ENROQUE_NET} -DOUTPUT=${ENROQUE_NET_CPP}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed.cmake
    DEPENDS ${ENROQUE_NET} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed.cmake
    COMMENT "Integrando la red NNUE")
  target_sources(enroque_lib PRIVATE ${ENROQUE_NET_CPP})
  target_compile_definitions(enroque_lib PRIVATE ENROQUE_HAS_EMBEDDED_NET)
endif()

# --- Motor UCI
add_executable(enroque src/main.cpp)
target_link_libraries(enroque PRIVATE enroque_lib)
if(MSVC AND ENROQUE_PGO STREQUAL "GENERATE")
  target_link_options(enroque PRIVATE /GENPROFILE:PGD=${CMAKE_BINARY_DIR}/enroque.pgd)
elseif(MSVC AND ENROQUE_PGO STREQUAL "USE")
  target_link_options(enroque PRIVATE /USEPROFILE:PGD=${CMAKE_BINARY_DIR}/enroque.pgd)
endif()

# --- Tests
enable_testing()
file(GLOB ENROQUE_TESTS CONFIGURE_DEPENDS tests/*.cpp)
add_executable(enroque_tests ${ENROQUE_TESTS})
target_include_directories(enroque_tests PRIVATE tests)
target_link_libraries(enroque_tests PRIVATE enroque_lib)
add_test(NAME unit COMMAND enroque_tests)

# --- Herramientas (partidas, generacion de datos, entrenamiento)
file(GLOB ENROQUE_TOOLS CONFIGURE_DEPENDS tools/*.cpp)
foreach(tool_src ${ENROQUE_TOOLS})
  get_filename_component(tool_name ${tool_src} NAME_WE)
  add_executable(${tool_name} ${tool_src})
  target_link_libraries(${tool_name} PRIVATE enroque_lib)
endforeach()
