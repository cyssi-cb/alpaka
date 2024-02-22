cmake_minimum_required(VERSION 3.22)
set(_TARGET_NAME lulesh_orig)

project(${_TARGET_NAME} LANGUAGES CUDA CXX)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

if (WITH_MPI)
  find_package(MPI REQUIRED)
  include_directories(${MPI_C_INCLUDE_PATH} ${MPI_CXX_INCLUDE_PATH})
  add_definitions("-DUSE_MPI=1")
  list(APPEND LULESH_EXTERNAL_LIBS ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES})
else()
  add_definitions("-DUSE_MPI=0")
endif()

if (WITH_OPENMP)
  find_package(OpenMP REQUIRED)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

set(LULESH_SOURCES ./src/allocator.h ./src/allocator.cu ./src/lulesh-comms.cu ./src/lulesh-comms-gpu.cu ./src/lulesh.cu  ./src/lulesh.h ./src/util.h ./src/vector.h)
add_executable(${_TARGET_NAME} ${LULESH_SOURCES})
target_link_libraries(
    ${_TARGET_NAME}
    PUBLIC alpaka::alpaka)
set_target_properties(${_TARGET_NAME} PROPERTIES FOLDER lulesh)
set_target_properties(${_TARGET_NAME} PROPERTIES CUDA_ARCHITECTURES "60")

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
