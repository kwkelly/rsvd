cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(RSVD CXX C)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

add_executable(rsvd_test src/rsvd_test.cpp )

find_package(MPI REQUIRED)
include_directories(${MPI_CXX_INCLUDE_PATH})
find_package(Elemental REQUIRED)
include_directories(${Elemental_INCLUDE_DIRS})
find_package(OpenMP REQUIRED)

include_directories("include")

set(LIBS ${LIBS} ${Elemental_LIBRARIES})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,-rpath,$ENV{TACC_IMPI_LIB} -std=c++11 -g -O0 ${MPI_CXX_COMPILE_FLAGS} ${OpenMP_CXX_FLAGS}")

target_link_libraries(rsvd_test ${LIBS})
