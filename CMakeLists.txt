cmake_minimum_required(VERSION 3.18)
project(vbatten_x VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(USE_CUDA    "Enable CUDA support"    OFF)
option(USE_MPI     "Enable MPI support"     OFF)
option(USE_OPENMP  "Enable OpenMP support"  ON)

find_package(Eigen3 3.4 REQUIRED)

if(USE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    endif()
endif()

include_directories(${CMAKE_SOURCE_DIR} ${EIGEN3_INCLUDE_DIR})

add_library(vbatten_x SHARED src/c_api.cc)
target_include_directories(vbatten_x PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_link_libraries(vbatten_x PRIVATE Eigen3::Eigen)

set_target_properties(vbatten_x PROPERTIES
    VERSION   ${PROJECT_VERSION}
    SOVERSION 0
    OUTPUT_NAME "vbatten_x")

install(TARGETS vbatten_x LIBRARY DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)

enable_testing()
find_package(GTest QUIET)
if(GTest_FOUND)
    add_executable(vbx_tests
        tests/cpp/test_field_state.cc
        tests/cpp/test_topology_ops.cc
        tests/cpp/test_dim_transitions.cc)
    target_include_directories(vbx_tests PRIVATE ${CMAKE_SOURCE_DIR})
    target_link_libraries(vbx_tests PRIVATE vbatten_x GTest::gtest_main Eigen3::Eigen)
    add_test(NAME vbx_unit COMMAND vbx_tests)
endif()
