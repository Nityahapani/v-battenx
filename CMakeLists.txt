cmake_minimum_required(VERSION 3.18)
project(vbatten_x VERSION 5.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(USE_CUDA   "Enable CUDA support"    OFF)
option(USE_MPI    "Enable MPI support"     OFF)
option(USE_OPENMP "Enable OpenMP support"  ON)

find_package(Eigen3 3.4 REQUIRED)

if(USE_OPENMP)
    find_package(OpenMP QUIET)
    if(OpenMP_CXX_FOUND)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    endif()
endif()

if(USE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_definitions(-DVBATTENX_CUDA)
endif()

if(USE_MPI)
    find_package(MPI REQUIRED)
    add_definitions(-DVBATTENX_MPI)
endif()

include_directories(
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIR}
)

add_library(vbatten_x SHARED
    src/vbatten_x_impl.cc
    src/c_api.cc
)

target_include_directories(vbatten_x PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(vbatten_x PRIVATE Eigen3::Eigen)

if(USE_CUDA)
    target_link_libraries(vbatten_x PRIVATE CUDA::cublas CUDA::cudart)
endif()
if(USE_MPI)
    target_link_libraries(vbatten_x PRIVATE MPI::MPI_CXX)
endif()

set_target_properties(vbatten_x PROPERTIES
    VERSION   ${PROJECT_VERSION}
    SOVERSION 5
    OUTPUT_NAME "vbatten_x"
    POSITION_INDEPENDENT_CODE ON
)

install(TARGETS vbatten_x
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
install(DIRECTORY include/ DESTINATION include)

enable_testing()
find_package(GTest QUIET)
if(GTest_FOUND)
    add_executable(vbx_cpp_tests tests/cpp/test_dim_transitions.cc)
    target_include_directories(vbx_cpp_tests PRIVATE ${CMAKE_SOURCE_DIR})
    target_link_libraries(vbx_cpp_tests PRIVATE Eigen3::Eigen)
    add_test(NAME vbx_dim_transitions COMMAND vbx_cpp_tests)
endif()
