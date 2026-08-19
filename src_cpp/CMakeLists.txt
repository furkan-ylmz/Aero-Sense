cmake_minimum_required(VERSION 3.16)
project(AeroSenseInference LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Cross-Platform AVX2 Optimization and Warning Flags
if(MSVC)
    add_compile_options(/O2 /arch:AVX2 /W4)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
else()
    add_compile_options(-O3 -mavx2 -Wall -Wextra)
endif()

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../proto/generated
)

# Optional ONNX Runtime include / lib path configuration
if(DEFINED ONNXRUNTIME_ROOT_DIR)
    include_directories(${ONNXRUNTIME_ROOT_DIR}/include)
    link_directories(${ONNXRUNTIME_ROOT_DIR}/lib)
endif()

# Optional Protobuf
find_package(Protobuf QUIET)
if(Protobuf_FOUND)
    include_directories(${Protobuf_INCLUDE_DIRS})
endif()

set(SOURCES
    src/main.cpp
)

add_executable(maneuver_inference_engine ${SOURCES})

# Cross-Platform Socket & Threading Libraries
if(WIN32)
    target_link_libraries(maneuver_inference_engine PRIVATE ws2_32)
else()
    target_link_libraries(maneuver_inference_engine PRIVATE pthread)
endif()

if(Protobuf_FOUND)
    target_link_libraries(maneuver_inference_engine PRIVATE ${Protobuf_LIBRARIES})
endif()
