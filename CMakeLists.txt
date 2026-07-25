# Charul - 002535330
# Spring 2026
# CS 5330 Computer Vision
# Project 2: Content-Based Image Retrieval

cmake_minimum_required(VERSION 3.16)
project(Project2)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

set(OpenCV_DIR "C:/Users/Charul/Downloads/opencv/build/x64/vc16/lib")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/include)

#shared library files used by both executables
set(SHARED_SOURCES
    src/features.cpp
    src/distances.cpp
    src/csv_util.cpp
)

add_executable(computeFeatures src/computeFeatures.cpp ${SHARED_SOURCES})
target_link_libraries(computeFeatures ${OpenCV_LIBS})

add_executable(matchImage src/matchImage.cpp ${SHARED_SOURCES})
target_link_libraries(matchImage ${OpenCV_LIBS})