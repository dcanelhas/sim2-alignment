#(c) Daniel Ricão Canelhas 
#BSD licensed, see align.cpp for details.
cmake_minimum_required(VERSION 2.8)
project(align)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)

#set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(CMAKE_BUILD_TYPE Release)
#set(CMAKE_BUILD_TYPE Debug)

#set the default path for built executables to the "bin" directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

set(CMAKE_CXX_FLAGS "-O3 -march=native -fopenmp")


include_directories(${EIGEN3_INCLUDE_DIRS})
include_directories(${OPENCV_INCLUDE_DIRS})
link_directories(${OPENCV_LIBRARY_DIRS})
add_definitions(${OPENCV_DEFINITIONS})

add_executable(align src/align.cpp)
target_link_libraries(align ${OpenCV_LIBS})
