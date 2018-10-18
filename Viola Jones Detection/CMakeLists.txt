cmake_minimum_required(VERSION 3.10)
project(ViolaDetection)

set(CMAKE_CXX_STANDARD 11)
find_package( OpenCV REQUIRED )

add_executable(ViolaDetection face_detection.cpp Image.cpp Image.h Classifieur.cpp Classifieur.h)

include_directories(/usr/include/mpi/)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(ViolaDetection ${OpenCV_LIBS})