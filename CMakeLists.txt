cmake_minimum_required(VERSION 3.14)
project(ANN)

set(CMAKE_CXX_STANDARD 14)

add_subdirectory(clBP)

add_executable(ANN main.cpp)

find_package(iir)

target_link_libraries(ANN clBP fir iir::iir ${OpenCV_LIBS})