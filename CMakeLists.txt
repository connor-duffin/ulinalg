cmake_minimum_required(VERSION 3.15...3.31)
project(uLinalg VERSION 0.0.1
                DESCRIPTION "A Micro 2D array library"
                LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

add_executable(main main.cpp src/array.cpp)
