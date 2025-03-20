cmake_minimum_required(VERSION 3.15...3.31)
project(uLinalg VERSION 0.0.1
                DESCRIPTION "A Micro 2D array library"
                LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

# compile the array library
add_library(array STATIC src/array.cpp)

# add the executable and link it
add_executable(main main.cpp)
target_link_libraries(main PUBLIC array)
