cmake_minimum_required(VERSION 3.15...3.31)
project(uLinalg VERSION 0.0.1
                DESCRIPTION "A Micro 2D array library"
                LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

# add testing support for this build,
# setting BUILD_TESTING = TRUE
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  include(CTest)
endif()

# compile the array library
add_library(array STATIC src/array.cpp src/array.hpp)

# add the executable and link it
add_executable(main main.cpp)
target_link_libraries(main PUBLIC array)

# add the tests to be built
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND BUILD_TESTING)
  add_subdirectory(tests)
endif()
