set(TEST_SOURCES test_array.cpp)

add_executable(TestULinalg ${TEST_SOURCES})
target_link_libraries(TestULinalg array)
add_test(NAME RunTests COMMAND TestULinalg)
