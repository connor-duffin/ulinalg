find_package(Catch2 3 REQUIRED)

set(TEST_SOURCES test_array.cpp test_decomp.cpp)

add_executable(TestULinalg ${TEST_SOURCES})
target_link_libraries(TestULinalg PRIVATE array decomp)
target_link_libraries(TestULinalg PRIVATE Catch2::Catch2WithMain)

add_test(NAME RunTests COMMAND TestULinalg)
