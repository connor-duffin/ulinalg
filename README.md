# ulinalg

A small C++ library for array operations. This is an educational library designed to mimic the operations of popular 2D array packages (`numpy`, base `R`, etc) to better understand what operations happen under the hood.

## Development plan

- Basic array operations and matrix algebra.
- LU decomposition for square `Array` objects.
- Cholesky decomposition for square `Array` objects.

## Build

The project uses [CMake](https://cmake.org/) to build. To do so from this directory:

```bash
cmake -S . -B build
cmake --build build
cd build
make
```

## Testing

The project uses [Catch2](https://github.com/catchorg/Catch2) for unit testing. To build the project and test use:

```bash
cmake -S . -B build
cmake --build build
cd build
ctest
```
