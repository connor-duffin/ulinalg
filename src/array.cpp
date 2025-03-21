#include "array.hpp"

#include <iostream>
#include <stdexcept>

// Array class initialization
Array::Array(int nrows, int ncols) {
  nrow = nrows;
  ncol = ncols;
  vals = std::vector<int>(nrow * ncol);
}

// Get the number of rows in the array object
int Array::get_nrow() { return nrow; }

// Get the number of cols in the array object
int Array::get_ncol() { return ncol; }

// Get the values from the object
std::vector<int> Array::get_vals() { return vals; }

// Set the elements to zeros
void Array::set_zeros() {
  for (auto it = vals.begin(); it != vals.end(); ++it) {
    *it = 0;
  }
}

// Set the elements to ones
void Array::set_ones() {
  for (auto it = vals.begin(); it != vals.end(); ++it) {
    *it = 1;
  }
}

// Set the values of an array from a vector:
// The values array has to be of the exact same size as expected
void Array::set_vals(std::vector<int> &values) {
  int size_in = values.size();
  int size_out = vals.size();

  if (size_in != size_out) {
    throw std::invalid_argument(
      "Input vector dimensions do not match!"
    );
  } else {
    for (int i = 0; i < size_in; ++i) {
      vals[i] = values[i];
    }
  }
}

// Pretty print the output array
void Array::pprint() {
  for (size_t i = 0; i < vals.size(); ++i) {
    std::cout << vals[i];

    if ((i + 1) % ncol == 0) {
      std::cout << std::endl;
    } else {
      std::cout << ", ";
    }
  }
}

// broadcast the input array to match this array
// arr.bcast({1}) = {{1, 1, 1, 1}, {1, 1, 1, 1}}
// we can only broadcast a 2d array if the dimensions match up OK
// bcast((4x1)) -> 4x4
// bcast((1x4)) -> 4x4
Array Array::bcast(Array& input) {
  // initialize this input array
  Array res(nrow, ncol);

  // store the number of rows/columns, and the values
  unsigned int input_nrow = input.get_nrow();
  unsigned int input_ncol = input.get_ncol();
  std::vector<int> input_vals = input.get_vals();
  std::vector<int> input_val_vector(nrow * ncol);

  if (input_nrow == 1 && input_ncol == 1) {
    // get scalar values, and broadcast to vector
    int input_val_scalar = input_vals[0];

    for (auto it = input_val_vector.begin(); it != input_val_vector.end();
         ++it) {
      *it = input_val_scalar;
    }

  } else if (input_nrow == nrow && input_ncol == 1) {
    // TODO: fix this as it currently does not work...
    for (auto i = 0; i < input_val_vector.size(); ++i) {
      input_val_vector[i] = input_vals[i / ncol];
    }
  } else if (input_nrow == 1 && input_ncol == ncol) {
    for (auto i = 0; i < input_val_vector.size(); ++i) {
      input_val_vector[i] = input_vals[i % ncol];
    }
  } else {
    throw std::invalid_argument("Dimensions prohibit broadcasting");
  }

  res.set_vals(input_val_vector);

  return res;
}

Array Array::operator+(Array const& summand) {
  // Store result in new object with same dimension as operand
  Array res(nrow, ncol);

  for (size_t i = 0; i < vals.size(); ++i) {
    res.vals[i] = vals[i] + summand.vals[i];
  }

  return res;
}

int* Array::operator[](int r) {
  return &vals[r * ncol];
}
