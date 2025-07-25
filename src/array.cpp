#include "array.hpp"

#include <iostream>
#include <stdexcept>
#include <vector>

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

Array Array::mult(Array &m) {
  // compute A = this @ m (= this.mult(m))
  int nrow_l = nrow;
  int ncol_l = ncol;

  // right-hand dimensions
  int nrow_r = m.nrow;
  int ncol_r = m.ncol;

  if (ncol_l != nrow_r) {
    throw std::invalid_argument("Dimensions prohibit matrix multiplication");
  }

  Array res = Array(nrow_l, ncol_r);
  res.set_zeros();

  for (int i = 0; i < nrow_l; ++i) {
    for (int j = 0; j < ncol_r; ++j) {
      for (int k = 0; k < ncol_l; ++k) {
        res[i][j] += vals[i + nrow_l * k] * m[k][j];
      }
    }
  }

  return res;
}

Array Array::operator+(Array summand) {
    int ncol_right = summand.get_ncol();
    int ncol_left = get_ncol();

    int nrow_right = summand.get_nrow();
    int nrow_left = get_nrow();

    int ncol_out = 1;
    int nrow_out = 1;

    // As in numpy: check right to left (columns then rows)
    // First check across the column counts
    if (ncol_right == ncol_left) {
        ncol_out = ncol_right;
    } else if (ncol_left == 1) {
        ncol_out = ncol_right;
    } else if (ncol_right == 1) {
        ncol_out = ncol_left;
    } else {
        throw std::invalid_argument("Columns prohibit broadcasting");
    }

    // Now check along the rows
    if (nrow_right == nrow_left) {
        nrow_out = nrow_right;
    } else if (nrow_right == 1) {
        nrow_out = nrow_right;
    } else if (nrow_right == 1) {
        nrow_out = nrow_left;
    } else {
        throw std::invalid_argument("Rows prohibit broadcasting");
    }

    // Initialize our output
    Array res(nrow_out, ncol_out);
    res.set_ones();
    std::vector<int> vals = res.get_vals();

    // Set the values of the sum
    for (size_t i = 0; i < vals.size(); ++i) {
        res.vals[i] = vals[i];
    }

    return res;
}

int* Array::operator[](int r) {
  return &vals[r * ncol];
}


// broadcast the input array to match this array
// arr.bcast({1}) = {{1, 1, 1, 1}, {1, 1, 1, 1}}
// we can only broadcast a 2d array if the dimensions match up OK
// bcast((4x1)) -> 4x4
// bcast((1x4)) -> 4x4
Array bcast(Array& input, int nrow, int ncol) {
  // store the number of rows/columns, and the values
  int input_nrow = input.get_nrow();
  int input_ncol = input.get_ncol();

  if (input_nrow == nrow && input_ncol == ncol) {
    return input;
  } else {
    // initialize this input array
    Array res(nrow, ncol);

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
}

