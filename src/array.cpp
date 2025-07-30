#include "array.hpp"

#include <iostream>
#include <stdexcept>
#include <vector>

// Array class initialization
Array::Array(int nrows, int ncols)
    : nrow(nrows), ncol(ncols), vals(nrow * ncol) {}

// Array class initialization: if values isn't the right length, recycle it so
// that it is
Array::Array(const std::vector<double> &values, int nrows, int ncols)
    : nrow(nrows), ncol(ncols), vals(nrow * ncol) {
  int n = values.size();
  int n_out = nrow * ncol;
  for (int i = 0; i < n_out; ++i) {
    vals[i] = values[i % n];
  }
}

// Get the number of rows in the array object
int Array::get_nrow() const { return nrow; }

// Get the number of cols in the array object
int Array::get_ncol() const { return ncol; }

// Get the values from the object
std::vector<double> Array::get_vals() const { return vals; }

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

// Set the elements to have ones along the main diagonal
void Array::eye() {
  set_zeros();
  if (ncol == 1) {
    vals[0] = 1;
  } else {
    for (int i = 0; i < nrow; ++i) {
      vals[i * (1 + ncol)] = 1;
    }
  }
}

// Set the values of an array from a vector:
// The values array has to be of the exact same size as expected
void Array::set_vals(std::vector<double> &values) {
  int size_in = values.size();
  int size_out = vals.size();

  if (size_in != size_out) {
    throw std::invalid_argument("Input vector dimensions do not match!");
  } else {
    for (int i = 0; i < size_in; ++i) {
      vals[i] = values[i];
    }
  }
}

// Copy input into 'this' Array
void Array::copy(Array &input) {
  int nrow_in = input.get_nrow();
  int ncol_in = input.get_ncol();

  if (nrow_in != nrow && ncol_in != ncol) {
    throw std::invalid_argument("Input dimensions do not match: can't copy");
  } else {
    for (int i = 0; i < nrow; ++i) {
      for (int j = 0; j < ncol; ++j) {
        vals[i * ncol + j] = input[i][j];
      }
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

// Matrix multiplication
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
        res[i][j] += vals[ncol_l * i + k] * m[k][j];
      }
    }
  }

  return res;
}

// Add two arrays together
Array operator+(const Array &a1, const Array &a2) {
  // Go from right to left as in numpy --- doesn't really matter in terms of
  // function calls but it is easier to maintain consistency
  int ncol_out = array_detail::get_op_ncol_out(a1, a2);
  int nrow_out = array_detail::get_op_nrow_out(a1, a2);

  std::vector<int> left_idx =
      array_detail::get_bcast_idx(a1, nrow_out, ncol_out);

  std::vector<int> right_idx =
      array_detail::get_bcast_idx(a2, nrow_out, ncol_out);

  // Set the return Array
  Array res(nrow_out, ncol_out);
  for (size_t i = 0; i < nrow_out * ncol_out; ++i) {
    res.vals[i] = a1.vals[left_idx[i]] + a2.vals[right_idx[i]];
  }

  return res;
}

// Allow for indexing operations e.g. a[1][2]
// It works by first returning a pointer which starts at the specified row; we
// the slice this row as required, to get our value. Arithmetically:
//
// a[i][j] === &vals[i * ncol][j] === *(&vals[i * ncol] + 2)
//
// This works because of the way pointer arithmetic works in C++:
// x[10] === *(x + 10) ==== *(10 + x) === 10[x] (!)
double *Array::operator[](int r) { return &vals[r * ncol]; }

// Take an array and broadcast it into a new Array
Array array_detail::bcast(Array &input, int nrow, int ncol) {
  std::vector<double> input_vals = input.get_vals();
  std::vector<double> input_val_vector(nrow * ncol);
  std::vector<int> idx_bcast = get_bcast_idx(input, nrow, ncol);

  for (auto i = 0; i < input_val_vector.size(); ++i) {
    input_val_vector[i] = input_vals[idx_bcast[i]];
  }

  // Initialize the output object
  Array res(input_val_vector, nrow, ncol);

  return res;
}

std::vector<int> array_detail::get_bcast_idx(const Array &a, int nrow_out,
                                             int ncol_out) {
  int nrow_in = a.get_nrow();
  int ncol_in = a.get_ncol();
  int n_elements = nrow_out * ncol_out;
  std::vector<int> idx(n_elements);
  if (nrow_in == nrow_out && ncol_in == ncol_out) {
    for (int i = 0; i < n_elements; ++i) {
      idx[i] = i;
    }
  } else if (nrow_in == 1 && ncol_in == 1) {
    for (int i = 0; i < n_elements; ++i) {
      idx[i] = 0;
    }
  } else if (nrow_in == nrow_out && ncol_in == 1) {
    for (int i = 0; i < n_elements; ++i) {
      idx[i] = i / ncol_out;
    }
  } else if (nrow_in == 1 && ncol_in == ncol_out) {
    for (int i = 0; i < n_elements; ++i) {
      idx[i] = i % ncol_out;
    }
  } else {
    throw std::invalid_argument("Dimensions prohibit broadcasting");
  }

  return idx;
}

int array_detail::get_op_ncol_out(const Array &a1, const Array &a2) {
  // Initialize return value
  int ncol_out(1);

  // Get the output columns
  int ncol_right = a2.get_ncol();
  int ncol_left = a1.get_ncol();

  if (ncol_right == ncol_left) {
    ncol_out = ncol_right;
  } else if (ncol_left == 1) {
    ncol_out = ncol_right;
  } else if (ncol_right == 1) {
    ncol_out = ncol_left;
  } else {
    throw std::invalid_argument("Columns prohibit broadcasting");
  }

  return ncol_out;
}

int array_detail::get_op_nrow_out(const Array &a1, const Array &a2) {
  // Initialize return value
  int nrow_out(1);

  // Get the output rows
  int nrow_right = a2.get_nrow();
  int nrow_left = a1.get_nrow();

  if (nrow_right == nrow_left) {
    nrow_out = nrow_right;
  } else if (nrow_left == 1) {
    nrow_out = nrow_right;
  } else if (nrow_right == 1) {
    nrow_out = nrow_left;
  } else {
    throw std::invalid_argument("Rows prohibit broadcasting");
  }

  return nrow_out;
}
