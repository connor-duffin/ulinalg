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
int Array::get_nrow() const { return nrow; }

// Get the number of cols in the array object
int Array::get_ncol() const { return ncol; }

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

// TODO: write tests for this
// Set the elements to have ones along the main diagonal
void Array::eye() {
  set_zeros();
  for (int i = 0; i < nrow; ++i) {
    vals[i * (1 + ncol)] = 1;
  }
}

// Set the values of an array from a vector:
// The values array has to be of the exact same size as expected
void Array::set_vals(std::vector<int> &values) {
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

// TODO: add ability to do this in place if dimensions allow for it
Array Array::operator+(Array summand) {
  int ncol_right = summand.get_ncol();
  int ncol_left = get_ncol();

  int nrow_right = summand.get_nrow();
  int nrow_left = get_nrow();

  int ncol_out;
  int nrow_out;

  // Get the output columns
  if (ncol_right == ncol_left) {
    ncol_out = ncol_right;
  } else if (ncol_left == 1) {
    ncol_out = ncol_right;
  } else if (ncol_right == 1) {
    ncol_out = ncol_left;
  } else {
    throw std::invalid_argument("Columns prohibit broadcasting");
  }

  // Get the output rows
  if (nrow_right == nrow_left) {
    nrow_out = nrow_right;
  } else if (nrow_left == 1) {
    nrow_out = nrow_right;
  } else if (nrow_right == 1) {
    nrow_out = nrow_left;
  } else {
    throw std::invalid_argument("Rows prohibit broadcasting");
  }

  std::cout << "Left bcasting" << nrow_out << ncol_out << std::endl;
  std::vector<int> left_idx =
    array_detail::get_bcast_idx(nrow_left, ncol_left, nrow_out, ncol_out);

  std::cout << "Right bcasting" << std::endl;
  std::vector<int> right_idx =
    array_detail::get_bcast_idx(nrow_right, ncol_right, nrow_out, ncol_out);

  // Set the return Array
  Array res(nrow_out, ncol_out);
  for (size_t i = 0; i < nrow_out * ncol_out; ++i) {
    res.vals[i] = vals[left_idx[i]] + summand.vals[right_idx[i]];
  }

  return res;
}

int *Array::operator[](int r) { return &vals[r * ncol]; }

Array array_detail::bcast(Array &input, int nrow, int ncol) {
  // store the number of rows/columns, and the values
  int nrow_in = input.get_nrow();
  int ncol_in = input.get_ncol();

  if (nrow_in == nrow && ncol_in == ncol) {
    return input;
  } else {
    // initialize this input array
    Array res(nrow, ncol);

    std::vector<int> input_vals = input.get_vals();
    std::vector<int> input_val_vector(nrow * ncol);
    std::vector<int> idx_bcast = get_bcast_idx(nrow_in, ncol_in, nrow, ncol);

    for (auto i = 0; i < input_val_vector.size(); ++i) {
      input_val_vector[i] = input_vals[idx_bcast[i]];
    }
    res.set_vals(input_val_vector);

    return res;
  }
}

std::vector<int> array_detail::get_bcast_idx(int nrow_in, int ncol_in,
                                             int nrow_out, int ncol_out) {
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
