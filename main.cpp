#include <iostream>
#include <stdexcept>
#include <vector>

// Array abstraction (generic for row-major vectors/matrices)
class Array {
private:
  int nrow, ncol;
  std::vector<int> vals;

public:
  Array(int, int);
  void set_zeros();
  void set_ones();
  void set_vals(std::vector<int>&);
  void pprint();
  Array operator+(Array const &obj);
};

Array::Array(int nrows, int ncols) {
  nrow = nrows;
  ncol = ncols;
  vals = std::vector<int>(nrow * ncol);
}

void Array::set_zeros() {
  for (auto it = vals.begin(); it != vals.end(); ++it) {
    *it = 0;
  }
}

void Array::set_ones() {
  for (auto it = vals.begin(); it != vals.end(); ++it) {
    *it = 1;
  }
}

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

// Set the values of an array from a vector:
// The values array has to be of the exact same size as expected
void Array::set_vals(std::vector<int>& values) {
  int size_in = values.size();
  int size_out = vals.size();

  if (size_in != size_out) {
    throw std::invalid_argument("Smaller object not multiple of larger object");
  } else {
    // If we get to the end of values, reset to the start
    for (int i = 0; i < size_in; ++i) {
      vals[i] = values[i];
    }
  }
}

// Add two arrays together, of the same dimensions
Array Array::operator+(Array const& summand) {
  // Store result in new object with same dimension as operand
  Array res(nrow, ncol);

  for (size_t i = 0; i < vals.size(); ++i) {
    res.vals[i] = vals[i] + summand.vals[i];
  }

  return(res);
}

int main() {
  std::cout << "Initialize 2x4 ones" << std::endl;
  Array v(5, 1);
  v.set_ones();
  v.pprint();

  std::cout << "Initialize 1:8 matrix" << std::endl;
  std::vector<int> vals = {1, 2, 3, 4, 5, 6, 7, 8};
  Array u(2, 4);
  u.set_vals(vals);
  u.pprint();

  // std::vector<int> small_vals = {1, 2, 3};
  // u.set_vals(small_vals);

  std::cout << "Sum of the above results" << std::endl;
  Array w = u + v;
  w.pprint();
  return 0;
}
