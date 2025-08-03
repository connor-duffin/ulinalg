#include "decomp.hpp"
#include "array.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

Decomp::Decomp(Array &A, int dim) : n(dim), M(dim, dim) {
  // Check that dimensions are square
  int nrow = M.get_nrow();
  int ncol = M.get_ncol();
  if (n != nrow && n != ncol) {
    std::invalid_argument("nrows != ncols: This class only works for square "
                          "arrays (square matrices)!");
  }

  // If dimensions are OK, initialize the M matrix
  std::vector<double> vals = A.get_vals();
  M.set_vals(vals);
}

int Decomp::get_nrows() const { return n; }

int Decomp::get_ncols() const { return n; }

std::vector<double> Decomp::get_vals() { return M.get_vals(); }

// Only need to initialize the pivot vector
LUDecomp::LUDecomp(Array &A, int dim) : Decomp(A, dim), p(dim) {
  for (int i = 0; i < dim; ++i) {
    p[i] = i;
  }
}

// In-place LU decomposition with partial (column) pivoting
void LUDecomp::decompose() {
  double max_curr;
  double l_mult{1};
  double temp_dbl;

  // initialize pivot temps
  int pivot_row;
  int temp_pivot;

  for (int i = 0; i < (n - 1); ++i) {
    max_curr = 0.0;
    pivot_row = i;

    // Find the pivot row (having the maximal entry)
    for (int m = i + 1; m < n; ++m) {
      if (std::abs(M[m][i]) >= max_curr) {
        max_curr = M[m][i];
        pivot_row = m;
      }
    }

    // Fail if the pivot is less than tolerance
    if (std::abs(max_curr) <= 1e-8) {
      throw std::runtime_error("Not able to proceed as pivot is below tol");
    }

    // Swap the rows
    for (int m = 0; m < n; ++m) {
      temp_dbl = M[i][m];
      M[i][m] = M[pivot_row][m];
      M[pivot_row][m] = temp_dbl;
    }

    // Save the swaps with the pivot vector
    std::swap(p[i], p[pivot_row]);

    for (int j = i + 1; j < n; ++j) {
      // Compute the multiplier for the current row
      l_mult = M[j][i] / M[i][i];
      for (int k = i; k < n; ++k) {
        // Loop over the row, eliminating elements as we go
        M[j][k] -= l_mult * M[i][k];
      }
      // After writing this, then write over zeros with the multiplier
      M[j][i] = l_mult;
    }
  }
}

// Solve using the LU decomposition
Array LUDecomp::solve(Array &b) {
  // Check input dimension align
  int len_b = b.get_nrow();
  int width_b = b.get_ncol();
  if (len_b != n || width_b != 1) {
    throw std::invalid_argument("Input dimensions incompatible");
  }

  // Permute the rows of b, into x
  Array x(n, 1);
  for (int i = 0; i < n; ++i) {
    x[i][0] = b[p[i]][0];
  }

  // First forward solve
  for (int i = 1; i < n; ++i) {
    for (int j = 0; j < i; ++j) {
      x[i][0] -= M[i][j] * x[j][0];
    }
  }

  // Then backsolve to finish up
  x[n - 1][0] /= M[n - 1][n - 1];
  for (int i = n - 2; i >= 0; --i) {
    for (int j = n - 1; j > i; --j) {
      x[i][0] -= M[i][j] * x[j][0];
    }
    x[i][0] /= M[i][i];
  }

  return x;
}

Cholesky::Cholesky(Array &A, int dim) : Decomp(A, dim) {}

// In-place Cholesky decomposition, returning upper-triangular matrix
void Cholesky::decompose() {
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j <= i; ++j) {
      double sum = 0;

      for (int k = 0; k < j; k++) {
        sum += M[i][k] * M[j][k];
      }

      if (i == j) {
        M[i][j] = std::sqrt(M[i][i] - sum);
      } else {
        M[i][j] = (1.0 / M[j][j] * (M[i][j] - sum));
      }
    }
  }
}

Array Cholesky::solve(Array &b) {
  // Initialize output array
  std::vector<double> b_vals = b.get_vals();
  Array x(b_vals, n, 1);

  // First forward solve
  x[0][0] /= M[0][0];
  for (int i = 1; i < n; ++i) {
    for (int j = 0; j < i; ++j) {
      x[i][0] -= M[i][j] * x[j][0];
    }
    x[i][0] /= M[i][i];
  }

  // Then backsolve (via transposed M) to finish up
  x[n - 1][0] /= M[n - 1][n - 1];
  for (int i = n - 2; i >= 0; --i) {
    for (int j = n - 1; j > i; --j) {
      x[i][0] -= M[j][i] * x[j][0];
    }
    x[i][0] /= M[i][i];
    std::cout << x[i][0] << std::endl;
  }

  return x;
}
