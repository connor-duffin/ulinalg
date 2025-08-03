#include "../src/array.hpp"
#include "../src/decomp.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

using namespace Catch::Matchers;

TEST_CASE("Decomposition initialization works as expected", "[LUDecomp]") {
  Array A(4, 4);
  std::vector<double> vals = {2, 1, 1, 0, 4, 3, 3, 1, 8, 7, 9, 5, 6, 7, 9, 8};
  A.set_vals(vals);
  LUDecomp LU = LUDecomp(A, 4);

  // check dimensions
  int nrow = LU.get_nrows();
  int ncol = LU.get_ncols();
  REQUIRE(nrow == 4);
  REQUIRE(ncol == 4);
}

TEST_CASE("Non-pivoting LU decomposition works as expected: 4x4",
          "[LUDecomp][decompose]") {
  std::vector<double> vals = {2, 1, 1, 0, 4, 3, 3, 1, 8, 7, 9, 5, 6, 7, 9, 8};
  Array A(vals, 4, 4);
  LUDecomp LU(A, 4);
  LU.decompose();

  // inplace true LU decomposition
  std::vector<double> LU_vals_true = {
      8,         7,          9,         5,          3.0 / 4.0,  7.0 / 4.0,
      9.0 / 4.0, 17.0 / 4.0, 0.5,       -2.0 / 7.0, -6.0 / 7.0, -2.0 / 7.0,
      1.0 / 4.0, -3.0 / 7.0, 1.0 / 3.0, 2.0 / 3.0};
  std::vector<double> LU_vals = LU.get_vals();
  for (int i = 0; i < LU_vals.size(); ++i) {
    REQUIRE_THAT(LU_vals[i], WithinAbs(LU_vals_true[i], 1e-6));
  }
}

TEST_CASE("LU solve works (same rows): 4x4", "[LUDecomp][solve]") {
  std::vector<double> vals = {2, 1, 1, 0, 4, 3, 3, 1, 8, 7, 9, 5, 6, 7, 9, 8};
  Array A(vals, 4, 4);
  LUDecomp LU(A, 4);
  LU.decompose();

  Array b(4, 1);
  b.set_ones();
  Array x = LU.solve(b);

  std::vector<double> x_vals = x.get_vals();
  std::vector<double> x_vals_true = {1.5, -1.0, -1.0, 1.0};
  for (int i = 0; i < x_vals.size(); ++i) {
    REQUIRE_THAT(x_vals[i], WithinAbs(x_vals_true[i], 1e-6));
  }
}

TEST_CASE("LU solve works (w/row pivoting): 4x4", "[LUDecomp][solve]") {
  std::vector<double> vals = {2, 1, 1, 0, 4, 3, 3, 1, 8, 7, 9, 5, 6, 7, 9, 8};
  Array A(vals, 4, 4);
  LUDecomp LU(A, 4);
  LU.decompose();

  std::vector<double> b_vals = {1.0, 2.0, 3.0, 4.0};
  Array b(b_vals, 4, 1);
  Array x = LU.solve(b);

  std::vector<double> x_vals = x.get_vals();
  std::vector<double> x_vals_true = {1.0, 0.5, -1.5, 1.0};
  for (int i = 0; i < x_vals.size(); ++i) {
    REQUIRE_THAT(x_vals[i], WithinAbs(x_vals_true[i], 1e-6));
  }
}

TEST_CASE("LU solve works (w/row pivoting): 5x5") {
  std::vector<double> vals = {2, 1, 0, 0, 0, 1, 2, 1, 0, 0, 0, 1, 2,
                              1, 0, 0, 0, 1, 2, 1, 0, 0, 0, 1, 2};
  Array A(vals, 5, 5);
  LUDecomp LU(A, 5);
  LU.decompose();

  std::vector<double> b_vals = {1.0, 2.0, 3.0, 2.0, 1.0};
  Array b(b_vals, 5, 1);
  Array x = LU.solve(b);
  std::vector<double> x_vals = x.get_vals();
  std::vector<double> x_vals_true = {0.5, 0.0, 1.5, 0.0, 0.5};
  for (int i = 0; i < x_vals.size(); ++i) {
    REQUIRE_THAT(x_vals[i], WithinAbs(x_vals_true[i], 1e-6));
  }
}

TEST_CASE("Cholesky decomposition initialization: 4x4 trivial",
          "[Cholesky][decompose]") {
  Array A(4, 4);
  std::vector<double> vals = {2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2};
  A.set_vals(vals);
  Cholesky chol = Cholesky(A, 4);

  // check dimensions
  int nrow = chol.get_nrows();
  int ncol = chol.get_ncols();
  REQUIRE(nrow == 4);
  REQUIRE(ncol == 4);
}

TEST_CASE("Cholesky decomposition: 4x4 trivial", "[Cholesky][decompose]") {
  Array A(4, 4);
  std::vector<double> vals = {2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2};
  A.set_vals(vals);
  Cholesky chol = Cholesky(A, 4);
  chol.decompose();

  std::vector<double> chol_vals = chol.get_vals();
  std::vector<double> chol_vals_true = {
      std::sqrt(2), 0, 0, 0, 0, std::sqrt(2), 0, 0, 0, 0,
      std::sqrt(2), 0, 0, 0, 0, std::sqrt(2)};

  for (int i = 0; i < chol_vals.size(); ++i) {
    REQUIRE_THAT(chol_vals[i], WithinAbs(chol_vals_true[i], 1e-6));
  }
}

TEST_CASE("Cholesky decomposition works: 3x3 totally nonnegative",
          "[Cholesky][decompose]") {
  Array A(3, 3);
  std::vector<double> vals = {4, 6, 2, 6, 13, 5, 2, 5, 6};
  A.set_vals(vals);
  Cholesky chol = Cholesky(A, 3);
  chol.decompose();

  // won't overwrite other entries with zeros to save space
  std::vector<double> chol_vals = chol.get_vals();
  std::vector<double> chol_vals_true = {2, 6, 2, 3, 2, 5, 1, 1, 2};

  for (int i = 0; i < chol_vals.size(); ++i) {
    REQUIRE_THAT(chol_vals[i], WithinAbs(chol_vals_true[i], 1e-6));
  }
}

TEST_CASE("Cholesky solve works: 3x3 totally nonnegative",
          "[Cholesky][solve]") {
  std::vector<double> vals = {4, 6, 2, 6, 13, 5, 2, 5, 6};
  Array A(vals, 3, 3);

  std::vector<double> b_vals = {1, 1, 1};
  Array b(b_vals, 3, 1);
  Cholesky chol = Cholesky(A, 3);
  chol.decompose();
  Array x = chol.solve(b);

  // won't overwrite other entries with zeros to save space
  std::vector<double> x_vals = x.get_vals();
  std::vector<double> x_vals_true = {0.484375, -0.21875, 0.1875};

  for (int i = 0; i < x_vals.size(); ++i) {
    REQUIRE_THAT(x_vals[i], WithinAbs(x_vals_true[i], 1e-6));
  }
}
