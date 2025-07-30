#include "../src/array.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <cstdint>
#include <vector>

TEST_CASE("Array initializations work OK", "[array]") {
  // Empty initialization
  Array a(2, 3);
  int nrow = a.get_nrow();
  int ncol = a.get_ncol();
  REQUIRE(nrow == 2);
  REQUIRE(ncol == 3);

  std::vector<double> vals = a.get_vals();
  std::vector<double> vals_true = {0, 0, 0, 0, 0, 0};
  REQUIRE(vals == vals_true);

  // Matrix initialization
  std::vector<double> b_vals = {1, 2, 3, 4, 5, 6};
  Array b(b_vals, 3, 2);
  REQUIRE(b.get_nrow() == 3);
  REQUIRE(b.get_ncol() == 2);
}

TEST_CASE("Basic array operations work", "[array]") {
  Array a(2, 3);
  a.set_ones();
  std::vector<double> v = a.get_vals();
  for (auto it = v.begin(); it != v.end(); ++it) {
    REQUIRE(*it == 1);
  }

  v = a.get_vals();
  for (auto it = v.begin(); it != v.end(); ++it) {
    REQUIRE(*it == 1);
  }

  // double the array via addition
  Array a2 = a + a;
  v = a2.get_vals();
  for (auto it = v.begin(); it != v.end(); ++it) {
    REQUIRE(*it == 2);
  }

  // check operator[] works for setting elements
  a[0][0] = 5;
  a[1][2] = 5;
  a.pprint();
  v = a.get_vals();
  REQUIRE(v[0] == 5);
  REQUIRE(v[5] == 5);
}

TEST_CASE("Addition with broadcasting works as expected", "[array][add]") {
  std::vector<double> vals = {1, 2, 3, 4, 5, 6, 7, 8};
  Array u(2, 4);
  u.set_vals(vals);

  Array z(1, 4);
  z.set_ones();

  Array upp = u + z;
  std::vector<double> valspp = upp.get_vals();
  for (size_t i = 0; i < valspp.size(); ++i) {
    REQUIRE(valspp[i] == (vals[i] + 1));
  }
}

TEST_CASE("Checking that broadcast addition is symmetric", "[array][add]") {
  std::vector<double> vals = {1, 2, 3};
  Array u(1, 3);
  u.set_vals(vals);
  Array v(3, 1);
  v.set_vals(vals);

  Array out = u + v;
  std::vector<double> vals_true = {2, 3, 4, 3, 4, 5, 4, 5, 6};
  REQUIRE(out.get_vals() == vals_true);

  Array out_sym = v + u;
  REQUIRE(out_sym.get_vals() == vals_true);
}

TEST_CASE("Array broadcasting works", "[array][bcast]") {
  // Scalar ones
  Array x(1, 1);
  x.set_ones();

  // bcast scalar to 2x4 matrix
  Array x_bcast = array_detail::bcast(x, 2, 4);
  REQUIRE(x_bcast.get_nrow() == 2);
  REQUIRE(x_bcast.get_ncol() == 4);

  std::vector<double> x_vals = x_bcast.get_vals();
  for (auto it = x_vals.begin(); it != x_vals.end(); ++it) {
    REQUIRE(*it == 1);
  }

  // 2x1 to be broadcast across `u`
  Array y(2, 1);
  std::vector<double> vals = {3, 4};
  y.set_vals(vals);
  Array y_bcast = array_detail::bcast(y, 2, 4);
  REQUIRE(y_bcast.get_nrow() == 2);
  REQUIRE(y_bcast.get_ncol() == 4);
  std::vector<double> y_vals = y_bcast.get_vals();

  // first row is 3, second is 4
  for (size_t i = 0; i < y_vals.size(); ++i) {
    if (i <= 3) {
      REQUIRE(y_vals[i] == 3);
    } else {
      REQUIRE(y_vals[i] == 4);
    }
  }

  // 1x4 to be broadcast
  Array z(1, 4);
  Array z_bcast = array_detail::bcast(z, 2, 4);
  std::vector<double> z_vals = z_bcast.get_vals();
  REQUIRE(z_bcast.get_nrow() == 2);
  REQUIRE(z_bcast.get_ncol() == 4);
}

TEST_CASE("More intricate broadcasting", "[array][bcast]") {
  Array z(1, 2);
  std::vector<double> vals = {10, 10};
  z.set_vals(vals);

  // broadcast along the proper dimensions
  Array z_bcast = array_detail::bcast(z, 4, 2);
  int nrow = z_bcast.get_nrow();
  int ncol = z_bcast.get_ncol();
  REQUIRE(nrow == 4);
  REQUIRE(ncol == 2);

  // make sure all values are equal
  std::vector<double> z_vals = z_bcast.get_vals();
  for (size_t i = 0; i < z_vals.size(); ++i) {
    REQUIRE(z_vals[i] == 10);
  }
}

TEST_CASE("Can copy between array objects", "[array][copy]") {
  std::vector<double> vals = {1, 2, 3, 4};
  Array b(vals, 4, 1);
  Array x(4, 1);
  x.copy(b);
  std::vector<double> vals_x = x.get_vals();
  REQUIRE(vals == vals_x);

  // Now check more 2 dimensions
  Array c(vals, 4, 4);
  Array d(4, 4);
  d.copy(c);
  std::vector<double> d_vals = d.get_vals();
  std::vector<double> d_vals_true = {1, 2, 3, 4, 1, 2, 3, 4,
                                     1, 2, 3, 4, 1, 2, 3, 4};
  REQUIRE(d_vals == d_vals_true);
}

TEST_CASE("Matrix multiplication works as expected", "[array][mult]") {
  Array u(2, 4);
  u.set_ones();

  Array v(4, 2);
  v.set_ones();

  Array res = u.mult(v);
  REQUIRE(res.get_nrow() == 2);
  REQUIRE(res.get_ncol() == 2);

  std::vector<double> vals = res.get_vals();
  for (int i = 0; i < vals.size(); ++i) {
    REQUIRE(vals[i] == 4);
  }
}

TEST_CASE("Matrix multiplication: 2x2 * 2x2", "[array][mult]") {
  Array A(2, 2);
  Array B(2, 2);

  // A = [[1, 2], [3, 4]]
  std::vector<double> a_vals = {1, 2, 3, 4};
  A.set_vals(a_vals);

  // B = [[5, 6], [7, 8]]
  std::vector<double> b_vals = {5, 6, 7, 8};
  B.set_vals(b_vals);

  // Expected: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
  Array C = A.mult(B);
  std::vector<double> expected = {19, 22, 43, 50};
  REQUIRE(C.get_vals() == expected);
}

TEST_CASE("Matrix multiplication: 2x3 * 3x2", "[array][mult]") {
  Array A(2, 3);
  Array B(3, 2);

  // A = [[1, 2, 3], [4, 5, 6]]
  std::vector<double> a_vals = {1, 2, 3, 4, 5, 6};
  A.set_vals(a_vals);
  // B = [[7, 8], [9, 10], [11, 12]]
  std::vector<double> b_vals = {7, 8, 9, 10, 11, 12};
  B.set_vals(b_vals);

  // Expected:
  // [[1*7 + 2*9 + 3*11, 1*8 + 2*10 + 3*12],
  //  [4*7 + 5*9 + 6*11, 4*8 + 5*10 + 6*12]]
  // = [[58, 64], [139, 154]]
  Array C = A.mult(B);
  std::vector<double> expected = {58, 64, 139, 154};
  REQUIRE(C.get_vals() == expected);
}

TEST_CASE("Matrix multiplication throws on incompatible sizes",
          "[array][mult]") {
  std::vector<double> a_vals = {1, 2, 3, 4, 5, 6};
  std::vector<double> b_vals = {1, 2, 3, 4};
  Array A(a_vals, 2, 3);
  Array B(b_vals, 2, 2);

  REQUIRE_THROWS_AS(A.mult(B), std::invalid_argument);
}

TEST_CASE("Matrix multiplication: 1x3 * 3x1 (dot product as 1x1 matrix)",
          "[array][mult]") {
  std::vector<double> a_vals = {1, 2, 3};
  std::vector<double> b_vals = {4, 5, 6};
  Array A(a_vals, 1, 3);
  Array B(b_vals, 3, 1);

  // Expected: [[1*4 + 2*5 + 3*6]] = [[32]]
  Array C = A.mult(B);
  std::vector<double> expected = {32};
  REQUIRE(C.get_vals() == expected);
}

TEST_CASE("Matrix multiplication: identity matrix", "[array][mult]") {
  std::vector<double> a_vals = {9, 8, 7, 6};
  std::vector<double> I_vals = {1, 0, 0, 1};
  Array A(a_vals, 2, 2);
  Array I(I_vals, 2, 2);
  Array C = A.mult(I);
  REQUIRE(C.get_vals() == A.get_vals());
}

TEST_CASE("Array::eye() sets ones along the main diagonal for square matrix",
          "[array][eye]") {
  Array arr(3, 3);
  arr.eye();

  std::vector<double> expected = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  REQUIRE(arr.get_vals() == expected);
}

TEST_CASE("Array::eye() sets diagonal for rectangular tall matrix",
          "[array][eye]") {
  Array arr(4, 3);
  arr.eye();

  std::vector<double> expected = {1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0};
  REQUIRE(arr.get_vals() == expected);
}

TEST_CASE("Array::eye() sets diagonal for rectangular wide matrix",
          "[array][eye]") {
  Array arr(2, 4);
  arr.eye();

  std::vector<double> expected = {1, 0, 0, 0, 0, 1, 0, 0};
  REQUIRE(arr.get_vals() == expected);
}

TEST_CASE("Array::eye() works for 1x1 matrix", "[array][eye]") {
  Array arr(1, 1);
  arr.eye();

  std::vector<double> expected = {1};
  REQUIRE(arr.get_vals() == expected);
}

TEST_CASE("Array::eye() works for 1xN matrix", "[array][eye]") {
  Array arr(1, 5);
  arr.eye();

  std::vector<double> expected = {1, 0, 0, 0, 0};
  REQUIRE(arr.get_vals() == expected);
}

TEST_CASE("Array::eye() works for Nx1 matrix", "[array][eye]") {
  Array arr(4, 1);
  arr.eye();

  std::vector<double> expected = {1, 0, 0, 0};
  REQUIRE(arr.get_vals() == expected);
}
