#include "../src/array.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <cstdint>
#include <vector>

TEST_CASE("Basic array operations work", "[array]") {
  Array a(2, 3);

  REQUIRE(a.get_nrow() == 2);
  REQUIRE(a.get_ncol() == 3);
  REQUIRE(a.get_vals()[0] == 0); // uninitialized should be 0

  a.set_ones();
  std::vector<int> v = a.get_vals();
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

TEST_CASE("Addition with broadcasting works as expected", "[array]") {
  std::vector<int> vals = {1, 2, 3, 4, 5, 6, 7, 8};
  Array u(2, 4);
  u.set_vals(vals);

  Array z(1, 4);
  z.set_ones();

  Array upp = u + z;
  std::vector<int> valspp = upp.get_vals();
  for (size_t i = 0; i < valspp.size(); ++i) {
    REQUIRE(valspp[i] == (vals[i] + 1));
  }
}

TEST_CASE("Checking that broadcast addition is symmetric", "[array]") {
  std::vector<int> vals = {1, 2, 3};
  Array u(1, 3);
  u.set_vals(vals);
  Array v(3, 1);
  v.set_vals(vals);

  Array out = u + v;
  std::vector<int> vals_true = {2, 3, 4, 3, 4, 5, 4, 5, 6};
  REQUIRE(out.get_vals() == vals_true);

  Array out_sym = v + u;
  REQUIRE(out_sym.get_vals() == vals_true);
}

TEST_CASE("Array broadcasting works", "[array]") {
  // Scalar ones
  Array x(1, 1);
  x.set_ones();

  // bcast scalar to 2x4 matrix
  Array x_bcast = array_detail::bcast(x, 2, 4);
  REQUIRE(x_bcast.get_nrow() == 2);
  REQUIRE(x_bcast.get_ncol() == 4);

  std::vector<int> x_vals = x_bcast.get_vals();
  for (auto it = x_vals.begin(); it != x_vals.end(); ++it) {
    REQUIRE(*it == 1);
  }

  // 2x1 to be broadcast across `u`
  Array y(2, 1);
  std::vector<int> vals = {3, 4};
  y.set_vals(vals);
  Array y_bcast = array_detail::bcast(y, 2, 4);
  REQUIRE(y_bcast.get_nrow() == 2);
  REQUIRE(y_bcast.get_ncol() == 4);
  std::vector<int> y_vals = y_bcast.get_vals();

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
  std::vector<int> z_vals = z_bcast.get_vals();
  REQUIRE(z_bcast.get_nrow() == 2);
  REQUIRE(z_bcast.get_ncol() == 4);
}

TEST_CASE("More intricate broadcasting", "[array]") {
  Array z(1, 2);
  std::vector<int> vals = {10, 10};
  z.set_vals(vals);

  // broadcast along the proper dimensions
  Array z_bcast = array_detail::bcast(z, 4, 2);
  int nrow = z_bcast.get_nrow();
  int ncol = z_bcast.get_ncol();
  REQUIRE(nrow == 4);
  REQUIRE(ncol == 2);

  // make sure all values are equal
  std::vector<int> z_vals = z_bcast.get_vals();
  for (size_t i = 0; i < z_vals.size(); ++i) {
    REQUIRE(z_vals[i] == 10);
  }
}

TEST_CASE("Matrix multiplication works as expected", "[array][mult]") {
  Array u(2, 4);
  u.set_ones();

  Array v(4, 2);
  v.set_ones();

  Array res = u.mult(v);
  REQUIRE(res.get_nrow() == 2);
  REQUIRE(res.get_ncol() == 2);

  std::vector<int> vals = res.get_vals();
  for (int i = 0; i < vals.size(); ++i) {
    REQUIRE(vals[i] == 4);
  }
}

TEST_CASE("Matrix multiplication: 2x2 * 2x2", "[array][mult]") {
  Array A(2, 2);
  Array B(2, 2);

  // A = [[1, 2], [3, 4]]
  A.set_vals({1, 2, 3, 4});
  // B = [[5, 6], [7, 8]]
  B.set_vals({5, 6, 7, 8});

  // Expected: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
  Array C = A.mult(B);
  std::vector<int> expected = {19, 22, 43, 50};
  REQUIRE(C.get_vals() == expected);
}

TEST_CASE("Matrix multiplication: 2x3 * 3x2", "[array][mult]") {
  Array A(2, 3);
  Array B(3, 2);

  // A = [[1, 2, 3], [4, 5, 6]]
  A.set_vals({1, 2, 3, 4, 5, 6});
  // B = [[7, 8], [9, 10], [11, 12]]
  B.set_vals({7, 8, 9, 10, 11, 12});

  // Expected:
  // [[1*7 + 2*9 + 3*11, 1*8 + 2*10 + 3*12],
  //  [4*7 + 5*9 + 6*11, 4*8 + 5*10 + 6*12]]
  // = [[58, 64], [139, 154]]
  Array C = A.mult(B);
  std::vector<int> expected = {58, 64, 139, 154};
  REQUIRE(C.get_vals() == expected);
}

TEST_CASE("Matrix multiplication throws on incompatible sizes",
          "[array][mult]") {
  Array A(2, 3);
  Array B(2, 2);
  A.set_vals({1, 2, 3, 4, 5, 6});
  B.set_vals({1, 2, 3, 4});

  REQUIRE_THROWS_AS(A.mult(B), std::invalid_argument);
}

TEST_CASE("Matrix multiplication: 1x3 * 3x1 (dot product as 1x1 matrix)",
          "[array][mult]") {
  Array A(1, 3);
  Array B(3, 1);

  // A = [[1, 2, 3]]
  A.set_vals({1, 2, 3});
  // B = [[4], [5], [6]]
  B.set_vals({4, 5, 6});

  // Expected: [[1*4 + 2*5 + 3*6]] = [[32]]
  Array C = A.mult(B);
  std::vector<int> expected = {32};
  REQUIRE(C.get_vals() == expected);
}

TEST_CASE("Matrix multiplication: identity matrix", "[array][mult]") {
  Array A(2, 2);
  Array I(2, 2);

  // A = [[9, 8], [7, 6]]
  A.set_vals({9, 8, 7, 6});
  // I = [[1, 0], [0, 1]]
  I.set_vals({1, 0, 0, 1});

  Array C = A.mult(I);
  REQUIRE(C.get_vals() == A.get_vals());
}

TEST_CASE("Array::eye() sets ones along the main diagonal for square matrix",
          "[array][eye]") {
  Array arr(3, 3);
  arr.eye();

  std::vector<int> expected = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  REQUIRE(arr.get_vals() == expected);
}

TEST_CASE("Array::eye() sets diagonal for rectangular tall matrix",
          "[array][eye]") {
  Array arr(4, 3);
  arr.eye();

  std::vector<int> expected = {1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0};
  REQUIRE(arr.get_vals() == expected);
}

TEST_CASE("Array::eye() sets diagonal for rectangular wide matrix",
          "[array][eye]") {
  Array arr(2, 4);
  arr.eye();

  std::vector<int> expected = {1, 0, 0, 0, 0, 1, 0, 0};
  REQUIRE(arr.get_vals() == expected);
}

TEST_CASE("Array::eye() works for 1x1 matrix", "[array][eye]") {
  Array arr(1, 1);
  arr.eye();

  std::vector<int> expected = {1};
  REQUIRE(arr.get_vals() == expected);
}

TEST_CASE("Array::eye() works for 1xN matrix", "[array][eye]") {
  Array arr(1, 5);
  arr.eye();

  std::vector<int> expected = {1, 0, 0, 0, 0};
  REQUIRE(arr.get_vals() == expected);
}

TEST_CASE("Array::eye() works for Nx1 matrix", "[array][eye]") {
  Array arr(4, 1);
  arr.eye();

  std::vector<int> expected = {1, 0, 0, 0};
  REQUIRE(arr.get_vals() == expected);
}
