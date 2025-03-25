#include "../src/array.hpp"

#include <cstddef>
#include <vector>
#include <catch2/catch_test_macros.hpp>

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

TEST_CASE("Array broadcasting works", "[array]") {
  // Scalar ones
  Array x(1, 1);
  x.set_ones();

  // Initialize matrix
  std::vector<int> vals = {1, 2, 3, 4, 5, 6, 7, 8};
  Array u(2, 4);
  u.set_vals(vals);

  // bcast scalar to matrix
  Array x_bcast = u.bcast(x);
  REQUIRE(x_bcast.get_nrow() == 2);
  REQUIRE(x_bcast.get_ncol() == 4);

  std::vector<int> x_vals = x_bcast.get_vals();
  for (auto it = x_vals.begin(); it != x_vals.end(); ++it) {
    REQUIRE(*it == 1);
  }

  // 2x1 to be broadcast across `u`
  Array y(2, 1);
  vals.resize(2);
  vals = {3, 4};
  y.set_vals(vals);

  Array y_bcast = u.bcast(y);
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
  vals.resize(4);
  vals = {3, 4, 5, 6};
  z.set_vals(vals);

  // Bcast row vector to matrix
  Array z_bcast = u.bcast(z);
  std::vector<int> z_vals = z_bcast.get_vals();
  for (size_t i = 0; i < z_vals.size(); ++i) {
    REQUIRE(z_vals[i] == vals[i % 4]);
  }
}

TEST_CASE("Addition with broadcasting works as expected", "[array]") {
  std::vector<int> vals = {1, 2, 3, 4, 5, 6, 7, 8};
  Array u(2, 4);
  u.set_vals(vals);

  Array z(1, 1);
  z.set_ones();

  Array upp = u + z;
  std::vector<int> valspp = upp.get_vals();
  for (size_t i = 0; i < valspp.size(); ++i) {
    REQUIRE(valspp[i] == (vals[i] + 1));
  }
}

TEST_CASE("Matrix multiplication works as expected", "[array]") {
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
