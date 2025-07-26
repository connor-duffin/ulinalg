#include "src/array.hpp"

#include <vector>
#include <iostream>

int main() {
  std::cout << "Ones" << std::endl;
  Array v(5, 1);
  v.set_ones();
  v.pprint();

  std::cout << "Scalar ones" << std::endl;
  Array x(1, 1);
  x.set_ones();
  x.pprint();

  std::cout << "Initialize 1:8 matrix" << std::endl;
  std::vector<int> vals = {1, 2, 3, 4, 5, 6, 7, 8};
  Array u(2, 4);
  u.set_vals(vals);
  u.pprint();

  std::cout << "2x1 to be broadcast" << std::endl;
  Array y(2, 1);
  vals = {3, 4};
  y.set_vals(vals);
  y.pprint();

  std::cout << "1x4 to be broadcast" << std::endl;
  Array z(1, 4);
  vals = {3, 4, 5, 6};
  z.set_vals(vals);
  z.pprint();

  std::cout << "Sum of the above results" << std::endl;
  Array w = u + v;
  w.pprint();
  return 0;
}
