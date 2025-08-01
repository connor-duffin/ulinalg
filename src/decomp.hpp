#ifndef DECOMP_HPP
#define DECOMP_HPP

#include "array.hpp"
#include <vector>

class LUDecomp {
private:
  int n;
  Array M;
  std::vector<int> p;

public:
  // All operations are in-place
  LUDecomp(Array &, int);
  void decompose();
  Array solve(Array &);

  // Simple access metadata functions
  int get_nrows() const;
  int get_ncols() const;
  std::vector<double> get_vals();
};

// class Cholesky {
// private:
//   int n;
//   Array M;
//
// public:
//   // All operations in-place
//   Cholesky(Array &M);
//   void decompose();
//   Array solve(Array &b);
//
//  // Simple access metadata functions
//  int get_nrows() const;
//  int get_ncols() const;
//  std::vector<double> get_vals();
// };

#endif
