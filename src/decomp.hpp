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
  Array invert(Array &);

  // simple access metadata functions
  int get_nrows() const;
  int get_ncols() const;
  std::vector<double> get_vals();
};

/*
 * class Cholesky {
 * private:
 *   int n;
 *   Array M;
 *   Array fact;
 *
 * public:
 *   // Factorize M -> fact
 *   Cholesky(Array &M);
 *   void decompose();
 *   Array solve(Array &b);
 *   Array invert();
 * };
 */

#endif
