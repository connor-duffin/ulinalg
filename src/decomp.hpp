#ifndef DECOMP_HPP
#define DECOMP_HPP

#include "array.hpp"
#include <vector>

class Decomp {
protected:
  int n;
  Array M;

public:
  Decomp(Array &, int);

  int get_nrows() const;
  int get_ncols() const;
  std::vector<double> get_vals();
};

class LUDecomp : public Decomp {
private:
  std::vector<int> p;

public:
  // All operations are in-place
  LUDecomp(Array &, int);
  void decompose();
  Array solve(Array &);
};

class Cholesky : public Decomp {
public:
  Cholesky(Array &, int);
  void decompose();
  Array solve(Array &);
};

#endif
