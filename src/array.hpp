#ifndef ARRAY_HPP
#define ARRAY_HPP

#include <vector>

// Generic row-major 2D arrays (vectors/matrices)
class Array {
private:
  int nrow, ncol;
  std::vector<int> vals;

public:
  Array(int, int);
  int get_nrow();
  int get_ncol();
  std::vector<int> get_vals();
  void set_zeros();
  void set_ones();
  void set_vals(std::vector<int>&);
  Array bcast(Array&);
  Array operator+(Array const&);
  void pprint();
};

#endif
