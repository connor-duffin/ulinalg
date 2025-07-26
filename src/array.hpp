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
  int get_nrow() const;
  int get_ncol() const;
  std::vector<int> get_vals();

  void set_zeros();
  void set_ones();
  void eye();
  void set_vals(std::vector<int> &);

  Array mult(Array &);

  Array operator+(Array);
  // Array operator-(Array);
  // Array operator*(Array);
  // Array operator/(Array);
  int *operator[](int r);

  void pprint();
};

namespace array_detail {
Array bcast(Array &input, int nrow, int ncol);
std::vector<int> get_bcast_idx(int nrow_in, int ncol_in, int nrow_out,
                               int ncol_out);
} // namespace array_detail

#endif
