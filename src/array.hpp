#ifndef ARRAY_HPP
#define ARRAY_HPP

#include <vector>

// Generic row-major 2D arrays (vectors/matrices)
class Array {
private:
  int nrow, ncol;
  std::vector<double> vals;

public:
  Array(int, int);
  Array(const std::vector<double> &, int, int);

  // Basic array attributes
  int get_nrow() const;
  int get_ncol() const;
  std::vector<double> get_vals() const;

  // Standard setters/initializations
  void set_zeros();
  void set_ones();
  void eye();
  void copy(Array &);
  void set_vals(std::vector<double> &);

  // Matrix multiplication
  Array mult(Array &);

  // Pretty print the array
  void pprint();

  // Binary operations via friend functions
  double *operator[](int r);
  friend Array operator+(const Array &, const Array&);
  // Array operator-(Array);
  // Array operator*(Array);
  // Array operator/(Array);

};

// Internally used broadcasting rules: not put inside class defn to avoid
// namespace pollution
namespace array_detail {
Array bcast(Array &input, int nrow, int ncol);
std::vector<int> get_bcast_idx(const Array &, int nrow_out, int ncol_out);
} // namespace array_detail

#endif
