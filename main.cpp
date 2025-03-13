#include <iostream>
#include <vector>

using namespace std;

// Array abstraction (generic for row-major vectors/matrices)
class Array {
private:
  int nrow, ncol;
  vector<int> vals;

public:
  Array(int, int);
  void set_zeros();
  void set_ones();
  void pprint();
};

Array::Array(int nrows, int ncols) {
  nrow = nrows;
  ncol = ncols;
  vals = vector<int>(nrow * ncol);
}

void Array::set_zeros() {
  for (vector<int>::iterator it = vals.begin(); it != vals.end(); ++it) {
    *it = 0;
  }
}

void Array::set_ones() {
  for (vector<int>::iterator it = vals.begin(); it != vals.end(); ++it) {
    *it = 1;
  }
}

void Array::pprint() {
  for (size_t i = 0; i < vals.size(); ++i) {
    cout << vals[i];

    if ((i + 1) % ncol == 0) {
      cout << endl;
    } else {
      cout << ", ";
    }
  }
}

int main() {
  Array v(3, 4);
  v.set_ones();
  v.pprint();
  return 0;
}
