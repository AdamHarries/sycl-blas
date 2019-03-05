#ifndef TEST_UNITTEST_UTIL_MATRIX_DEBUG_TOOLS_H
#define TEST_UNITTEST_UTIL_MATRIX_DEBUG_TOOLS_H

template <bool ColMajor, typename T>
struct MatrixGenerator {
  // SFINAE
};

template <typename T>
struct MatrixGenerator<true, T> {
  template <typename IxType>
  static std::vector<T> eval(IxType n, T init, T inc) {
    std::vector<T> ret;

    return ret;
  }
};

template <typename T>
struct MatrixGenerator<false, T> {
  template <typename IxType>
  static std::vector<T> eval(IxType n, T init, T inc) {
    std::vector<T> ret;
    T val = init;

    for (IxType i = 0; i < n; i++) {
      ret.push_back(val);
      val += inc;
    }
    return ret;
  }
};

// ---------------------------
// Utilities to print matrices
// ---------------------------
template <bool ColMajor>
struct MatrixPrinter {
  // SFINAE
};

template <>
struct MatrixPrinter<true> {
  template <typename IxType, typename VectorT>
  static inline void eval(IxType w, IxType h, VectorT v) {
#ifdef VERBOSE
    for (IxType i = 0; i < h; i++) {
      std::cerr << "[";
      for (IxType j = 0; j < w; j++) {
        if (j != 0) {
          std::cerr << ", ";
        }
        std::cerr << v[i + (j * h)];
      }
      std::cerr << "]\n";
    }
#endif
  }
};

template <>
struct MatrixPrinter<false> {
  template <typename IxType, typename VectorT>
  static inline void eval(IxType w, IxType h, VectorT v) {
#ifdef VERBOSE
    for (IxType i = 0; i < h; i++) {
      std::cerr << "[";
      for (IxType j = 0; j < w; j++) {
        if (j != 0) {
          std::cerr << ", ";
        }
        std::cerr << v[(i * w) + j];
      }
      std::cerr << "]\n";
    }
#endif
  }
};

#endif