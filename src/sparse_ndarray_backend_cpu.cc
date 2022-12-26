#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace needle {
namespace sparse_cpu {

#define ALIGNMENT 256
#define TILE 8
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);

/**
 * This is a utility structure for maintaining an array aligned to ALIGNMENT boundaries in
 * memory.  This alignment should be at least TILE * ELEM_SIZE, though we make it even larger
 * here by default.
 */
struct AlignedArray {
  AlignedArray(const size_t nnz) {
    int ret1 = posix_memalign((void**)&ptr_value, ALIGNMENT, nnz * ELEM_SIZE);
    if (ret1 != 0) throw std::bad_alloc();
    int ret2 = posix_memalign((void**)&ptr_loc, ALIGNMENT, nnz * ELEM_SIZE);
    if (ret2 != 0) throw std::bad_alloc();
    this->nnz = nnz;
  }
  ~AlignedArray() {
      free(ptr_value);
      free(ptr_loc);
  }
  size_t ptr_value_as_int() {return (size_t)ptr_value; }
  size_t ptr_loc_as_int() {return (size_t)ptr_loc; }
  scalar_t* ptr_value;
  scalar_t* ptr_loc;
  size_t nnz;
};


void EwiseAdd(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  /**
   * Set entries in out to be the sum of correspondings entires in a and b.
   */
}

void ScalarAdd(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  /**
   * Set entries in out to be the sum of corresponding entry in a plus the scalar val.
   */
}

/**
 * In the code the follows, use the above template to create analogous element-wise
 * and and scalar operators for the following functions.  See the numpy backend for
 * examples of how they should work.
 *   - EwiseMul, ScalarMul
 *   - EwiseDiv, ScalarDiv
 *   - ScalarPower
 *   - EwiseMaximum, ScalarMaximum
 *   - EwiseEq, ScalarEq
 *   - EwiseGe, ScalarGe
 *   - EwiseLog
 *   - EwiseExp
 *   - EwiseTanh
 *
 * If you implement all these naively, there will be a lot of repeated code, so
 * you are welcome (but not required), to use macros or templates to define these
 * functions (however you want to do so, as long as the functions match the proper)
 * signatures above.
 */

/// BEGIN YOUR SOLUTION

/// END YOUR SOLUTION

void Matmul(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, uint32_t m, uint32_t n,
            uint32_t p) {
  /**
   * Multiply two matrices into an output matrix.  
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: compact 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   m: rows of a / out
   *   n: columns of a / rows of b
   *   p: columns of b / out
   */
  }

void ReduceMax(const AlignedArray& a, AlignedArray* out) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.
   */

  /// BEGIN YOUR SOLUTION

  /// END YOUR SOLUTION
}

void ReduceSum(const AlignedArray& a, AlignedArray* out) {
  /**
   * Reduce by taking sum over `reduce_size` contiguous blocks.
   */

  /// BEGIN YOUR SOLUTION

  /// END YOUR SOLUTION
}

}  // namespace sparse_cpu
} // namespace needle


PYBIND11_MODULE(sparse_ndarray_backend_cpu, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace sparse_cpu;

  m.attr("__device_name__") = "cpu";
  m.attr("__tile_size__") = TILE;

  py::class_<AlignedArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def("ptr_value", &AlignedArray::ptr_value_as_int)
      .def("ptr_loc", &AlignedArray::ptr_loc_as_int)
      .def_readonly("nnz", &AlignedArray::size);

  // return numpy array (with copying for simplicity, otherwise garbage
  // collection is a pain)
  m.def("to_numpy_value", [](const AlignedArray& a) {
    // need implementation
  });

  m.def("to_numpy_loc", [](const AlignedArray& a, std::vector<size_t> shape) {
    // need implementation
  });

  // convert from numpy (with copying)
  m.def("from_numpy", [](py::array_t<scalar_t> a_value, py::array_t<scalar_t> a_loc, AlignedArray* out) {
    // need implementation
  });

  m.def("ewise_add", EwiseAdd);
  m.def("scalar_add", ScalarAdd);

  m.def("ewise_mul", EwiseMul);
  m.def("scalar_mul", ScalarMul);
  m.def("ewise_div", EwiseDiv);
  m.def("scalar_div", ScalarDiv);
  m.def("scalar_power", ScalarPower);

  m.def("ewise_maximum", EwiseMaximum);
  m.def("scalar_maximum", ScalarMaximum);
  m.def("ewise_eq", EwiseEq);
  m.def("scalar_eq", ScalarEq);
  m.def("ewise_ge", EwiseGe);
  m.def("scalar_ge", ScalarGe);

  m.def("ewise_log", EwiseLog);
  m.def("ewise_exp", EwiseExp);
  m.def("ewise_tanh", EwiseTanh);

  m.def("matmul", Matmul);
  m.def("matmul_tiled", MatmulTiled);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);
}



