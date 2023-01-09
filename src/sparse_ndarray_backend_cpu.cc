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
const size_t LOC_SIZE = sizeof(scalar_t);

/**
 * This is a utility structure for maintaining an array aligned to ALIGNMENT boundaries in
 * memory.  This alignment should be at least TILE * ELEM_SIZE, though we make it even larger
 * here by default.
 */
struct AlignedArray {
  AlignedArray(const size_t nnz) {
    int ret1 = posix_memalign((void**)&ptr_value, ALIGNMENT, nnz * ELEM_SIZE);
    if (ret1 != 0) throw std::bad_alloc();
    int ret2 = posix_memalign((void**)&ptr_loc, ALIGNMENT, nnz * LOC_SIZE);
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
  size_t index_a = 0, index_b = 0, out_size = 0;
  while(index_a < a.nnz || index_b < b.nnz){
    if (index_b == b.nnz || a.ptr_loc[index_a] < b.ptr_loc[index_b]){
      out->ptr_loc[out_size] = a.ptr_loc[index_a];
      out->ptr_value[out_size] = a.ptr_value[index_a];
      index_a++;
    } else if (index_a == a.nnz || a.ptr_loc[index_a] > b.ptr_loc[index_b]){
      out->ptr_loc[out_size] = b.ptr_loc[index_b];
      out->ptr_value[out_size] = b.ptr_value[index_b];
      index_b++;
    } else {
      out->ptr_loc[out_size] = a.ptr_loc[index_a];
      out->ptr_value[out_size] = a.ptr_value[index_a] + b.ptr_value[index_b];
      index_a++;
      index_b++;
    }
    out_size++;
  }
}

void ScalarAdd(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  /**
   * Set entries in out to be the sum of corresponding entry in a plus the scalar val.
   */
  for(size_t i = 0; i < a.nnz; i++) {
    out->ptr_loc[i] = a.ptr_loc[i];
    out->ptr_value[i] = a.ptr_value[i] + val;
  }
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
void EwiseMul(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  size_t index_a = 0, index_b = 0, out_size = 0;
  while(index_a < a.nnz || index_b < b.nnz){
    if (index_b == b.nnz || a.ptr_loc[index_a] < b.ptr_loc[index_b]){
      index_a++;
    } else if (index_a == a.nnz || a.ptr_loc[index_a] > b.ptr_loc[index_b]){
      index_b++;
    } else {
      out->ptr_loc[out_size] = a.ptr_loc[index_a];
      out->ptr_value[out_size] = a.ptr_value[index_a] * b.ptr_value[index_b];
      index_a++;
      index_b++;
      out_size++;
    }
  }
}

void ScalarMul(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  for(size_t i = 0; i < a.nnz; i++) {
    out->ptr_loc[i] = a.ptr_loc[i];
    out->ptr_value[i] = a.ptr_value[i] * val;
  }
}

void ScalarDiv(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  for(size_t i = 0; i < a.nnz; i++) {
    out->ptr_loc[i] = a.ptr_loc[i];
    out->ptr_value[i] = a.ptr_value[i] / val;
  }
}

void ScalarPower(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  for(size_t i = 0; i < a.nnz; i++) {
    out->ptr_loc[i] = a.ptr_loc[i];
    out->ptr_value[i] = pow(a.ptr_value[i], val);
  }
}

void EwiseMaximum(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  size_t index_a = 0, index_b = 0, out_size = 0;
  while(index_a < a.nnz || index_b < b.nnz){
    if (index_b == b.nnz || a.ptr_loc[index_a] < b.ptr_loc[index_b]){
      out->ptr_loc[out_size] = a.ptr_loc[index_a];
      out->ptr_value[out_size] = (a.ptr_value[index_a] > 0) ? a.ptr_value[index_a] : 0;
      index_a++;
    } else if (index_a == a.nnz || a.ptr_loc[index_a] > b.ptr_loc[index_b]){
      out->ptr_loc[out_size] = b.ptr_loc[index_b];
      out->ptr_value[out_size] = (b.ptr_value[index_b] > 0) ? b.ptr_value[index_b] : 0;
      index_b++;
    } else {
      out->ptr_loc[out_size] = a.ptr_loc[index_a];
      out->ptr_value[out_size] = (a.ptr_value[index_a] > b.ptr_value[index_b]) ? a.ptr_value[index_a] : b.ptr_value[index_b];
      index_a++;
      index_b++;
    }
    out_size++;
  }
}

void EwiseTanh(const AlignedArray& a, AlignedArray* out) {
  for(size_t i = 0; i < a.nnz; i++) {
    out->ptr_loc[i] = a.ptr_loc[i];
    out->ptr_value[i] = tanh(a.ptr_value[i]);
  }
}

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

void ReduceMax(const AlignedArray& a, AlignedArray* out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.
   */

  /// BEGIN YOUR SOLUTION
  float maxValue = 0;
  size_t out_size  = 0, out_loc = 0;
  for(size_t i = 0; i < a.nnz; i++) {
    if (i == 0 || a.ptr_loc[i] / reduce_size != a.ptr_loc[i - 1] / reduce_size) {
      out_loc = a.ptr_loc[i] / reduce_size;
      maxValue = a.ptr_value[i];
    } else {
      if (maxValue < a.ptr_value[i]) {
        maxValue = a.ptr_value[i];
      }
    }
    if (i == a.nnz - 1 || a.ptr_loc[i] / reduce_size != a.ptr_loc[i + 1] / reduce_size) {
      out->ptr_loc[out_size] = out_loc;
      out->ptr_value[out_size] = maxValue;
      out_size++;
    }
  }
  /// END YOUR SOLUTION
}

void ReduceSum(const AlignedArray& a, AlignedArray* out, size_t reduce_size) {
  /**
   * Reduce by taking sum over `reduce_size` contiguous blocks.
   */

  /// BEGIN YOUR SOLUTION
  float sumValue = 0;
  size_t out_size  = 0, out_loc = 0;
  for(size_t i = 0; i < a.nnz; i++) {
    if (i == 0 || a.ptr_loc[i] / reduce_size != a.ptr_loc[i - 1] / reduce_size) {
      out_loc = a.ptr_loc[i] / reduce_size;
      sumValue = a.ptr_value[i];
    } else {
      sumValue += a.ptr_value[i];
    }
    if (i == a.nnz - 1 || a.ptr_loc[i] / reduce_size != a.ptr_loc[i + 1] / reduce_size) {
      out->ptr_loc[out_size] = out_loc;
      out->ptr_value[out_size] = sumValue;
      out_size++;
    }
  }
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
      .def_readonly("nnz", &AlignedArray::nnz);

  // return numpy array (with copying for simplicity, otherwise garbage
  // collection is a pain)
  m.def("to_numpy_value", [](const AlignedArray& a) {
    // need implementation
    std::vector<size_t> shape;
    shape.push_back(a.nnz);

    std::vector<size_t> strides;
    strides.push_back(ELEM_SIZE);
    
    // FOR DEBUGGING
    // for (size_t i=0; i<5; i++){
    //   printf("%f", a.ptr_value[i]);
    // }

    return py::array_t<scalar_t>(shape, strides, a.ptr_value);
  });

  m.def("to_numpy_loc", [](const AlignedArray& a) {
    
    std::vector<size_t> shape;
    shape.push_back(a.nnz);

    std::vector<size_t> strides;
    strides.push_back(ELEM_SIZE);
    
    // FOR DEBUG
    // for (size_t i=0; i<5; i++){
    //   printf("%f", a.ptr_loc[i]);
    // }
    
    return py::array_t<scalar_t>(shape, strides, a.ptr_loc);
  });

  // convert from numpy (with copying)
  m.def("from_numpy", [](py::array_t<scalar_t> a_value, py::array_t<scalar_t> a_loc, AlignedArray* out) {
    std::memcpy(out->ptr_value, a_value.request().ptr, out->nnz * ELEM_SIZE);
    std::memcpy(out->ptr_loc, a_loc.request().ptr, out->nnz * LOC_SIZE);
  });

  m.def("ewise_add", EwiseAdd);
  m.def("scalar_add", ScalarAdd);

  m.def("ewise_mul", EwiseMul);
  m.def("scalar_mul", ScalarMul);
  m.def("scalar_div", ScalarDiv);
  m.def("scalar_power", ScalarPower);

  m.def("ewise_maximum", EwiseMaximum);

  m.def("ewise_tanh", EwiseTanh);

  m.def("matmul", Matmul);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);
}



