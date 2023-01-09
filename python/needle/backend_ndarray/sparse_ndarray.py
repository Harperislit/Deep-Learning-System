import operator
import math
from functools import reduce
import numpy as np
from . import NDArray
from . import sparse_ndarray_backend_cpu
from . import ndarray


# math.prod not in Python 3.7
def prod(x):
    return reduce(operator.mul, x, 1)


class BackendDevice:
    """A backend device, wrapps the implementation module.
       Currenly only CPU and float32 is supported."""

    def __init__(self, name, mod):
        self.name = name
        self.mod = mod

    def __eq__(self, other):
        return self.name == other.name

    def __repr__(self):
        return self.name + "()"

    def __getattr__(self, name):
        return getattr(self.mod, name)

    def enabled(self):
        return self.mod is not None

    def one_hot(self, n, i, dtype="float32"):
        raise NotImplementedError()

    def empty(self, shape, dtype="float32"):
        dtype = "float32" if dtype is None else dtype
        assert dtype == "float32"
        raise NotImplementedError()


def cpu():
    """Return cpu device"""
    return BackendDevice("cpu", sparse_ndarray_backend_cpu)


def default_device():
    return cpu()


def all_devices():
    """return a list of all available devices"""
    return [cpu()]


class SparseNDArray:
    def __init__(self, locations: np.ndarray, values: np.ndarray, shape, device = None):
        
        self._ndim = len(shape)

        device = device if device is not None else default_device()
        array = self.make(shape, len(values), device=device)
        array.device.from_numpy(np.ascontiguousarray(values), 
                                np.ascontiguousarray(SparseNDArray.convert_loc_to_1d(shape, locations)),
                                array._handle)
        self._init(array)
        
    
    def _init(self, other):
        self._shape = other._shape
        self._nnz = other._nnz
        self._handle = other._handle
        self._device = other._device     
    
    @staticmethod
    def compact_strides(shape):
        """ Utility function to compute compact strides """
        stride = 1
        res = []
        for i in range(1, len(shape) + 1):
            res.append(stride)
            stride *= shape[-i]
        return tuple(res[::-1])

    @staticmethod        
    def convert_loc_to_1d(shape, locations):
        """ Utility function to convert locations from (ndim, nnz) to 1d array
            recording the location of non-zero values in a flattened array.
        """
        strides = SparseNDArray.compact_strides(shape)
        return strides@locations

    @staticmethod        
    def convert_1d_to_loc(shape, loc_1d):
        """ Inverse operation of the above
        """
        strides = SparseNDArray.compact_strides(shape)
        locations = np.zeros((len(shape), len(loc_1d)))
        for icol, loc in enumerate(loc_1d):
            locs = []
            for s in strides:
                locs.append(loc // s)
                loc = loc % s
            locations[:, icol] = locs
        return locations


    @staticmethod
    def make(shape, nnz, handle = None, device = None):
      """Create a new SparseNDArray with the given properties.  This will allocation the
        memory if handle=None, otherwise it will use the handle of an existing
        array."""
      array = SparseNDArray.__new__(SparseNDArray)
      array._shape = tuple(shape)
      array._nnz = nnz
      array._device = device if device is not None else default_device()
      if handle is None:
          array._handle = array.device.Array(nnz)
      else:
          array._handle = handle
      array._device = device if device is not None else default_device()
      return array

    ### Properies and string representations
    @property
    def shape(self):
        return self._shape
  
    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        # only support float32 for now
        return "float32"

    @property
    def ndim(self):
        """ Return number of dimensions. """
        return len(self._shape)

    @property
    def size(self):
        return prod(self._shape)

    @property
    def nnz(self):
        return self._nnz

    def __repr__(self):
        ### display in COO format
        ### use to_numpy_value and to_numpy_loc
        return "NDArray(locations=" + self.numpy_location().__str__() + "\n" + \
                "values=" + self.numpy_value().__str__() + \
                f", device={self.device})"

    def __str__(self):
        raise NotImplementedError()

    def numpy_value(self):
        return self.device.to_numpy_value(self._handle)

    def numpy_1d_location(self):
        return self.device.to_numpy_loc(self._handle)

    def numpy_location(self):
        return SparseNDArray.convert_1d_to_loc(self._shape, self.numpy_1d_location())

    ### Basic array manipulation
    def to_dense(self):
        ### Use some low-level functions to convert the sparse array to a dense array and assign it to "out"
        dense_array = np.zeros(self.shape)

        nz_values = self.numpy_value()
        nz_locations = self.numpy_location()
        
        for i in range(len(nz_values)):
            idx = tuple(nz_locations[:,i].astype(int))            
            dense_array[idx] = nz_values[i]
        print(type(dense_array), dense_array)

        if self.device.name == 'cpu':
            device = ndarray.cpu()
        else:
            raise NotImplementedError()
        return NDArray(dense_array, device=device)

    def reshape(self, new_shape):
        raise NotImplementedError()

    def permute(self, new_axes):
        raise NotImplementedError()
    
    def broadcast_to(self, new_shape):
        raise NotImplementedError()

    @property
    def flat(self):
        return self.reshape((self.size,))

    def __getitem__(self, idxs):
        ### may be not used
        raise NotImplementedError()
    
    def __setitem__(self, idxs, other):
        ### may be not used
        raise NotImplementedError() 

    ### Collection of elementwise and scalar function: add, multiply, boolean, etc
    def ewise_or_scalar(self, other, ewise_func, scalar_func):
        """Run either an elementwise or scalar version of a function,
        depending on whether "other" is a SparseNDArray or scalar
        
        Note: we assume these operations will NOT create new non-zero entries other 
        than those already in either self or other.
        """
        if isinstance(other, SparseNDArray):
            this_1d_locations = self.numpy_1d_location()
            other_1d_locations = other.numpy_1d_location()
            if ewise_func == self.device.ewise_mul:
                all_1d_locations = list(set(this_1d_locations).intersection(set(other_1d_locations)))
                print(all_1d_locations)
            else:
                all_1d_locations = list(set(this_1d_locations).union(set(other_1d_locations)))
            ewise_nnz = len(all_1d_locations) 
            out = SparseNDArray.make(self.shape, ewise_nnz, device=self.device)
            assert self.shape == other.shape, "operation needs two equal-sized arrays"
            ewise_func(self._handle, other._handle, out._handle)
        else:
            out = SparseNDArray.make(self.shape, self.nnz, device=self.device)
            scalar_func(self._handle, other, out._handle)
        return out

    def __add__(self, other):
        return self.ewise_or_scalar(
            other, self.device.ewise_add, self.device.scalar_add
        )

    __radd__ = __add__

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __mul__(self, other):
        return self.ewise_or_scalar(
            other, self.device.ewise_mul, self.device.scalar_mul
        )

    __rmul__ = __mul__

    def __truediv__(self, other):
        return self.ewise_or_scalar(
            other, self.device.ewise_div, self.device.scalar_div
        )

    def __neg__(self):
        return self * (-1)

    def __pow__(self, other):
        out = SparseNDArray.make(self.shape, self.nnz, device=self.device)
        self.device.scalar_power(self._handle, other, out._handle)
        return out

    def maximum(self, other):
        return self.ewise_or_scalar(
            other, self.device.ewise_maximum, self.device.scalar_maximum
        )

    ### Binary operators all return (0.0, 1.0) floating point values, could of course be optimized
    def __eq__(self, other):
        return self.ewise_or_scalar(other, self.device.ewise_eq, self.device.scalar_eq)

    def __ge__(self, other):
        return self.ewise_or_scalar(other, self.device.ewise_ge, self.device.scalar_ge)

    def __ne__(self, other):
        return 1 - (self == other)

    def __gt__(self, other):
        return (self >= other) * (self != other)

    def __lt__(self, other):
        return 1 - (self >= other)

    def __le__(self, other):
        return 1 - (self > other)

    ### Elementwise functions
    def log(self):
        out = SparseNDArray.make(self.shape, self.nnz, device=self.device)
        self.device.ewise_log(self._handle, out._handle)
        return out

    def exp(self):
        out = SparseNDArray.make(self.shape, self.nnz, device=self.device)
        self.device.ewise_exp(self._handle, out._handle)
        return out

    def tanh(self):
        out = SparseNDArray.make(self.shape, self.nnz, device=self.device)
        self.device.ewise_tanh(self._handle, out._handle)
        return out

    ### Matrix multiplication
    def __matmul__(self, other):
        """Matrix multplication of two arrays.  This requires that both arrays
        be 2D (i.e., we don't handle batch matrix multiplication), and that the
        sizes match up properly for matrix multiplication.
        """

        assert self.ndim == 2 and other.ndim == 2
        assert self.shape[1] == other.shape[0]
        raise NotImplementedError() 

    def sum(self, axis=None, keepdims=False):
        raise NotImplementedError() 

    def max(self, axis=None, keepdims=False):
        raise NotImplementedError() 

