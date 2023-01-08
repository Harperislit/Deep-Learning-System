import operator
import math
from functools import reduce
import numpy as np
from . import NDArray
from . import sparse_ndarray_backend_cpu


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
        device = device if device is not None else default_device()
        array = self.make(shape, len(values), device=device)
        array.device.from_numpy(np.ascontiguousarray(values), np.ascontiguousarray(locations), array._handle)
        self._init(array)
    
    def _init(self, other):
        self._shape = other._shape
        self._nnz = other._nnz
        self._handle = other._handle
        self._device = other._device     

    @staticmethod
    def make(shape, nnz, handle = None, device = None):
      """Create a new SparseNDArray with the given properties.  This will allocation the
        memory if handle=None, otherwise it will use the handle of an existing
        array."""
      array = SparseNDArray.__new__(SparseNDArray)
      array._shape = tuple(shape)
      array._nnz = nnz
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
        return prod(self._shape)

    def __repr__(self):
        ### display in COO format
        ### use to_numpy_value and to_numpy_loc
        raise NotImplementedError()

    def __str__(self):
        raise NotImplementedError()

    ### Basic array manipulation
    def to_dense(self):
        out = NDArray.make(self.shape, device=self.device)
        ### Use some low-level functions to convert the sparse array to a dense array and assign it to "out"
        raise NotImplementedError()

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
        """
        out = SparseNDArray.make(self.shape, device=self.device)
        if isinstance(other, SparseNDArray):
            assert self.shape == other.shape, "operation needs two equal-sized arrays"
            ewise_func(self._handle, other._handle, out._handle)
        else:
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
        out = SparseNDArray.make(self.shape, device=self.device)
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
        out = SparseNDArray.make(self.shape, device=self.device)
        self.device.ewise_log(self._handle, out._handle)
        return out

    def exp(self):
        out = SparseNDArray.make(self.shape, device=self.device)
        self.device.ewise_exp(self._handle, out._handle)
        return out

    def tanh(self):
        out = SparseNDArray.make(self.shape, device=self.device)
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

