"""Operatpr table."""
# Global operator table.
from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
from . import init
import numpy

from .backend_selection import array_api, NDArray


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple([out_grad[i] for i in range(len(out_grad))])


def make_tuple(*args):
    return MakeTensorTuple()(*args)


class TupleGetItem(TensorOp):
    def __init__(self, index):
        self.index = index

    def __call__(self, a: TensorTuple, fold_const=True) -> Value:
        assert isinstance(a, TensorTuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTensorTuple):
            return a.inputs[self.index]
        return Tensor.make_from_op(self, [a])

    def compute(self, a):
        return a[self.index]

    def gradient(self, out_grad, node):
        index = self.index
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(init.zeros_like(value))
            else:
                in_grad.append(out_grad)
        return MakeTensorTuple()(*in_grad)


def tuple_get_item(value, index):
    return TupleGetItem(index)(value)


class FusedAddScalars(TensorTupleOp):
    def __init__(self, c0: float, c1: float):
        self.c0 = c0
        self.c1 = c1

    def compute(self, a):
        return a + self.c0, a + self.c1

    def gradient(self, out_grad, node):
        return out_grad[0] + out_grad[1]


def fused_add_scalars(x, c0, c1):
    return FusedAddScalars(c0, c1)(x)


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + numpy.float32(self.scalar)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * numpy.float32(self.scalar)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a**self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad*self.scalar*(node.inputs[0]**(self.scalar-1))
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a/b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        return out_grad/rhs, out_grad*(-lhs*rhs**(-2))
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a/self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad/self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.axes is not None:
          axes = list(range(a.ndim))
          axes[self.axes[0]] = self.axes[1]
          axes[self.axes[1]] = self.axes[0]
          return array_api.permute(a, axes=tuple(axes))
        else:
          return array_api.permute(a, axes=tuple(range(a.ndim)[:-2])+tuple(range(a.ndim)[-1:-3:-1]))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return transpose(out_grad, self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        self.origin_shape = a.shape
        return array_api.reshape(a.compact(), self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return reshape(out_grad, self.origin_shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape).compact()

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        if node.inputs[0].shape==out_grad.shape:
            return out_grad
        axes = []
        input_ptr = len(node.inputs[0].shape)-1
        for i in range(len(out_grad.shape), 0, -1):
            ax = i-1   #ax of self.shape
            if (input_ptr<0):
                axes = [ax]+axes
            elif node.inputs[0].shape[input_ptr]!=out_grad.shape[ax]:
                axes = [ax]+axes
            input_ptr-=1
        
        result = summation(out_grad,tuple(axes))
        if result.shape != node.inputs[0].shape:
            result = reshape(result, node.inputs[0].shape)
        return result
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        if (isinstance(axes, int)):
            axes = tuple([axes])
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        self.origin_shape = a.shape
        result = array_api.summation(a, axis=self.axes)
        return result
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        
        #construct a dim (eg. if out_grad is (3, 2), origin_shape is (3, 4, 2), we construct a (3, 1, 2) )
        #if out_grad is (3, ), origin_shape is (4, 3), we construct (1, 3)
        #if out_grad is (1, ), origin_shape is (1, 5, 5), we construct (1, 1, 1)

        #first, we make sure all axes are positive
        if self.axes is None:
            result = reshape(out_grad, tuple([1]*len(node.inputs[0].shape)))
        else:
            lst_axes = []
            for ax in self.axes:
                if ax >=0:
                    lst_axes.append(ax)
                else:
                    lst_axes.append(len(node.inputs[0].shape)+ax)
            lst_axes.sort()
            tmp_dim = list(out_grad.shape)
            for ax in lst_axes:
                tmp_dim.insert(ax, 1)
            result = reshape(out_grad, tmp_dim)
        #broadcast to original shape
        result = broadcast_to(result, node.inputs[0].shape)
        return result
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a@b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        ndiml, ndimr = len(lhs.shape), len(rhs.shape)
        if ndiml==ndimr:
            return matmul(out_grad, transpose(rhs)), matmul(transpose(lhs), out_grad)
        elif ndiml<ndimr:
            dimdiff = ndimr - ndiml
            return summation(matmul(out_grad, transpose(rhs)), tuple(range(dimdiff))), matmul(transpose(lhs), out_grad)
        else:
            dimdiff = ndiml -ndimr
            return matmul(out_grad, transpose(rhs)), summation(matmul(transpose(lhs), out_grad), tuple(range(dimdiff)))
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        self.origin_shape = a.shape
        return -1*a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        data = -1*array_api.full(self.origin_shape, fill_value=1, device=out_grad.device)
        return multiply(out_grad, Tensor.make_const(data))
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION 
        numerator = Tensor.make_const(array_api.ones((node.inputs[0].shape)))
        return (multiply(out_grad, divide(numerator, node.inputs[0])), )
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return multiply(out_grad, node)
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a*(a>0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        gradient = Tensor.make_const((node.inputs[0].realize_cached_data()>0))
        return multiply(out_grad, gradient)
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        if isinstance(axes, int):
            axes = tuple([axes])
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        max_Z = array_api.max_uni(Z, axis=self.axes)
        max_shape = list(Z.shape)
        if self.axes is None:
            self.axes = tuple(range(len(Z.shape)))
        for ax in self.axes:
            max_shape[ax] = 1
        self.max_shape = max_shape
        max_Z_broadcast = array_api.broadcast_to(array_api.reshape(max_Z, max_shape), Z.shape)
        tmp_exp = array_api.exp(Z-max_Z_broadcast)

        prod = 1
        for i in tmp_exp.shape:
            prod *= i
        if (prod==1) & (len(self.axes)>1):
            tmp_sum = array_api.flat(tmp_exp)
        else:
            tmp_sum = array_api.summation(tmp_exp, self.axes)

        return array_api.log(tmp_sum) + max_Z
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        max_Z = Tensor(array_api.max_uni(node.inputs[0].cached_data, axis=self.axes), device=node.device)
        test =node - max_Z
        sum_exp = exp(test)
        max_Z = broadcast_to(reshape(max_Z, self.max_shape), node.inputs[0].shape)
        sum_exp = broadcast_to(reshape(sum_exp, self.max_shape), node.inputs[0].shape)
        exp_z = exp(node.inputs[0]-max_Z)
        grad = divide(exp_z, sum_exp)
        new_out_grad = broadcast_to(reshape(out_grad, self.max_shape), node.inputs[0].shape)
        return multiply(new_out_grad, grad)
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)


class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.tanh(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return multiply(out_grad, 1-node**2)
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: TensorTuple) -> Tensor:
        ### BEGIN YOUR SOLUTION
        shape = list(args[0].shape)
        shape.insert(self.axis, len(args))
        shape = tuple(shape)
        result = array_api.empty(shape, device=args[0].device)
        slicing = [slice(None, None, None) for i in range(len(shape))]
        start = 0
        for i in range(len(args)):
            slicing[self.axis] = slice(start, start+1, None)
            start = start+1
            result[tuple(slicing)] = args[i]
        return result
        ### END YOUR SOLUTION


    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return split(out_grad, self.axis)
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        result = []
        slicing = [slice(None, None, None) for i in range(len(A.shape))]
        shape = list(A.shape)
        shape.pop(self.axis)
        shape = tuple(shape)
        for i in range(A.shape[self.axis]):
            slicing[self.axis] = slice(i, i+1, None)
            subarray = array_api.empty(shape, device=A.device)
            subslicing = [slice(None, None, None) for i in range(len(shape))]
            subarray[tuple(subslicing)] = A[tuple(slicing)]
            result += [subarray]
        return result
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return stack(out_grad, self.axis)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.flip(a, self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return flip(out_grad, self.axes)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)



class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        result = a
        for axes in self.axes:
            if axes>=len(a.shape):
                continue
            shape = list(result.shape)
            shape[axes]  =shape[axes]*(self.dilation+1)
            shape = tuple(shape)
            new_result = array_api.full(shape=shape, fill_value=0, device=a.device)
            slicing = [slice(None, None, None) for i in range(len(new_result.shape))]
            new_slicing = [slice(None, None, None) for i in range(len(new_result.shape))]
            for col in range(result.shape[axes]):
                new_slicing[axes] = col*(self.dilation+1)
                slicing[axes] = col
                new_result[tuple(new_slicing)] = result[tuple(slicing)]
            result = new_result
        return result
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return undilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)

class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        result = a
        for axes in self.axes:
            if axes>= len(result.shape):
                continue
            shape = list(result.shape)
            shape[axes]  = int(shape[axes]/(self.dilation+1))
            shape = tuple(shape)
            new_result = array_api.full(shape=shape, fill_value=0, device=a.device)
            slicing = [slice(None, None, None) for i in range(len(new_result.shape))]
            new_slicing = [slice(None, None, None) for i in range(len(new_result.shape))]
            for col in range(new_result.shape[axes]):
                slicing[axes] = col*(self.dilation+1)
                new_slicing[axes] = col
                new_result[tuple(new_slicing)] = result[tuple(slicing)]
            result = new_result
        return result
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return dilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        if self.padding>0:
            A = array_api.pad(A, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0,0)))
        N,H,W,C_in = A.shape
        K,_,_,C_out = B.shape
        out = array_api.full(shape=(N,int((H-K)/self.stride+1), int((W-K)/self.stride+1), C_out), fill_value=0, device=A.device)
        for i in range(K):
            for j in range(K):
                A_reshape = A[:, i:i+H-K+1:self.stride, j:j+W-K+1:self.stride, :].compact().reshape((N*((H-K)//self.stride+1)*((W-K)//self.stride+1), C_in))
                B_reshape = B[i, j, :, :].compact().reshape((C_in, C_out))
                tmp_out = A_reshape @ B_reshape
                
                out += tmp_out.compact().reshape((N, (H-K)//self.stride+1, (W-K)//self.stride+1, C_out))
        return out

        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        A, B = node.inputs
        K,_,_,_ = B.shape
        B_gradA = transpose(B, (3,2))
        if self.stride>1:
            out_gradA = dilate(out_grad, (1,2), self.stride-1)
        else:
            out_gradA = out_grad
        A_grad = conv(out_gradA, flip(B_gradA, (0, 1)), stride=1, padding=K-1-self.padding)
        A_gradB = transpose(A, (3,0))
        out_gradB = transpose(out_grad, (0,1))
        out_gradB = transpose(out_gradB, (1,2)) #(1,2,0,3)
        if self.stride>1:
            out_gradB = dilate(out_gradB, (0,1), self.stride-1)

        B_grad = conv(A_gradB, out_gradB, stride=1, padding=self.padding)
        B_grad = transpose(B_grad, (0,1))
        B_grad = transpose(B_grad, (1,2)) #(1,2,0,3)
        return A_grad, B_grad
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)



