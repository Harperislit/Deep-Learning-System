"""The module.
"""
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
import math


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(self.in_features, self.out_features, device=device))
        if bias:
            self.bias = init.kaiming_uniform(self.out_features, 1, device=device)
            self.bias = Parameter(ops.reshape(self.bias, (1, self.out_features)))
        else:
            self.bias = None
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.bias is None:
            return ops.matmul(X, self.weight)
        else:
            return ops.matmul(X, self.weight) + ops.broadcast_to(self.bias, (X.shape[0], self.out_features))
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        prod = 1
        for i in X.shape[1:]:
            prod *= i
        new_X = ops.reshape(X, (X.shape[0], prod))
        return new_X
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.tanh(x)
        ### END YOUR SOLUTION


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        one = init.ones(*x.shape, device=x.device, dtype=x.dtype)
        return one/(one+ops.exp(-x))
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for module in self.modules:
            x = module(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        print("softmax: ", y.device)
        one_hot = init.one_hot(logits.shape[1], y, y.device)
        softmax_vector = ops.logsumexp(logits, axes=(1,)) - ops.summation(ops.multiply(logits, one_hot), axes=(1,))
        softmax = ops.summation(softmax_vector, axes=(0,))/logits.shape[0]
        return softmax
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(*(1, self.dim), device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(*(1, self.dim), device=device, dtype=dtype))
        self.running_mean = init.zeros(*(self.dim, ), device=device, dtype=dtype)
        self.running_var = init.ones(*(self.dim, ), device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            sum_x = ops.summation(x, axes=0)
            mean_x = sum_x/x.shape[0]
            self.running_mean = (1-self.momentum)*self.running_mean.detach() + self.momentum*mean_x.detach()
            mean_x = ops.reshape(mean_x, (1, x.shape[1]))
            mean_x = ops.broadcast_to(mean_x, x.shape)

            x_demean = x - mean_x
            var_x = ops.summation(ops.power_scalar(x_demean, 2), axes=0)/x.shape[0]
            self.running_var = (1-self.momentum)*self.running_var.detach() + self.momentum*var_x.detach()
            
        
            sqrt_x = ops.power_scalar(var_x+self.eps, 0.5)
            sqrt_x = ops.reshape(sqrt_x, (1, x.shape[1]))
            sqrt_x = ops.broadcast_to(sqrt_x, x.shape)
            norm_x = (x-mean_x)/sqrt_x
        else:
            running_mean = ops.broadcast_to(self.running_mean, (x.shape[0], self.dim))
            sqrt = ops.power_scalar(self.running_var+self.eps, 0.5)
            sqrt = ops.broadcast_to(sqrt, (x.shape[0], self.dim))
            norm_x = (x-running_mean)/sqrt
        
        w = ops.broadcast_to(self.weight, (x.shape[0], self.dim))
        b = ops.broadcast_to(self.bias, (x.shape[0], self.dim))

        return ops.multiply(w, norm_x)+b

        ### END YOUR SOLUTION


class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(Tensor.make_const(np.ones((1, self.dim))))
        self.bias = Parameter(Tensor.make_const(np.zeros((1, self.dim))))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        sum_x = ops.summation(x, axes=1)
        mean_x = sum_x/self.dim
        mean_x = ops.reshape(mean_x, (x.shape[0], 1))
        mean_x = ops.broadcast_to(mean_x, x.shape)

        x_demean = x - mean_x
        var_x = ops.summation(ops.power_scalar(x_demean, 2), axes=1)/self.dim
        sqrt_x = ops.power_scalar(var_x+self.eps, 0.5)
        sqrt_x = ops.reshape(sqrt_x, (x.shape[0], 1))
        sqrt_x = ops.broadcast_to(sqrt_x, x.shape)
        
        norm_x = (x-mean_x)/sqrt_x
        w = ops.broadcast_to(self.weight, (x.shape[0], self.dim))
        b = ops.broadcast_to(self.bias, (x.shape[0], self.dim))
        return ops.multiply(w, norm_x)+b
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            ratio = init.randb(*x.shape, p=1-self.p)
            ratio = Tensor(1/(1-self.p)*ratio)
            new_x = ops.multiply(ratio, x)
            return new_x
        else:
            return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x)+x
        ### END YOUR SOLUTION

class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        ### BEGIN YOUR SOLUTION
        weight = init.kaiming_uniform(self.kernel_size**2*self.in_channels, self.kernel_size**2*self.out_channels, (self.kernel_size, self.kernel_size, self.in_channels, self.out_channels), device=device, dtype=dtype)
        self.weight = Parameter(Tensor.make_const(weight))
        if bias:
            bound = 1/(self.in_channels*self.kernel_size**2)**0.5
            bias = init.rand(*(self.out_channels, ), low=-bound, high=bound, device=device, dtype=dtype)
            self.bias = Parameter(Tensor.make_const(bias))
        else:
            self.bias = None
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if len(x.shape)==3:
            new_shape = tuple([1]+list(x.shape))
            x = ops.reshape(x, new_shape)
        N, _, H, W = x.shape
        self.padding = ((H-1)+self.kernel_size-H)/2
        if self.padding==int(self.padding):
            self.padding=int(self.padding)
        else:
            self.padding=int(self.padding)+1
        x_trans = ops.transpose(x, (1, 2))
        x_trans = ops.transpose(x_trans, (2,3))
        result = ops.conv(x_trans, self.weight, self.stride, self.padding)
        _, H_new, W_new, _ = result.shape
        if self.bias is not None:
            bias = ops.reshape(self.bias, (1, self.out_channels))
            bias = ops.broadcast_to(bias, (N*H_new*W_new, self.out_channels))
            bias = ops.reshape(bias, (N, H_new, W_new, self.out_channels))
            result += bias
        result = ops.transpose(result, (2, 3))
        result = ops.transpose(result, (1, 2))
        return result
        ### END YOUR SOLUTION



class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        k = 1/hidden_size
        self.hidden_size = hidden_size
        self.device = device
        self.dtype = dtype
        self.bias = bias
        self.W_ih = Parameter(init.rand(*(input_size, hidden_size), low=-math.sqrt(k), high=math.sqrt(k), device=device, dtype=dtype))
        self.W_hh = Parameter(init.rand(*(hidden_size, hidden_size), low=-math.sqrt(k), high=math.sqrt(k), device=device, dtype=dtype))
        if self.bias == True:
            self.bias_ih = Parameter(init.rand(*(1, hidden_size), low=-math.sqrt(k), high=math.sqrt(k), device=device, dtype=dtype))
            self.bias_hh = Parameter(init.rand(*(1, hidden_size), low=-math.sqrt(k), high=math.sqrt(k), device=device, dtype=dtype))
        if nonlinearity == "tanh":
            self.nonlinearity = Tanh()
        elif nonlinearity == "relu":
            self.nonlinearity = ReLU()
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        if h is None:
            h = init.zeros(*(X.shape[0], self.hidden_size), device=self.device, dtype=self.dtype)
        X = X @ self.W_ih
        h = h @ self.W_hh
        if self.bias == True:
            if len(self.bias_ih.shape)==1:
                self.bias_ih = ops.reshape(self.bias_ih, (1, self.hidden_size))
            if len(self.bias_hh.shape)==1:
                self.bias_hh = ops.reshape(self.bias_hh, (1, self.hidden_size))
            if X.shape[0]>1:
                bias_ih = ops.broadcast_to(self.bias_ih, (X.shape[0], self.hidden_size))
                bias_hh = ops.broadcast_to(self.bias_hh, (X.shape[0], self.hidden_size))
            else:
                bias_ih = self.bias_ih
                bias_hh = self.bias_hh
            h = self.nonlinearity(X + bias_ih + h + bias_hh)
        else:
            h = self.nonlinearity(X + h)
        return h
        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = device
        self.dtype = dtype
        self.rnn_cells = []
        for layer in range(num_layers):
            if layer==0:
                self.rnn_cells.append(RNNCell(input_size, hidden_size, bias, nonlinearity=nonlinearity, device=device, dtype=dtype))
            else:
                self.rnn_cells.append(RNNCell(hidden_size, hidden_size, bias, nonlinearity=nonlinearity, device=device, dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        inputs = ops.split(X, 0)
        if h0 is None:
            h0 = init.zeros(*(self.num_layers, X.shape[1], self.hidden_size), device=self.device, dtype=self.dtype)
        hs = ops.split(h0, 0)  #initial hidden states at time 0
        hs = [i for i in hs]
        h_t = []
        for t in range(len(inputs)):
            x_prev = inputs[t]
            for layer in range(self.num_layers):
                h_prev = hs[layer]
                hs[layer] = self.rnn_cells[layer](x_prev, h_prev)
                x_prev = hs[layer]
            h_t.append(hs[self.num_layers-1])

        h_n = ops.stack(hs, 0)
        h_t = ops.stack(h_t, 0)
        return h_t, h_n
        ### END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        k = 1/hidden_size
        self.hidden_size = hidden_size
        self.device = device
        self.dtype = dtype
        self.bias = bias
        self.W_ih = Parameter(init.rand(*(input_size, hidden_size*4), low=-math.sqrt(k), high=math.sqrt(k), device=device, dtype=dtype))
        self.W_hh = Parameter(init.rand(*(hidden_size, hidden_size*4), low=-math.sqrt(k), high=math.sqrt(k), device=device, dtype=dtype))
        self.W_ih_shape = (input_size, hidden_size*4)
        self.W_hh_shape = (hidden_size, hidden_size*4)
        if self.bias == True:
            self.bias_ih = Parameter(init.rand(*(1, hidden_size*4), low=-math.sqrt(k), high=math.sqrt(k), device=device, dtype=dtype))
            self.bias_hh = Parameter(init.rand(*(1, hidden_size*4), low=-math.sqrt(k), high=math.sqrt(k), device=device, dtype=dtype))
            self.bias_shape = (1, hidden_size*4)
        self.sigmoid = Sigmoid()
        self.tanh = Tanh()
        ### END YOUR SOLUTION


    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        bs = X.shape[0]
        if h is None:
            h0 = init.zeros(*(bs, self.hidden_size), device=self.device, dtype=self.dtype)
            c0 = init.zeros(*(bs, self.hidden_size), device=self.device, dtype=self.dtype)
        else:
            h0, c0 = h

        if self.W_ih.shape!=self.W_ih_shape:
            self.W_ih = ops.reshape(self.W_ih, self.W_ih_shape)
        if self.W_hh.shape!=self.W_hh_shape:
            self.W_hh = ops.reshape(self.W_hh, self.W_hh_shape)

        if self.bias==True:
            if self.bias_ih.shape!=self.bias_shape:
                self.bias_ih = ops.reshape(self.bias_ih, self.bias_shape)
            if self.bias_hh.shape!=self.bias_shape:
                self.bias_hh = ops.reshape(self.bias_hh, self.bias_shape)
            if bs>1:
                bias_ih = ops.broadcast_to(self.bias_ih, (bs, self.hidden_size*4))
                bias_hh = ops.broadcast_to(self.bias_hh, (bs, self.hidden_size*4))
            else:
                bias_ih = self.bias_ih
                bias_hh = self.bias_hh
            
            linear_res = X @ self.W_ih + bias_ih + h0 @ self.W_hh + bias_hh

        else:
            linear_res = X @ self.W_ih + h0 @ self.W_hh

        
        linear_res_split = ops.split(linear_res, 1)
        linear_res_split = [i for i in linear_res_split]
        gates = [] #i, f, g, o
        for i in range(4):     
            gates.append(ops.stack(linear_res_split[i*self.hidden_size: (i+1)*self.hidden_size], 1))
        i, f, g, o = gates
        i, f, g, o = self.sigmoid(i), self.sigmoid(f), self.tanh(g), self.sigmoid(o)
        c_out = f*c0 + i*g
        h_out = o * self.tanh(c_out)

        return (h_out, c_out)
        ### END YOUR SOLUTION


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        ### BEGIN YOUR SOLUTION
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = device
        self.dtype = dtype
        self.lstm_cells = []
        for layer in range(num_layers):
            if layer==0:
                self.lstm_cells.append(LSTMCell(input_size, hidden_size, bias, device=device, dtype=dtype))
            else:
                self.lstm_cells.append(LSTMCell(hidden_size, hidden_size, bias, device=device, dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            c_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        inputs = ops.split(X, 0)
        bs = X.shape[1]
        if h is None:
            h0 = init.zeros(*(self.num_layers, bs, self.hidden_size), device=self.device, dtype=self.dtype)
            c0 = init.zeros(*(self.num_layers, bs, self.hidden_size), device=self.device, dtype=self.dtype)
        else:
            h0, c0 = h
        hs = ops.split(h0, 0)  #initial hidden states at time 0
        cs = ops.split(c0, 0)  #initial cell states at time 0
        hs = [i for i in hs]
        cs = [i for i in cs]
        h_t, c_t = [], []

        for t in range(len(inputs)):
            x_prev = inputs[t]
            for layer in range(self.num_layers):
                h_prev, c_prev = hs[layer], cs[layer]
                hs[layer], cs[layer] = self.lstm_cells[layer](x_prev, (h_prev, c_prev))
                x_prev = hs[layer]
            h_t.append(hs[self.num_layers-1])
        h_n = ops.stack(hs, 0)
        c_n = ops.stack(cs, 0)
        h_t = ops.stack(h_t, 0)
        return h_t, (h_n, c_n)
        ### END YOUR SOLUTION


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        self.device= device
        self.dtype = dtype
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = init.randn(*(num_embeddings, embedding_dim), device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs = x.shape
        output = init.zeros(*(seq_len, bs, self.embedding_dim))
        x_cols = ops.split(x, 1)
        x_col_embed = []
        for col in range(len(x_cols)):
            x_col = x_cols[col]
            xs = ops.split(x_col, 0)
            x_row_embed = []
            for row in range(len(xs)):
                x = xs[row]
                x_row_embed.append(init.one_hot(self.num_embeddings, x, device=self.device, dtype=self.dtype))
            x_col_embed.append(ops.stack(x_row_embed, 0) @ self.weight)
        output = ops.stack(x_col_embed, 1)

        return output
        ### END YOUR SOLUTION
