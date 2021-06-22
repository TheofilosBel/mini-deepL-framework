# To allow Tensor functions return Tensor types
from __future__ import annotations

# Python imports
from torch import Tensor as TorchTensor # Only for type hints
from torch import empty as torchEmpty

from numbers import Number
from typing import List, Union

from scripts.autog.utils import is_column_vector, is_matrix, is_row_vector, is_scalar, is_vector

# Hack Import of functional
class Fn: pass
class autogard: pass

#
# No grad context wrapper
#
class TensorNoGrad:
    def __init__(self, *args):
        """
        Use with the keyword `with` and pass the tensors to switch off their gradient:

        eg.:
        loss.backward()

        # Weight update
        with TensorNoGrad(ww,bb):
            w += -gamma_a * w.grad
            b += -gamma_b * b.grad
        """
        self.tensors = [t for t in args if t.with_grad == True]

    def __enter__(self):
        for t in self.tensors:
            t.with_grad = False

    def __exit__(self, *args, **kwargs):
        for t in self.tensors:
            t.with_grad = True

#
# Tensor class
#
class Tensor:

    #
    # Static methods for initialization
    #
    @classmethod
    def empty(cls, shape, with_grad=False):
        """ Initialize an empty tensor """
        return cls(torchEmpty(shape), with_grad)

    @classmethod
    def normal(cls, shape, with_grad=False):
        """ Initialize a tensor with a normal distribution """
        return cls(torchEmpty(shape).normal_(), with_grad)

    @classmethod
    def ones(cls, shape, with_grad=False):
        """ Initialize a tensor of ones with the given shape """
        return cls(torchEmpty(shape).fill_(1), with_grad)

    @classmethod
    def zeros(cls, shape, with_grad=False):
        """ Initialize a tensor of zeros with the given shape """
        return cls(torchEmpty(shape).fill_(0), with_grad)

    #
    # Constructor
    #
    def __init__(self, tensor: TorchTensor, with_grad=False) -> None:
        """ Init with TorchTensor """
        self.with_grad = with_grad

        # Wrap a torch tensor
        self.data = tensor
        self.shape = tensor.shape

        # Keep info for auto grad
        self.grad = None
        self.grad_fn = None


        # Info about the node
        # We only keep the backward information
        self.backward_nodes: List[Union[Tensor, Number]] = []

    #
    # Accessors
    #
    def item(self, int: Number = 0):
        assert is_scalar(self.data)
        return self.data.item()

    #
    # Modificators
    #
    def switchGrad(self):
        self.with_grad = not self.with_grad

    def narrow(self, dim, start, length):
        return Tensor(self.data.narrow(dim, start, length), self.with_grad)

    #
    # Gradient functions
    #
    def backward(self, cur_partial_grad: TorchTensor = torchEmpty(1).fill_(1)):
        # Assert that its a scalar
        # print(cur_partial_grad.shape)
        # print(self.data.shape)
        # assert cur_partial_grad.shape == self.data.shape

        # Call backward on the node
        autograd.backward(self, cur_partial_grad)

    def add_grad_func(self, func_ptr):
        """ Adds a function for the backward pass """
        self.grad_fn = func_ptr

    def add_backward_node(self, x: Tensor):
        """ Adds a backward node """
        self.backward_nodes.append(x)

    #
    # Convenient functional calls
    #
    def sum(self):
        """ Sum of vector elements """
        return Fn.sum(self)

    def add(self, x: Tensor):
        """ Elementwise addition """
        if is_matrix(self.data) and is_matrix(x.data):
            return Fn.ewMatAdd(self,x)
        elif is_matrix(self.data) and is_vector(x.data):
            return Fn.ewMatBVecAdd(self, x)
        elif is_vector(self.data) and is_scalar(x.data):
            return Fn.ewVecBScalarAdd(self, x)
        elif (is_column_vector(self.data) and is_column_vector(x.data)) or\
            (is_row_vector(self.data) and is_row_vector(x.data)) or \
            (is_scalar(self.data) and is_scalar(x.data)):
            # Abuse name vec vec even for scalars
            return Fn.ewVecAdd(self, x)
        else:
            raise ValueError("Tensor Add cannot handle sizes:", self.shape, x.shape)

    def ewSub(self, x: Tensor):
        """ Elementwise subtraction """
        return Fn.ewSub(self, x)

    def scalarMul(self, y: Number):
        """Scalar multiplication"""
        return Fn.scalarMul(self, y)

    def ewVecMul(self, vector:Tensor):
        return Fn.ewVecMul(self, vector)

    def matmul(self, x: Tensor):
        """
        Look utils to see what we define as vectors and as arrays

        If self is matrix and x is scalar then we have the unique case of a matrix
        """
        if (is_column_vector(self.data) and is_row_vector(x.data)) or \
            (is_row_vector(self.data) and is_column_vector(x.data)):
            return Fn.matmatMul(self,x)
        elif is_vector(self.data) and is_vector(x.data):
            return Fn.vecVecInnerMul(self, x)
        elif is_matrix(self.data) and is_vector(x.data):
            return Fn.matvecMul(self, x)
        elif is_vector(self.data) and is_matrix(x.data):
            return Fn.vecmatMul(self, x)
        elif is_matrix(self.data) and is_matrix(x.data):
            return Fn.matmatMul(self, x)
        else:
            raise ValueError('MatMul error >> Cannot handle shapes passed:', self.shape, x.shape)

    def power(self, exp):
        """ Raise the vector elements to the given exponent """
        return Fn.power(self, exp)

    def relu(self):
        """ Rectified linear unit """
        return Fn.relu(self)

    def tanh(self):
        """ Apply tanh """
        return Fn.tanh(self)

    def sigmoid(self):
        """ Apply sigmoid """
        return Fn.sigmoid(self)

    #
    # Python math functions overloads
    #
    def __iadd__(self, other):
        # This is veeery hacky. Cases:
        # 1. When self has grad:
        #    use _add_ in order to create new node in the graph.
        #    If we don't create new node, and addition as defined takes
        #    only 2 arguments then a chain of += would case an assertion
        #    in the addition definition
        # 2. When self has no grad:
        #    We just do the operations in the data fields of the tensor
        if self.with_grad:
            return self.__add__(other)
        else:
            if isinstance(other, Number):
                self.data.add_(other)
            elif type(other) is type(self):
                self.data.add_(other.data)
            else:
                raise ValueError('Tensor += overload >> Object type is not not supported:', type(other))
            return self

    def __add__(self, other) -> Tensor:
        if isinstance(other, Number):
            # If adding vector with scalar
            return Fn.scalarAdd(self, other)
        elif type(other) is type(self):
            # If adding two tensors
            return self.add(other)
        else:
            raise ValueError('Tensor +/+= overload >> Object type is not not supported:', type(other))

    def __sub__(self, other) -> Tensor:
        # Check if number
        if isinstance(other, Number):
            return Fn.scalarAdd(self, -1 * other)
        elif type(other) is type(self):
            return self.add(Fn.scalarMul(other,-1)) # Subtraction is addition with negative
        else:
            raise ValueError('Tensor - overload >> Object type is not not supported:', type(other))

    def __neg__(self) -> Tensor:
        return self.scalarMul(-1)

    def __mul__(self, other) -> Tensor:
        # Check if number
        if isinstance(other, Number):
            return Fn.scalarMul(self, other)
        elif type(other) is type(self):
            return Fn.ewVecMul(self, other)
        else:
            raise ValueError('Tensor * overload >> Object type is not not supported:', type(other))

    def __div__(self,other) -> Tensor:
        # Check if number
        if isinstance(other, Number):
            return Fn.scalarMul(self, 1/other)
        else:
            raise ValueError('Tensor / overload >> Object type is not not supported:', type(other))

    def __radd__(self, other) -> Tensor:
        return self.__add__(other)

    def __rsub__(self, other) -> Tensor:
        return self.__neg__().__add__(other)

    def __rmul__(self, other) -> Tensor:
        return self.__mul__(other)

    def __truediv__(self, other) -> Tensor:
        return self.__div__(other)

    def __len__(self):
        return self.data.shape[0]

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.n <= self.shape[0]:
            return Tensor(self.data[self.index], with_grad=self.with_grad)
        else:
            raise StopIteration

    def __getitem__(self, index):
        return Tensor(self.data[index], with_grad=self.with_grad)

    #
    # Printing functions
    #
    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        rep = f"{self.data.__repr__()}, with_grad={self.with_grad}"
        if self.grad_fn:
            rep += f" grad_fn:{self.grad_fn}"
        return rep

# Hack Import
import scripts.autog.functional.functions as Fn
import scripts.autog.autograd.autograd as autograd