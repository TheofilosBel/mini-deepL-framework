import math

from scripts.autog.tensor import Tensor
import scripts.autog.functional.functions as Fn
import scripts.autog.modules.weight_init as init

from abc import ABC, abstractmethod

class Module(ABC):
    """
    An abstract class that all other modules inherit from
    """

    @abstractmethod
    def forward(self, *input):
        pass

    def __call__(self, *input):
        return self.forward(*input)

    @abstractmethod
    def params(self):
        pass

class Linear(Module):
    """
    A fully connected linear layer
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Tensor.empty((in_features, out_features), with_grad = True)

        if bias:
            self.bias = Tensor.zeros((1, out_features), with_grad = True)
        else:
            self.bias = None

        # Do the weight init
        self.reset_params()

    def reset_params(self):
        init.xavier_init(self.weights, self.in_features, self.out_features)

        if self.bias:
            bound = 1 / math.sqrt(self.in_features)
            self.bias.data.uniform_(-1 * bound, bound)


    def forward(self, x: Tensor) -> Tensor:
        """
        Applies the relu function element-wise

        Input: x a 1 dim tensor
        Output: an 1x1 tensor
        """
        return Fn.linear(x, self.weights, self.bias)

    def params(self):
        return [self.weights, self.bias]

class ReLU(Module):
    """
    The rectified linear unit module
    """
    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x: Tensor) -> Tensor:
        """
        Applies the relu function element-wise.

        Input: x a 1 dim tensor
        Output: an 1x1 tensor
        """
        return Fn.relu(x)

    def reset_params(self):
        pass

    def params(self):
        return []

class Tanh(Module):
    """
    The tanh module
    """
    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, x: Tensor) -> Tensor:
        """
        Applies the tanh function element-wise.

        Input: x a tensor
        Output: same size tensor as input
        """
        return Fn.tanh(x)

    def reset_params(self):
        pass

    def params(self):
        return []

class Sigmoid(Module):
    """
    The sigmoid module
    """
    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, x: Tensor) -> Tensor:
        """
        Applies the sigmoid function element-wise.

        Input: x a tensor
        Output: same size tensor as input
        """
        return Fn.sigmoid(x)

    def reset_params(self):
        pass

    def params(self):
        return []

class LossMSE(Module):
    """
    The mean square error loss function.
    """
    def __init__(self):
        super(LossMSE, self).__init__()

    def forward(self, y: Tensor, target: Tensor) -> Tensor:
        """
        Computes the mse loss.

        Input: y a 1 dim tensor
        Output: a 1 dim tensor
        """
        return Fn.mse(y, target)

    def reset_params(self):
        pass

    def params(self):
        return []

class Sequential(Module):
    """
    Saves and calls a list of modules in a sequential order.
    """
    def __init__(self, *args: Module):
        super(Sequential, self).__init__()
        self.modules = []
        for module in args:
            self.modules.append(module)

    def forward(self, x: Tensor) -> Tensor:
        """
        Calls the module in sequential order by passing the ouput of each one as input to the next.
        """
        for module in self.modules:
            x = module(x)
        return x

    def reset_params(self):
        for module in self.modules:
            module.reset_params()

    def params(self):
        parameters =[]
        for mod in self.modules:
            for p in mod.params():
                parameters.append(p)

        return parameters