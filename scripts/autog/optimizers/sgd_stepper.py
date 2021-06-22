from scripts.autog.optimizers.optimizer import Optimizer
from scripts.autog.tensor import TensorNoGrad

class SGDStepper(Optimizer):
    '''
        This Optimizer simply uses the `.grad` filed of a tensor
        to update it's values.

        The Stochastic part should come from the way of training and
        is left to the user to descide.
    '''

    def __init__(self, parameters, lr) -> None:
        super().__init__(parameters, lr=lr)

    def step(self) -> None:

        with TensorNoGrad(*self.parameters):
            for p in self.parameters:
                p += -self.lr * p.grad

    def zero_grad(self) -> None:
        for p in self.parameters:
            if p.grad != None:
                p.grad.data.zero_()

