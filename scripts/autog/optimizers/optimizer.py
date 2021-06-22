from abc import ABC, abstractmethod
from typing import List

class Optimizer(ABC):
    """
    An abstract class that all other optimizers inherit from
    """

    def __init__(self, parameters, lr=0.01) -> None:
        super().__init__()
        self.parameters = parameters
        self.lr = lr

    @abstractmethod
    def step(self) -> None:
        pass

    @abstractmethod
    def zero_grad(self) -> None:
        pass