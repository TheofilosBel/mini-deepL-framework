from typing import List, Union
from torch import Tensor as TorchTensor # Only for type hints
from numbers import Number

_TorchOrNum = Union[TorchTensor, Number]