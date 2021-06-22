''' This file provides weight init functions for modules (mainly linear)'''
import math
import torch
from scripts.autog.tensor import Tensor


def xavier_init(w: Tensor, fan_in, fan_out) -> None:
    w.data.uniform_(-1,1) * math.sqrt(6./(fan_in + fan_out))
