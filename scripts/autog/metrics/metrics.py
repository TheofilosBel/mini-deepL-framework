from scripts.autog.tensor import Tensor


def accuracy(y, target):
    '''
        Returns how many elements in y are the same as target.
        Input: x,y Tensors
        Output: A double in range [0,1]
    '''
    assert type(y[0]) == type(target[0])
    assert y.shape[0] == target.shape[0]

    return (y == target).sum().item() / y.shape[0]
