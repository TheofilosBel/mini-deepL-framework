from torch import Tensor as TorchTensor # Only for type hints

def is_matrix(x: TorchTensor):
    '''
    A matrix is a tensor with 2 dims.
    If the first dim is 1 then its not a matrix, but a vector.
    If the first dim is larger then 1 then we have a matrix.
    '''
    return len(x.shape) == 2 and x.shape[0] > 1 and x.shape[1] > 1

def is_vector(x: TorchTensor):
    '''
    Returns true if x is a vector
    '''
    return (len(x.shape) == 1 and x.shape[0] > 1)\
        or is_column_vector(x) or is_row_vector(x)\

def is_column_vector(x: TorchTensor):
    '''
    We will say that 1xN is also a vector
    '''
    return (len(x.shape) == 2 and x.shape[1] == 1 and x.shape[0] > 1)

def is_row_vector(x: TorchTensor):
    '''
    We will say that 1xN is also a vector
    '''
    return (len(x.shape) == 2 and x.shape[0] == 1 and x.shape[1] > 1)

def is_scalar(x: TorchTensor):
    return len(x.shape) == 0 or \
        (len(x.shape) == 1 and x.shape[0] == 1) or\
        (len(x.shape) == 2 and x.shape[0] == 1 and x.shape[1] == 1)
