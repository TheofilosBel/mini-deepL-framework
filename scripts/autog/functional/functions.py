from typing import Optional
from numbers import Number

# Framework imports
import scripts.autog.functional.definitions as defs
from scripts.autog.tensor import Tensor
from scripts.autog.utils import is_matrix, is_scalar, is_vector, is_row_vector, is_column_vector


#
# Additions
#

def sum(x: Tensor) -> Tensor:
    """
    Returns the sum of the elements of the a tensor x
    """
    result = Tensor(defs.Summation.forward(x.data), x.with_grad)

    # Check for gradient monitoring
    # If x requires grad then result requires grad also
    if result.with_grad:
        result.add_backward_node(x)
        result.add_grad_func(defs.Summation.backward)

    return result

def scalarAdd(x: Tensor, y: Number) -> Tensor:
    """
    Performs the element-wise addition between a scalar and a vector.
    ## Inputs:
     - x: 1 dim tensors (vectors)
     - y: scalar
    ## Returns: a tensor of same shape as x.
    """

    # Compute result and keep grad only if x or y has grad
    result = Tensor(defs.ScalarAddition.forward(x.data, y), x.with_grad)

    # Check for gradient monitoring
    # If x has grad then result has grad also
    if result.with_grad:
        result.add_backward_node(x)
        result.add_backward_node(y)
        result.add_grad_func(defs.ScalarAddition.backward)

    return result

def ewVecAdd(x: Tensor, y: Tensor) -> Tensor:
    """
    Element-wise addition between two vectors.
    Input: x, y are 1 dim tensors (vectors or scalars)
    Output: a tensor of same shape as x (and y).
    """
    # Two vectors or 2 scalars can be added together in ew manner
    assert (is_vector(x.data) and is_vector(y.data)) or\
        (is_scalar(x.data) and is_scalar(y.data))

    # Compute result and keep grad only if one of x or y has grad
    result = Tensor(defs.ElementWiseVectorAddition.forward(x.data, y.data), x.with_grad or y.with_grad)

    # Check for gradient monitoring
    # If x has grad then result has grad also
    if result.with_grad:
        result.add_backward_node(x)
        result.add_backward_node(y)
        result.add_grad_func(defs.ElementWiseVectorAddition.backward)

    return result

def ewVecBScalarAdd(x: Tensor, y: Tensor) -> Tensor:
    """
    Adds one vector elementwise with a broadcasted scalar and produce a vector
    Input:
        - x: an 1 dim tensor
        - y: an 1x1 scalar
    Output: an 1 dim tensor (same size as x)
    """
    # Two vectors or 2 scalars can be added together in ew manner
    assert (is_vector(x.data) and is_scalar(y.data))

    # Compute result and keep grad only if one of x or y has grad
    result = Tensor(defs.ElementWiseVectorBroadcastedScalarAddition.forward(x.data, y.data), x.with_grad or y.with_grad)

    # Check for gradient monitoring
    # If x has grad then result has grad also
    if result.with_grad:
        result.add_backward_node(x)
        result.add_backward_node(y)
        result.add_grad_func(defs.ElementWiseVectorBroadcastedScalarAddition.backward)

    return result

def ewMatBVecAdd(x: Tensor, y: Tensor) -> Tensor:
    """
    Element-wise addition between matrix and broadcasted vector.
    Input:
        - x: an NxM matrix
        - y: an Mx1 vector
    Output: one NxM Matrix
    """

    # Compute result and keep grad only if one of x or y has grad
    result = Tensor(defs.ElementWiseMatrixBroadcastedVectorAddition.forward(x.data, y.data),
         x.with_grad or y.with_grad)

    # Check for gradient monitoring
    # If x or y has grad then result has grad also
    if result.with_grad:
        result.add_backward_node(x)
        result.add_backward_node(y)
        result.add_grad_func(defs.ElementWiseMatrixBroadcastedVectorAddition.backward)

    return result

def ewMatAdd(x: Tensor, y: Tensor) -> Tensor:
    """
    Element-wise addition between two matrixes

    Input:
        - x: an NxM matrix
        - y: an NxM matrix
    Output: an NxM Matrix
    """

    # Compute result and keep grad only if one of x or y has grad
    result = Tensor(defs.ElementWiseMatrixAddition.forward(x.data, y.data), x.with_grad or y.with_grad)

    # Check for gradient monitoring
    # If x or y has grad then result has grad also
    if result.with_grad:
        result.add_backward_node(x)
        result.add_backward_node(y)
        result.add_grad_func(defs.ElementWiseMatrixAddition.backward)

    return result

#
# Multiplications
#

def scalarMul(x: Tensor, y: Number) -> Tensor:
    """
    Performs the element-wise multiplication of a scalar and a vector

    Inputs:
     - x: 1 dim tensors (vectors)
     - y: scalar
    Returns: a tensor of same shape as x.
    """

    # Compute result and keep grad only if x or y has grad
    result = Tensor(defs.ScalarMultiplication.forward(x.data, y), x.with_grad)

    # Check for gradient monitoring
    # If x has grad then result has grad also
    if result.with_grad:
        result.add_backward_node(x)
        result.add_backward_node(y)
        result.add_grad_func(defs.ScalarMultiplication.backward)

    return result

def ewVecMul(x: Tensor, y: Tensor) -> Tensor:
    """
    Performs the element-wise multiplication between two vectors

    Input: x,y 1 dim tensors (vectors)
    Output: a tensor of same shape as x (and y).
    """
    assert is_vector(x.data) and is_vector(y.data)

    # Compute result and keep grad only if one of x or y has grad
    result = Tensor(defs.ElementWiseVectorMultiplication.forward(x.data, y.data), x.with_grad or y.with_grad)

    # Check for gradient monitoring
    # If x has grad then result has grad also
    if result.with_grad:
        result.add_backward_node(x)
        result.add_backward_node(y)
        result.add_grad_func(defs.ElementWiseVectorMultiplication.backward)

    return result

def vecVecInnerMul(x: Tensor, y: Tensor) -> Tensor:
    """
    Performs the element-wise multiplication between two vectors and then a sum

    Input: x,y 1 dim tensors (vectors)
    Output: a scalar
    """
    assert is_vector(x.data) and is_vector(y.data)
    return sum( ewVecMul(x,y) )

def matvecMul(matrix: Tensor, vector: Tensor) -> Tensor:
    """
    Performs the matrix-vector multiplication operation.
    Input:
      - x: a NxM dim tensor
      - y: a M dim vector tensor
    Returns: an N dim vector tensor.
    """
    assert is_matrix(matrix.data) and is_vector(vector.data)

    # Compute result and keep grad only if x or y has grad
    result = Tensor(defs.MatrixVectorMultiplication.forward(matrix.data, vector.data),
        matrix.with_grad or vector.with_grad)

    # Check for gradient monitoring
    # If x has grad then result has grad also
    if result.with_grad:
        result.add_backward_node(matrix)
        result.add_backward_node(vector)
        result.add_grad_func(defs.MatrixVectorMultiplication.backward)

    return result

def vecmatMul(vector: Tensor, matrix: Tensor,) -> Tensor:
    """
    Performs the vector-matrix multiplication operation.
    Input:
      - vector: a M dim vector tensor
      - matrix: a NxM dim tensor
    Returns: an N dim vector tensor.
    """
    assert is_vector(vector.data) and is_matrix(matrix.data)

    # Compute result and keep grad only if x or y has grad
    result = Tensor(defs.VectorMatrixMultiplication.forward(vector.data, matrix.data,),
        matrix.with_grad or vector.with_grad)

    # Check for gradient monitoring
    # If x has grad then result has grad also
    if result.with_grad:
        result.add_backward_node(vector)
        result.add_backward_node(matrix)
        result.add_grad_func(defs.VectorMatrixMultiplication.backward)

    return result

def matmatMul(x: Tensor, y: Tensor) -> Tensor:
    """
    Performs the matrix-vector multiplication operation

    Input:
        - x: a NxM dim tensor
        - y: a MxC dim tensor
    Returns: an NxC dim tensor
    """
    assert is_matrix(x.data) and is_matrix(y.data) or \
        (is_column_vector(x.data) and is_row_vector(y.data)) or \
        (is_row_vector(x.data) and is_column_vector(y.data))

    # Compute result and keep grad only if x or y has grad
    result = Tensor(defs.MatrixMatrixMultiplication.forward(x.data, y.data),
      x.with_grad or y.with_grad)

    # Check for gradient monitoring
    # If x has grad then result has grad also
    if result.with_grad:
        result.add_backward_node(x)
        result.add_backward_node(y)
        result.add_grad_func(defs.MatrixMatrixMultiplication.backward)

    return result

#
# Other functions
#

def power(x: Tensor, exp: Number) -> Tensor:
    """
    Returns the sum of the elements of the 1 dim tensor x
    """
    result = Tensor(defs.Power.forward(x.data, exp), x.with_grad)

    # Check for gradient monitoring
    # If x requires grad then result requires grad also
    if result.with_grad:
        result.add_backward_node(x)
        result.add_backward_node(exp)
        result.add_grad_func(defs.Power.backward)

    return result

def linear(x: Tensor, weights: Tensor, bias: Optional[Tensor] = None) -> Tensor:
    """
    Applies a linear transformation x*w + b
    """
    if bias is None:
        return x.matmul(weights)
    else:
        return x.matmul(weights).add(bias)

def mse(y: Tensor, target: Tensor) -> Tensor:
    """
    Computes the mean square error loss
    """
    assert (y.shape == target.shape)

    if is_matrix(y):
        # If y has shape NxD, compute mean along each point's dimensions (along D)
        return power(y - target, 2).sum() * (1 / y.shape[1]) * (1 / y.shape[0])
    else:
        return power(y - target, 2).sum() * (1 / y.shape[0])

def relu(x: Tensor) -> Tensor:
    """
    Returns the tensor after applying a rectified linear unit
    """

    # Compute result
    result = Tensor(defs.ReLU.forward(x.data), x.with_grad)

    # Check for gradient monitoring
    # If x requires grad then result requires grad also
    if result.with_grad:
        result.add_backward_node(x)
        result.add_grad_func(defs.ReLU.backward)

    return result


def tanh(x: Tensor) -> Tensor:
    """
    Return the tensor after applying tanh
    """
    # Compute result
    result = Tensor(defs.Tanh.forward(x.data), x.with_grad)

    # Check for gradient monitoring
    # If x requires grad then result requires grad also
    if result.with_grad:
        result.add_backward_node(x)
        result.add_grad_func(defs.Tanh.backward)

    return result

def sigmoid(x: Tensor) -> Tensor:
    """
    Return the tensor after applying sigmoid
    """
    # Compute result
    result = Tensor(defs.Sigmoid.forward(x.data), x.with_grad)

    # Check for gradient monitoring
    # If x requires grad then result requires grad also
    if result.with_grad:
        result.add_backward_node(x)
        result.add_grad_func(defs.Sigmoid.backward)

    return result