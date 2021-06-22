""" This file stores the defintions of functions """
from torch import Tensor as TorchTensor # Only for type hints
from torch import empty as torchEmpty

from enum import Enum
from typing import List
from numbers import Number

from scripts.autog.utils import is_column_vector, is_matrix, is_row_vector
from scripts.autog.types import _TorchOrNum


class ChainRuleMulWay(Enum):
    """
    Hints to the chain rule what type of matrix multiplication to use when multiplying the gradients
    """
    NORMAL = 0
    OUTER = 1
    ELEMENTWISE = 2
    REVERSE_OUTER_INNER_GRAD_SEQ = 3
    SKIP_INNER_GRAD = 4                 # Skips inner grad because it is just multiplying by 1
                                        # and sized remain the same


class PostProcessDueToBroadcast(Enum):
    SUM_GRAD_ROWS = 1  # Sums the rows (colapses them) of a matrix gradient to handle bradcasting
    SUM_GRAD_COLS = 2  # Sums the on the columns (colapses them) of a matrix gradient to handle bradcasting

class Context:
    """
    Context communicates info between the gradient-implementations and
    the gradient merging (chain_rule) parts of the framework

    1. Hints to the function definition gradient implementations which inputs
       of an operation need gradient (1, 2, or both)
    2. Transfers a message to the gradient combination (chain rule) part of the framework
       to the what matrix multiplication function to use when merging
       the returned partial gradients with the gradient of the previous node
    """
    def __init__(self, requires_partial: List[bool]) -> None:
        self.requires_partial: List[bool] = requires_partial
        self.how_to_chain: List[ChainRuleMulWay] = [None for _ in requires_partial]
        self.post_process: List[PostProcessDueToBroadcast] = [None for _ in requires_partial]

class Summation:
    """
    Sums all the elements of a vector matrix
    """

    @staticmethod
    def forward(x: TorchTensor) -> TorchTensor:
        """
        Input: a tensor of max 2 dim
        Output: an 1x1 tensor
        """
        return x.sum()

    @staticmethod
    def backward(ctx: Context, forward_input: List[TorchTensor]) -> List[TorchTensor]:
        """
        Returns the gradient of sum

        Input: the input passed to forward
        Output: a vector of 1s equal to the size of the forward pass vector
        """
        assert len(forward_input) == 1
        ctx.how_to_chain[0] = ChainRuleMulWay.NORMAL

        return [torchEmpty(forward_input[0].shape).fill_(1)]

class ElementWiseVectorAddition:
    """
    Adds two vectors elementwise and produces another vector with the same size, as result
    """

    @staticmethod
    def forward(x: TorchTensor, y: TorchTensor) -> TorchTensor:
        """
        Input:
          - x: an 1 dim tensor
          - y: another 1 dim tensor (same size as x)
        Output: an 1 dim tensor (same size as x and y)
        """
        return x.add(y)

    @staticmethod
    def backward(ctx: Context, forward_input: List[TorchTensor]) -> List[TorchTensor]:
        """
        Returns the gradient of the addtition
        We expect 2 tensors in the forward_input.

        Input: the input passed to forward
        Output: A list of partial_derivatives wrt to inputs that need a partial derivative

        Theory:
        Elementwise addition Jacobian wrt both inputs is a diagonal matrix, specificity the Identity matrix.
        In order to avoid costly matrix multiplications we return the diagonal of the matrix
        and use it with elementwise product in the cain rule.
        """
        assert len(forward_input) == 2
        shape = forward_input[0].shape
        partial_jacobians = [None, None]

        # We could add a for, but it slows down computations
        # First vector
        if (ctx.requires_partial[0]):
            # The partial derivative of pointwise sum is the diag(1) (identity matrix)
            ctx.how_to_chain[0] = ChainRuleMulWay.ELEMENTWISE
            partial_jacobians[0] = torchEmpty(shape).fill_(1)

        # Second the broadcasted scalar
        if (ctx.requires_partial[1]):
            # The partial derivative of pointwise sum is the diag(1) (identity matrix)
            ctx.how_to_chain[1] = ChainRuleMulWay.ELEMENTWISE
            partial_jacobians[1] = torchEmpty(shape).fill_(1)

        return partial_jacobians

class ElementWiseVectorBroadcastedScalarAddition:
    """
    Adds one vector elementwise with a broadcasted scalar and produce a vector
    """

    @staticmethod
    def forward(x: TorchTensor, y: TorchTensor) -> TorchTensor:
        """
        Input:
          - x: an 1 dim tensor
          - y: an 1x1 scalar
        Output: an 1 dim tensor (same size as x)
        """
        return x.add(y)

    @staticmethod
    def backward(ctx: Context, forward_input: List[TorchTensor]) -> List[TorchTensor]:
        """
        Returns the gradient of the addtition
        We expect 2 tensors in the forward_input.

        Input: the input passed to forward
        Output: A list of partial_derivatives wrt to inputs that need a partial derivative

        Theory:
        Elementwise addition Jacobian wrt both inputs is a diagonal matrix, specificity the Identity matrix.
        In order to avoid costly matrix multiplications we return the diagonal of the matrix
        and use it with elementwise product in the cain rule.

        For the broadcased scalar we handle it like it was a vector. So its grad is also a vector,
        but we hint the autograd to accumulate (sum) those values. To explain this, its like if
        we used a for loop to make additions:  Sum_i (scalar_i + scalar_b) of size N (= vector size) and then
        requested the grad of scalar_b. This would be the sum( (scalar_i + scalar_n)' )
        """
        assert len(forward_input) == 2
        shape = forward_input[0].shape
        partial_jacobians = [None, None]

        # We could add a for, but it slows down computations
        # First vector
        if (ctx.requires_partial[0]):
            # The partial derivative of pointwise sum is the diag(1) (identity matrix)
            ctx.how_to_chain[0] = ChainRuleMulWay.ELEMENTWISE
            partial_jacobians[0] = torchEmpty(shape).fill_(1)

        # Second vector
        if (ctx.requires_partial[1]):
            # The partial derivative of pointwise sum is the diag(1) (identity matrix)
            ctx.how_to_chain[1] = ChainRuleMulWay.ELEMENTWISE
            partial_jacobians[1] = torchEmpty(shape).fill_(1)

            # Decide the summation of the broadcasted term for the gradient
            if (is_column_vector(forward_input[0])):
                ctx.post_process[1] = PostProcessDueToBroadcast.SUM_GRAD_ROWS
            else:
                ctx.post_process[1] = PostProcessDueToBroadcast.SUM_GRAD_COLS

        return partial_jacobians


class ElementWiseMatrixAddition:
    """ Add two matrixes elementwise and produce another matrix with the same size as result """

    @staticmethod
    def forward(x: TorchTensor, y: TorchTensor) -> TorchTensor:
        """
        Input:
          - x: an NxM matrix
          - y: an NxM matrix
        Output: one NxM Matrix
        """
        assert (is_matrix(x.data) and is_matrix(y.data))
        return x.add(y)

    @staticmethod
    def backward(ctx: Context, forward_input: List[TorchTensor]) -> List[TorchTensor]:
        """
        Returns the gradient of the addtition
        We expect 2 tensors in the forward_input.

        Input: the input passed to forward
        Output: A list of partial_derivatives wrt to inputs that need a partial derivative

        Theory:
        Elementwise matrix addition Jacobian can be quite tricky. But we simplify it by
        knowing that there are other operations after that that will lead to a scalar loss.

        We assume:
        z = W + X, but we also have, some_func(....prev_func(z)) -> scalar.
        By abusing the notation of a jacobian a little bit we can say that d prev_func / dz = delta,
        where delta is a matrix in order to fit size of a NxM (result of z).

        We have computed that:
        d some_func / dw_ij = d some_func / d prev_func * d prev_func/ dz * dz/dw_ij = delta_ij (same for x_ij).

        From that we can construct the vector notation (also abusing the jacobian notation) like:
        d prev_func / dW = delta (same of X)
        """
        assert len(forward_input) == 2
        partial_jacobians = [None, None]


        # Matrix 1
        if (ctx.requires_partial[0]):
            ctx.how_to_chain[0] = ChainRuleMulWay.SKIP_INNER_GRAD

        # Matrix 2
        if (ctx.requires_partial[1]):
            ctx.how_to_chain[1] = ChainRuleMulWay.SKIP_INNER_GRAD

        return partial_jacobians

class ElementWiseMatrixBroadcastedVectorAddition:
    """
        Add one matrix and a broadcasted vector to produce another matrix of same size of the first.
        The vector should match the size of a column of the matrix and will be used like having a
        matrix of multiple same columns.
    """

    @staticmethod
    def forward(x: TorchTensor, y: TorchTensor) -> TorchTensor:
        """
        Input:
          - x: an NxM matrix
          - y: an Mx1 vector
        Output: one NxM Matrix
        """
        assert (is_matrix(x.data) and is_row_vector(y.data))
        return x.add(y)

    @staticmethod
    def backward(ctx: Context, forward_input: List[TorchTensor]) -> List[TorchTensor]:
        """
        Returns the gradient of the addtition.
        We expect 2 tensors in the forward_input.

        Input: the input passed to forward
        Output: A list of partial_derivatives wrt to inputs that need a partial derivative

        Theory:
        Read ElementWiseMatrixAddition matrix addition for the details of the gradient.

        We just add a context hint for the autograd to sum the rows of the gradint
        in order to give to the broadcasted vector the correct gradient which will
        be later used in an update (see also ElementWiseVectorBroadcastedScalarAddition)
        """
        assert len(forward_input) == 2
        partial_jacobians = [None, None]

        # Fist, Matrix
        if (ctx.requires_partial[0]):
            ctx.how_to_chain[0] = ChainRuleMulWay.SKIP_INNER_GRAD

        # Then broadcasted vector
        if (ctx.requires_partial[1]):
            ctx.how_to_chain[1] = ChainRuleMulWay.SKIP_INNER_GRAD
            # ITS FIXED: this is the only case happening in a Linear layer so we hardcoded it!
            # we only use row_vectors, it can be extended for column also but wasn't needed here
            ctx.post_process[1] = PostProcessDueToBroadcast.SUM_GRAD_ROWS

        return partial_jacobians


class ScalarMultiplication:
    """ Multiply with a scalar """

    @staticmethod
    def forward(x: TorchTensor, y: Number) -> TorchTensor:
        """
        Input:
         - x a tensor
         - y a number
        Output: a tensor (same size as x)
        """
        return x * y

    @staticmethod
    def backward(ctx: Context, forward_input: List[_TorchOrNum]) -> List[TorchTensor]:
        """
        Returns the gradient of the scalar multiplication
        We expect 2 tensors as input

        Input: the input passed to forward
        Output: A list of partial_derivatives wrt to inputs that need a partial derivative

        Theory: if g(x) = x * a, where x in Rn and x in R
                then dg/dx = a * [[1, ... , 1], ...] (same size as x)
        """
        assert len(forward_input) == 2

        ctx.how_to_chain[0] = ChainRuleMulWay.NORMAL
        return [torchEmpty(forward_input[0].shape).fill_(1) * forward_input[1]]

class ScalarAddition:
    """ Adds a scalar to a tensor"""

    @staticmethod
    def forward(x: TorchTensor, y: Number) -> TorchTensor:
        """
        Input:
         - x a tensor
         - y a number
        Output: a 1 dim tensor (same size as x)
        """
        return x + y

    @staticmethod
    def backward(ctx: Context, forward_input: List[_TorchOrNum]) -> List[TorchTensor]:
        """
        Returns the gradient of the scalar addtition
        We expect 2 tensors as input

        Input: the input passed to forward
        Output: A list of partial_derivatives wrt to inputs that need a partial derivative

        Theory: if g(x) = x + a, where x in Rn and x in R
                then dg/dx = [[1, ... , 1], ...] (same size as x)
        """
        assert len(forward_input) == 2
        ctx.how_to_chain[0] = ChainRuleMulWay.NORMAL
        return [torchEmpty(forward_input[0].shape).fill_(1)]

class ElementWiseVectorMultiplication:
    """ Multiplies two vectors elementwise and produce another vector with the same size as result """

    @staticmethod
    def forward(x: TorchTensor, y: TorchTensor) -> TorchTensor:
        """
        Input:
          - x: an 1 dim tensor
          - y: an 1 dim tensor
        Output: an 1 dim tensor (same size as x and y)
        """
        return x * y

    @staticmethod
    def backward(ctx: Context, forward_input: List[TorchTensor]) -> List[TorchTensor]:
        """
        Returns the gradient of the addtition
        We expect 2 tensors in the forward_input.

        Input: the input passed to forward
        Output: A list of partial_derivatives wrt to inputs that need a partial derivative

        Theory:
        Elementwise multiplication Jacobian wrt both inputs is a diagonal matrix,
        with entries of the diagonal to be the other vector whose grad we are computing.
        In order to avoid costly matrix multiplications we return the diagonal of the matrix
        and use it with elementwise product in the cain rule.
        """
        assert len(forward_input) == 2
        partial_jacobians = [None, None]

        # Fist vector
        if ctx.requires_partial[0]:
            # Return second vector as grad
            ctx.how_to_chain[0] = ChainRuleMulWay.ELEMENTWISE
            partial_jacobians[0] = forward_input[1]

        # Second vector
        if ctx.requires_partial[1]:
            # Return second vector as grad
            ctx.how_to_chain[1] = ChainRuleMulWay.ELEMENTWISE
            partial_jacobians[1] = forward_input[0]

        return partial_jacobians

class MatrixVectorMultiplication:
    """ Multiply a matrix with a vector and get a vector as a result """

    @staticmethod
    def forward(x: TorchTensor, y: TorchTensor) -> TorchTensor:
        """
        Input:
         - x: a NxM dim tensor
         - y: a Mx1 dim tensor
        Output: an Nx1 dim tensor
        """
        return x.matmul(y)

    @staticmethod
    def backward(ctx: Context, forward_input: List[TorchTensor]) -> List[TorchTensor]:
        """
        Returns the gradient of the multiplication

        Input: the input passed to forward (2 tensors)
        Output: A list of partial_derivatives wrt to inputs that need a partial derivative
        """
        assert len(forward_input) == 2

        partial_jacobians = [None, None]

        # First, the matrix
        if ctx.requires_partial[0]:
            """
            The first is the matrix.

            The gradient of the operation g = w*x where:
                - w is a matrix in R^m*n and
                - x is a vector in R^n
            for a scalar loss L, that exists in some node after this computation, wrt the matrix w is
            dL/dw = dL/dg * dg/dw = delta.T * x.T (known as outer product)
            where dL/dg is equal to delta, a vector of size R^n (same as the result of g=w*x).

            This function returns only x and also sets that we need to use outer product to multiply delta*x.
            """
            ctx.how_to_chain[0] = ChainRuleMulWay.OUTER
            partial_jacobians[0] = forward_input[1] # the vector

        # The second is the vector:
        if ctx.requires_partial[1]:
            """
            The gradient of a vector v in Rn and a matrix m in Rm*n of the operation g = m*v , is:
            dg/dx = w

            For the matrix to be transposed and we need to change the inner and outter grad order
            """
            ctx.how_to_chain[1] = ChainRuleMulWay.REVERSE_OUTER_INNER_GRAD_SEQ
            partial_jacobians[1] = forward_input[0].T # the matrix

        return partial_jacobians


class VectorMatrixMultiplication:
    """ Multiply a vector with a matrix and get a vector as a result """

    @staticmethod
    def forward(x: TorchTensor, y: TorchTensor) -> TorchTensor:
        """
        Input:
         - x: a Mx1 dim tensor
         - y: a NxM dim tensor
        Output: an Nx1 dim tensor
        """
        return x.matmul(y)

    @staticmethod
    def backward(ctx: Context, forward_input: List[TorchTensor]) -> List[TorchTensor]:
        """
        Returns the gradient of the multiplication

        Input: the input passed to forward (2 tensors)
        Output: A list of partial_derivatives wrt to inputs that need a partial derivative
        """
        assert len(forward_input) == 2

        partial_jacobians = [None, None]

        # First, the vector
        if ctx.requires_partial[0]:
            """
            The gradient of a vector v in Rn and a matrix m in Rm*n of the operation g = m*v , is:
            dg/dx = w

            For the matrix to be transposed and we need to change the inner and outter grad order
            """
            ctx.how_to_chain[0] = ChainRuleMulWay.NORMAL
            partial_jacobians[0] = forward_input[1].T # the matrix transposed

        # The second is the matrix:
        if ctx.requires_partial[1]:
            """
            The second is the matrix.

            The gradient of the operation g = w*x where:
                - w is a matrix in R^m*n and
                - x is a vector in R^n
            for a scalar loss L, that exists in some node after this computation, wrt the matrix w is
            dL/dw = dL/dg * dg/dw = delta.T * x.T (known as outer product)
            where dL/dg is equal to delta, a vector of size R^n (same as the result of g=w*x).

            This function returns only x and also sets that we need to use outer product to multiply delta*x.
            """
            ctx.how_to_chain[1] = ChainRuleMulWay.REVERSE_OUTER_INNER_GRAD_SEQ
            partial_jacobians[1] = forward_input[0].T # the vector transposed

        return partial_jacobians


class MatrixMatrixMultiplication:
    """ Multiply a matrix with a matrix and get a matrix as a result """

    @staticmethod
    def forward(x: TorchTensor, y: TorchTensor) -> TorchTensor:
        """
        Input:
         - x: a NxM dim tensor
         - y: a MxC dim tensor
        Output: an NxC dim tensor
        """
        return x.matmul(y)

    @staticmethod
    def backward(ctx: Context, forward_input: List[TorchTensor]) -> List[TorchTensor]:
        """
        Returns the gradient of the multiplication

        Input: the input passed to forward (2 tensors)
        Output: A list of partial_derivatives wrt to inputs that need a partial derivative

        IMPORTANT: The order of forward must be the same for backward too
        """
        assert len(forward_input) == 2
        partial_jacobians = [None, None]

        """
            The gradient of a matrix-matrix multiplication is a 4-th order tensor.
            But we simplify things by assuming that we will have another computation
            after the mat-mat that will have a gradint `delta` and find the partial
            derivative of each point of the second matrix for that `delta`.

            Assume:  z = X * W,
                - X is an NxM dim matrix
                - W is an MxC dim matrix
                - z will be an NxC matrix

            Also assume that  func(other_func(...prev_func(z))) --after many steps--> scalar
            Under that contraint the prev_func will input an NxC matrix and output whatever.
            The gradient d prev_func/dz (abuse of notation here) will be an NxC matrix.
            Let's call that `delta`
        """

         # The first is the matrix
        if ctx.requires_partial[0]:
            """
            We computed that:
            d prev_func /d x_i,j = SUM_{k=1}^{C} delta_i,k * W_j,k

            If we once again abuse the gradient notation to match W.grad with W size we can say that
            d prev_func / dW = delta * W.T, the result will be an NxM matrix, like X.
            """
            # We know that autograd respects the oreder outer_grad * inner_grad
            # So we just issue a normal multiplication
            ctx.how_to_chain[0] = ChainRuleMulWay.NORMAL
            partial_jacobians[0] = forward_input[1].T # the W.T


        # The second matrix:
        if ctx.requires_partial[1]:
            """
            We computed that:
            d prev_func /d w_i,j = SUM_{k=1}^{N} delta_k,j * X_k,i

            If we once again abuse the gradient notation to match gradW with W size we can say that
            d prev_func / dW = X.T * delta, the result will be an MxC matrix, like W.
            """

            # We say explicitly that we need to first multiply our gradient
            # and then use the gradient of d prev_func / dz with the following flag
            ctx.how_to_chain[1] = ChainRuleMulWay.REVERSE_OUTER_INNER_GRAD_SEQ
            partial_jacobians[1] = forward_input[0].T # the X.T

        return partial_jacobians

class Power:
    """ Raises all the elements of a vector to an exponent """

    @staticmethod
    def forward(x: TorchTensor, exp: Number) -> TorchTensor:
        """
        Input: x tensor
        Output: same size tensor as input raised elementwise to the of exp
        """
        return x.pow(exp)

    @staticmethod
    def backward(ctx: Context, forward_input: List[TorchTensor]) -> List[TorchTensor]:
        """
        Returns the gradient of power

        Input: the input passed to forward
        Output: a vector of 1s equal to the size of the forward pass vector
        """
        assert len(forward_input) == 2
        ctx.how_to_chain[0] = ChainRuleMulWay.NORMAL
        exp = forward_input[1]
        return [exp * forward_input[0].pow(exp - 1)]

class ReLU:
    """ Applies a rectified linear unit """

    @staticmethod
    def forward(x: TorchTensor) -> TorchTensor:
        """
        Input: x a 1 dim tensor
        Output: an 1x1 tensor
        """
        return x.relu()

    @staticmethod
    def backward(ctx: Context, forward_input: List[TorchTensor]) -> List[TorchTensor]:
        """
        Returns the gradient of power

        Input: the input passed to forward
        Output: a vector of 1s equal to the size of the forward pass vector
        """
        assert len(forward_input) == 1
        ctx.how_to_chain[0] = ChainRuleMulWay.ELEMENTWISE

        x = forward_input[0].relu()
        x[x > 0] = 1

        return [x]


class Tanh:
    """ Applies a hyperbolic tangent """

    @staticmethod
    def forward(x: TorchTensor) -> TorchTensor:
        """
        Input: x, a tensor
        Output: A same size tensor as input
        """
        return x.tanh()

    @staticmethod
    def backward(ctx: Context, forward_input: List[TorchTensor]) -> List[TorchTensor]:
        """
        Returns the gradient of power

        Input: the input passed to forward
        Output: a tensor of size equal to the size of the forward pass tensor
        """
        assert len(forward_input) == 1
        ctx.how_to_chain[0] = ChainRuleMulWay.ELEMENTWISE
        return [1 - forward_input[0].tanh().pow(2)]


class Sigmoid:
    """ Applies a sigmoid function """

    @staticmethod
    def forward(x: TorchTensor) -> TorchTensor:
        """
        Input: x, a tensor
        Output: A same size tensor as input
        """
        return x.sigmoid()

    @staticmethod
    def backward(ctx: Context, forward_input: List[TorchTensor]) -> List[TorchTensor]:
        """
        Returns the gradient of power

        Input: the input passed to forward
        Output: a tensor of size equal to the size of the forward pass tensor
        """
        assert len(forward_input) == 1
        ctx.how_to_chain[0] = ChainRuleMulWay.ELEMENTWISE
        return [forward_input[0].sigmoid() * (1-forward_input[0].sigmoid())]