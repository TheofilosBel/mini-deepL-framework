from torch import Tensor as TorchTensor # Only for type hints

# Framework imports
from scripts.autog.tensor import Tensor
from scripts.autog.functional.definitions import Context, ChainRuleMulWay, PostProcessDueToBroadcast
from scripts.autog.utils import is_matrix, is_scalar, is_vector

# Forward declaration of Edge/Tensor class
def calculate_chain_rule(mul_way: ChainRuleMulWay, post_proc: PostProcessDueToBroadcast,
     outer_grad: TorchTensor, inner_grad: TorchTensor):
    '''
    Generally we do the multiplication between the gradients with the following way:
    - outer_grad () inner_grad.
    The () can be defined by `ChainRuleMulWay`
    Also the order can change if `ChainRuleMulWay` says so.

    # Inputs:
    - outer_grad: Is the gradient calculated first in the cain rule
        (closer to the top of the graph and not the leafs)
    - inner_grad: Is the gradient calculated using the
        next node, whose gradient is outer_grad, with respect to the
        operation variable that these node takes as input.
    '''
    chain_grad = None

    if mul_way == ChainRuleMulWay.OUTER:
        # Since we got that hint we want to create an outer product of two vectors.
        # To avoid struggling with diff format (1xN or Nx1) we squeeze both
        # The result is always going to be a matrix
        assert is_vector(outer_grad) and is_vector(inner_grad)
        chain_grad = outer_grad.squeeze().outer(inner_grad.squeeze())

    elif mul_way == ChainRuleMulWay.ELEMENTWISE:
        # If we got an elementwise hint suppose that Gradient backward functions
        # know what they are doing and leave pytorch's broadcast to do the magic
        chain_grad = outer_grad * inner_grad

    elif mul_way == ChainRuleMulWay.REVERSE_OUTER_INNER_GRAD_SEQ:
        # As hinted we reverse the outer and inner grad sequence for this gradient
        chain_grad = inner_grad.matmul(outer_grad)

    elif mul_way == ChainRuleMulWay.NORMAL:
        # For normal we need to discover if we need mat mul or maybe simple elementwise
        if is_vector(outer_grad) and is_matrix(inner_grad):
            # Most of the times inner_grad will be a matrix (Jacobian) and outer grad a vector
            assert outer_grad.shape[1] == inner_grad.shape[0]
            chain_grad = outer_grad.matmul(inner_grad)
        elif is_matrix(outer_grad) and is_matrix(inner_grad):
            chain_grad = outer_grad.matmul(inner_grad)
        elif is_vector(outer_grad) and is_vector(inner_grad):
            chain_grad = outer_grad * inner_grad
        elif is_scalar(outer_grad) and\
            (is_vector(inner_grad) or is_matrix(inner_grad) or is_scalar(inner_grad)):
            # The common case were the cain rule starts with one scalar [1]
            # and the chain_rule grad will always be an elementwise mul
            chain_grad = outer_grad * inner_grad
        else:
            raise ValueError("Chain rule, Normal hint has un-recognized sizes", outer_grad.shape, inner_grad.shape)

    elif mul_way == ChainRuleMulWay.SKIP_INNER_GRAD:
        chain_grad = outer_grad
    else:
        raise ValueError("Chain rule doesn't recognize grad sizes", outer_grad.shape, inner_grad.shape)


    # Apply an post processing if exists
    if post_proc == PostProcessDueToBroadcast.SUM_GRAD_COLS:
        chain_grad = chain_grad.sum(axis=1).unsqueeze(1) # Add unsqueeze to have same shape as vector
        # Vector should be column_vector to sum grad columns
    elif post_proc == PostProcessDueToBroadcast.SUM_GRAD_ROWS:
        chain_grad = chain_grad.sum(axis=0).unsqueeze(0) # Add unsqueeze to have same shape as vector
        # Vector should be row_vector to sum grad rows

    return chain_grad


# Returns if partial grad is needed
def requires_partial(x):
    if isinstance(x, Tensor):
        return x.with_grad
    else:
        return False

# Extract the data part from a tensor (or return x if its a number)
def extract_data(x):
    if isinstance(x, Tensor):
        return x.data
    else:
        return x


def backward(tensor: Tensor, cur_partial_grad: TorchTensor):
    '''
    Get our partial gradient from the calling node (next in forward pass).
    Calculate the gradient of the previous node wrt to the current node (this).
    Call backwards on the previous nodes and pass them the partial grad until this point

    Important:
    While we use our Tensor class, when we do calculations we use
    a TorchTensor in order to get the efficiency of the library
    '''

    # If the grad is None that means we called backward for the 1st time
    if tensor.grad == None:
        tensor.grad = Tensor(cur_partial_grad)
    else:
        tensor.grad.data += cur_partial_grad

    # If node is leaf terminate the backward calls
    if len(tensor.backward_nodes) == 0:
        return
    else:
        # Calculate partial grads of previous nodes wrt to current node

        # Create the context: Check which of the previous nodes require grad
        ctx = Context([requires_partial(x) for x in tensor.backward_nodes])

        # Compute the partial grads wrt to the previous nodes
        partial_grads = tensor.grad_fn(ctx, [extract_data(x) for x in tensor.backward_nodes])

        # Call backward for the previous nodes
        for i, previous_tensor in enumerate(tensor.backward_nodes):
            # Skip nodes that don't require partial
            if ctx.requires_partial[i] == False: continue

            # Create the chain rule product
            chain_rule_grad = calculate_chain_rule(ctx.how_to_chain[i], ctx.post_process[i],
                 tensor.grad.data, partial_grads[i])

            # Call backward to previous nodes passing them their partial gradients
            previous_tensor.backward(chain_rule_grad)
