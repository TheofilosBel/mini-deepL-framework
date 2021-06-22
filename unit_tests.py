from scripts.autog.tensor import Tensor, TensorNoGrad
import torch
import math

############### TEST UTILS ########################
def generate_disc_set(nb):
    center = torch.empty(2).fill_(0.5)
    x = torch.empty((nb, 2)).uniform_(0, 1)
    y = (x - center).pow(2).sum(1).sub(1/(2*math.pi)).sign().add(1).div(2)#.long()
    return x, y

####################################################

###############################################################
# Test name:    1 layer Matrix Matrix mull Gradient test      #
# action :      X * W                                         #
# inputs:       both X & W are matrixes                       #
###############################################################
x_shape = (2,200)
w_shape = (200,300)
assert x_shape[1] == w_shape[0]

#  ---> torch way
X = torch.rand(x_shape, requires_grad=True)
W = torch.rand(w_shape, requires_grad=True)

X.matmul(W).sum().backward()

# print(X.grad)
# print(W.grad)

# ----> our way
XX = Tensor(X, with_grad=True)
WW = Tensor(W, with_grad=True)

XX.matmul(WW).sum().backward()

# print(X.grad)
# print(W.grad)

try:
  assert torch.sum(XX.grad.data == X.grad).item() == X.grad.shape[0] * X.grad.shape[1]
  assert torch.sum(WW.grad.data == W.grad).item() == W.grad.shape[0] * W.grad.shape[1]
  print('[test] 1 layer MatMat gradient: PASSED !')
except AssertionError:
  print('[test] 1 layer MatMat gradient: FAILED ! :(')


###############################################################
# Test name:    2 layer Matrix Matrix mull Gradient test      #
# action :      X * W                                         #
# inputs:       both X & W are matrixes                       #
###############################################################

x_shape = (100,20)
w1_shape = (20,333)
w2_shape = (333,1)
assert x_shape[1] == w1_shape[0] and w1_shape[1] == w2_shape[0]

#  ---> torch way
X = torch.rand(x_shape, requires_grad=True)
W1 = torch.rand(w1_shape, requires_grad=True)
W2 = torch.rand(w2_shape, requires_grad=True)

X.matmul(W1).matmul(W2).sum().backward()

# print(X.grad)
# print(W1.grad)
# print(W2.grad)

# ----> our way
XX = Tensor(X, with_grad=True)
WW1 = Tensor(W1, with_grad=True)
WW2 = Tensor(W2, with_grad=True)

XX.matmul(WW1).matmul(WW2).sum().backward()

# print(XX.grad.data)
# print(WW1.grad.data)
# print(WW2.grad.data)

try:
  assert torch.sum(XX.grad.data == X.grad).item() == X.grad.shape[0] * X.grad.shape[1]
  assert torch.sum(WW1.grad.data == W1.grad).item() == W1.grad.shape[0] * W1.grad.shape[1]
  assert torch.sum(WW2.grad.data == W2.grad).item() == W2.grad.shape[0] * W2.grad.shape[1]
  print('[test] 2 layer MatMat gradient: PASSED !')
except AssertionError:
  print('[test] 2 layer MatMat gradient: FAILED ! :(')


###############################################################
# Test name:    2 layer Circle learning test                  #
# action :      backprop                                      #
###############################################################

data, target = generate_disc_set(1000)
target = target.unsqueeze(1)


# -----> Torch way
W1 = torch.zeros((2,200), requires_grad=True)
W2 = torch.zeros((200,1), requires_grad=True)

with torch.no_grad():
    W1.normal_(0, 0.7)
    W2.normal_(0, 0.7)

# Copy for our way
w1_cp = torch.clone(W1)
w2_cp = torch.clone(W2)

# Keep for comaprison
losses = []
w1_grads = []

gama = 0.001
EPOCH = 100

# Loop
for e in range(EPOCH):

    # Pass all datapoints through the W1 W1
    output = data.matmul(W1).matmul(W2)
    # loss = loss_func(output, target)
    loss = (output - target).pow(2).sum() * (1 / data.shape[0])

    if W1.grad != None:
        W1.grad.zero_()
        W2.grad.zero_()

    loss.backward()
    # optimizer.step()

    with torch.no_grad():
        W1 += -gama * W1.grad
        W2 += -gama * W2.grad

    if e % 50 == 0 or e == EPOCH -1:
      losses.append(loss.item())
      w1_grads.append(W1.grad)

# ----> our way
my_data = Tensor(data)
my_target = Tensor(target)

W1 = Tensor(w1_cp, with_grad = True)
W2 = Tensor(w2_cp, with_grad = True)

our_losses = []
our_w1_grads = []

for e in range(EPOCH):

    # Pass all datapoints through the W1 W1
    output = my_data.matmul(W1).matmul(W2)
    loss = (output - my_target).power(2).sum() * (1 / data.shape[0])

    if W1.grad != None:
        W1.grad.data.zero_()
        W2.grad.data.zero_()

    loss.backward()

    with TensorNoGrad(W1,W2):
        W1 += -gama * W1.grad
        W2 += -gama * W2.grad

    if e % 50 == 0 or e == EPOCH -1:
      our_losses.append(loss.item())
      our_w1_grads.append(W1.grad.data)

try:
  assert losses == our_losses
  for i in range(len(our_w1_grads)):
    assert torch.sum(our_w1_grads[i] == w1_grads[i]).item() \
       == w1_grads[i].shape[0] * w1_grads[i].shape[1]
  print(f'[test] 2-layer backprop - {EPOCH} epoch: PASSED !')
except AssertionError:
  print(f'[test] 2-layer backprop - {EPOCH} epoch: FAILED ! :(')