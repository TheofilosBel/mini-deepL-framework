from scripts.autog.tensor import Tensor, TensorNoGrad
from scripts.autog.modules import module as nn
import scripts.autog.optimizers as optim
from scripts.autog.metrics.metrics import accuracy


from torch import empty as torchEmpty
import random
import math

#
# Utility functions
#
def generate_disc_set(nb):
    center = torchEmpty(2).fill_(0.5)
    x = torchEmpty((nb, 2)).uniform_(0, 1)
    y = (x - center).pow(2).sum(1).sub(1/(2*math.pi)).sign().add(1).div(2)#.long()
    return x, y

def shuffle(data, target):
    assert data.shape[0] == target.shape[0]
    indices = list(range(data.shape[0]))
    random.shuffle(indices)

    return data[indices], target[indices]

#
# Data generation and training
#

# Params
nb_data = 2000
nb_train_points = 1000
mini_batch_size = 50

RUNS = 10
EPOCHS = 100
lr = 0.01

# Data generation
input, target = generate_disc_set(nb_data)
target = target.unsqueeze(1) # unsqueeze for loss function correctness

# Split to train and test set
# Also, use our Tensor wrappers to enable our autograd framework
train_input, test_input = Tensor(input[:nb_train_points]), Tensor(input[nb_train_points:])
train_target, test_target = Tensor(target[:nb_train_points]), Tensor(target[nb_train_points:])


# Define the model
model = nn.Sequential(
    nn.Linear(2, 25),
    nn.ReLU(),
    nn.Linear(25, 25),
    nn.ReLU(),
    nn.Linear(25, 25),
    nn.ReLU(),
    nn.Linear(25, 1),
    nn.Sigmoid()
)

# Define loss and optimizer
loss_func = nn.LossMSE()
optimizer = optim.SGDStepper(model.params(), lr=lr)

# Run multiple
acc_scores_train = torchEmpty(RUNS)
acc_scores_test = torchEmpty(RUNS)
err_train = torchEmpty(RUNS)
err_test = torchEmpty(RUNS)
for run in range(RUNS):
    # Reset paremeters
    model.reset_params()

    # start training
    for epoch in range(EPOCHS):
        # Shuffle data in every epoch
        train_input, train_target = shuffle(train_input, train_target)

        # Break to batches
        for b in range(0, train_input.shape[0], mini_batch_size):
            # Calculate output and loss
            output = model(train_input.narrow(0, b, mini_batch_size))
            loss = loss_func(output, train_target.narrow(0, b, mini_batch_size))

            # Training
            optimizer.zero_grad() # clear gradients before backprop
            loss.backward()       # backprop
            optimizer.step()      # update parameters

    # Report accuracy
    with TensorNoGrad(*model.params()):
        test_out =  model(test_input)
        train_out =  model(train_input)

        classes_test = (test_out.data > 0.5).int()
        classes_train = (train_out.data > 0.5).int()

        acc_scores_test[run] = accuracy(classes_test, test_target.data)
        acc_scores_train[run] = accuracy(classes_train, train_target.data)

        err_test[run] = loss_func(test_out, test_target).item()
        err_train[run] = loss_func(train_out, train_target).item()

        print(f"Run[{run}], train_accuracy: {acc_scores_train[run]:.2f}, test_accuracy: {acc_scores_test[run]:.2f},"+
            f" train_error: {err_train[run]:.2f}, test_error: {err_test[run]:.2f}")

# Print results for all runs
print('~~~~ Results ~~~~')
print(f"Train mean accuracy: {acc_scores_train.mean():.2f} +- {acc_scores_train.std():.2f}")
print(f"Test mean accuracy: {acc_scores_test.mean():.2f} +- {acc_scores_test.std():.2f}")
print(f"Train mean error: {err_train.mean():.2f} +- {err_train.std():.2f}")
print(f"Test mean error: {err_test.mean():.2f} +- {err_test.std():.2f}")
