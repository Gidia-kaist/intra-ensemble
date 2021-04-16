#from torchvision import datasets
from torch.utils.data import DataLoader, TensorDataset
#import torchvision.transforms as transforms
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import wandb

# CUDA 처리
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu") # GPU 사용 가능하면 사용하고 아니면 CPU 사용
print("다음 기기로 학습합니다:", device)

# COMMAND
hidden_size = 512
switch_ensemble = 0
probability = 0.3
lr = 0.1
wandb.init(project="ensemble_1", config = {"hidden_size": hidden_size, "switch_ensemble": switch_ensemble, "probability": probability, "lr": lr})
wandb.log({"probability": probability})
hidden_size = wandb.config.hidden_size
switch_ensemble = wandb.config.switch_ensemble
probability = 1- wandb.config.probability
lr = wandb.config.lr
mask2 = torch.bernoulli(probability*torch.ones(hidden_size, 784)).to(device)
print(mask2.sum()/(hidden_size*784))


if switch_ensemble == 1:
    switch_ensemble = True
elif switch_ensemble == 0:
    switch_ensemble = False

# for reproducibility
random.seed(777)
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

# Hyperparameters
batch_size = 16

MNIST_data = np.load("/home/kdh/MNIST_data/MNIST_treated.npz")
MNIST_data.files
test_data = MNIST_data['arr_0']
train_data = MNIST_data['arr_1']
test_labels = MNIST_data['arr_2']
train_labels = MNIST_data['arr_3']

test_X, train_X = test_data / 255.0, train_data / 255.0
test_Y, train_Y = test_labels, train_labels

train_X = train_X.reshape((len(train_X), 1, 28, 28))
test_X = test_X.reshape((len(test_X), 1, 28, 28))

train_X = torch.from_numpy(train_X).float().to(device)
train_Y = torch.from_numpy(train_Y).long().to(device)

test_X = torch.from_numpy(test_X).float().to(device)
test_Y = torch.from_numpy(test_Y).long().to(device)

test = TensorDataset(test_X, test_Y)
train = TensorDataset(train_X, train_Y)

test_loader = DataLoader(test, batch_size=10000)
train_loader = DataLoader(train, batch_size=batch_size)



## Define the NN architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x



# initialize the NN
model = Net()
model.cuda()
print(model)


## Specify loss and optimization functions

# specify loss function
criterion = nn.CrossEntropyLoss()

# specify optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# number of epochs to train the model
n_epochs = 50  # suggest training between 20-50 epochs
epoch_counter = 0
model.train()  # prep model for training


print('ensemble? :', switch_ensemble)
print('probability? :', 1-probability)
for epoch in range(n_epochs):
    # monitor training loss
    torch.cuda.empty_cache()
    train_loss = 0.0
    ###################
    # train the model #
    ###################

    for data, target in train_loader:

        if switch_ensemble is True:
            # print("Ensemble:", switch_ensemble)
            weight_fc1 = model.fc1.weight
            weight_fc2 = model.fc2.weight
#            print(weight)
#            print(F.dropout(weight, p=probability)*(1-probability))
            model.fc1.weight = torch.nn.Parameter(F.dropout(weight_fc1, p=probability)*(1-probability))
            model.fc2.weight = torch.nn.Parameter(F.dropout(weight_fc2, p=probability) * (1 - probability))
        # linear layer (n_hidden -> hidden_2)
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        if switch_ensemble is True:
            model.fc1.weight = torch.nn.Parameter(weight_fc1)
            model.fc2.weight = torch.nn.Parameter(weight_fc2)
        # calculate the loss
        loss = criterion(output, target)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update running training loss
        train_loss += loss.item() * data.size(0)

    # print training statistics
    # calculate average loss over an epoch
    train_loss = train_loss / len(train_loader.dataset)
    wandb.log({"train_loss": train_loss, "n_epochs": n_epochs})
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(
        epoch + 1,
        train_loss
    ))

# initialize lists to monitor test loss and accuracy
test_loss = 0.0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

model.eval() # prep model for *evaluation*

for data, target in test_loader:
    torch.cuda.empty_cache()
    # forward pass: compute predicted outputs by passing inputs to the model
    output = model(data)
    # calculate the loss
    loss = criterion(output, target)
    # update test loss
    test_loss += loss.item()*data.size(0)
    # convert output probabilities to predicted class
    _, pred = torch.max(output, 1)
    # compare predictions to true label
    correct = np.squeeze(pred.eq(target.data.view_as(pred)))
    # calculate test accuracy for each object class
    for i in range(10000):
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

# calculate and print avg test loss
test_loss = test_loss/len(test_loader.dataset)
wandb.log({"test_loss": test_loss})
print('Test Loss: {:.6f}\n'.format(test_loss))

for i in range(10):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            str(i), 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))

test_acc = 100. * np.sum(class_correct) / np.sum(class_total)
print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))
wandb.log({"test_acc": test_acc})