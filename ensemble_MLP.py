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
hidden_size = 1024
switch_ensemble = 1
probability = 0.5
learning_rate = 0.3
batch_size = 16
Z = 1000
# number of epochs to train the model
n_epochs = 50   # suggest training between 20-50 epochs
wandb.init(project="intra-ensemble", entity='fust', config = {"hidden_size": hidden_size,
                                                              "switch_ensemble": switch_ensemble,
                                                              "probability": probability,
                                                              "learning_rate": learning_rate,
                                                              "batch_size": batch_size,
                                                              "n_epochs": n_epochs})

hidden_size = wandb.config.hidden_size
switch_ensemble = wandb.config.switch_ensemble
probability = wandb.config.probability

learning_rate = wandb.config.learning_rate
n_epochs = wandb.config.n_epochs
batch_size = wandb.config.batch_size


if switch_ensemble == 1:
    switch_ensemble = True
elif switch_ensemble == 0:
    switch_ensemble = False

# for reproducibility
random.seed(777)
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)


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
train_X = train_X[0:500].to(device)
train_Y = train_Y[0:500].to(device)


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
        self.fc1 = nn.Linear(28 * 28, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 10)

    def forward(self, x):
        # flatten image input
        x = x.view(-1, 28 * 28)
        # add hidden layer, with relu activation function


        if switch_ensemble is True and inference is True:
            # fc1
            mean = probability * torch.matmul(x, torch.transpose(self.fc1.weight.data, 0, 1))
            std = torch.sqrt(probability * (1 - probability) * torch.matmul(x ** 2,
                                                                            torch.transpose(self.fc1.weight.data ** 2,
                                                                                            0, 1)))
            tmp = torch.normal(mean, std)
            for i in range(Z - 1):
                tmp += torch.normal(mean, std)
            x = tmp / Z
            x = F.relu(x)

            # fc2
            mean = probability * torch.matmul(x, torch.transpose(self.fc2.weight.data, 0, 1))
            std = torch.sqrt(probability * (1 - probability) * torch.matmul(x ** 2,
                                                                            torch.transpose(self.fc2.weight.data ** 2,
                                                                                            0, 1)))
            tmp = torch.normal(mean, std)
            for i in range(Z - 1):
                tmp += torch.normal(mean, std)
            x = tmp / Z

            return x

        else:
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x



# initialize the NN
model = Net()
model.cuda()
print(model)

running_loss_history = []
running_correct_history = []
validation_running_loss_history = []
validation_running_correct_history = []

criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=learning_rate)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
print('ensemble? :', switch_ensemble)
print('probability? :', probability)
print(len(train_loader))
for epoch in range(n_epochs):

    train_loss = 0.0
    train_correct = 0.0
    val_loss = 0.0
    val_correct = 0.0

    ###################
    # train the model #
    ###################
    inference = False
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        if switch_ensemble is True:
            with torch.no_grad():
                mask1 = torch.bernoulli(probability * torch.ones(hidden_size, 784)).to(device)
                mask2 = torch.bernoulli(probability * torch.ones(10, hidden_size)).to(device)
                # mask3 = torch.bernoulli(probability * torch.ones(10, 512)).to(device)
                model.fc1.weight.data = torch.mul(mask1, model.fc1.weight)
                model.fc2.weight.data = torch.mul(mask2, model.fc2.weight)
                # model.fc3.weight.data = torch.mul(mask3, model.fc3.weight)

        outputs = model(data)
        loss = criterion(outputs, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        _, preds = torch.max(outputs, 1)

        train_correct += torch.sum(preds == target.data)
        train_loss += loss.item()

        if switch_ensemble is True:
            model.fc1.weight.data = model.fc1.weight.data + torch.mul(1 - mask1, model.fc1.weight)
            model.fc2.weight.data = model.fc2.weight.data + torch.mul(1 - mask2, model.fc2.weight)
            # model.fc3.weight.data = model.fc3.weight.data + torch.mul(1 - mask3, model.fc3.weight)

    inference = False
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)


            output = model(data)
            loss = criterion(output, target)

            _, val_preds = torch.max(output, 1)
            val_loss += loss.item() * len(data)
            val_correct += torch.sum(val_preds == target.data)


    epoch_loss = train_loss / len(train_X)
    epoch_acc = train_correct.float() / len(train_X)

    running_loss_history.append(epoch_loss)
    running_correct_history.append(epoch_acc)

    val_epoch_loss = val_loss / len(test_X)
    val_epoch_acc = val_correct.float() / len(test_X)
    validation_running_loss_history.append(val_epoch_loss)
    validation_running_correct_history.append(val_epoch_acc)

    print("===================================================")
    print("epoch: ", epoch + 1)
    print("training loss: {:.5f}, acc: {:5f}".format(epoch_loss, epoch_acc))
    print("validation loss: {:.5f}, acc: {:5f}".format(val_epoch_loss, val_epoch_acc))