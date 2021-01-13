# from torchvision import datasets
import os

import torchvision
from torch.backends import cudnn
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import wandb
import copy

# CUDA 처리
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")  # GPU 사용 가능하면 사용하고 아니면 CPU 사용
print("다음 기기로 학습합니다:", device)

# COMMAND
hidden_size = 512
switch_ensemble = 1
probability = 0.5
learning_rate = 0.001
batch_size = 128


n_epochs = 10
wandb.init(project="intra-ensemble", entity='fust', config={"hidden_size": hidden_size,
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

transformer = transforms.Compose([transforms.Resize((32, 32)),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                 ])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transformer)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=100,
                                          shuffle=True, num_workers=4)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transformer)
test_loader = torch.utils.data.DataLoader(testset, batch_size=100,
                                         shuffle=False, num_workers=4)


## Define the NN architecture
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(5 * 5 * 50, 500)
        # self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        # flatten
        x = x.view(-1, 5 * 5 * 50)
        x = F.relu(self.fc1(x))
        # x = self.dropout1(x)
        x = self.fc2(x)
        return x


model = Net()
model.to(device)

print(model)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

running_loss_history = []
running_correct_history = []
validation_running_loss_history = []
validation_running_correct_history = []

print('ensemble? :', switch_ensemble)
print('probability? :', probability)

for epoch in range(n_epochs):

    train_loss = 0.0
    train_correct = 0.0
    val_loss = 0.0
    val_correct = 0.0

    ###################
    # train the model #
    ###################

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        if switch_ensemble is True:
            with torch.no_grad():
                mask1 = torch.bernoulli(probability * torch.ones(500, 5 * 5 * 50)).to(device)
                mask2 = torch.bernoulli(probability * torch.ones(10, 500)).to(device)
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


    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)


            output = model(data)
            loss = criterion(output, target)

            _, val_preds = torch.max(output, 1)
            val_loss += loss.item()
            val_correct += torch.sum(val_preds == target.data)


    epoch_loss = train_loss / len(train_loader)
    epoch_acc = train_correct.float() / len(train_loader)
    running_loss_history.append(epoch_loss)
    running_correct_history.append(epoch_acc)

    val_epoch_loss = val_loss / len(test_loader)
    val_epoch_acc = val_correct.float() / len(test_loader)
    validation_running_loss_history.append(val_epoch_loss)
    validation_running_correct_history.append(val_epoch_acc)

    print("===================================================")
    print("epoch: ", epoch + 1)
    print("training loss: {:.5f}, acc: {:5f}".format(epoch_loss, epoch_acc))
    print("validation loss: {:.5f}, acc: {:5f}".format(val_epoch_loss, val_epoch_acc))
