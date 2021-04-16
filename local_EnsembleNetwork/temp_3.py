# from torchvision import datasets
from torch.utils.data import DataLoader, TensorDataset
# import torchvision.transforms as transforms
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
probability = 0.7
learning_rate = 0.05
batch_size = 128
# number of epochs to train the model
n_epochs = 5  # suggest training between 20-50 epochs
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
        self.fc1 = nn.Linear(28 * 28, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 10)
        # linear layer (n_hidden -> 10)
        # dropout layer (p=0.2)
        # dropout prevents overfitting of data
        # self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # flatten image input
        x = x.view(-1, 28 * 28)
        # add hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# initialize the NN
model = Net()
model.cuda()
print(model)

## Specify loss and optimization functions

# specify loss function
criterion = nn.CrossEntropyLoss()

# specify optimizer
optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=learning_rate)

epoch_counter = 0
# model.train()  # prep model for training


print('ensemble? :', switch_ensemble)
print('probability? :', probability)
for epoch in range(n_epochs):
    # monitor training loss
    torch.cuda.empty_cache()
    train_loss = 0.0
    test_loss = 0.0
    val_loss = 0.0
    count = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    ###################
    # train the model #
    ###################

    for data, target in train_loader:

        if switch_ensemble is True:
            with torch.no_grad():
                mask1 = torch.bernoulli(probability * torch.ones(hidden_size, 784)).to(device)
                mask2 = torch.bernoulli(probability * torch.ones(hidden_size, hidden_size)).to(device)
                a = torch.mul(mask1, model.fc1.weight)
                b = torch.mul(mask2, model.fc2.weight)
                c = torch.mul(1 - mask1, model.fc1.weight)
                d = torch.mul(1 - mask2, model.fc2.weight)
                model.fc1.weight.data = a
                model.fc2.weight.data = b

        optimizer.zero_grad()
        output = model(data)


        # calculate the loss
        loss = criterion(output, target)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)

        # print("#1", (model.fc1.weight.sum()))
        optimizer.step() # 마스크 숭숭
        # print("#2", (model.fc1.weight.sum()))
        # update running training loss
        if switch_ensemble is True:
            model.fc1.weight.data = model.fc1.weight.data + c
            model.fc2.weight.data = model.fc2.weight.data + d

        train_loss += loss.item() * data.size(0)

    for data, target in test_loader:
        torch.cuda.empty_cache()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update test loss
        val_loss += loss.item() * data.size(0)
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)
        # compare predictions to true label
        correct = np.squeeze(pred.eq(target.data.view_as(pred)))
        # calculate test accuracy for each object class
        for i in range(10000):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1
    val_acc = 100. * np.sum(class_correct) / np.sum(class_total)

    # print training statistics
    # calculate average loss over an epoch
    train_loss = train_loss / len(train_loader.dataset)
    val_loss = val_loss / len(test_loader.dataset)
    wandb.log({"train_loss": train_loss, "val_loss": val_loss, "val_err: ": 100 - val_acc, "n_epochs": n_epochs})
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch + 1,
        train_loss,
        val_loss
    ))

# initialize lists to monitor test loss and accuracy
test_loss = 0.0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

# model.eval() # prep model for *evaluation*

for data, target in test_loader:
    torch.cuda.empty_cache()
    # forward pass: compute predicted outputs by passing inputs to the model
    output = model(data)
    # calculate the loss
    loss = criterion(output, target)
    # update test loss
    test_loss += loss.item() * data.size(0)
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
test_loss = test_loss / len(test_loader.dataset)
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
wandb.log({"test_err": 100 - test_acc})

torch.save(model.state_dict(), '/home/gidia/anaconda3/envs/myenv/projects/Ensemble_git/models/temp3_model_lr_'
           +str(wandb.config.learning_rate)+'_ep_'+str(wandb.config.n_epochs)+'_p_'+str(wandb.config.probability))