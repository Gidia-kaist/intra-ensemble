from torchvision import datasets
import torchvision.transforms as transforms
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tracemalloc
import wandb

# CUDA 처리
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu") # GPU 사용 가능하면 사용하고 아니면 CPU 사용
print("다음 기기로 학습합니다:", device)

tracemalloc.start(10)

# COMMAND
switch_ensemble = True
probability = 0.7


mask = torch.bernoulli(probability*torch.ones(512, 784))




# for reproducibility
random.seed(777)
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

# Hyperparameters
batch_size = 64
hidden_size = 512
transform = transforms.ToTensor()
train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='data', train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=4)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=4)

wandb.init(project="ensemble_normal", config={"hidden_size": hidden_size, "probability": probability})
hidden_size = wandb.config.hidden_size
probability = wandb.config.probability

## Define the NN architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, hidden_size)

        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # linear layer (n_hidden -> 10)
        self.fc3 = nn.Linear(hidden_size, 10)
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
with torch.autograd.profiler.profile(use_cuda=True) as prof:
    model = Net()




## Specify loss and optimization functions

# specify loss function
criterion = nn.CrossEntropyLoss()

# specify optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# number of epochs to train the model
n_epochs = 50  # suggest training between 20-50 epochs
epoch_counter = 0
model.train()  # prep model for training
temp = model.fc1.weight
temp3 = model.fc1.weight


for epoch in range(n_epochs):
    # monitor training loss
    train_loss = 0.0
    ###################
    # train the model #
    ###################
    '''
        if switch_ensemble is True:
        if epoch % 10 == 0:
            temp2 = model.fc1.weight
            temp = torch.mul(temp, 1 - mask) + torch.mul(temp2, mask)
            model.fc1.weight = torch.nn.Parameter(temp)
            print("MASK CHANGE", epoch_counter)
            mask = np.random.choice([0, 1], size=(512, 784), p=[1-probability, probability])
            mask = torch.tensor(mask)
            print(mask)
            epoch_counter = epoch_counter + 1
    '''

    for data, target in train_loader:
        '''
                if switch_ensemble is True:
                    temp2 = model.fc1.weight
                    temp = torch.mul(temp, 1 - mask) + torch.mul(temp2, mask)
                    model.fc1.weight = torch.nn.Parameter(temp)
                    # print("MASK CHANGE", epoch_counter)
                    mask = np.random.choice([0, 1], size=(512, 784), p=[1 - probability, probability])
                    mask = torch.tensor(mask)
                    drop = model.fc1.weight
                    drop = drop.mul(mask)
                    model.fc1.weight = torch.nn.Parameter(drop)
        '''

        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update running training loss
        train_loss += loss.item() * data.size(0)
        torch.cuda.empty_cache()

        del data, target
    # print training statistics
    # calculate average loss over an epoch
    train_loss = train_loss / len(train_loader.dataset)
    wandb.log({"train_loss": train_loss})
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(
        epoch + 1,
        train_loss
    ))
if switch_ensemble is True:
    temp2 = model.fc1.weight
    temp = torch.mul(temp, 1-mask) + torch.mul(temp2, mask)
    model.fc1.weight = torch.nn.Parameter(temp)
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
    for i in range(batch_size):
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

# calculate and print avg test loss
test_loss = test_loss/len(test_loader.dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))
wandb.log({"test_loss": test_loss})
for i in range(10):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            str(i), 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))