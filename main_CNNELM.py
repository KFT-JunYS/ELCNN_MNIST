from __future__ import print_function
import argparse
import torch.utils.data.dataloader
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable
import time
from pseudoInverse import pseudoInverse
import numpy as np

# Training settings
parser = argparse.ArgumentParser(description='PyTorch ELM MNIST Example')
parser.add_argument('--batch-size', type=int, default=60000, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 1)')

args = parser.parse_args()
torch.manual_seed(args.seed)

print('Start!!')

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0,), (1,))
                   ])),
    batch_size=args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0,), (1,))
                   ])),
    batch_size=args.batch_size, shuffle=True)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5, padding=1)
        self.conv2 = nn.Conv2d(10, 450, kernel_size=4, padding=1)
        self.fc2 = nn.Linear(1000, 10, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x,kernel_size=2)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x,kernel_size=2)
        x = F.relu(x)

        x = x.view(-1, self.num_flat_features(x))

        x = self.fc2(x)

        return x

    def forwardToHidden(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x,kernel_size=2)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x,kernel_size=2)
        x = F.relu(x)

        print('features = ', self.num_flat_features(x))

        x = x.view(-1, self.num_flat_features(x))

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        print('size = ', size)
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

model = Net()
optimizer= pseudoInverse(params=model.parameters(),C=1e-5)


def train():
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)

        print('shape of image = ', data.size())
        hiddenOut = model.forwardToHidden(data)
        optimizer.train(inputs=hiddenOut, targets=target)


def train_accuracy():
    model.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        output = model.forward(data)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()
    ending = time.time()
    print('training time: {:.2f}sec'.format(ending - init))
    print('\nTrain set accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))


def test():
    model.eval()
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        output = model.forward(data)
        pred=output.data.max(1)[1]
        #print(pred)
        correct += pred.eq(target.data).cpu().sum()
    print('\nTest set accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


init=time.time()
train()
train_accuracy()
test()
ending=time.time()

print(ending-init)