import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import os
import shutil
import sys
from utils.optimizer_utils import *
from utils.data_prep import *



class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(4096, 512)
        self.fc2 = nn.Linear(512, 10)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1) 
        x = self.fc1(x)
        x = self.fc2(x)
        return x

def test(self, net, testloader, epoch):
	test_losses = []
	test_loss = 0
	correct = 0

	# with torch.no_grad(): # we don't need Autograd for testing. commented out for l2_grad computation.
	for batch_idx, (data, target) in enumerate(testloader):
		data = data.cuda()
		target = target.cuda()
		output = net.forward(data)
		pred = output.data.max(1, keepdim=True)[1] #prediction
		correct += pred.eq(target.data.view_as(pred)).sum() #view_as --> use same dim?
	test_loss /= len(testloader.dataset)
	test_losses.append(test_loss)
	accuracy = 100.*correct/len(testloader.dataset)

	print('Epoch: {}, \t Avg.loss: {:.4f},\t Test Accuracy: {}/{}({:.0f}%)'.format(epoch, test_loss, correct, len(testloader.dataset), accuracy))

	return accuracy

#Load datasets
transform_train = transforms.Compose([
	transforms.RandomCrop(32, padding=4),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR100(root="../../data", train=True, download=False, transform=transform_train)
testset = torchvision.datasets.CIFAR100(root="../../data", train=False, download=False, transform=transform_test)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size_train, shuffle=True, num_workers=2)
testloader  = torch.utils.data.DataLoader(testset, batch_size=args.batch_size_test, shuffle=False, num_workers=2)


net = Net()
net.cuda()
optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=0)
for epoch in range(80):
	for batch_idx, (data, target) in enumerate(trainloader):
		data = data.cuda()
		target = target.cuda()
		output = net.forward(data)
		loss = nn.CrossEntropyLoss()
		optimizer.zero_grad()
		loss.backward()
		acc = test(net, testloader, epoch)
