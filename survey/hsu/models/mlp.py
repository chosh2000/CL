import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP_drop(nn.Module):
    def __init__(self, out_dim=10, in_channel=1, img_sz=32, hidden_dim=256):
        super(MLP_drop, self).__init__()
        #dimensions
        self.in_dim = in_channel*img_sz*img_sz
        self.hidden_dim = hidden_dim
        #Layer
        self.fc1 = nn.Linear(self.in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, out_dim)  # Subject to be replaced dependent on task
        #masks
        self.fc1_mask = torch.ones(hidden_dim).cuda()
        self.fc2_mask = torch.ones(hidden_dim).cuda()
        self.mask_list = {'fc1.weight': self.fc1_mask, 'fc2.weight':self.fc2_mask}
        self.rho = {'fc1.weight': 1.0, 'fc2.weight': 1.0}

    def forward(self, x):
        x = x.view(-1,self.in_dim)
        x = F.relu(self.fc1(x))
        x = x*self.fc1_mask
        x = F.relu(self.fc2(x))
        x = x*self.fc2_mask
        x = self.head(x)
        return x


class MLP(nn.Module):
    def __init__(self, out_dim=10, in_channel=1, img_sz=32, hidden_dim=256):
        super(MLP, self).__init__()
        self.in_dim = in_channel*img_sz*img_sz
        self.linear = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim),
            #nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            #nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.last = nn.Linear(hidden_dim, out_dim)  # Subject to be replaced dependent on task
    def features(self, x):
        x = self.linear(x.view(-1,self.in_dim))
        return x
    def logits(self, x):
        x = self.last(x)
        return x
    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x


def MLP100():
    return MLP(hidden_dim=100)


def MLP400():
    return MLP(hidden_dim=400)


def MLP1000():
    return MLP(hidden_dim=1000)


def MLP2000():
    return MLP(hidden_dim=2000)


def MLP5000():
    return MLP(hidden_dim=5000)


def MLP1000_drop():
    return MLP_drop(hidden_dim=1000)


def MLP2000_drop():
    return MLP_drop(hidden_dim=2000)

