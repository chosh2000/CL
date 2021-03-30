import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
	def __init__(self, args, in_channel=1, img_size=28, out_dim=10):
		super(MLP, self).__init__()
		self.device = torch.device("cuda:0" if args.use_gpu else "cpu")
		#Dimensions
		if args.padding:
			img_size = 32
		self.input_dim = in_channel*img_size*img_size
		self.hidden_dim = args.mlp_size
		self.out_dim = out_dim

		#Parameters
		self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
		self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
		self.fc_head = nn.Linear(self.hidden_dim, out_dim)

		#Masks
		self.rho = {}
		self.fc1_mask = torch.ones(self.hidden_dim).to(self.device)
		self.fc2_mask = torch.ones(self.hidden_dim).to(self.device)
		self.mask_list = {'fc1.weight': self.fc1_mask, 'fc2.weight': self.fc2_mask}
		assert len(args.rho) == len(self.mask_list), "Number of masks must match rho size"
		for i, key in enumerate(self.mask_list.keys()):
			self.rho[key] = args.rho[i]

	def forward(self, x):
		# x = x.view(x.size(0), -1)
		x = x.view(-1, self.input_dim)
		x = F.relu(self.fc1(x))
		x = x * self.fc1_mask
		x = F.relu(self.fc2(x))
		x = x * self.fc2_mask
		x = self.fc_head(x)
		return x

	def SIM_forward(self, x):
		x = x.view(x.size(0), -1)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		# does not have the head layer
		return x



class CNN(nn.Module):
	def __init__(self, args):
		super(CNN, self).__init__()
		self.args = args
		self.device = torch.device("cuda:0" if args.use_gpu else "cpu")
		self.conv1_feat = 32
		self.conv2_feat = 32
		self.conv3_feat = 64
		self.conv4_feat = 64

		self.conv1_mask = torch.ones([self.conv1_feat, 32, 32]).to(self.device)
		self.conv2_mask = torch.ones([self.conv2_feat, 16, 16]).to(self.device)
		self.conv3_mask = torch.ones([self.conv3_feat, 16, 16]).to(self.device)
		self.conv4_mask = torch.ones([self.conv4_feat, 8, 8]).to(self.device)

		self.fc1_mask = torch.ones(512).to(self.device)
		self.rho = {}
		self.mask_list = {
							'conv1_layer.0.weight':self.conv1_mask, 
							'conv2_layer.0.weight':self.conv2_mask,
							'conv3_layer.0.weight':self.conv3_mask,
							'conv4_layer.0.weight':self.conv4_mask,
							'fc1_layer.0.weight'  :self.fc1_mask,
						}
		assert len(self.args.rho) == len(self.mask_list), "make sure mask number matches"
		for i, key in enumerate(self.mask_list.keys()):
			self.rho[key] = self.args.rho[i]

		# Conv Layer block 1
		self.conv1_layer = nn.Sequential(
			nn.Conv2d(in_channels=3, out_channels=self.conv1_feat, kernel_size=3, padding=1),
			# nn.BatchNorm2d(32),
			nn.ReLU(),
			)

		self.conv2_layer = nn.Sequential(
			nn.Conv2d(in_channels=self.conv1_feat, out_channels=self.conv2_feat, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2),
			)
		
		# Conv Layer block 2
		self.conv3_layer = nn.Sequential(
			nn.Conv2d(in_channels=self.conv2_feat, out_channels=self.conv3_feat, kernel_size=3, padding=1),
			# nn.BatchNorm2d(64),
			nn.ReLU(),
			)

		self.conv4_layer = nn.Sequential(
			nn.Conv2d(in_channels=self.conv3_feat, out_channels=self.conv4_feat, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2),
			# nn.Dropout2d(p=0.05),
			)

		# FC layer
		self.fc1_layer = nn.Sequential(
			# nn.Dropout(p=0.1),   ############## commented out for nodropout
			nn.Linear(self.conv2_mask.numel(), 512),  #nn.Linear(4096,1024) with 3 conv layers
			nn.ReLU(),
			)

		# Head layer
		self.fc_head = nn.Linear(512, 10)

	def forward(self, x):
		# conv layers
		x = self.conv1_layer(x)
		x = x * self.conv1_mask
		x = self.conv2_layer(x)
		x = x * self.conv2_mask
		x = self.conv3_layer(x)
		x = x * self.conv3_mask
		x = self.conv4_layer(x)
		x = x * self.conv4_mask

		# fc layer
		x = x.view(x.size(0), -1) # output: [channel, 8192] or [batch_size, 4096]
		x = self.fc1_layer(x) # output: [batch_size, 1024]
		print(x.shape)
		x = x * self.fc1_mask

		# head layer
		x = self.fc_head(x)

		return x

	def SIM_forward(self, x):
		x = self.conv1_layer(x)
		x = self.conv2_layer(x)
		x = self.conv3_layer(x)
		x = self.conv4_layer(x)
		x = x.view(x.size(0), -1)
		x = self.fc1_layer(x)
		# x = self.fc_head(x) # does not have the head layer
		return x

