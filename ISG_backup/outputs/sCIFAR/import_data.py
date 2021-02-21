import os
import numpy as np
import matplotlib.pyplot as plt
import torch

# path = "/home/sanghyun/Documents/CL/ISG/outputs/sCIFAR/Dec_16_20h_50m/"
path = "/home/sanghyun/Documents/CL/ISG/outputs/sCIFAR/unscripted/"

data = torch.load(path+"SIM_mask[0.5, 0.5, 0.5].pth", map_location=torch.device('cpu'))
for task_num, mask_list in data.items():
	print("\n\n task number: ",task_num)
	for layer, mask in mask_list.items():
		print("layer     : ",layer)
		print("mask.shape: ",mask.shape)
		print("mask.numel: ",mask.numel())
		print("mask.sum  : ",mask.sum())
