import os
import sys
import argparse
import numpy as np
import torch
import json
# from utils.optimizer_utils import *
from utils.train_utils import *
from utils.reg_utils import *
from utils.model_utils import *
from utils.data_prep import *
from utils.network_utils import *



def SIM_CIFAR_train(args):
    #Initialize model
    model = CNN(args)
    network = shared_model(model, args)
    acc_list = {}

    #save paths
    save_path = os.path.join(os.getcwd(),args.out_dir)
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    with open(save_path+'/args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    acc_save_path = os.path.join(save_path,"SIM_acc"+str(args.rho)+".pth" )
    mask_save_path = os.path.join(save_path,"SIM_mask"+str(args.rho)+".pth" )
    stat_save_path = os.path.join(save_path,"SIM_stat"+str(args.rho)+".pth" )
    # param_save_path = os.path.join(save_path,"SIM_param"+str(args.rho)+".pth" )
    # reg_save_path = os.path.join(save_path,"SIM_reg"+str(args.rho)+".pth" )

    #Initialize network with CIFAR10 dataset
    task_num = -1
    trainloader, testloader = load_datasets(args, task_num)
    init_train(network, args, task_num, trainloader, testloader)

    #Training split CIFAR100
    for task_num in range(args.num_task):
        #Re-import the dataset associated with the task_num
        trainloader, testloader = load_datasets(args, task_num)
        #Training
        train_SIM(network, args, task_num, trainloader, testloader)

        #Inference test on past tasks
        if args.multi_head:
            acc_list[task_num] = {}
            for loaded_task in range(task_num + 1):
                network.tmodel.eval() #TODO: is this line necessary
                network.load_head(loaded_task)
                network.load_mask(loaded_task)
                _, testloader = load_datasets(args, loaded_task)
                accuracy = test(network, loaded_task, testloader, -1)
                acc_list[task_num][loaded_task] = accuracy
                network.lift_mask()
                print("Trained Task:{}, Loaded task:{}, Accuracy:{:.1f}%".format(task_num, loaded_task, accuracy))
        else:
            raise("not implemented yet")
        #Save data
        torch.save(acc_list, acc_save_path)
        torch.save(network.task_masks, mask_save_path)
        torch.save(network.stat, stat_save_path)
        # torch.save(network.params, param_save_path)
        # torch.save(network.reg_params, reg_save_path)


def get_args(argv):
    # TO read from a file
    # parser = ArgumentParser()
    # args = parser.parse_args()
    # with open('commandline_args.txt', 'r') as f:
    #     args.__dict__ = json.load(f)

    # This function prepares the variables shared across demo.py
    parser = argparse.ArgumentParser()
    
    #experiment
    parser.add_argument('--use_gpu', type=bool, default=True, help="Use_gpu")
    parser.add_argument('--out_dir', type=str, default="outputs/sCIFAR/unscripted", help="output directory")
    parser.add_argument('--repeat', type=int, default=1, help="Repeat the experiment N times")

    #network config
    parser.add_argument('--method', type=str, default="MAS", help="Continual Learning Algorithm used")
    parser.add_argument('--model_type', type=str, default='cnn',help="The type (mlp|cnn|lenet|vgg|resnet) of backbone network")
    parser.add_argument('--mlp_size', type=int, default=1000)
    parser.add_argument('--optimizer', type=str, default='Adam', help="SGD|Adam|RMSprop|amsgrad|Adadelta|Adagrad|Adamax ...")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--multi_head', type=bool, default=True)
    parser.add_argument('--momentum', type=float, default=0)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--model_weights', type=str, default=None, help="The path to the file for the model weights (*.pth).")
    
    #dataset
    parser.add_argument('--dataset', type=str, default='sCIFAR100', help="pMNIST|CIFAR10|sCIFAR100")
    parser.add_argument('--num_task', type=int, default=10, help="number of tasks")
    parser.add_argument('--schedule', nargs="+", type=int, default=[40], help="The list of epoch numbers to reduce learning rate by factor of 0.1. Last number is the end epoch")
    parser.add_argument('--batch_size_train', type=int, default=128)
    parser.add_argument('--batch_size_test', type=int, default=1000)
    parser.add_argument('--batch_size_fisher', type=int, default=100)
    parser.add_argument('--print_freq', type=float, default=10, help="Print the log at every x iteration")
    parser.add_argument('--workers', type=int, default=3, help="#Thread for dataloader")
    parser.add_argument('--train_aug', dest='train_aug', default=False, action='store_true',help="Allow data augmentation during training")
    parser.add_argument('--dataroot', type=str, default='data', help="The root folder of dataset or downloaded data")
    parser.add_argument('--padding', type=bool, default=True, help="apply padding to input data")

    #regularization
    parser.add_argument('--reglambda', type=float, default=0, help="Lambda: regularization strength")
    parser.add_argument('--online_reg', type=bool, default=True, help="Flag for online regularization computation")
    parser.add_argument('--omega_multiplier', type=float, default = 1, help="Determines how fast omega accumulates")

    #masking
    parser.add_argument('--rho', nargs="+", type=float, default=[1, 0.5, 0.5], help="ratio of 1 in mask")
    parser.add_argument('--xi', type=float, default=0.001, help="Xi, damping factor to avoid divison by zero")
    parser.add_argument('--alpha', type=float, default=0.5, help="Alpha, stability-plasticity tradeoff")

    args = parser.parse_args(argv)
    return args


if __name__ =="__main__":
    args = get_args(sys.argv[1:])
    SIM_CIFAR_train(args)
    # plot_result(num_task, save_path)
    # clear_models()
