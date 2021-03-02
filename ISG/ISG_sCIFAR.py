import os
import sys
import argparse
import numpy as np
import torch
import json
# from utils.optimizer_utils import *
import utils.model_utils
import utils.train_utils
from utils.train_utils import *
from utils.reg_utils import *
from utils.data_prep import *
from utils.network_utils import *
from utils.result_utils import *


def SIM_CIFAR_train(args, ob, repeat):
    #Initialize model
    model = CNN(args)
    network = utils.model_utils.__dict__[args.method](model, args)
    
    #save paths
    save_path = os.path.join(os.getcwd(),args.out_dir)
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    with open(save_path+'/args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    acc_save_path = os.path.join(save_path,"SIM_acc"+str(args.rho)+".pth" )
    model_save_path = os.path.join(os.getcwd(), "models", "model_pretrained_cifar10.pth")
    # mask_save_path = os.path.join(save_path,"SIM_mask"+str(args.rho)+".pth" )


    #Initialize network with CIFAR10 dataset
    if args.init_model:
        task_num = -1
        trainloader, testloader = load_datasets(args, task_num)
        init_train(network, args, task_num, trainloader, testloader)
        torch.save(network.tmodel.state_dict(), model_save_path)
        
    #Load initialized model
    if args.use_gpu:
        network.tmodel.load_state_dict(torch.load(model_save_path))
    else:
        network.tmodel.load_state_dict(torch.load(model_save_path, map_location=torch.device('cpu')))

    #Training split CIFAR100
    acc_list = {}
    acc_avg_list = []
    for task_num in range(args.num_task):
        #Re-import the dataset associated with the task_num
        trainloader, testloader = load_datasets(args, task_num)
        #Training
        train(network, args, task_num, trainloader, testloader)

        #Inference test on past tasks
        network.tmodel.eval()
        if args.multi_head:
            acc_list[task_num] = []
            for loaded_task in range(task_num + 1):
                network.load_head(loaded_task)
                if args.apply_SIM:
                    network.load_mask(loaded_task)
                _, testloader = load_datasets(args, loaded_task)
                accuracy = test(network, loaded_task, testloader, -1)
                acc_list[task_num].append(accuracy.data.item())
                # acc_list[task_num][loaded_task] = accuracy
                network.lift_mask()
                print("Trained Task:{}, Loaded task:{}, Accuracy:{:.1f}%".format(task_num, loaded_task, accuracy))
            acc_avg = np.around(sum(acc_list[task_num])/len(acc_list[task_num]), 1)
            acc_avg_list.append(acc_avg)
            print("Trained Task:{}, Avg. Accuracy: {:.1f} \n".format(task_num, acc_avg))
        else:
            raise("not implemented yet")

        #Save data to the observer
        print("List of avg. accuracy: {}".format(acc_avg_list))
        ob.ACC.append(acc_avg_list)
        ob.SAT.append(network.SAT)


        # if args.apply_SIM:
            # torch.save(network.task_masks, mask_save_path)

    torch.save(acc_avg_list, acc_save_path)
    

def get_args(argv):
    parser = argparse.ArgumentParser()
    
    #experiment
    parser.add_argument('--use_gpu', type=int, default=0, help="Use_gpu")
    parser.add_argument('--out_dir', type=str, default="outputs/sCIFAR/unscripted", help="output directory")
    parser.add_argument('--repeat', type=int, default=1, help="Repeat the experiment N times")

    #network config
    parser.add_argument('--init_model', type=int, default=0)
    parser.add_argument('--random_drop', type=int, default=0)
    parser.add_argument('--method', type=str, default="MAS", help="CL algorithm (MAS|SI|EWC)")
    parser.add_argument('--model_type', type=str, default='cnn',help="The type (mlp|cnn|lenet|vgg|resnet) of backbone network")
    parser.add_argument('--mlp_size', type=int, default=1000)
    parser.add_argument('--optimizer', type=str, default='Adam', help="SGD|Adam|RMSprop|amsgrad|Adadelta|Adagrad|Adamax ...")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--multi_head', type=bool, default=True)
    parser.add_argument('--momentum', type=float, default=0)
    parser.add_argument('--weight_decay', type=float, default=0)
    # parser.add_argument('--model_weights', type=str, default=None, help="The path to the file for the model weights (*.pth).")
    
    #dataset
    parser.add_argument('--dataset', type=str, default='sCIFAR100', help="pMNIST|CIFAR10|sCIFAR100")
    parser.add_argument('--num_task', type=int, default=10, help="number of tasks")
    parser.add_argument('--schedule', nargs="+", type=int, default=[60, 80], help="The list of epoch numbers to reduce learning rate by factor of 0.1. Last number is the end epoch")
    parser.add_argument('--batch_size_train', type=int, default=128)
    parser.add_argument('--batch_size_test', type=int, default=1000)
    parser.add_argument('--batch_size_fisher', type=int, default=100)
    parser.add_argument('--print_freq', type=float, default=10, help="Print the log at every x iteration")
    parser.add_argument('--dataroot', type=str, default='data', help="The root folder of dataset or downloaded data")
    parser.add_argument('--padding', type=bool, default=True, help="apply padding to input data")
    # parser.add_argument('--workers', type=int, default=3, help="#Thread for dataloader")
    # parser.add_argument('--train_aug', dest='train_aug', default=False, action='store_true',help="Allow data augmentation during training")

    #regularization
    parser.add_argument('--reglambda', type=float, default=1, help="Lambda: regularization strength")
    parser.add_argument('--online_reg', type=bool, default=True, help="Flag for online regularization computation")
    parser.add_argument('--omega_multiplier', type=float, default = 1, help="Determines how fast omega accumulates")

    #SIM related
    parser.add_argument('--apply_SIM', type=int, default=1, help="flag to apply SIM")
    parser.add_argument('--dropmethod', type=str, default="rho", help="Drop method (rho | prob | dist| random_even)")
    # parser.add_argument('--dist_num', type=int, default=1, help="how many hist bins to include for the dist. method")
    parser.add_argument('--inhib', type=float, default=0, help="Print the log at every x iteration")
    parser.add_argument('--rho', nargs="+", type=float, default=[1, 0.4, 0.4], help="ratio of 1 in mask")
    parser.add_argument('--xi', type=float, default=0.1, help="Xi, damping factor to avoid divison by zero")
    parser.add_argument('--alpha', type=float, default=0, help="Alpha, stability-plasticity tradeoff")
    parser.add_argument('--beta', type=float, default=0.9, help="Beta, stability-plasticity tradeoff")


    args = parser.parse_args(argv)
    return args


if __name__ =="__main__":
    args = get_args(sys.argv[1:])
    ob = observer(args) #records all experimental results

    for repeat in range(args.repeat):
        SIM_CIFAR_train(args, ob, repeat)

    ob.to_csv() #saves all csv files
    # ob.plot_results()
