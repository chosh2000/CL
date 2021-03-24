import os
import sys
import argparse
import numpy as np
import torch
import json
# from utils.optimizer_utils import *
# import utils.model_utils
# import utils.train_utils
import utils
from utils.train_utils import *
from utils.reg_utils import *
from utils.data_prep import *
from utils.network_utils import *
from utils.result_utils import *


def SIM_train(args, ob):
    #Initialize model
    # model = CNN(args)
    model   = utils.network_utils.__dict__[args.model_type](args)
    model.cuda()
    network = utils.model_utils.__dict__[args.method](model, args, ob)
    
    #save paths
    save_path = os.path.join(os.getcwd(),args.out_dir)
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    with open(save_path+'/args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    if args.model_type == 'CNN':
        model_save_path = os.path.join(os.getcwd(), "models", "model_pretrained_cifar10.pth")
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
        acc_list[task_num] = []
        BWT = 0
        for loaded_task in range(task_num + 1):
            network.load_head(loaded_task)
            if args.apply_SIM:
                network.load_mask(loaded_task)
            trainloader, testloader = load_datasets(args, loaded_task)
            accuracy = network.test(loaded_task, testloader, -1)
            acc_list[task_num].append(accuracy.data.item())
            print("Trained Task:{}, Loaded task:{}, Accuracy:{:.1f}%".format(task_num, loaded_task, accuracy))
            #Added to BWT calculation
            if loaded_task < task_num:
                # print("Backward (BWT) ", end='')
                # acc_bwt  = network.finetune_head(loaded_task, trainloader, testloader)
                BWT     += accuracy / network.Rii[loaded_task] -1
            network.lift_mask()

        network.BWT.append(BWT)
        #Information Packing (IPK)
        ipk = (1-network.FWT[-1]+network.BWT[-1])/(network.SAT[-1]/network.PTB[-1])
        network.IPK.append(ipk)
        acc_avg = np.around(sum(acc_list[task_num])/len(acc_list[task_num]), 1)
        acc_avg_list.append(acc_avg)
        print("Trained Task:{}, Avg. Accuracy: {:.1f} \n".format(task_num, acc_avg))
        print("List of avg. accuracy: {}".format(acc_avg_list))


    #Save data to the observer
    ob.ACC.append(acc_avg_list)
    ob.FWT.append(network.FWT)
    ob.BWT.append(network.BWT)
    ob.SAT.append(network.SAT)
    ob.PTB.append(network.PTB)
    ob.IPK.append(network.IPK)



def get_args(argv):
    parser = argparse.ArgumentParser()
    
    #experiment
    parser.add_argument('--use_gpu', type=int, default=1, help="Use_gpu")
    parser.add_argument('--out_dir', type=str, default="outputs/sCIFAR/unscripted", help="output directory")
    parser.add_argument('--repeat', type=int, default=1, help="Repeat the experiment N times")
    parser.add_argument('--finetune_epoch', type=int, default=20, help="defines the number of epochs used for finetuning the head")
    parser.add_argument('--revert_head', type=int, default = 1, help="revert finetuning of headlayer during FWT computation ")

    #network config
    parser.add_argument('--init_model', type=int, default=0)
    parser.add_argument('--random_drop', type=int, default=0)
    parser.add_argument('--method', type=str, default="MAS", help="CL algorithm (MAS|SI|EWC)")
    parser.add_argument('--model_type', type=str, default='CNN',help="The type (MLP|CNN|lenet|vgg|resnet) of backbone network")
    parser.add_argument('--mlp_size', type=int, default=100)
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
    parser.add_argument('--apply_SIM', type=int, default=0, help="flag to apply SIM")
    parser.add_argument('--dropmethod', type=str, default="rho", help="Drop method (rho | prob | dist| random_even)")
    parser.add_argument('--rho', nargs="+", type=float, default=[1, 0.4, 0.4], help="ratio of 1 in mask")
    parser.add_argument('--xi', type=float, default=1, help="Xi, damping factor to avoid divison by zero")
    parser.add_argument('--alpha', type=float, default=1, help="Alpha, stability-plasticity tradeoff")
    # parser.add_argument('--dist_num', type=int, default=1, help="how many hist bins to include for the dist. method")
    # parser.add_argument('--inhib', type=float, default=0, help="Print the log at every x iteration")
    # parser.add_argument('--beta', type=float, default=0.9, help="Beta, stability-plasticity tradeoff")


    args = parser.parse_args(argv)
    return args


if __name__ =="__main__":
    args = get_args(sys.argv[1:])
    ob = observer(args) #records all experimental results

    for repeat in range(args.repeat):
        SIM_train(args, ob)
        ob.to_csv() #saves all csv files

    # ob.plot_results()
