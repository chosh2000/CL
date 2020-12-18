import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import autograd
import numpy as np
from torch.utils.data import DataLoader
from utils.data_prep import *



def EWC_update_fisher_params(network, current_ds, use_gpu, shuffle_idx, task_num, batch_size, num_batch):
    dl = DataLoader(current_ds, batch_size, shuffle=True)
    log_liklihoods = []
    device = torch.device("cuda:0" if use_gpu else "cpu")

    for i, (data, target) in enumerate(dl):
        if i > num_batch:
            break
        data = permute_MNIST(data, shuffle_idx, task_num)
        data = data.to(device)
        target = target.to(device)
        
        output = F.log_softmax(network.tmodel(data), dim=1)
        log_liklihoods.append(output[:, target])
    log_likelihood = torch.cat(log_liklihoods).mean()
    grad_log_liklihood = autograd.grad(log_likelihood, network.tmodel.parameters())
    for i, (name, param) in enumerate(network.tmodel.named_parameters()):
        network.reg_params[i]['omega'] = grad_log_liklihood[i].data.clone() ** 2
    return network


class ElasticWeightConsolidation:

    def __init__(self, model, criteria, optimizer, lr=0.001, weight=1000000):
        self.model = model
        self.weight = weight
        self.criteria = criteria
        # self.optimizer = optim.Adam(self.model.parameters(), lr)
        self.optimizer = optimizer
        self.a = a

    def _update_mean_params(self):
        for param_name, param in self.model.named_parameters():
            _buff_param_name = param_name.replace('.', '__')
            self.model.register_buffer(_buff_param_name+'_estimated_mean', param.data.clone())

    def _update_fisher_params(self, current_ds, batch_size, num_batch):
        dl = DataLoader(current_ds, batch_size, shuffle=True)
        log_liklihoods = []
        for i, (input, target) in enumerate(dl):
            if i > num_batch:
                break
            output = F.log_softmax(self.model(input), dim=1)
            log_liklihoods.append(output[:, target])
        log_likelihood = torch.cat(log_liklihoods).mean()
        grad_log_liklihood = autograd.grad(log_likelihood, self.model.parameters())
        _buff_param_names = [param[0].replace('.', '__') for param in self.model.named_parameters()]
        for _buff_param_name, param in zip(_buff_param_names, grad_log_liklihood):
            self.model.register_buffer(_buff_param_name+'_estimated_fisher', param.data.clone() ** 2)

    def register_ewc_params(self, dataset, batch_size, num_batches):
        self._update_fisher_params(dataset, batch_size, num_batches)
        self._update_mean_params()

    def _compute_consolidation_loss(self, weight):
        try:
            losses = []
            for param_name, param in self.model.named_parameters():
                _buff_param_name = param_name.replace('.', '__')
                estimated_mean = getattr(self.model, '{}_estimated_mean'.format(_buff_param_name))
                estimated_fisher = getattr(self.model, '{}_estimated_fisher'.format(_buff_param_name))
                losses.append((estimated_fisher * (param - estimated_mean) ** 2).sum())
            return (weight / 2) * sum(losses)
        except AttributeError:
            return 0

    def forward_backward_update(self, input, target):
        output = self.model(input)
        loss = self.criteria(output, target)
        # loss = self._compute_consolidation_loss(self.weight) + self.criteria(output, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, filename):
        torch.save(self.model, filename)

    def load(self, filename):
        self.model = torch.load(filename)
