'''
Code from Double Bubble, Toil against Trouble: Enhancing Certified Robustness with Transitivity
By Andrew C. Cullen, Paul Montague, Shijie Liu, Sarah M. Erfani and Benjamin I.P. Rubinstein
'''

from torchvision.models.resnet import ResNet, Bottleneck
from torch.cuda.amp import autocast, GradScaler

from torch.nn.parallel import DistributedDataParallel as DDP

import argparse
import os

import sys

import time
import datetime

import torch
torch.pi = torch.acos(torch.zeros(1)).item() * 2

import numpy as np

from collections import OrderedDict

import torch.backends.cudnn as cudnn

import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


from statsmodels.stats.proportion import proportion_confint as binom_conf
from statsmodels.stats.proportion import multinomial_proportions_confint as multi_conf



torch.manual_seed(0)
import random
random.seed(0)

def grad_phi(E):
    '''
    Analytic calculation of the derivative of Phi^(-1)(x)
    '''
    return np.sqrt(2*np.pi)*torch.exp( ( torch.erfinv(2*E - 1) )**2)
    
def dE_ds(classes, x, x_1, x_2, x_3, sigma, r):
    '''
    Analytic approximation of derivatives
    '''
    N = classes.shape[0]
    
    coeff = -(r / (torch.linalg.norm(x_1 - x_2) * N*sigma**2))
    vector = torch.sum(((x_1 - x_2)*(x-x_3)).reshape(N, -1), dim=1)
        
    return coeff * torch.sum(classes * vector)
          
def multi_conf_wrapper(count_set, alpha=0.05, method='goodman', depth=0):
    '''
    Adjustments to avoid failure modes of statsmodels.stats.proportion.multinomial_proportions_confint
    Note that these adjustments are not designed to guarantee preservation of expectations or the gap between expectations
    Rather the aim is to remove failure modes in a fashion that /disadvantages/ the highest class expectation, thus producing a conservative largest class expectation in the case of failure
    '''
    
    if method == 'sison-glaz':
        start_time = time.time()
        if depth > 7:
            print('Multi Conf count_set: ', count_set)
            raise ValueError('Multi Conf has failed')
        try:
            out = multi_conf(count_set, alpha=alpha, method=method)
            print(time.time() - start_time, depth, '#')
            return out
        except:
            for i in range(4):
                loc_zero = np.argmax(count_set == 0) 
                if (loc_zero > 0):
                    count_set[loc_zero] += 1
                elif (i > 0): 
                    break
            out = multi_conf_wrapper(count_set, alpha=alpha, method=method, depth=depth+1)
            return out        
    elif method == 'goodman':
        count_set_save = np.sort(count_set)
        sorted_indices = np.argsort(count_set) # Low to high
        ix_max, ix_second = sorted_indices[-1], sorted_indices[-2]   
    
        out_temp = multi_conf(count_set, alpha=alpha, method='goodman')    
        less_sum = np.sum(count_set[count_set < 5])
        set_greater = count_set[count_set >= 5]
        len_set_smaller = int(np.max((1, np.floor(less_sum/5))))
        if len_set_smaller > 1:
            len_difference = less_sum%5
        else:
            len_difference = 0

        count_set = np.zeros(len(set_greater) + len_set_smaller)
        count_set[:len_set_smaller] = 5
        count_set[0] = 5 + len_difference
        count_set[len_set_smaller:] = set_greater
        out_temp = multi_conf(np.sort(count_set), alpha=alpha, method='goodman')
        out = np.zeros((len(sorted_indices), 2))
        #out[ix_max, 0] = out_temp[-1, 0]
        #out[ix_second, 1] = out_temp[-2, 1] 
        out[:, 0] = out_temp[-1, 0]
        out[:, 1] = out_temp[-2, 1] 
        

        return out
    elif method == 'goodman-truncated': # Goodman truncated
        reference_out = multi_conf(count_set, alpha=alpha, method='goodman')
        sorted_indices = np.argsort(count_set) # Low to high
        ix_max, ix_second = sorted_indices[-1], sorted_indices[-2]
                
        count_set_l_5 = count_set <= 5
        sum_subset = np.sum(count_set[count_set_l_5])
        if np.sum(sum_subset < 5):
            sum_subset = 5
        count_set_new = np.zeros(len(count_set) - np.sum(count_set_l_5) + 1)
        count_set_new[0] = sum_subset
        count_set_new[1:] = count_set[~count_set_l_5]
        out = multi_conf(np.sort(count_set_new), alpha=alpha, method='goodman')
        out2 = multi_conf(np.sort(count_set), alpha=alpha, method='goodman')               
        
        reframed_out = np.zeros((len(sorted_indices),2))

        reframed_out[:, 0] = out[-1, 0]
        reframed_out[:, 1] = out[-2, 1]
        

        return reframed_out                
    elif method == 'goodman-short':
        sorted_indices = np.argsort(count_set) # Low to high
        ix_max, ix_second = sorted_indices[-1], sorted_indices[-2]
        sorted_set = count_set[sorted_indices] # Low to high
        cumsum_sorted = np.cumsum(sorted_set)
        offset = -3
        if sorted_set[offset] < 5:
            count = cumsum_sorted[offset] #np.sum(sorted_set[:(offset + 1)] # Commented part is equivalent
            count_set = np.zeros(-1*offset)
            count_set[0] = count
            count_set[1:] = sorted_set[-2:]
            count_set = np.sort(count_set)
            count_set[count_set < 5] = 5
            out = multi_conf(count_set, alpha=alpha, method='goodman')
        else:  
            ix = np.argmin(cumsum_sorted < sorted_set[offset])
            count_set = np.zeros(len(count_set) - ix + 1)
            count_set[0] = cumsum_sorted[ix - 1]        
            count_set[1:] = sorted_set[ix:]
            count_set = np.sort(count_set)
            if count_set[0] < 5:
                count_set[count_set < 5] = 5
            out = multi_conf(count_set, alpha=alpha, method='goodman')
                        
        reframed_out = np.zeros((len(sorted_indices),2))
        reframed_out[ix_max, 0] = out[-1, 0]
        reframed_out[ix_second, 1] = out[-2, 1]

            
        return reframed_out
    else: 
        return ValueError('Technique Not Implemented')
        
        
        
        
            
        
        
def return_wrapper(model, inpt, shape, validate=False):
    '''
    Evaluates model(inpt) over a range of monte carlo samples, using either argmax or gumbel_softmax (as a differentiable approximation to the argmax)
    '''

    device = inpt.device
    samples = inpt.shape[0]

    if len(shape) == 4:
        inpt = inpt.reshape(samples, shape[1], shape[2], shape[3])    
    else:
        inpt = inpt.reshape(samples, shape[0], shape[1], shape[2])  

    flag = False    
    if (inpt.shape[1] == 1) and (len(shape) == 4):
        flag = True
        inpt = inpt.repeat(1, 3, 1, 1)
    
    if validate:
        direct_out = model(inpt)       
        ix = torch.argmax(direct_out, dim=1)
        indices, counts = torch.unique(ix, return_counts=True)
        return indices, counts, ix
    else:
        gs = F.gumbel_softmax(100*model(inpt), tau=1, hard=False, dim=1) # wrapper_module(model, inpt)-
        x = inpt.detach().clone()        
        sf = torch.sum(gs, dim=0).reshape(1, -1)
        if flag:
            x = x[:,0,:,:].unsqueeze(1)
        return (sf[0,:] / samples).to(device), x, (torch.argmax(gs, dim=1)).to(device)
        
def cohen_calculator(E0, E1, sigma, uncertainties=None, alpha=None, binary=False, expectation_set=None, max_class=None, second_class=None, return_expectations=False):
    '''
    Calculates the certification and class expectations (including adjustments for confidence intervals) as per Cohen, using the multiclass formulation
    '''

    if (binary is True) and (uncertainties is not None):
        E0_temp = torch.tensor(binom_conf(int((E0*uncertainties).detach().cpu().numpy()), uncertainties, alpha=alpha, method='beta')[0])
        E1_temp = torch.tensor(binom_conf(int((E1*uncertainties).detach().cpu().numpy()), uncertainties, alpha=alpha, method='beta')[1])        
        delta_E0 = E0.detach().clone() - E0_temp
        delta_E1 = E1.detach().clone() - E1_temp
        E0 -= delta_E0
        E1 += delta_E1
    elif (binary is False):
        count_set = np.round((expectation_set*uncertainties).detach().cpu().numpy())             
        
        expectation_set = multi_conf_wrapper(count_set, alpha=alpha, method='goodman')
        #E0 = torch.tensor(expectation_set[max_class, 0]) 
        #E1 = torch.tensor(expectation_set[second_class, 1])
        E0_temp_save = E0.detach().clone() # Just for the moment
        E1_temp_save = E1.detach().clone() # Just for the moment
        delta_E0 = E0.detach().clone() - torch.tensor(expectation_set[max_class, 0]) 
        E0 -= delta_E0
        
        delta_E1 = -1.*E1.detach().clone() + torch.tensor(expectation_set[second_class, 1])
        E1 += delta_E1
                
        if (E0 > E0_temp_save) or (E1 < E1_temp_save):
            print(max_class, second_class, '*'*20, flush=True)
            print(np.argmax(expectation_set, axis=0), flush=True)
            print('E0 before: {}, after: {}, delta: {}, E1 before: {}, after: {}, delta: {}'.format(E0_temp_save, E0, delta_E0, E1_temp_save, E1, delta_E1), flush=True)
            raise ValueError('Something has gone wrong with passing the derivatives around')
        

    output =  (sigma / 2.) * (np.sqrt(2.)*(torch.erfinv(2*E0 - 1) - torch.erfinv(2*E1 - 1))) 

    if return_expectations:
        return output, E0, E1

    return output
        
        
def extract_classes(model, x, samples, shape, sigma, binary=False):
    '''
    Identify two classes with the largest expectations
    '''

    if len(shape) == 4:
        x = x.reshape(samples, shape[1], shape[2], shape[3])    
    else:
        x = x.reshape(samples, shape[0], shape[1], shape[2])  


    expectation_set, x, pred_classes = return_wrapper(model, x, shape)
    vals, indices = torch.topk(expectation_set, 2, sorted=True)

    max_class, second_class = indices[0], indices[1] 
    x = x.reshape(samples, -1)   

    E0, E1 = vals[0], vals[1] #vals[0]
    
    if binary:
        class0 = 1.*(pred_classes == max_class)
        class1 = 1 - class0
        E1 = 1 - E0
    else:

        class0, class1 = 1.*(pred_classes == max_class), 1.*(pred_classes == second_class)
    cohen_val = cohen_calculator(E0, E1, sigma, alpha=0.05, uncertainties=samples, binary=binary, expectation_set=expectation_set, max_class=max_class, second_class=second_class)        
    return cohen_val, E0, E1, class0, class1, max_class, second_class, expectation_set
