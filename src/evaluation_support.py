'''
Code from Double Bubble, Toil against Trouble: Enhancing Certified Robustness with Transitivity
By Andrew C. Cullen, Paul Montague, Shijie Liu, Sarah M. Erfani and Benjamin I.P. Rubinstein
'''

import argparse
import os
import sys

import time
import datetime

import torch
torch.pi = torch.acos(torch.zeros(1)) * 2. #.item() * 2

import numpy as np

import torch.backends.cudnn as cudnn

import torch

from evaluation_common import extract_classes, cohen_calculator, grad_phi, dE_ds 

torch.manual_seed(0)
import random
random.seed(0)    

def sampler(y, y_original, samples, sigma, label, model, device, shape, approximate_gradients=True, approximate_beta=False, binary=False):
    '''
    Code for taking a single gradient based step for identifying complimentary certification points
    '''

    y = y.detach().clone()

    if approximate_gradients is False:
        y.requires_grad = True
    if y.shape[0] != samples:
        x = y.repeat(samples, 1, 1, 1)
    
    if len(shape) == 4:
        x = x.reshape(samples, shape[1], shape[2], shape[3])    
    else:
        x = x.reshape(samples, shape[0], shape[1], shape[2])      
    

    noise = torch.randn_like(x) * sigma
    x += noise
        
    cohen_val, E0, E1, class0, class1, max_class, second_class, expectation_classes = extract_classes(model, x, samples, shape, sigma, binary=binary)

    if max_class == label:
        flag = True
        if approximate_gradients:
            y = y.reshape(-1)
            x = x.reshape(samples, -1) 
            noise = noise.reshape(samples, -1)               
            cohen_grad = (np.sqrt(np.pi) / (np.sqrt(2)*sigma*samples)) * (torch.sum(class0[:, None] * noise, dim=0) / inv_cdf_grad(E0) - torch.sum(class1[:,None] * noise, dim=0)/ inv_cdf_grad(E1))
        else:
            cohen_grad = torch.autograd.grad(cohen_val, y)[0]
        cohen_grad = cohen_grad.reshape(y_original.shape)
    else:
        flag = False
        cohen_grad = torch.tensor(0.)
                
    y = y.reshape(y_original.shape) # Oct 7
    delta_y_norm = torch.linalg.norm(y - y_original) # This is suboptimal
    diff_grad = -(y - y_original) / delta_y_norm 
    
    objective_val = cohen_val - delta_y_norm
                  
    combined_grad = cohen_grad + diff_grad
    combined_grad = combined_grad 
                  
    return objective_val, flag, E0, E1, cohen_val, combined_grad, max_class, second_class, expectation_classes    
    
def inv_cdf_grad(expectation):
    '''
    Evaluating analytic derivative of Phi^(-1)(x)
    '''
    phi_inverse = np.sqrt(2.)*torch.erfinv(2*expectation - 1)
    phi_prime = torch.exp(-0.5*(phi_inverse**2))
    
    return phi_prime
    

def secondary_ball_simple_estimator(y_centre, y_offset, r_adjusted, r_2, samples, sigma, model, iterations, alpha, device, shape, gamma_fixed=False, binary=False, gamma_start=0.01, autograd=True):
    '''
    Calculates update steps for finding the second supplementary certification region
    '''

    offset = torch.tensor(0.9)
    offset.requires_grad = True
    adjustment_vector = r_adjusted*((y_centre - y_offset) / torch.linalg.norm(y_centre - y_offset))
    adjustment_vector.requires_grad = False
    
    d_prime_max = None
    offset_max = torch.tensor(0.)
    r_prime_max = torch.tensor(0.)

    gamma = gamma_start
    
    a = torch.linalg.norm(y_centre - y_offset)
    
    offset_old, grad_old, grad = None, None, None
    best_iter = -1
    for ix in range(iterations):
        yp = y_centre + offset * adjustment_vector
        d = torch.linalg.norm(yp - y_centre)
        adv_input = yp.repeat(samples, 1, 1, 1)
        adv_input += torch.randn_like(adv_input) * sigma
        if (torch.sum(torch.isnan(adv_input)) > 0):
            if (d_prime_max is not None):
                return d_prime_max.cpu(), best_iter  
            else:
                return torch.tensor(0.), best_iter
                
        r_3, E_0, E_1, classes_0, classes_1, _, _, _ = extract_classes(model, adv_input, samples, shape, sigma, binary=False)
        
        if torch.isnan(r_3): 
            if (d_prime_max is not None):
                return d_prime_max.cpu(), best_iter      
            else:
                return torch.tensor(0.), best_iter
                
        if autograd:
            if (r_3 + torch.linalg.norm(yp - y_offset)) < r_2: # Contained circle, no intersection
                quantity = r_3 + torch.linalg.norm(yp - y_offset)
            else:
                d_2 = torch.linalg.norm(y_centre - y_offset)
                d_3 = torch.linalg.norm(y_centre - yp) 
                d_prime = torch.sqrt( (d_2 * r_3**2 - d_2 * d_3**2 + d_3*r_2**2 -d_3 * d_2**2) / (d_2 + d_3) )                   

                quantity = d_prime   
                
                if (d_prime_max is None) or (d_prime > d_prime_max):
                    d_prime_max = d_prime.clone().detach()
                    best_iter = ix

                            
            grad = torch.autograd.grad(quantity, offset)[0]
        else:
            grad_r3 = (sigma / 2) * (grad_phi(E_0) * dE_ds(classes_0, adv_input, y_centre, y_offset, yp, sigma, r_adjusted) - grad_phi(E_1) * dE_ds(classes_1, adv_input, y_centre, y_offset, yp, sigma, r_adjusted))
            if (r_3 + torch.linalg.norm(yp - y_offset)) < r_2: # Contained circle, no intersection
                quantity = r_3 + torch.linalg.norm(yp - y_offset)
                grad = grad_r3 + r_adjusted
            else:
                d_2 = torch.linalg.norm(y_centre - y_offset)
                d_3 = torch.linalg.norm(y_centre - yp) 
                d_prime = torch.sqrt( (d_2 * r_3**2 - d_2 * d_3**2 + d_3*r_2**2 -d_3 * d_2**2) / (d_2 + d_3) )
                
                grad = (grad_r3 * (2*d_2 * r_3 * (d_2 + d_3)) - r_adjusted*((d_2 + d_3)**2 + r_3**2 - r_2**2)) / (2*(d_2 + d_3)*d_prime)

        normalised = False
        if normalised:
            grad = torch.sign(grad)
        
        if (grad_old is not None) and (gamma_fixed is False): # First iteration we use a fixed step size
            if normalised: 
                gamma = 0.8*gamma
            else:
                d_grad = grad - grad_old
                gamma = torch.abs(torch.sum((offset - offset_old) * (d_grad))) / torch.sum(d_grad * d_grad)  
                gamma = torch.min(gamma, torch.tensor(5*gamma_start))
                if autograd is False:
                    print('Not using autograd gamma and ix :', gamma, ix, '##', offset - offset_old, d_grad, offset, flush=True)

        offset_old = offset.clone().detach()
        grad_old = grad.clone().detach()
        
        offset = torch.clip((offset + gamma*grad), 0., 1.).detach().clone()
        offset.requires_grad = True                            
        
        if gamma < 1e-3:
            break   
                   
    if d_prime_max is None:
        return torch.tensor(0.), best_iter
    else:       
        return d_prime_max.cpu(), best_iter
        


def simple_estimator(y, samples, sigma, label, model, iterations, alpha, device, shape, gamma_fixed=False, binary=False, approximate_gradients=True, gamma_start=0.01):
    '''
    Iterator for finding the first supplementary certification region
    '''


    y_original = y.detach().clone()

    yp = (y + torch.randn_like(y) * 1e-6).detach().clone()
    yp = torch.clip(yp, 1e-7, 1. - 1e-7)        
    objective_val, objective_max, objective_trigger, cohen_max = torch.tensor(0.), None, torch.tensor(0.), torch.tensor(0.)
    y_max = y.detach().clone()
    yp_old, grad_old, grad = None, None, None
    gamma = torch.tensor(gamma_start)
    best_iter = -1
    
    for ix in range(iterations):
        objective_val, flag, E0, E1, cohen_val, grad, max_class, second_class, expectation_set = sampler(yp, y_original, samples, sigma, label, model, device, shape, binary=binary, approximate_gradients=approximate_gradients)
        if (objective_max is None) or (objective_val > objective_max):
            objective_max = objective_val.clone().detach()
            cohen_max = cohen_val
            y_max = yp.clone().detach()
            best_iter = ix
     
            
        if (grad_old is not None) and (gamma_fixed is False): # First iteration we use a fixed step size
            d_grad = grad - grad_old
            gamma = torch.abs(torch.sum((yp - yp_old) * (d_grad))) / torch.sum(d_grad * d_grad) 
            
        yp_old, grad_old = yp.detach().clone(), grad.detach().clone()


        yp = torch.clip(yp + gamma*grad, 1e-7, 1. - 1e-7).detach().clone()
        yp.requires_grad = True
        
        if (gamma < 1e-7) or torch.isnan(gamma) or (torch.sum(torch.isnan(grad)) > 0):
            break  
                      
    if objective_max is None:
        objective_max = torch.tensor(0.)
    return objective_max, y_max, (y_max != torch.tensor(0.)), cohen_max, best_iter

