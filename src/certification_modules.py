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
torch.pi = torch.acos(torch.zeros(1)).item() * 2

import numpy as np


import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


from evaluation_common import multi_conf_wrapper as multi_conf
from evaluation_common import extract_classes

from scipy.stats import norm as stats_norm 

torch.manual_seed(0)
import random
random.seed(0)

from logging_manager import LogSet

from evaluation_common import return_wrapper
from evaluation_support import simple_estimator, secondary_ball_simple_estimator

from certify_simple import certify

class CertificationModule:
    '''
    General certification measure. Takes inputs:
    model - Pytorch model
    dataloader - loader for both validation images and labels
    samples - number of iterations used to construct Monte Carlo estimations of expectations
    classes - number of classes in the dataset
    filename - output filename for logging
    sigma - level of noise added
    device - pytorch device (cpu or gpu)
    z_val - z_score for confidence intervals
    gamma_start - control parameter for certification searfch
    '''
    def __init__(self, model, dataloader, samples, classes, certifications, filename, sigma, device, z_val=2.58, gamma_start=0.01): # Most of these can be replaced by calling from models model_settings
        super(CertificationModule, self).__init__()    
        self.model = model
        self.dataloader = iter(dataloader)
        self.samples = samples
        self.filename = filename
        self.device = device
        self.sigma = sigma
        self.alpha = 0.5*(1 - stats_norm.cdf(z_val))  
        self.classes = classes 
        self.gamma_start = gamma_start
        
        certifications = ['iter', 'label', 'E0'] + certifications + [i + '_t' for i in certifications]
        if 'simple' in certifications:
            certifications = certifications + ['mod', 'f_mod']  
        self.log = LogSet(self.filename, decimals=4, normalise='cohen', means=True, console=True)
    
        self.shape = None
                
                
        self.use_cohen = False
        self.use_single = False
        self.use_double = False
        self.use_simplified = False
        self.use_double_simplified = False
        self.use_simplified_calculate_gradients = False
        self.double_gradients = False
        for cert in certifications:
            if cert == 'cohen':
                self.use_cohen = True
            if cert == 'single_bubble':
                self.use_single = True
            if cert == 'double_bubble':
                self.use_double = True
            if cert == 'simple':
                self.use_simplified = True
            if cert == 'd_sim':
                self.use_double_simplified = True
            if cert == 'f_sim':
                self.use_simplified_calculate_gradients = True  
            if cert == 'f_d_sim':
                self.double_gradients = True
            
                
    def certify(self, max_count):
        '''
        Loops through (max_count) validation samples and certifies them across the range of techniques 
        '''
        count = 0
        reject_count = 0
        im_min = torch.tensor(10.)
        im_max = torch.tensor(-10.)
        old_image = None
        while count < max_count:
            # Load next sample
            
            images, labels = next(self.dataloader)
            if self.shape is None:
                self.shape = images.shape

            images = images.to(self.device)
                        
            labels = torch.tensor(labels).to(self.device)                       
            time_base_predict = time.time()

            adv_input = images.repeat(self.samples, 1, 1, 1)
            adv_input = adv_input + torch.randn_like(adv_input) * self.sigma
            pred_baseline = self.basic_predict(adv_input)
            time_base_predict = time.time() - time_base_predict
            
            if pred_baseline == labels:
                self.log.append(count, 'iter')
                self.log.append(labels, 'label')
                self.log.append(reject_count, 'rej')

                correct_hard, cert_hard, correct_soft, cert_soft = certify(self.model, self.device, images, labels, self.classes, mode='hard', sigma=self.sigma, N0=100, N=self.samples, alpha=0.005, batch=100, verbose=False, beta=1.0)
                correct_multiclass, cert_multiclass = certify(self.model, self.device, images, labels, self.classes, mode='hard', sigma=self.sigma, N0=100, N=self.samples, alpha=0.005, batch=100, verbose=False, beta=1.0, multi=True)                
        
                self.log.append(correct_hard, 'co_h'), self.log.append(cert_hard, 'c_h'), self.log.append(correct_soft, 'co_s'), self.log.append(cert_soft, 'c_s'), self.log.append(correct_multiclass, 'co_mu'), self.log.append(cert_multiclass, 'c_mu')
                
                
                if self.use_cohen:
                    cohen_result, cohen_time, E0 = self.cohen_certify(adv_input)
                    self.log.append(E0, 'E0'), self.log.append(cohen_result, 'cohen'), self.log.append(cohen_time + time_base_predict, 'cohen_t')                
                
                if self.use_simplified:
                    simple_radius, simple_time, simple_mod, y_offset, t2, best_iter = self.simple_certify(labels, images, cohen_result)                    

                    simple_radius = np.max([simple_radius, cohen_result])                    
                    simple_mod = np.max([simple_mod, simple_radius])
                    self.log.append(simple_radius, 'sim'), self.log.append(simple_time, 'sim_t'), self.log.append(t2, 'sim_t2')
                    self.log.append(simple_mod, 'mod'), self.log.append(best_iter, 'b')
                    
                    if self.use_double_simplified:
                        double_radius, double_time, best_iter = self.double_simple_certify(images, y_offset, torch.tensor(np.max([simple_radius, simple_mod])), self.shape)

                        double_radius = np.max([double_radius, simple_radius])
                        double_time += simple_time
                        self.log.append(double_time, 'd_sim_t'), self.log.append(double_radius, 'd_sim'), self.log.append(best_iter, 'd_b')


                if self.use_simplified_calculate_gradients:
                    false_radius, false_time, false_mod, false_y_offset, t2, best_iter = self.simple_certify(labels, images, cohen_result, approximate_gradients=False)                    

                    false_radius = np.max([false_radius, cohen_result])                    
                    false_mod = np.max([false_mod, false_radius])
                    self.log.append(false_radius, 'f_sim'), self.log.append(false_time, 'f_sim_t'), self.log.append(t2, 'f_sim_t2')
                    self.log.append(false_mod, 'f_mod'), self.log.append(best_iter, 'f_b')
                    if self.double_gradients:                  
                        double_radius, double_time, best_iter = self.double_simple_certify(images, false_y_offset, torch.tensor(np.max([false_radius, false_mod])), self.shape)
                        double_radius = np.max([double_radius, false_radius])
                        double_time += false_time
                        self.log.append(double_time, 'f_d_sim_t'), self.log.append(double_radius, 'f_d_sim'), self.log.append(best_iter, 'f_d_b')
                                                                                             
                self.log.print()
                
                count += 1
            else:
                 reject_count += 1
        print('Total samples counted : {}, and rejected: {}'.format(count, reject_count), flush=True)           
                
    def cohen_certify(self, adv_input, binary=False):
        time_used = time.time()
        cohen_val, E0, E1, class0, class1, max_class, second_class, expectation_classes = extract_classes(self.model, adv_input, self.samples, self.shape, self.sigma, binary=binary)
        cohen_val = cohen_val.detach().cpu().numpy()
        cohen_val = np.max([cohen_val, 0.])

        time_used = time.time() - time_used
        return cohen_val, time_used, E0
                
    def simple_certify(self, labels, yp, baseline_cohen, approximate_gradients=True):
        time_used = time.time()
        # Load in samples
        
        shape = list(yp.shape)
        yp = yp.reshape(shape[0], -1)        

        simple_radius, first_bubble_x, _, cohen_max, best_iter = simple_estimator(yp, self.samples, self.sigma, labels, self.model, 50, self.alpha, self.device, self.shape, approximate_gradients=approximate_gradients, gamma_start=self.gamma_start)    # Why is this only using 500, as

        time_used_1 = time.time() - time_used
        
        simple_prime = self.cap_adjustment(yp, first_bubble_x, cohen_max) #self.cap_adjustment(yp, simple_radius, first_bubble_x, cohen_max)

        time_used_2 = time.time() - time_used
        
        return simple_radius.detach().cpu().numpy(), time_used_1, simple_prime.detach().cpu().numpy(), first_bubble_x, time_used_2, best_iter
        
    def double_simple_certify(self, y_centre, y_offset, r_adjusted, shape, binary=False, approximate_gradients=True, secondary_autograd=True):
        time_used = time.time()
        
        if len(shape) == 4:
            y_centre = y_centre.reshape(shape[0], shape[1], shape[2], shape[3])    
            y_offset = y_offset.reshape(shape[0], shape[1], shape[2], shape[3])
        else:
            y_centre = y_centre.reshape(1, shape[0], shape[1], shape[2])                      
            y_offset = y_offset.reshape(1, shape[0], shape[1], shape[2])          
                
        r_offset = r_adjusted + torch.linalg.norm(y_centre - y_offset)
                
        d_prime, best_iter = secondary_ball_simple_estimator(y_centre, y_offset, r_adjusted, r_offset, self.samples, self.sigma, self.model, 30, self.alpha, self.device, shape, binary=binary, gamma_start=0.025, autograd=secondary_autograd)#self.gamma_start)
        
        time_used = time.time() - time_used

        return d_prime.detach().clone().cpu().numpy(), time_used, best_iter
        
    def basic_predict(self, images):
        images = images.detach().clone()
                        
        _, _, pred_baseline = return_wrapper(self.model, images, self.shape)
            
        indices, counts = torch.unique(pred_baseline, sorted=True, return_counts=True)
        pred_baseline = indices[torch.argmax(counts)]
        
        return pred_baseline.detach()    
        
                
    def cap_adjustment(self, x, x_p, r_p):
        x = x.reshape(-1)
        x_p = x_p.reshape(-1)
                
        if (torch.min(x_p) < 0) or (torch.max(x_p) > 1):
            raise ValueError('Projection has failed before this point')        
            
        z = torch.zeros_like(x)
        z[x > 0.5] = 1
        
        r_adjusted = torch.sqrt( (x - z)**2 + ( torch.sqrt(r_p**2 - (x_p - z)**2) - torch.sqrt( (torch.linalg.norm(x_p - x))**2 - (x_p - x)**2 ) )**2 )
        r_adjusted[torch.isnan(r_adjusted)] = 0

        return torch.max(r_adjusted)

