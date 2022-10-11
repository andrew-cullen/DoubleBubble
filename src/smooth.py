'''
Code from Double Bubble, Toil against Trouble: Enhancing Certified Robustness with Transitivity
By Andrew C. Cullen, Paul Montague, Shijie Liu, Sarah M. Erfani and Benjamin I.P. Rubinstein
'''

from math import ceil

import numpy as np
from scipy.stats import norm
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.proportion import proportion_confint as binom_conf
from statsmodels.stats.proportion import multinomial_proportions_confint as multi_conf
import torch
import torch.nn.functional as F

NORMICDF = lambda x: torch.distributions.Normal(0,1).icdf(x)


class Smooth(object):
    def __init__(self, base_classifier, num_classes, sigma, device, bubble_factor, max_step, alpha, exact_gradients=True):
        self.base_classifier = base_classifier
        self.num_classes = num_classes
        self.sigma = sigma
        self.device = device
        self.exact_gradients = exact_gradients
        self.bubble_factor = bubble_factor
        self.max_step = max_step
        self.alpha = alpha
        self.mode = 'hard'
                
    def certify(self, x, num, alpha, batch_size, calculate_gradients=False):#, target=None):    
        result_hard = torch.zeros(self.num_classes, dtype=float).to(self.device)
        result_soft = torch.zeros(self.num_classes, dtype=float).to(self.device)
        num_orig = num

        for temp in range(ceil(num / batch_size)): 
            this_batch_size = min(batch_size, num)
            num -= this_batch_size
            batch = x.repeat((this_batch_size, 1, 1, 1))
            noise = torch.randn_like(batch, device=self.device) * self.sigma
            predictions = self.base_classifier(batch + noise)

            if self.mode == 'hard':
                p_hard = F.gumbel_softmax(10*predictions, tau=1, hard=False, dim=1) 
                result_hard += p_hard.sum(0)
                                    
            if self.mode == 'soft':
                p_soft = F.softmax(predictions, 1)
                result_soft += p_soft.sum(0)
        classes_hard = result_hard.argsort().flip(0)[:2] # Flip creates a new item in memory       
        classes_soft = result_soft.argsort().flip(0)[:2]
                
        E_0_hard, E_1_hard = self._confidence_bounds(result_hard, classes_hard)
        E_0_soft, E_1_soft = self._confidence_bounds(result_soft, classes_soft)        
        
        if E_0_hard > E_1_hard:
            r_hard = 0.5 * self.sigma * (NORMICDF(E_0_hard) - NORMICDF(E_1_hard))
        else:
            r_hard = -1
        if E_0_soft > E_1_soft:
            r_soft = 0.5 * self.sigma * (NORMICDF(E_0_soft) - NORMICDF(E_1_soft))
        else:
            r_soft = -1
            
        offset_0_hard, offset_1_hard = result_hard[classes_hard[0]] - E_0_hard, result_hard[classes_hard[1]] - E_1_hard
        offset_0_soft, offset_1_soft = result_hard[classes_soft[0]] - E_0_soft, result_hard[classes_soft[1]] - E_1_soft        

        return (r_hard, r_soft), (result_hard, result_soft), (classes_hard, classes_soft), (E_0_hard, E_1_hard), (E_0_soft, E_1_soft), ((offset_0_hard, offset_1_hard), (offset_0_soft, offset_1_soft))
    
    def single_bubble(self, x, label, bubble_iters, bubble_num, num, alpha, batch_size):
        inpt_flag = True if x.shape[1] == 1 else False
        class_set = torch.zeros(10).to(self.device)
        class_set[0] = label#[0]
        class_dummy = torch.arange(10).to(self.device)
        class_set[1:] = class_dummy[class_dummy != label]#[0]]
                  
        x_origin = x.detach().clone()
        x_min = x.detach().clone()
        x.requires_grad_()
        
        distance_min = 1e6
        success = False
        
        if self.mode == 'hard':
            ix = 0
        else:
            ix = 1        
        
        for _ in range(bubble_iters):
            r_set, results_set, classes_set, E_hard_set, E_soft_set, offset_set = self.certify(x, bubble_num, alpha, batch_size)           

            if classes_set[ix][0] == label:
                grad = -1.*self.evaluate_gradients(x, results_set[ix], classes_set[ix][0], classes_set[ix][1], offset_set[ix][0], offset_set[ix][1])
                if r_set[ix] > 0:
                    stepsize = np.min([(1 + self.bubble_factor)*np.asarray(r_set[ix].detach().cpu()), np.asarray(self.max_step)])
                else:
                    stepsize = np.asarray(0.1)
                stepsize = torch.tensor(stepsize)
            else:
                if r_set[ix] > 0:
                    # Assume move towards origin, we want to minimise the certification
                    grad = x_origin - x
                    stepsize = torch.tensor(np.min([(1 - self.bubble_factor)*np.asarray(r_set[ix].detach().cpu()), np.asarray(self.max_step)]))
                    
                    distance = torch.linalg.norm(grad)
                    if distance < distance_min:
                        distance_min = distance.detach().clone()
                        x_min = x.detach().clone()
                        success = True                    
                else:
                    # This is when we're in a mixed class environment, but where we're below the clear certification threshold
                    grad = self.evaluate_gradients(x, results_set[0], classes_set[0][0], classes_set[0][1], offset_set[0][0], offset_set[0][1])
                    stepsize = torch.tensor(0.1) #np.asarray(0.1)
            
            x_new = x + stepsize*(grad / torch.linalg.norm(grad))
            x = x_new.detach().clone().requires_grad_()
          
        return x_min, distance_min, success          
        
    def evaluate_gradients(self, x, results, class_0, class_1, offset_0, offset_1):
        if self.exact_gradients:
            certification_distance = 0.5 * self.sigma * (NORMICDF(results[class_0] - offset_0) - NORMICDF(results[class_1] - offset_1))                    
            grad = torch.autograd.grad(certification_distance, x)[0]
        else:
            raise ValueError('Not yet implemented')
            
        return grad
        
    def _confidence_bounds(self, results, classes):
    	# To preserve differentiability across the confidence bounds, we do not simply return E_hat = f(E), as f(X) is not differentiable.
    	# Instead we calculate that E_hat' = E - f(E), and then E_hat = E - E_hat'. Thus dE_hat = dE - dE_hat' ~ dE
    	# As E_hat is proportional to E, the difference between derivatives should be small and consistent
        binary = True
        E0, E1 = results[classes[0]], results[classes[1]]
        if binary:   
            if E0.int().detach().cpu().numpy() > 0:                
                E0_temp = torch.tensor(proportion_confint(E0.int().detach().cpu().numpy(), torch.sum(results).int().detach().cpu().numpy(), alpha=self.alpha, method='beta')[0])
            else:
                E0_temp = 0
            if E1.int().detach().cpu().numpy() > 0:                                
                E1_temp = torch.tensor(proportion_confint(E1.int().detach().cpu().numpy(), torch.sum(results).int().detach().cpu().numpy(), alpha=self.alpha, method='beta')[1])        
            else:
                E1_temp = 0
            delta_E0 = E0.detach().clone() - E0_temp
            delta_E1 = E1.detach().clone() - E1_temp
            delta_E0.requires_grad, delta_E1.requires_grad = False, False
            E0 -= delta_E0
            E1 -= delta_E1
        else: 
            expectation_set = multi_conf(np.ceil((results).detach().cpu().numpy()), alpha=self.alpha, method='goodman')
            E0_temp_save = E0.detach().clone()
            E1_temp_save = E1.detach().clone()
            delta_E0 = E0.detach().clone() - torch.tensor(expectation_set[classes[0], 0]) 
            E0 -= delta_E0
            
            delta_E1 = E1.detach().clone() - torch.tensor(expectation_set[classes[1], 1])
            E1 -= delta_E1
        return E0, E1
    

