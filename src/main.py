'''
Code from Double Bubble, Toil against Trouble: Enhancing Certified Robustness with Transitivity
By Andrew C. Cullen, Paul Montague, Shijie Liu, Sarah M. Erfani and Benjamin I.P. Rubinstein
'''

import random
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import argparse

from collections import OrderedDict

from datasets import DATASETS
from models import model_settings
from train import train
from certification_modules import CertificationModule


def rename_state_dict(base_dict, new_prefix):
    new_dict = OrderedDict()
    for key, value in base_dict.items():
        new_key = new_prefix + key.partition('.')[2].partition('.')[2] # Corrects key to match loading in here
        new_dict[new_key] = value
        
    return new_dict

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean Value Expected')

PARALLEL_CHOICES = ['never', 'always', 'eval']
SAVED_LOC = ""

torch.manual_seed(0)
import random
random.seed(0)

parser = argparse.ArgumentParser(description='Certifying examples as per Smoothing')
parser.add_argument('--dataset', type=str, choices=DATASETS)
parser.add_argument('--filename', type=str, default='testing1234')
parser.add_argument('--parallel', type=str, choices=PARALLEL_CHOICES, help='Never if parallel will never be used, always = Training & Eval, eval = only at evaluation')
parser.add_argument('--batch_size', type=int, default=0, help='Batch Size (0 == Model default)')
parser.add_argument('--certification_iters', type=int, default=100, help='Batch Size (0 == Model default)')
parser.add_argument('--lr', type=float, default=0, help='Learning Rate (0 == Model default)')

parser.add_argument('--sigma', type=float, default=0.0, help='Noise level')
parser.add_argument('--gamma_start', type=float, default=0.0005, help='Step size starting point')
parser.add_argument('--samples', type=int, default=1500, help='Number of samples')
parser.add_argument('--epochs', type=int, default=80, help='Training Epochs')
parser.add_argument('--total_cutoff', type=int, default=250, help='Number of samples tested over')

parser.add_argument('--train', action='store_true', help='If training is required')
parser.add_argument('--eval', action='store_true', help='If evaluation is required')
parser.add_argument('--cutoff_experiment', action='store_true', help='If cutoff experiment is performed')
parser.add_argument('--new_cr', action='store_true', help='If improved cr experiment is performed')
parser.add_argument('--plotting', type=str2bool, nargs='?', const=True, default=True, help='If cutoff experiment is performed')

args = parser.parse_args()

cudnn.benchmark = True

def to_dataparallel(model):
    cuda_device_count = torch.cuda.device_count()        
    print('Cuda device count: ', cuda_device_count)
    model = model.to("cpu")
    model = torch.nn.DataParallel(model)
    device = torch.device("cuda:0")
    model.to(device)
    return model, device


if __name__ == "__main__":
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    # Preload model settings
    model, loss, optimizer, lr_scheduler, train_loader, val_loader, test_loader, device, classes = model_settings(args.dataset, args)
    
        
    # Train
    if args.train:
        print('Training', flush=True)
        if args.parallel == 'always':
            model, device = to_dataparallel(model)                    
        model, cutoff = train(device, model, optimizer, lr_scheduler, args.epochs, train_loader, val_loader, args, args.dataset, val_cutoff=1e6)
    else: 
        print('Loading Model', flush=True)
        pth = SAVED_LOC + args.dataset + '-' + str(args.sigma) + '-weight.pth'               
        loc = 'cuda'#:{}'.format(args.gpu_num)
    
        checkpoint = torch.load(pth)                  
        model.load_state_dict(checkpoint)
        model.eval()     
                     
    del train_loader, val_loader
   
    if args.eval or args.new_cr:      
        print('Evaluating attacks')
        if args.parallel == 'eval' or ((args.parallel == 'always') and (args.train is False)):
            model, device = to_dataparallel(model)        
            
        if args.new_cr: 
            certifications = ['cohen', 'simple', 'd_sim', 'f_sim', 'f_d_sim'] 
            filename = args.filename + '-' + str(args.gamma_start) + '-' + str(args.sigma)
            cert = CertificationModule(model, test_loader, args.samples, classes, certifications, filename, args.sigma, device, z_val=2.58, gamma_start=args.gamma_start)
            cert.certify(args.certification_iters)
        
