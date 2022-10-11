'''
Code from MACER: Attack-free and Scalable Robust Training via Maximizing Certified Radius
ICLR 2020 Submission

Modified for Double Bubble, Toil against Trouble: Enhancing Certified Robustness with Transitivity
By Andrew C. Cullen, Paul Montague, Shijie Liu, Sarah M. Erfani and Benjamin I.P. Rubinstein
'''

import argparse
import numpy as np
import time
import os

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10, SVHN

from macer import macer_train
from macer_model import resnet110, LeNet
from certify import certify

from certification_modules import CertificationModule

if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='MACER Train and Test')
  parser.add_argument('--name', default='default', type=str, help='')  
  parser.add_argument('--task', default='train',
                      type=str, help='Task: train or test')
  parser.add_argument('--root', default='data', type=str, help='Dataset path')
  parser.add_argument('--dataset', default='cifar10', type=str, help='Dataset')
  parser.add_argument('--resume_ckpt', default='none', type=str,
                      help='Checkpoint path to resume')
  parser.add_argument('--ckptdir', default='none', type=str,
                      help='Checkpoints save directory')

  parser.add_argument('--epochs', default=440,
                      type=int, help='Number of training epochs')
  parser.add_argument('--gauss_num', default=16, type=int,
                      help='Number of Gaussian samples per input')
  parser.add_argument('--batch_size', default=64, type=int, help='Batch size')

  # params for train
  parser.add_argument('--lr', default=0.01, type=float, help='Initial learning rate')
  parser.add_argument('--sigma', default=0.25, type=float,
                      help='Standard variance of gaussian noise (also used in test)')
  parser.add_argument('--lbd', default=12.0, type=float,
                      help='Weight of robustness loss')
  parser.add_argument('--gamma', default=8.0, type=float,
                      help='Hinge factor')
  parser.add_argument('--beta', default=16.0, type=float,
                      help='Inverse temperature of softmax (also used in test)')

  # params for test
  parser.add_argument('--start_img', default=500,
                      type=int, help='Image index to start (choose it randomly)')
  parser.add_argument('--num_img', default=500, type=int,
                      help='Number of test images')
  parser.add_argument('--skip', default=1, type=int,
                      help='Number of skipped images per test image')

  args = parser.parse_args()

  ckptdir = None if args.ckptdir == 'none' else args.ckptdir

  if ckptdir is not None and not os.path.isdir(ckptdir):
    os.makedirs(ckptdir)
  checkpoint = None if args.resume_ckpt == 'none' else args.resume_ckpt
  
  V2 = True


  # Load dataset and build model
  if args.dataset == 'mnist':
    base_model = LeNet()
    transform_train = transforms.Compose([
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainset = MNIST(
        root=args.root, train=True, download=True, transform=transform_train)
    testset = MNIST(
        root=args.root, train=False, download=True, transform=transform_test)

  elif args.dataset == 'cifar10':
    base_model = resnet110()
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainset = CIFAR10(
        root=args.root, train=True, download=True, transform=transform_train)
    testset = CIFAR10(
        root=args.root, train=False, download=True, transform=transform_test)

  elif args.dataset == 'svhn':
    base_model = resnet110()
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainset = SVHN(
        root=args.root, split='train', download=True, transform=transform_train)
    testset = SVHN(
        root=args.root, split='test', download=True, transform=transform_test)

  else:
    raise NotImplementedError

  trainloader = torch.utils.data.DataLoader(
      trainset, batch_size=args.batch_size, shuffle=True, num_workers=1)

  num_classes = 10

  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  if device == 'cuda':
    cudnn.benchmark = True
    model = torch.nn.DataParallel(base_model)
  else:
    model = base_model

  optimizer = optim.SGD(
      model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
  scheduler = MultiStepLR(
      optimizer, milestones=[200, 400], gamma=0.1)


  # Resume from checkpoint if required
  start_epoch = 0
  if checkpoint is not None:
    print('==> Resuming from checkpoint..')
    print(checkpoint)
    checkpoint = torch.load(checkpoint)
    base_model.load_state_dict(checkpoint['net'])
    start_epoch = checkpoint['epoch']
    scheduler.step(start_epoch)


  # Main routine
  if args.task == 'train':
    # Training routine
    for epoch in range(start_epoch + 1, args.epochs + 1):
      print('===train(epoch={})==='.format(epoch))
      t1 = time.time()
      model.train()

      macer_train(args.sigma, args.lbd, args.gauss_num, args.beta,
                      args.gamma, num_classes, model, trainloader, optimizer, device)
                      
      scheduler.step()     
      
      if (epoch >= 200) and (args.lbd < 1):
        args.lbd = 12

      t2 = time.time()
      print('Elapsed time: {}'.format(t2 - t1))

      if epoch % 20 == 0 and epoch >= 200:
        t1 = time.time()
        model.eval()
        certify(model, device, testset, transform_test, num_classes,
                mode='hard', start_img=args.start_img, num_img=args.num_img, 
                sigma=args.sigma, beta=args.beta, 
                V2=V2)
        t2 = time.time()
        print('Elapsed time: {}'.format(t2 - t1))      
      
      
        # Certify test
        print('===test(epoch={})==='.format(epoch))
        if ckptdir is not None:
          # Save checkpoint
          print('==> Saving {}.pth..'.format(epoch))
          try:
            state = {
                'net': base_model.state_dict(),
                'epoch': epoch,
            }
            torch.save(state, '{}/{}.pth'.format(ckptdir, epoch))
          except OSError:
            print('OSError while saving {}.pth'.format(epoch))
            print('Ignoring...')        
        
        




  else:
    filename = args.name + '-' + str(args.sigma)
    certifications =  ['cohen', 'single_bubble', 'double_bubble', 'simple', 'd_sim', 'f_sim', 'f_d_sim']
    
    cert = CertificationModule(model, testset, 500, 10, certifications, filename, args.sigma, device, z_val=2.58, gamma_start=0.01)
    cert.certify(1000)
  

