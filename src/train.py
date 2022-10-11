'''
Code from Double Bubble, Toil against Trouble: Enhancing Certified Robustness with Transitivity
By Andrew C. Cullen, Paul Montague, Shijie Liu, Sarah M. Erfani and Benjamin I.P. Rubinstein
'''

import time
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from torchvision import datasets

from torchmetrics import Accuracy

SAVE_LOC = ""


def train(device, model, optimizer, lr_scheduler, num_epochs, train_loader, val_loader, args, name, val_cutoff=1e6):
    loss_fn = torch.nn.CrossEntropyLoss()
    train_accuracy = Accuracy().cuda()
    val_accuracy = Accuracy().cuda()

    for epoch in range(num_epochs):
        model.train()
        total_batch = len(train_loader)
        train_accuracy.reset()
        start_time = time.time()
        
        for i, (data, target) in enumerate(train_loader):
            optimizer.zero_grad() 

            data = data + torch.randn_like(data) * args.sigma
            
            if data.shape[1] == 1:
                data = data.repeat(1, 3, 1, 1)            
            
            data, target = data.to(device), target.to(device)
                        
            pred = model(data)        
            loss = loss_fn(pred, target)
            
            loss.backward()
            optimizer.step()
                    
            batch_accuracy = train_accuracy(pred, target)
            
            secondary_accuracy = torch.sum(torch.argmax(pred, dim=1) == target) / pred.shape[0]
                   
            if (i+1) % 100 == 0:
                print('Epoch [%d/%d], lter [%d/%d], Loss: %.4f, Acc: %.4f, Time: %.4f'
                     %(epoch+1, num_epochs, i+1, total_batch, loss.item(), 100*batch_accuracy, time.time() - start_time))                   
             
        epoch_accuracy = train_accuracy.compute()
        print('Epoch Accuracy: {}'.format(100*epoch_accuracy))
        
        val_accuracy.reset()
        model.eval()
        total_batch = len(val_loader)
        val_loss = 0
        secondary_val_accuracy = 0.
        total_shape = 0.
        for i, (data, target) in enumerate(val_loader):        
            data = data + torch.randn_like(data) * args.sigma
            if data.shape[1] == 1:
                data = data.repeat(1, 3, 1, 1)
                            
            data, target = data.to(device), target.to(device)

            pred = model(data)        
            loss = loss_fn(pred, target)
            val_loss += loss.item() * pred.shape[0]
                            
            batch_val_accuracy = val_accuracy(pred, target)
            
            secondary_val_accuracy += torch.sum(torch.argmax(pred, dim=1) == target) 
            total_shape += pred.shape[0]
            
        secondary_val_accuracy = 100 * secondary_val_accuracy / total_shape
        val_loss = val_loss / total_shape
        print('VAL Epoch [%d/%d], Loss: %.4f, Acc: %.4f, Acc2: %.4f'
                 %(epoch+1, num_epochs, val_loss , 100*val_accuracy.compute(), secondary_val_accuracy))
                 
        if val_loss < val_cutoff:
            print('Saving at epoch: {}'.format(epoch+1))
            torch.save(model.state_dict(), SAVE_LOC + name + '-' + str(args.sigma) + '-weight.pth')
            val_cutoff = val_loss

             
        if lr_scheduler is not None:
            lr_scheduler.step()

    return model, val_cutoff

