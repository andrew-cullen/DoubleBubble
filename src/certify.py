'''
Original code from MACER: Attack-free and Scalable Robust Training via Maximizing Certified Radius
ICLR 2020 Submission


Modified for Double Bubble, Toil against Trouble: Enhancing Certified Robustness with Transitivity
By Andrew C. Cullen, Paul Montague, Shijie Liu, Sarah M. Erfani and Benjamin I.P. Rubinstein
'''
      
import numpy as np
from PIL import Image
import scipy.io as sio

from core import Smooth
from smooth import Smooth as SmoothV2

#V2 = True


def certify(model, device, dataset, transform, num_classes,
            mode='hard', start_img=0, num_img=500, skip=1, sigma=0.25, N0=100, N=300,
            alpha=0.001, batch=100, verbose=False,
            grid=(0.25, 0.50, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25), beta=1.0, V2=False):
  print('===certify(N={}, sigma={}, mode={})==='.format(N, sigma, mode))

  model.eval()
  if V2:
    smoothed_net = SmoothV2(model, num_classes, sigma, device, 0.05, 100, .05)
  else:
    smoothed_net = Smooth(model, num_classes,
                        sigma, device, mode, beta)

  radius_hard = np.zeros((num_img,), dtype=np.float)
  radius_soft = np.zeros((num_img,), dtype=np.float)
  num_grid = len(grid)
  cnt_grid_hard = np.zeros((num_grid + 1,), dtype=np.int)
  cnt_grid_soft = np.zeros((num_grid + 1,), dtype=np.int)
  s_hard = 0.0
  s_soft = 0.0

  for i in range(num_img):
    img, target = dataset[start_img + i * skip]
    img = img.to(device)

    if mode == 'both':
      p_hard, r_hard, p_soft, r_soft = smoothed_net.certify(
          img, N0, N, alpha, batch)
      correct_hard = int(p_hard == target)
      correct_soft = int(p_soft == target)
      if verbose:
        if correct_hard == 1:
          print('Hard Correct: 1. Radius: {}.'.format(r_hard))
        else:
          print('Hard Correct: 0.')
        if correct_soft == 1:
          print('Soft Correct: 1. Radius: {}.'.format(r_soft))
        else:
          print('Soft Correct: 0.')
      radius_hard[i] = r_hard if correct_hard == 1 else -1
      radius_soft[i] = r_soft if correct_soft == 1 else -1
      if correct_hard == 1:
        cnt_grid_hard[0] += 1
        s_hard += r_hard
        for j in range(num_grid):
          if r_hard >= grid[j]:
            cnt_grid_hard[j + 1] += 1
      if correct_soft == 1:
        cnt_grid_soft[0] += 1
        s_soft += r_soft
        for j in range(num_grid):
          if r_soft >= grid[j]:
            cnt_grid_soft[j + 1] += 1
    else:
      if V2:
        (r_hard, r_soft), (result_hard, result_soft), (classes_hard, classes_soft), (E_0_hard, E_1_hard), (E_0_soft, E_1_soft), ((offset_0_hard, offset_1_hard), (offset_0_soft, offset_1_soft)) = smoothed_net.certify(img, N, alpha, batch)#, target=target.to(device))
        print('@'*40, flush=True)
        smoothed_net.single_bubble(img, target, 100, 100, 100, 2.58, 100) # x, label, bubble_iters, buble_num, num, alpha, batch_size)
        print('@#!'*20, flush=True)
        if mode == 'hard':
          radius = r_hard
          prediction = classes_hard[0]
          print('This is the radius ', radius, flush=True)
        else:
          radius = r_soft
          prediction = classes_soft[0]      
      else:
        prediction, radius = smoothed_net.certify(img, N0, N, alpha, batch)
      correct = int(prediction == target)
      if verbose:
        if correct == 1:
          print('Correct: 1. Radius: {}.'.format(radius))
        else:
          print('Correct: 0.')
      radius_hard[i] = radius if correct == 1 else -1
      if correct == 1:
        cnt_grid_hard[0] += 1
        s_hard += radius
        for j in range(num_grid):
          if radius >= grid[j]:
            cnt_grid_hard[j + 1] += 1

  print('===Certify Summary===')
  print('Total Image Number: {}'.format(num_img))
  if mode == 'both':
    print('===Hard certify===')
    print('Radius: 0.0  Number: {}  Acc: {}'.format(
        cnt_grid_hard[0], cnt_grid_hard[0] / num_img * 100))
    for j in range(num_grid):
      print('Radius: {}  Number: {}  Acc: {}'.format(
          grid[j], cnt_grid_hard[j + 1], cnt_grid_hard[j + 1] / num_img * 100))
    print('ACR: {}'.format(s_hard / num_img))
    print('===Soft certify===')
    print('Radius: 0.0  Number: {}  Acc: {}'.format(
        cnt_grid_soft[0], cnt_grid_soft[0] / num_img * 100))
    for j in range(num_grid):
      print('Radius: {}  Number: {}  Acc: {}'.format(
          grid[j], cnt_grid_soft[j + 1], cnt_grid_soft[j + 1] / num_img * 100))
    print('ACR: {}'.format(s_soft / num_img))

  else:
    print('Radius: 0.0  Number: {}  Acc: {}'.format(
        cnt_grid_hard[0], cnt_grid_hard[0] / num_img * 100))
    for j in range(num_grid):
      print('Radius: {}  Number: {}  Acc: {}'.format(
          grid[j], cnt_grid_hard[j + 1], cnt_grid_hard[j + 1] / num_img * 100))
    print('ACR: {}'.format(s_hard / num_img))

