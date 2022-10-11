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


def certify(model, device, img, target, num_classes,
            mode='hard', sigma=0.25, N0=100, N=1000,
            alpha=0.001, batch=100, verbose=False, beta=1.0, multi=False):
  model.eval()

  smoothed_net = Smooth(model, num_classes,
                        sigma, device, mode, beta)
  if multi:
    #print(smoothed_net.certify_multiclass(img, N, alpha, batch), 'ABCDEFGH', flush=True)
    p, r = smoothed_net.certify_multiclass(img, N, alpha, batch)
    correct = int(p == target)
    if correct == 1:
      return correct, r
    else:
      return correct, -1

  s_hard = 0.0
  s_soft = 0.0

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
    radius_hard = r_hard if correct_hard == 1 else -1
    radius_soft = r_soft if correct_soft == 1 else -1
    return correct_hard, radius_hard, correct_soft, radius_soft
  else:
    prediction, radius = smoothed_net.certify(img, N0, N, alpha, batch)
    
    correct = int(prediction == target)
    if correct == 1:
      return correct, radius, 0, -1
    else:
      return correct, -1, 0, -1
