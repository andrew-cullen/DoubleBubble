'''
Original code from MACER: Attack-free and Scalable Robust Training via Maximizing Certified Radius
ICLR 2020 Submission


Modified for Double Bubble, Toil against Trouble: Enhancing Certified Robustness with Transitivity
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

class Smooth(object):
  '''
  Smoothed classifier
  mode can be hard, soft or both
  beta is the inverse of softmax temperature
  '''

  # to abstain, Smooth returns this int
  ABSTAIN = -1

  def __init__(self, base_classifier: torch.nn.Module, num_classes: int,
               sigma: float, device, mode='hard', beta=1.0):
    self.base_classifier = base_classifier
    self.num_classes = num_classes
    self.sigma = sigma
    self.device = device
    self.mode = mode
    self.square_sum = None
    self.ss = 0
    self.beta = beta

  def certify(self, x: torch.tensor, n0: int, n: int, alpha: float, batch_size: int) -> (int, float):
    if self.mode == 'both':
      c_hard, c_soft = self.predict(x, n0, batch_size)
      o_hard, o_soft = self._sample_noise(x, n, batch_size)
      na_hard, na_soft = o_hard[c_hard].item(), o_soft[c_soft].item()
      self.ss = self.square_sum[c_soft]
      pa_hard = self._lower_confidence_bound(na_hard, n, alpha, 'hard')
      pa_soft = self._lower_confidence_bound(na_soft, n, alpha, 'soft')
      r_hard = 0.0
      r_soft = 0.0
      if pa_hard < 0.5:
        c_hard = Smooth.ABSTAIN
      else:
        r_hard = self.sigma * norm.ppf(pa_hard)
      if pa_soft < 0.5:
        c_soft = Smooth.ABSTAIN
      else:
        r_soft = self.sigma * norm.ppf(pa_soft)
      return c_hard, r_hard, c_soft, r_soft
    else:
      # make an initial prediction of the label
      cAHat = self.predict(x, n0, batch_size)
      # draw more samples of f(x + epsilon)
      observation = self._sample_noise(x, n, batch_size)
      # use these samples to estimate a lower bound on pA
      nA = observation[cAHat].item()
      if self.mode == 'soft':
        self.ss = self.square_sum[cAHat]
      pABar = self._lower_confidence_bound(nA, n, alpha, self.mode)
      if pABar < 0.5:
        return Smooth.ABSTAIN, 0.0
      else:
        radius = self.sigma * norm.ppf(pABar)
        return cAHat, radius
        
  def certify_multiclass(self, x: torch.tensor, n: int, alpha: float, batch_size: int) -> (int, float):
    #c_hard, c_soft = self.predict(x, n0, batch_size)
    o_hard = self._sample_noise(x, n, batch_size)
    probabilities = self._lower_confidence_bound(o_hard, None, alpha, 'hard', multi=True)
    class_2, class_1 = o_hard.argsort()[-2:]
    pa_hard = probabilities[class_1,0]
    pb_hard = probabilities[class_2,1]      
    if pa_hard < pb_hard:
      radius = Smooth.ABSTAIN, 0.0
    else:
      radius = 0.5 * self.sigma * (norm.ppf(pa_hard) - norm.ppf(pb_hard))  
    return class_1, radius    

  def predict(self, x: torch.tensor, n: int, batch_size: int) -> int:
    self.base_classifier.eval()
    if self.mode == 'both':
      result_hard, result_soft = self._sample_noise(x, n, batch_size)
      return result_hard.argsort()[::-1][0], result_soft.argsort()[::-1][0]
    else:
      result = self._sample_noise(x, n, batch_size)
      return result.argsort()[::-1][0]

  def _sample_noise(self, x: torch.tensor, num: int, batch_size) -> np.ndarray:
    with torch.no_grad():
      result_hard = np.zeros(self.num_classes, dtype=int)
      result_soft = np.zeros(self.num_classes, dtype=float)
      self.square_sum = np.zeros(self.num_classes, dtype=float)
      for _ in range(ceil(num / batch_size)):
        this_batch_size = min(batch_size, num)
        num -= this_batch_size

        batch = x.repeat((this_batch_size, 1, 1, 1))
        noise = torch.randn_like(batch, device=self.device) * self.sigma
        predictions = self.base_classifier(batch + noise)
        predictions *= self.beta
        if self.mode == 'hard' or self.mode == 'both':
          p_hard = predictions.argmax(1)
          result_hard += self._count_arr(p_hard.cpu().numpy(),
                                         self.num_classes)
        if self.mode == 'soft' or self.mode == 'both':
          p_soft = F.softmax(predictions, 1)
          p_soft_square = p_soft ** 2
          p_soft = p_soft.sum(0)
          p_soft_square = p_soft_square.sum(0)
          result_soft += p_soft.cpu().numpy()
          self.square_sum += p_soft_square.cpu().numpy()
      if self.mode == 'hard':
        return result_hard
      if self.mode == 'soft':
        return result_soft
      else:
        return result_hard, result_soft

  def _count_arr(self, arr: np.ndarray, length: int) -> np.ndarray:
    counts = np.zeros(length, dtype=int)
    for idx in arr:
      counts[idx] += 1
    return counts

  def _lower_confidence_bound(self, NA, N, alpha: float, mode, multi=False) -> float:
    if multi:
      if mode == 'hard':
        return multi_conf(NA, alpha=alpha, method='goodman')
      else:
        raise ValueError('Not Implemented')
    if mode == 'hard':
      return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]
    else:
      sample_variance = (self.ss - NA * NA / N) / (N - 1)
      if sample_variance < 0:
        sample_variance = 0
      t = np.log(2 / alpha)
      return NA / N - np.sqrt(2 * sample_variance * t / N) - 7 * t / 3 / (N - 1)
