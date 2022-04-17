'''
made by XHU-WNCG
2022.4
'''
import math
from torch.nn.modules import loss
import torch.nn.functional as F
from torch import argmax
import torch.nn as nn


class categorical_crossentropy(nn.Module):
  def __init__(self):
     super(categorical_crossentropy, self).__init__()
  def forward(self, input, target):
      loss = 0
      for i in range(len(input[0] - 1)):
          loss -= target[0][i]*math.log(input[0][i])


      return loss