import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import namedtuple, deque
import numpy as np
import math
from numpy.random import choice
import gym 
import operator
from copy import deepcopy

class ImitationAgent(nn.Module):
  def __init__(self, num_actions,batch_size=64):
    super(ImitationAgent, self).__init__()
    self.batch_size = batch_size
    self.layer1 = nn.Sequential(
        nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
        nn.BatchNorm2d(96),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 3, stride = 2))
    self.layer2 = nn.Sequential(
        nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 3, stride = 2))
    self.layer3 = nn.Sequential(
        nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(384),
        nn.ReLU())
    self.layer4 = nn.Sequential(
        nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(384),
        nn.ReLU())
    self.layer5 = nn.Sequential(
        nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 3, stride = 2))
    self.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(3840, 4096),
        nn.ReLU())
    self.fc1 = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(4096, 4096),
        nn.ReLU())
    self.fc2= nn.Sequential(
        nn.Linear(4096, num_actions))

  def forward(self, x):
      out = self.preproc_state(x)
      out = self.layer1(out)
      out = self.layer2(out)
      out = self.layer3(out)
      out = self.layer4(out)
      out = self.layer5(out)
      out = out.reshape(out.size(0), -1)
      out = self.fc(out)
      out = self.fc1(out)
      out = self.fc2(out)
      return out,F.softmax(out, dim=1)

  def act(self, state):
      # Stack 4 states
      #state = torch.vstack([self.preproc_state(state) for i in range(1)]).unsqueeze(0)
      
      # Get Action Probabilities
      probs,_ = self.forward(state)
      
      
      # Return Action and LogProb
      action = probs.argmax(-1)
      return action.item()
    
  def preproc_state(self, state):
      # State Preprocessing
      #state = state.transpose(2,0,1) #Torch wants images in format (channels, height, width)
      #state = torch.from_numpy(state)
      
      return state/255 # normalize