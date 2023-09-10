import torch
import torch.utils
import torch.nn as nn
import torch.nn.functional as F


class ImitationAgent(nn.Module):
  def __init__(self, num_actions, batch_size=64):
    super(ImitationAgent, self).__init__()
    self.batch_size = batch_size

    ## Activation functions
    self.relu = nn.ReLU()

    ## Convo Layers
    self.c1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7)
    self.c2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5)
    
    self.avgp1 = nn.AvgPool2d(kernel_size=2)
    self.avgp2 = nn.AvgPool2d(kernel_size=2)

    ## FC Layers
    self.fc1 = nn.Linear(in_features=32, out_features=32)
    self.fc2 = nn.Linear(in_features=32, out_features=num_actions)

  def forward(self, x):
    ## 1st Convo Layer
    x = self.c1(x)
    x = self.relu(x)
    x = self.avgp1(x)

    ## 2nd Convo Layer
    x = self.c2(x)
    x = self.relu(x)
    x = self.avgp2(x)
        
    ## 1st FC Layer
    batch_size = x.shape[0]
    x = x.reshape(batch_size, 32, -1).max(axis=2).values
    x = self.fc1(x)
    x = self.relu(x)

    ## 2nd FC Layer
    x = self.fc2(x)
    p = F.softmax(x, dim=1)

    return x,p

  def act(self, state):
      # Stack 4 states
      state = torch.vstack([self.preproc_state(state) for i in range(1)]).unsqueeze(0)
      
      # Get Action Probabilities
      _,probs = self.forward(state)
      
      
      # Return Action and LogProb
      action = probs.argmax(-1)
      return action.item()
    
  def preproc_state(self, state):
      # State Preprocessing
      state = state.transpose(2,0,1) #Torch wants images in format (channels, height, width)
      state = torch.from_numpy(state)
      
      return state/255 # normalize