import torch
import torch.utils
import torch.nn as nn
from torch.utils.data.dataset import random_split
import gym
import pickle

from expert_dataset import ExpertDataSet
from agent import ImitationAgent

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

objects = []
with (open("/content/drive/MyDrive/Colab Notebooks/RL/Project/expert_trace.pkl", "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break

obs = objects[0][0]
act = objects[0][1]
expert_dataset = ExpertDataSet(obs, act)

env = gym.make('MontezumaRevenge-v4', render_mode='rgb_array')
# Define relevant variables for the ML task
learning_rate = 0.005
num_epochs = 200
num_workers = 2
batch_size = 64
train_prop = 0.8
train_size = int(train_prop * len(expert_dataset))
test_size = len(expert_dataset) - train_size

train_expert_dataset, test_expert_dataset = random_split(expert_dataset, [train_size, test_size])
train_loader = torch.utils.data.DataLoader(  dataset=train_expert_dataset, batch_size=batch_size, shuffle=True,num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(  dataset=test_expert_dataset, batch_size=batch_size, shuffle=True,num_workers=num_workers)

student = ImitationAgent(env.action_space.n)
student.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(student.parameters(), lr=learning_rate, weight_decay = 0.005, momentum = 0.9)  

for epoch in range(num_epochs):
    for batch, (images,labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        results,_ = student(images)
        loss = criterion(results, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('Epoch {}/{} | Batch {}/{} | Training Loss: {:.4f}'.format(epoch+1, num_epochs, batch+1, batch_size, loss.item()))

        if epoch % 100 == 0:
          net_checkpoint = "/content/drive/MyDrive/Colab Notebooks/RL/Project/imit"+str(epoch+1)+".pt"
          torch.save(student.state_dict(), net_checkpoint)

net_checkpoint = "/content/drive/MyDrive/Colab Notebooks/RL/Project/imit_final.pt"
torch.save(student.state_dict(), net_checkpoint)