import torch
import torch.nn as nn
import numpy as np
import cv2
import gym
import gym.spaces
import numpy as np
import collections
import argparse
import time
import collections
import torch.optim as optim
from buffer import Experience, ExperienceBuffer
from dqn import DQN
from wrap import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_ENV_NAME = "MontezumaRevenge-v4"
#DEFAULT_ENV_NAME = "Pong-v4"
MEAN_REWARD_BOUND = 19.0
GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 10000
REPLAY_START_SIZE = 10000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 1000
EPSILON_DECAY_LAST_FRAME = 150000
EPSILON_START = 1.0
EPSILON_FINAL = 0.01
EPOCHS = 1000000

class Agent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.0

    @torch.no_grad()
    def play_step(self, net, epsilon=0.0, device="cpu"):
        done_reward = None

        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state_a = np.array([self.state], copy=False)
            state_v = torch.tensor(state_a).to(device)
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        # do step in the environment
        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward

        exp = Experience(self.state, action, reward,
                         is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward

def calc_loss(batch, net, tgt_net, device="cpu"):
      states, actions, rewards, dones, next_states = batch

      states_v = torch.tensor(np.array(states, copy=False)).to(device)
      next_states_v = torch.tensor(np.array(next_states, copy=False)).to(device)
      actions_v = torch.tensor(actions).to(device)
      rewards_v = torch.tensor(rewards).to(device)
      done_mask = torch.BoolTensor(dones).to(device)

      state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    
      with torch.no_grad():
        next_state_values = tgt_net(next_states_v).max(1)[0]
        next_state_values[done_mask] = 0.0
        next_state_values = next_state_values.detach()

      expected_state_action_values = next_state_values * GAMMA + rewards_v
      
      return nn.MSELoss()(state_action_values,expected_state_action_values)

env = make_env(DEFAULT_ENV_NAME)

net = DQN(env.observation_space.shape,env.action_space.n).to(device)
tgt_net = DQN(env.observation_space.shape,env.action_space.n).to(device)
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

buffer = ExperienceBuffer(REPLAY_SIZE)

agent = Agent(env, buffer)
epsilon = EPSILON_START

total_rewards = []
frame_idx = 0
ts_frame = 0
ts = time.time()
best_m_reward = None

for epoch in range(EPOCHS):
    frame_idx += 1
    epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)
    reward = agent.play_step(net, epsilon, device=device)

    if reward is not None:
        total_rewards.append(reward)
        ts_frame = frame_idx
        ts = time.time()
        speed = (frame_idx - ts_frame) / (time.time() - ts)
        m_reward = np.mean(total_rewards[-100:])
        
        print("%d: done %d games, reward %.3f, "
                "eps %.2f, speed %.2f f/s" % (
              frame_idx, len(total_rewards), m_reward, epsilon,
              speed
          ))
         
        if best_m_reward is None or best_m_reward < m_reward:
            torch.save(net.state_dict(), DEFAULT_ENV_NAME +
                        "-best_%.0f.dat" % m_reward)
            if best_m_reward is not None:
                print("Best reward updated %.3f -> %.3f" % (
                    best_m_reward, m_reward))
            best_m_reward = m_reward
        if m_reward > MEAN_REWARD_BOUND:
            print("Solved in %d frames!" % frame_idx)
            break

    if len(buffer) < REPLAY_START_SIZE:
        continue

    if frame_idx % SYNC_TARGET_FRAMES == 0:
        tgt_net.load_state_dict(net.state_dict())
        net_checkpoint = "/content/drive/MyDrive/Colab Notebooks/RL/Project/net"+str(epoch+1)+".pt"
        torch.save(net.state_dict(), net_checkpoint)
        tgt_checkpoint = "/content/drive/MyDrive/Colab Notebooks/RL/Project/tgt"+str(epoch+1)+".pt"
        torch.save(tgt_net.state_dict(), tgt_checkpoint)

    optimizer.zero_grad()
    batch = buffer.sample(BATCH_SIZE)
    loss_t = calc_loss(batch, net, tgt_net, device=device)
    loss_t.backward()
    optimizer.step()
