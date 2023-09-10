#!/usr/bin/env python3
import gym
import time
import argparse
import numpy as np
import collections
import torch
import pickle

from wrap import *
from dqn import DQN


#DEFAULT_ENV_NAME = "ALE/MontezumaRevenge-v5"
DEFAULT_ENV_NAME = "Pong-v4"
FPS = 25

states = []
actions = []

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, help="Model for let play the agent")
    args = parser.parse_args()

    env = make_env(DEFAULT_ENV_NAME)
    net = DQN(env.observation_space.shape,env.action_space.n)
    state = torch.load(args.model, map_location=lambda stg, _: stg)
    net.load_state_dict(state)
    env = gym.wrappers.RecordVideo(env, 'video')

    
    for i in range(20):
        state = env.reset()
        total_reward = 0.0
        c = collections.Counter()

        while True:
            start_ts = time.time()
            env.render()
            states.append(state)
            state_v = torch.tensor(np.array([state], copy=False))
            q_vals = net(state_v).data.numpy()[0]
            action = np.argmax(q_vals)
            actions.append(action)
            c[action] += 1
            state, reward, done, _ = env.step(action)
            
            total_reward += reward
            if done:
                break
            
            delta = 1/FPS - (time.time() - start_ts)
            if delta > 0:
                time.sleep(delta)

        print("Total reward: %.2f" % total_reward)
        print("Action counts:", c)

    with open('pong_expert_trace.pkl', 'wb') as f:
        pickle.dump([states,actions], f) 
