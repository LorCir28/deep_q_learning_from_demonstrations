import gym
import argparse
import torch

from agent import ImitationAgent
#from ex_ag import ImitationAgent
DEFAULT_ENV_NAME = "MontezumaRevenge-v4"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, help="Model for let play the agent")
    args = parser.parse_args()

    env = gym.make(DEFAULT_ENV_NAME, render_mode="rgb_array")
    net = ImitationAgent(env.action_space.n,1)
    #net = ImitationAgent((3,210,160),env.action_space.n)
    state = torch.load(args.model, map_location=lambda stg, _: stg)
    net.load_state_dict(state)
    env = gym.wrappers.RecordVideo(env, 'video')
    
    rewards = []
    
    for ep in range(100):
        done = False
        tot_rew = 0
        obs = env.reset()

        while not done:

            action = net.act(obs)
            obs, reward, done, info = env.step(action)
            env.render()


