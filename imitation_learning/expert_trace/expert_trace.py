""""
    Programma per realizzare le tracce dell'esperto. Per ora printa:
        - Lo stato attuale
        - L'azione compiuta
        - Il reward ottenuto
        - Lo stato successivo

    Tutti e due gli stati sono printati mostrando solo il loro primo elemento per motivi di spazio. La funzione compute_metric, oltre ad occuparsi della print
    suddetta, è utile per stampare dei grafici real time di alcune statistiche come il reward immediato e cumulativo e l'importanza dell'azione per il raggiungimento
    del reward ("Action Magnitude",penso).

    Le altre funzionalità di utils.play di Gym sono al seguente link:
        https://github.com/openai/gym/blob/master/gym/utils/play.py

"""

import gym
import numpy as np
from gym.utils.play import play, PlayPlot
import pickle
obs = list()
act = list()
tot_rew = list()

def compute_metrics(obs_t, obs_tp1, action, rew, done, info):
    #print("Prev state: " + str(obs_t[0][0]) +" || Action: " + str(action) + " || Reward: " + str(rew) + " || Next state:" + str(obs_tp1[0][0]))
    obs.append(obs_t)
    act.append(action)
    tot_rew.append(rew)
    return [rew, sum(tot_rew), np.linalg.norm(action)] #occhio ad action

#env = gym.make('ALE/MontezumaRevenge-v5', render_mode='rgb_array')
env = gym.make('Pong-v4', render_mode='rgb_array')

plotter = PlayPlot(compute_metrics, horizon_timesteps=200,plot_names=["Immediate Rew.", "Cumulative Rew.", "Action Magnitude"])
        
play(env,zoom=3,callback=plotter.callback)

with open('pong_expert_trace.pkl', 'wb') as f:
    pickle.dump([obs,act], f)