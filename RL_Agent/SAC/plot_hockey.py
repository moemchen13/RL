import pickle

import matplotlib.pyplot as plt
import numpy as np

with open("DSAC_Easy_easy-s42-e3000-stat.pkl", 'rb') as f:
    data = pickle.load(f)
    rewards = np.asarray(data["rewards"])
    q_losses =  np.asarray(data["q_losses"])
    pi_losses  = np.asarray(data["pi_losses"])
    temperature_losses = np.asarray(data["temperature_loss"])


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)    

plt.plot(running_mean(rewards,100))
plt.savefig("reward_SAC")