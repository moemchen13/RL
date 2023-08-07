import pickle

import matplotlib.pyplot as plt
import numpy as np

episode = 500
max_episode = 10000
all_rewards = []
all_q_losses = []
all_pi_losses = []
all_temperature_losses = []

while episode <= max_episode:
    with open(f"SAC_run_easy-s42-e{episode}-stat.pkl", 'rb') as f:
        data = pickle.load(f)
        rewards = np.asarray(data["rewards"])
        q_losses =  np.asarray(data["q_losses"])
        pi_losses  = np.asarray(data["pi_losses"])
        temperature_losses = np.asarray(data["temperature_loss"])
        episode += 500
        all_rewards.extend(rewards)
        all_q_losses.extend(q_losses)
        all_pi_losses.extend(pi_losses)
        all_temperature_losses.extend(temperature_losses)


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)    

plt.plot(running_mean(all_rewards,100))
plt.savefig("reward_SAC")