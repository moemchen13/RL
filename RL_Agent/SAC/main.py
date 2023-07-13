import pickle
import time

import gymnasium as gym
import numpy as np
import torch
from DR3 import DR3_Agent
from gymnasium import spaces
from sac import SAC_Agent


def save_statistics(rewards,lengths,q_losses,pi_losses,temperature_loss,env_name,random_seed,episode,regularized=False):
    version = ""
    if regularized:
        version = "DR3"
    with open(f"./results/SAC_{version}_{env_name}-s{random_seed}-e{episode}-stat.pkl", 'wb') as f:
        pickle.dump({"rewards" : rewards, "lengths": lengths, "train": train_iter,
                        "pi_losses": pi_losses, "q_losses": q_losses,
                        "temperature_loss":temperature_loss}, f)


def run_sac_agent_in_environment(env_name,log_interval,save_interval,max_episodes,
                                 max_timesteps,train_iter,random_seed,regularized=False):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    env = gym.make(env_name)

    if env_name == "LunarLander-v2":
        env = gym.make(env_name,continuous=True)
    else:
        env = gym.make(env_name)

    if regularized:
        agent = DR3_Agent(env.observation_space)
    else:
        agent = SAC_Agent(env.observation_space, env.action_space)
    
    rewards = []
    lengths = []
    q_losses = []
    policy_losses = []
    temperature_losses = []
    
    version = ""
    if regularized:
        version = "DR3"

    for episode in range(1,max_episodes+1):
        ob, _info = env.reset()
        total_reward=0
        for t in range(max_timesteps):
            timestep +=1
            done = False
            a = agent.act(ob)
            (ob_new,reward,done,trunc,_info) = env.step(a)
            total_reward += reward
            agent.store_transition((ob,a,reward,ob_new,done))
            ob=ob_new
            if done or trunc: break

        q_loss,pi_loss,temperature_loss = agent.train(train_iter)
        
        q_losses.extend(q_loss)
        policy_losses.extend(pi_loss)
        temperature_losses.extend(temperature_loss)
        rewards.append(total_reward)
        lengths.append(t)

        if episode % save_interval == 0:
            print("########### Save checkpoint ################")
            torch.save(agent.get_networks_states(),f'./results/SAC{version}_{env_name}-e{episode}-t{train_iter}-s{random_seed}.pth')
            save_statistics(rewards,lengths,q_losses,policy_losses,temperature_losses,env_name,random_seed,episode)

        if episode % log_interval == 0:
            avg_reward = np.mean(rewards[-log_interval:])
            avg_length = int(np.mean(lengths[-log_interval:]))
            print('Episode {} \t avg length: {} \t reward: {}'.format(episode, avg_length, avg_reward))
        
    save_statistics(rewards,lengths,q_losses,policy_losses,temperature_losses,env_name,random_seed,episode)
    


env_name = "Pendulum-v1"
#env_name = "LunarLander-v2"
log_interval = 20         # print avg reward in the interval
max_episodes = 2000 # max training episodes
max_timesteps = 2000         # max timesteps in one episode
save_interval = 20
train_iter = 32      # update networks for given batched after every episode
random_seed = 42
time_plot_intervall = 1000

run_sac_agent_in_environment(env_name,log_interval,save_interval,max_episodes,max_timesteps,train_iter,random_seed)
run_sac_agent_in_environment(env_name,log_interval,save_interval,max_episodes,max_timesteps,train_iter,random_seed,regularized=True)

env_name = "HalfCheetah-v4"
log_interval = 20         # print avg reward in the interval
max_episodes = 10000 # max training episodes
max_timesteps = 2000         # max timesteps in one episode
save_interval = 10000
train_iter = 32      # update networks for given batched after every episode
random_seed = 42

run_sac_agent_in_environment(env_name,log_interval,save_interval,max_episodes,max_timesteps,train_iter,random_seed)
run_sac_agent_in_environment(env_name,log_interval,save_interval,max_episodes,max_timesteps,train_iter,random_seed,regularized=True)