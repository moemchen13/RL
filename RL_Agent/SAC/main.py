import pickle
import time

import gymnasium as gym
import numpy as np
import pylab as plt
import torch
from DR3 import DR3_Agent
from gymnasium import spaces
from matplotlib import cm
from sac import SAC_Agent


def save_statistics(rewards,lengths,q_losses,pi_losses,temperature_loss,env_name,random_seed,episode,regularized=False):
    version = ""
    if regularized:
        version = "DR3"
    with open(f"./results/SAC_{version}_{env_name}-s{random_seed}-e{episode}-stat.pkl", 'wb') as f:
        pickle.dump({"rewards" : rewards, "lengths": lengths, "train": train_iter,
                        "pi_losses": pi_losses, "q_losses": q_losses,
                        "temperature_loss":temperature_loss}, f)


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)  


def plot(name,file,stepsize=50):
    with open(file, 'rb') as f:
        data = pickle.load(f)
        rewards = np.asarray(data["rewards"])
        q_losses =  np.asarray(data["q_losses"])
        pi_losses  = np.asarray(data["pi_losses"])
        temperature_losses = np.asarray(data["temperature_loss"])

    plt.plot(running_mean(q_losses,stepsize),label=f"Q loss")
    plt.plot(running_mean(pi_losses,stepsize),label=f"Pi loss")
    plt.plot(running_mean(temperature_losses,stepsize),label=f"Temp loss")
    plt.legend()
    plt.savefig(f"./{name}_loss_env_{env_name}_episode_{max_episodes}.jpg")

    plt.plot(running_mean(rewards,10),label=f"rewards")
    plt.legend()
    plt.savefig(f"./{name}_rewards_env_{env_name}_episode_{max_episodes}.jpg")


def plot_Q_function(q_function, observations, actions, plot_dim1=0, plot_dim2=2,
                    label_dim1="cos(angle)", label_dim2="angular velocity",filename="x.jpg"):
    plt.rcParams.update({'font.size': 12})
    observations_tensor = torch.from_numpy(observations)
    actions_tensor = torch.from_numpy(actions)
    values =q_function.get_min_Q_value(observations_tensor,actions_tensor).detach().numpy()
    
    fig = plt.figure(figsize=[10,8])
    ax = fig.add_subplot()
    surf = ax.scatter (observations[:,plot_dim1], observations[:,plot_dim2],  c = values, cmap=cm.coolwarm)
    ax.set_xlabel(label_dim1)
    ax.set_ylabel(label_dim2)

    fig.savefig(filename)

def run(env, agent, n_episodes=100):
    rewards = []
    observations = []
    actions = []
    agent.eval
    for ep in range(1, n_episodes+1):
        ep_reward = 0
        state, _info = env.reset()
        for t in range(2000):
            action = agent.act(state)
            state, reward, done, _trunc, _info = env.step(action)
            observations.append(state)
            actions.append(action)
            ep_reward += reward
            if done or _trunc:
                break
        rewards.append(ep_reward)
        ep_reward = 0
    print(f'Mean reward: {np.mean(rewards)}')
    observations = np.asarray(observations)
    actions = np.asarray(actions)
    return observations, actions, rewards


def plot_Q(filename_model,filename_plot,regularized=False):
    env = gym.make("Pendulum-v1")
    if regularized:
        agent = SAC_Agent(env.observation_space, env.action_space)
    else:
        agent = DR3_Agent(env.observation_space, env.action_space)
    agent.load_network_states(torch.load(filename_model))
    agent.eval()
    observations, actions, rewards = run(env,agent,100)
    plot_Q_function(agent.critic,observations,actions,filename=filename_plot)


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
        agent = DR3_Agent(env.observation_space,env.action_space)
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
            if regularized:
                torch.save(agent.get_networks_states(),f'./SAC_DR3_{version}_{env_name}-e{episode}-t{train_iter}-s{random_seed}.pth')
            else:
                torch.save(agent.get_networks_states(),f'./SAC_{version}_{env_name}-e{episode}-t{train_iter}-s{random_seed}.pth')
            save_statistics(rewards,lengths,q_losses,policy_losses,temperature_losses,env_name,random_seed,episode)

        if episode % log_interval == 0:
            avg_reward = np.mean(rewards[-log_interval:])
            avg_length = int(np.mean(lengths[-log_interval:]))
            print('Episode {} \t avg length: {} \t reward: {}'.format(episode, avg_length, avg_reward))
        
    save_statistics(rewards,lengths,q_losses,policy_losses,temperature_losses,env_name,random_seed,episode)
    

env_name = "Pendulum-v1"
#env_name = "LunarLander-v2"
log_interval = 20         # print avg reward in the interval
max_episodes = 1000 # max training episodes
max_timesteps = 2000         # max timesteps in one episode
save_interval = 1000
train_iter = 32      # update networks for given batched after every episode
random_seed = 42
time_plot_intervall = 1000

print(f"Start training on {env_name}")
run_sac_agent_in_environment(env_name,log_interval,save_interval,max_episodes,max_timesteps,train_iter,random_seed)
filename = f".SAC_{env_name}-s{random_seed}-e{max_episodes}-stat.pkl"
plot("SAC",filename)
filename_model = f"./SAC_{env_name}-e{int(max_episodes/save_interval)*save_interval}-t{train_iter}-s{random_seed}.pth"
filename_plot = "SAC_Q_Function.jpg"
plot_Q(filename_model,filename_plot,regularized=False)
print(f"Finished running normal SAC on {env_name}")

run_sac_agent_in_environment(env_name,log_interval,save_interval,max_episodes,max_timesteps,train_iter,random_seed,regularized=True)
filename = f"./SAC_DR3_{env_name}-s{random_seed}-e{max_episodes}-stat.pkl"
plot("DR3",filename)
filename_model = f"./SAC_DR3_{env_name}-e{int(max_episodes/save_interval)*save_interval}-t{train_iter}-s{random_seed}.pth"
filename_plot = "DR3_Q_Function.jpg"
plot_Q(filename_model,filename_plot,regularized=False)
print(f"Finished running DR3 regularized SAC on {env_name}")


env_name = "LunarLander-v2"
#env_name = "HalfCheetah-v4"
log_interval = 20         # print avg reward in the interval
max_episodes = 1000 # max training episodes
max_timesteps = 1000        # max timesteps in one episode
save_interval = 1000
train_iter = 32      # update networks for given batched after every episode
random_seed = 42

print(f"Start training on {env_name}")
run_sac_agent_in_environment(env_name,log_interval,save_interval,max_episodes,max_timesteps,train_iter,random_seed)
filename = f"./SAC_{env_name}-s{random_seed}-e{max_episodes}-stat.pkl"
plot("SAC",filename)
print(f"Finished running normal SAC on {env_name}")

run_sac_agent_in_environment(env_name,log_interval,save_interval,max_episodes,max_timesteps,train_iter,random_seed,regularized=True)
filename = f"./SAC_DR3_{env_name}-s{random_seed}-e{max_episodes}-stat.pkl"
plot("DR3",filename)
print(f"Finished running DR3 regularized SAC on {env_name}")


#env_name = "LunarLander-v2"
env_name = "HalfCheetah-v4"
log_interval = 20         # print avg reward in the interval
max_episodes = 3000 # max training episodes
max_timesteps = 2000        # max timesteps in one episode
save_interval = 3000
train_iter = 32      # update networks for given batched after every episode
random_seed = 42

print(f"Start training on {env_name}")
run_sac_agent_in_environment(env_name,log_interval,save_interval,max_episodes,max_timesteps,train_iter,random_seed)
filename = f"./SAC_{env_name}-s{random_seed}-e{max_episodes}-stat.pkl"
plot("SAC",filename)
print(f"Finished running normal SAC on {env_name}")

run_sac_agent_in_environment(env_name,log_interval,save_interval,max_episodes,max_timesteps,train_iter,random_seed,regularized=True)
filename = f"./SAC_DR3_{env_name}-s{random_seed}-e{max_episodes}-stat.pkl"
plot("DR3",filename)
print(f"Finished running DR3 regularized SAC on {env_name}")