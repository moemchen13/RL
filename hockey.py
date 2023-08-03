import argparse
import pickle
from importlib import reload

import gymnasium as gym
import laserhockey.hockey_env as h_env
import numpy as np
import pylab as plt
import torch
from IPython import display

from RL_Agent.SAC_Agents.SAC.DR3 import DR3_Agent
from RL_Agent.SAC_Agents.SAC.dsac import DSAC_Agent
from RL_Agent.SAC_Agents.SAC.sac import SAC_Agent


def save_statistics(rewards,lengths,q_losses,pi_losses,temperature_loss,env_name,random_seed,episode,name):
    
    with open(f"./{name}_{env_name}-s{random_seed}-e{episode}-stat.pkl", 'wb') as f:
        pickle.dump({"rewards" : rewards, "lengths": lengths,
                        "pi_losses": pi_losses, "q_losses": q_losses,
                        "temperature_loss":temperature_loss}, f)


def reward_shaping(reward,info,player1):
    #TODO: implement reward shaping from additional information
    if player1:
        return -reward
    return reward

def create_agent(agent):
    env = h_env.HockeyEnv()
    if agent == 'SAC':
        agent = SAC_Agent(env.observation_space,env.action_space)
    elif agent == 'DSAC':
        agent = DSAC_Agent(env.observation_space,env.action_space) 
    elif agent == 'DR3':
        agent = DR3_Agent(env.observation_space,env.action_space)
        env.close()
    return agent

def run_sac_agent_in_env_modes(agent,mode,log_interval,save_interval,max_episodes,
                                 max_timesteps,train_iter,random_seed,name=""):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    if mode == "Defense":
        env = h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_DEFENSE)
    elif mode == "Attack":
        env = h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_SHOOTING)
    obs,info = env.reset()
    
    rewards = []
    lengths = []
    q_losses = []
    policy_losses = []
    temperature_losses = []

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
            torch.save(agent.get_networks_states(),f'./{name}_{mode}-e{episode}-t{train_iter}-s{random_seed}.pth')
            save_statistics(rewards,lengths,q_losses,policy_losses,temperature_losses,mode,random_seed,episode,name)

        if episode % log_interval == 0:
            avg_reward = np.mean(rewards[-log_interval:])
            avg_length = int(np.mean(lengths[-log_interval:]))
            print('Episode {} \t avg length: {} \t reward: {}'.format(episode, avg_length, avg_reward))
        
    save_statistics(rewards,lengths,q_losses,policy_losses,temperature_losses,mode,random_seed,episode,name)
    

def run_sac_agent_hockey_game(agent,opponent,mode,log_interval,save_interval,max_episodes,
                                 max_timesteps,train_iter,random_seed,name="",show_both_logs=True):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    player2 = agent

    if mode == "easy":
        env = h_env.HockeyEnv()
        player1 = h_env.BasicOpponent(weak=True)
    elif mode == "hard":
        player1 = h_env.HockeyEnv(weak=False)
    elif mode == "self":
        player1 = opponent
    
        rewards1 = []
        lengths1 = []
        q_losses1 = []
        policy_losses1 = []
        temperature_losses1 = []
        raise NotImplementedError("Implement self training")


    obs,info = env.reset()
    
    rewards2 = []
    lengths2 = []
    q_losses2 = []
    policy_losses2 = []
    temperature_losses2 = []


    for episode in range(1,max_episodes+1):
        ob, _info = env.reset()
        ob_player_2 = env.obs_agent_two()
        total_reward=0
        for t in range(max_timesteps):
            done = False
            action_player_1 = player1.act(ob)
            action_player_2 = player2.act(ob_player_2)
            (ob_new,reward,done,trunc,info) = env.step([action_player_1,action_player_2])
            total_reward += reward
            player2.store_transition((ob,action_player_2,reward,ob_new,done))
            ob=ob_new
            ob_player_2=env.obs_agent_two() 
            if done or trunc: break

        q_loss,pi_loss,temperature_loss = player2.train(train_iter)
        
        q_losses2.extend(q_loss)
        policy_losses2.extend(pi_loss)
        temperature_losses2.extend(temperature_loss)
        rewards2.append(total_reward)
        lengths2.append(t)

        if mode=="self":
            q_loss,pi_loss,temperature_loss = player1.train(train_iter)
            
            q_losses1.extend(q_loss)
            policy_losses1.extend(pi_loss)
            temperature_losses1.extend(temperature_loss)
            rewards1.append(total_reward)
            lengths1.append(t)

        if episode % save_interval == 0:
            print("########### Save checkpoint ################")
            torch.save(player2.get_networks_states(),f'./{name}_{mode}-e{episode}-t{train_iter}-s{random_seed}-player2.pth')
            if mode == "self":
                torch.save(player2.get_networks_states(),f'./{name}_{mode}-e{episode}-t{train_iter}-s{random_seed}-player1.pth')
                save_statistics(rewards1,lengths1,q_losses1,policy_losses1,temperature_losses1,mode,random_seed,episode)
            save_statistics(rewards2,lengths2,q_losses2,policy_losses2,temperature_losses2,mode,random_seed,episode)

        if episode % log_interval == 0:
            avg_reward = np.mean(rewards2[-log_interval:])
            avg_length = int(np.mean(lengths2[-log_interval:]))
            print('Player2: Episode {} \t avg length: {} \t reward: {}'.format(episode, avg_length, avg_reward))
            
            if mode=="self" and show_both_logs:
                avg_reward = np.mean(rewards1[-log_interval:])
                avg_length = int(np.mean(lengths1[-log_interval:]))
                print('Player1: Episode {} \t avg length: {} \t reward: {}'.format(episode, avg_length, avg_reward))

        
    save_statistics(rewards2,lengths2,q_losses2,policy_losses2,temperature_losses2,mode,random_seed,episode)
    if mode =="self":
        save_statistics(rewards1,lengths1,q_losses1,policy_losses1,temperature_losses1,mode,random_seed,episode)

def main():
    parser = argparse.ArgumentParser(prog='RL Agents',
                    description='This programm trains the given agent on the Laserhockey environment',
                    epilog='Have a look at our repository at https://github.com/moemchen13/RL.git')

    parser.add_argument('-n', '--name', dest='name',default="SAC_run", help='Name for run')
    parser.add_argument('-m', '--mode',choices=["Attack","Defense","self","easy","hard"],default="Defense",
                         help='Possible modes to play (default %(default)s) Options: Attack/Defense/self/easy/hard')
    parser.add_argument('-t', '--train_iter',default=32,help='number of training batches per episode (default %(default)s)')
    parser.add_argument('-l', '--lr',default=0.0001,help='learning rate for actor/policy (default %(default)s)')
    parser.add_argument('-e', '--maxepisodes',default=2000,help='number of episodes (default %(default)i)')
    parser.add_argument('-u', '--update',default=1,help='number of episodes between target network updates (default %(default)s)')
    parser.add_argument('-s', '--seed',default=42,help='random seed (default %(default)s)')
    parser.add_argument('-a', '--agent',choices=["SAC","DR3","DSAC"],default='SAC',
                         help='Choose an Agent you wanna activate (default %(default)s) Options: SAC/DSAC/DR3')
    parser.add_argument('-o', '--opponent',choices=["SAC","DR3","DSAC"],default='SAC',
                         help='Choose an opponent only in self (default %(default)s) Options: SAC/DSAC/DR3')
    opts = parser.parse_args()
    ############## Hyperparameters ##############
    run_name = opts.name
    mode = opts.mode
    agent = opts.agent
    opponent = opts.opponent

    render = False
    log_interval = 20           # print avg reward in the interval
    max_episodes = int(opts.maxepisodes) # max training episodes
    max_timesteps = 2000         # max timesteps in one episode

    train_iter = int(opts.train_iter)      # update networks for given batched after every episode
    lr  = int(opts.lr)                # learning rate of DDPG policy
    random_seed = int(opts.seed)
    save_interval=500
    #############################################
    
    agent = create_agent(agent)    
    if mode == "self":
        opponent = create_agent(opponent)
    

    if mode=="Defense" or mode=="Attack":
        run_sac_agent_in_env_modes(agent,mode,log_interval,save_interval,max_episodes,
                                 max_timesteps,train_iter,random_seed,name=run_name)
    else:
        run_sac_agent_hockey_game(agent,opponent,mode,log_interval,save_interval,max_episodes,
                                 max_timesteps,train_iter,random_seed,run_name)

if __name__ == '__main__':
    main()