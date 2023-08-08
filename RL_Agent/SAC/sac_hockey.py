import argparse
import pickle

import laserhockey.hockey_env as h_env
import numpy as np
import torch
from DR3 import DR3_Agent
from dsac import DSAC_Agent
from sac import SAC_Agent


def save_statistics(rewards,lengths,q_losses,pi_losses,temperature_loss,env_name,random_seed,episode,name):
    
    with open(f"./{name}_{env_name}-s{random_seed}-e{episode}-stat.pkl", 'wb') as f:
        pickle.dump({"rewards" : rewards, "lengths": lengths,
                        "pi_losses": pi_losses, "q_losses": q_losses,
                        "temperature_loss":temperature_loss}, f)
    

def reward_shaping(reward,info,touched_puck):
    #reward shaping should encourage interaction with ball and penailze if none available
    reward += int(touched_puck) * 0.1 
    #shaping should improve movement to ball for defense and attack
    reward += info["reward_closeness_to_puck"]*1
    #want to encourage shoots on goal
    #But trying to let the direction lose by shoting onto boundaries
    reward += info["reward_puck_direction"]*0.01

def create_agent(agent,filename,from_cuda):
    env = h_env.HockeyEnv()
    if agent == 'SAC':
        agent = SAC_Agent(env.observation_space,env.action_space)
    elif agent == 'DSAC':
        agent = DSAC_Agent(env.observation_space,env.action_space) 
    elif agent == 'DR3':
        agent = DR3_Agent(env.observation_space,env.action_space)
        
    env.close()
    if filename != '':
        agent.load_network_states_from_file(filename,from_cuda)
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
    

def run_sac_agent_hockey_game(agent,mode,log_interval,save_interval,max_episodes,
                                 max_timesteps,train_iter,random_seed,name="",shape_rewards=True):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    env = h_env.HockeyEnv()
    if mode == "easy":  
        opponent = h_env.BasicOpponent(weak=True)
    elif mode == "hard":
        opponent = h_env.BasicOpponent(weak=False)
    
    rewards = []
    lengths = []
    q_losses = []
    policy_losses = []
    temperature_losses = []

    player = agent

    #leftsided
    win = 0
    loss = 0
    tie = 0

    for episode in range(1,max_episodes+1):
        ob, _info = env.reset()
        total_reward=0
        done = False
        touched_puck = False
        for t in range(max_timesteps):
            ob_opponent = env.obs_agent_two()
            action_player = player.act(ob)
            action_opponent = opponent.act(ob_opponent)
            action_environment = np.hstack([action_player,action_opponent])

            (ob_new,reward,done,trunc,info) = env.step(action_environment)
            
            if info["reward_touch_puck"]!=0:
                touched_puck=True

            reward_shaped = reward
            if shape_rewards:
                reward_shaped = reward_shaping(reward,info,touched_puck)

            total_reward += reward_shaped
            
            player.store_transition((ob,action_player,reward_shaped,ob_new,done))
            ob=ob_new


            if done or trunc: 
                if done:
                    if env.winner ==1:
                        win +=1
                    elif env.winner == -1: 
                        loss +=1
                    else:
                        tie +=1
                break

        q_loss,pi_loss,temperature_loss = player.train(train_iter)
        
        q_losses.extend(q_loss)
        policy_losses.extend(pi_loss)
        temperature_losses.extend(temperature_loss)
        rewards.append(total_reward)
        lengths.append(t)

        if episode % log_interval == 0:
            avg_reward = np.mean(rewards[-log_interval:])
            avg_length = int(np.mean(lengths[-log_interval:]))
            print(f'Player: Episode {episode} \t avg length: {avg_length} \t avg_reward: {avg_reward} \t wins: {win} \t losses: {loss} \t  ties: {tie} \t last reward: {int(rewards[-1])}')
        

        if episode % save_interval == 0:
            print("########### Save checkpoint ################")
            torch.save(player.get_networks_states(),f'./{name}_{mode}-e{episode}-t{train_iter}-s{random_seed}-player.pth')
            save_statistics(rewards,lengths,q_losses,policy_losses,temperature_losses,mode,random_seed,episode,name)
            rewards = []
            lengths = []
            q_losses = []
            policy_losses = []
            temperature_losses = []


    save_statistics(rewards,lengths,q_losses,policy_losses,temperature_losses,mode,random_seed,episode,name)


def run_sac_agent_against_yourself(agent,enemy,log_interval,save_interval,max_episodes,
                                 max_timesteps,train_iter,random_seed,name="",show_both_logs=True,shape_rewards=False):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    env = h_env.HockeyEnv()
    player = agent
    opponent = enemy

    rewards = []
    lengths = []
    q_losses = []
    policy_losses = []
    temperature_losses = []

    rewards_opponent = []
    lengths_opponent = []
    q_losses_opponent = []
    policy_losses_opponent = []
    temperature_losses_opponent = []

    #leftsided
    win = 0
    loss = 0
    tie = 0

    for episode in range(1,max_episodes+1):
        ob, _info = env.reset()
        ob_opponent = env.obs_agent_two()
        total_reward=0
        total_reward_opponent=0
        done = False
        touched_puck = False
        touched_puck_opponent = False
        for t in range(max_timesteps):
            
            action_player = player.act(ob)
            action_opponent = opponent.act(ob_opponent)
            (ob_new,reward,done,trunc,info) = env.step(np.hstack([action_player,action_opponent]))
            ob_new_opponent = env.obs_agent_two()
            info_opponent = env.get_info_agent_two()
            reward = env.get_reward(info)
            reward_opponent = env.get_reward_agent_two(info_opponent)
            if info["reward_touch_puck"] !=0:
                touched_puck=True
            if info_opponent["reward_touch_puck"] !=0:
                touched_puck_opponent=True
            
            reward_shaped = reward
            reward_shaped_opponent = reward_opponent
            if shape_rewards:
                reward_shaped = reward_shaping(reward,info,touched_puck)
                reward_shaped_opponent = reward_shaping(reward_opponent,info_opponent,touched_puck_opponent)
            
            total_reward += reward_shaped
            total_reward_opponent += reward_shaped_opponent
            
            player.store_transition((ob,action_player,reward_shaped,ob_new,done))
            opponent.store_transition((ob_opponent,action_opponent,reward_shaped_opponent,ob_new_opponent,done))
            ob=ob_new
            ob_opponent=env.obs_agent_two() 
            if done or trunc: 
                if done:
                    if env.winner ==1:
                        win +=1
                    elif env.winner == -1: 
                        loss +=1
                    else:
                        tie +=1
                break

        q_loss_opponent,pi_loss_opponent,temperature_loss_opponent = opponent.train(train_iter)
        q_loss,pi_loss,temperature_loss = player.train(train_iter)
        
        q_losses_opponent.extend(q_loss_opponent)
        policy_losses_opponent.extend(pi_loss_opponent)
        temperature_losses_opponent.extend(temperature_loss_opponent)
        rewards_opponent.append(total_reward_opponent)
        lengths_opponent.append(t)
            
        q_losses.extend(q_loss)
        policy_losses.extend(pi_loss)
        temperature_losses.extend(temperature_loss)
        rewards.append(total_reward)
        lengths.append(t)


        if episode % log_interval == 0:
            avg_reward = np.mean(rewards[-log_interval:])
            avg_length = int(np.mean(lengths[-log_interval:]))
            print(f'Player2: Episode {episode} \t avg length: {avg_length} \t reward: {avg_reward} \t wins: {win} \t losses: {loss} \t ties: {tie} \t last reward: {int(rewards[-1])}')
            
            if show_both_logs:
                avg_reward = np.mean(rewards_opponent[-log_interval:])
                avg_length = int(np.mean(lengths_opponent[-log_interval:]))
                print(f'Player1: Episode {episode} \t avg length: {avg_length} \t reward: {avg_reward} \t wins: {loss} \t losses: {win} \t ties: {tie} \t last reward: {int(rewards_opponent[-1])}')


        if episode % save_interval == 0:
            print("########### Save checkpoint ################")
            torch.save(player.get_networks_states(),f'./{name}_self_play-e{episode}-t{train_iter}-s{random_seed}-player2.pth')
            torch.save(opponent.get_networks_states(),f'./{name}_self_play-e{episode}-t{train_iter}-s{random_seed}-player1.pth')
            save_statistics(rewards,lengths,q_losses,policy_losses,temperature_losses,"self_play_player1",random_seed,episode,name)
            save_statistics(rewards_opponent,lengths_opponent,q_losses_opponent,policy_losses_opponent,temperature_losses_opponent,"self_play_player2",random_seed,episode,name)
            rewards = []
            lengths = []
            q_losses = []
            policy_losses = []
            temperature_losses = []

            rewards_opponent = []
            lengths_opponent = []
            q_losses_opponent = []
            policy_losses_opponent = []
            temperature_losses = []

        
    save_statistics(rewards_opponent,lengths_opponent,q_losses_opponent,policy_losses_opponent,temperature_losses_opponent,"self_play_player2",random_seed,episode,name)
    save_statistics(rewards,lengths,q_losses,policy_losses,temperature_losses,"self_play_player1",random_seed,episode,name)


def main():
    parser = argparse.ArgumentParser(prog='RL Agents',
                    description='This programm trains the given agent on the Laserhockey environment',
                    epilog='Have a look at our repository at https://github.com/moemchen13/RL.git')

    parser.add_argument('-n', '--name', dest='name',default="SAC_run", help='Name for run')
    parser.add_argument('-m', '--mode',choices=["Attack","Defense","self","easy","hard"],default="easy",
                         help='Possible modes to play (default %(default)s) Options: Attack/Defense/self/easy/hard')
    parser.add_argument('-t', '--train_iter',default=32,help='number of training batches per episode (default %(default)s)')
    parser.add_argument('-e', '--maxepisodes',default=2000,help='number of episodes (default %(default)i)')
    parser.add_argument('-u', '--update',default=1,help='number of episodes between target network updates (default %(default)s)')
    parser.add_argument('-s', '--seed',default=42,help='random seed (default %(default)s)')
    parser.add_argument('-a', '--agent',choices=["SAC","DR3","DSAC"],default='SAC',
                         help='Choose an Agent you wanna activate (default %(default)s) Options: SAC/DSAC/DR3')
    parser.add_argument('-o', '--opponent',choices=["SAC","DR3","DSAC"],default='SAC',
                         help='Choose an opponent only in self (default %(default)s) Options: SAC/DSAC/DR3')
    parser.add_argument('-f', '--file',default='',help='load Agent from file')
    parser.add_argument('-v','--enemyfile',default='',help='weights of the opponent')
    parser.add_argument('-c', '--cuda',default=False,help='load Agent from cuda file')
    parser.add_argument('-k', '--cudaenemy',default=False,help='load enemy from cuda file')
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
    random_seed = int(opts.seed)
    save_interval=500
    reward_shaping = False
    file_of_weights = opts.file
    from_cuda = opts.cuda
    from_cuda_enemy = opts.cudaenemy
    enemy_file = opts.enemyfile
    #############################################
    
    agent = create_agent(agent,file_of_weights,from_cuda)    
    if mode == "self":
        enemy = create_agent(opponent,enemy_file,from_cuda_enemy)
        run_sac_agent_against_yourself(agent,enemy,log_interval,save_interval,max_episodes,
                                 max_timesteps,train_iter,random_seed,name=run_name,shape_rewards=reward_shaping)
    elif mode=="Defense" or mode=="Attack":
        run_sac_agent_in_env_modes(agent,mode,log_interval,save_interval,max_episodes,
                                 max_timesteps,train_iter,random_seed,name=run_name)
    else:
        run_sac_agent_hockey_game(agent,mode,log_interval,save_interval,max_episodes,
                                 max_timesteps,train_iter,random_seed,run_name,shape_rewards=reward_shaping)

if __name__ == '__main__':
    main()