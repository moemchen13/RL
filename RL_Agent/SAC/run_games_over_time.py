import argparse
import os

import laserhockey.hockey_env as h_env
import matplotlib.pyplot as plt
import numpy as np
import torch
from dsac import DSAC_Agent
from sac import SAC_Agent


def play_hockey_game(agent,opponent,max_episode = 10,time_one_round=400):
    agent.eval()
    loss = 0
    win = 0
    tie = 0

    env = h_env.HockeyEnv()
    for run in range(0,max_episode):
        ob, info = env.reset()
        ob_opponent = env.obs_agent_two()
        for t in range(1,time_one_round):
            action = agent.act(ob)
            opponent_action = opponent.act(ob_opponent)
            #opponent_action = np.zeros_like(opponent_action)
            ob_new,reward,done,trunc,info = env.step(np.hstack([action,opponent_action]))
            
            if done:
                if env.winner ==1:
                    win +=1
                elif env.winner == -1: 
                    loss +=1
                else:
                    tie +=1
                break

            ob = ob_new
            ob_opponent = env.obs_agent_two()
    env.close()
    return win,loss,tie


def create_Agent(DSAC=True):
    torch.manual_seed(42)
    np.random.seed(42)
    if DSAC:
        agent = DSAC_Agent(h_env.HockeyEnv().observation_space,h_env.HockeyEnv().action_space)
    else:
        agent = SAC_Agent(h_env.HockeyEnv().observation_space,h_env.HockeyEnv().action_space)
    return agent


def create_opponent(hard=False):
    opponent = h_env.BasicOpponent(weak=True)
    if hard:
        opponent = h_env.BasicOpponent(weak=False)
    return opponent

def load_Agent(agent, file,cuda=True):
    if cuda:
        agent.load_network_states(torch.load(file, map_location=torch.device('cpu')))
    else:
        agent.load_network_states(torch.load(file))


def run_games(agent,opponent,step,points_in_plot,prefix,suffix,max_episode):
    
    win,loss,tie = play_hockey_game(agent,opponent,max_episode=max_episode)
    stats = np.array([win,loss,tie])

    for episode in range(step,points_in_plot*step+step,step):
        print(f'loaded agent for episode {episode}')
        loading_file = prefix + str(episode) + '-' + suffix
        load_Agent(agent,loading_file)
        win,loss,tie = play_hockey_game(agent,opponent,max_episode=max_episode)
        stats = np.vstack([stats,[win,loss,tie]])

    return stats

def test_loading(agent,prefix,suffix):
    load_Agent(agent,prefix+'1000'+suffix)
    print('found files')
    print(agent,prefix+'1000'+suffix)


def main():

    parser = argparse.ArgumentParser(prog='run_hockey')
    parser.add_argument('-f', '--file', dest='file',default="result.txt", help='file to save to')
    parser.add_argument('-p', '--prefix',dest='prefix',default='',help='prefix of the file')
    parser.add_argument('-s', '--suffix',dest='suffix',default='t32-s42-player.pth',help='file suffix (default %(default)s)')
    parser.add_argument('-d', '--dsac',dest='dsac',action='store_true',help='is DSAC or not')
    parser.add_argument('-e', '--episodesteps',dest='episodesteps',default=500,help='stepsize (default %(default)s)')
    parser.add_argument('-o', '--opponent_hard',dest='opponent',action='store_true',help='opponent hard')
    parser.add_argument('-i', '--points',dest='points',default=40,help='points time length the thing runs (default %(default)s)')
    parser.add_argument('-r', '--rounds',dest='rounds',default=1000,help='rounds to evaluate performance on (default %(default)s)')
    parser.add_argument('-t','--test',dest='test',action="store_true",help='checks if files is valid')
    opts = parser.parse_args()
    ############## Hyperparameters ##############
    weak = not bool(opts.opponent)
    prefix = opts.prefix
    suffix = opts.suffix
    suffix = '-' + suffix
    is_DSAC = bool(opts.dsac)
    steps = int(opts.episodesteps)
    points_in_plot = int(opts.points)
    file_name = opts.file
    max_episode = int(opts.rounds)
    test = opts.test
    opponent = create_opponent(weak)
    agent = create_Agent(is_DSAC)
    if test:
        test_loading(agent,prefix,suffix)
    else:
        print('start running games')
        stats = run_games(agent,opponent,steps,points_in_plot,prefix,suffix,max_episode=max_episode)
        np.save(file_name,stats,allow_pickle=True)

if __name__ == '__main__':
    main()