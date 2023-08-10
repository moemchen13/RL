import os

import laserhockey.hockey_env as h_env
import matplotlib.pyplot as plt
import numpy as np
import torch
from dsac import DSAC_Agent
from sac import SAC_Agent

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def calculate_winrate(win,loss,tie):
    return win/(win+loss+tie)

def create_winrate_timeline(wins,losses,draws):
    winrate = []
    for win,loss,draw in zip(wins,losses,draws):
        winrate.append(win/(loss+draw+win))
    return winrate

def play_hockey_game(agent,opponent,rounds = 1000,time_one_round=400):
    agent.eval()
    loss = 0
    win = 0
    tie = 0

    env = h_env.HockeyEnv()
    for run in range(0,rounds):
        ob, info = env.reset()
        ob_opponent = env.obs_agent_two()
        for t in range(1,time_one_round):
            action = agent.act(ob)
            opponent_action = opponent.act(ob_opponent)
            #opponent_action = np.zeros_like(opponent_action)
            ob_new,reward,done,trunc,info = env.step(np.hstack([action,opponent_action]))
            info_opponent = env.get_info_agent_two()
            
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

def create_opponent(hard=False):
    opponent = h_env.BasicOpponent(weak=True)
    if hard:
        opponent = h_env.BasicOpponent(weak=False)
    return opponent

def create_Agent(DSAC=True):
    torch.manual_seed(42)
    np.random.seed(42)
    if DSAC:
        agent = DSAC_Agent(h_env.HockeyEnv().observation_space,h_env.HockeyEnv().action_space)
    else:
        agent = SAC_Agent(h_env.HockeyEnv().observation_space,h_env.HockeyEnv().action_space)
    return agent

def load_Agent(agent, file,cuda=True):
    if cuda:
        agent.load_network_states(torch.load(file, map_location=torch.device('cpu')))
    else:
        agent.load_network_states(torch.load(file))

def create_timeline_winrate(agent,opponent,prefix_file,suffix_file,length = 40,step=500):
    wins = []
    losses = []
    draws = []
    win,loss,tie = play_hockey_game(agent,opponent)
    wins.append(win)
    losses.append(loss)
    draws.append(tie)

    for episode in range(step,length*step+step,step):
        loading_file = prefix_file + str(episode) + suffix_file
        load_Agent(agent,loading_file)
        win,loss,tie = play_hockey_game(agent,opponent)
        wins.append(win)
        losses.append(loss)
        draws.append(tie)
    return wins, losses,draws

def plot_winrates(filename,title,legendes,winrates,points):
    for legend_label,winrate in zip(legendes,winrates):
        stop = points*500+500
        plt.plot(range(0,stop,500),winrate,'-o')
        plt.legend(legend_label)
        #TODO add axis titles
    plt.suptitle(title)
    plt.savefig(filename)


def dumb_output(file,prefix,wins,losses,draws):
    f = open(file,"a")
    f.write(f"{prefix};{wins};{losses};{draws}\n")
    f.close()
    

def create_winrates(all_prefixes,all_suffixes,are_DSAC,opponent,steps,points=2):
    timelines = []
    for prefix,suffix,is_DSAC,step in zip(all_prefixes,all_suffixes,are_DSAC,steps):
        agent = create_Agent(is_DSAC)
        wins,losses,draws = create_timeline_winrate(agent,opponent,prefix,suffix,length=points,step=step)
        dumb_output('stats.txt',prefix,wins,losses,draws)
        timeline = create_winrate_timeline(wins,losses,draws)
        timelines.append(timeline)
    return timelines


def main():
    file = 'winrate-different_training.png'
    title = 'Winrate over time for different training methods against weak Opponent'
    legendes = ['DSAC against easy','DSAC against hard','SAC against easy','SAC against hard']

    opponent = create_opponent()
    file_prefixes = ['./results/DSAC/DSAC_easy-e','./results/DSAC_against_hard/DSAC_against_hard_hard-e','./results/SAC/SAC_easy-e','./results/SAC_against_hard/SAC_against_hard_hard-e',]
    files_suffixes = ['-t32-s42-player.pth','-t32-s42-player.pth','-t32-s42-player.pth','-t32-s42-player.pth']
    are_DSAC = [True,True,False,False]
    steps = [500,500,500,500]

    points_in_plot = 2
    winrates = create_winrates(file_prefixes,files_suffixes,are_DSAC,opponent,steps,points=points_in_plot)

    plot_winrates(file,title,legendes,winrates,points_in_plot)


if __name__ == '__main__':
    main()