import time

import laserhockey.hockey_env as h_env
import numpy as np
import pandas as pd
import torch
from dsac import DSAC_Agent
from sac import SAC_Agent


def evaluate(file,DSAC=True):
    torch.manual_seed(42)
    np.random.seed(42)

    
    if DSAC:
        agent = DSAC_Agent(h_env.HockeyEnv().observation_space,h_env.HockeyEnv().action_space)
        agent.load_network_states(torch.load(file, map_location=torch.device('cpu')))
    else:
        agent = SAC_Agent(h_env.HockeyEnv().observation_space,h_env.HockeyEnv().action_space)
        agent.load_network_states(torch.load(file, map_location=torch.device('cpu')))
    agent.eval()

    weak_opponent = h_env.BasicOpponent(weak=True)
    strong_opponent = h_env.BasicOpponent(weak=False)
    weak_win,weak_loss,weak_tie = play(agent,weak_opponent)
    strong_win,strong_loss,strong_tie = play(agent,strong_opponent)
    return [weak_win,weak_loss,weak_tie,strong_win,strong_loss,strong_tie]
    


def play(agent,opponent):
    loss = 0
    win = 0
    tie = 0

    env = h_env.HockeyEnv()
    for run in range(0,100):
        ob, info = env.reset()
        ob_opponent = env.obs_agent_two()
        for t in range(1,1000):
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


def main():
    all_SAC = ['./results/SAC_against_hard_hard-e12500-t32-s42-player.pth']
    all_DSAC = ['./results/DSAC_after_episode10000_easy-e10500-t32-s42-player.pth',
                './results/DSAC_after_episode10000_hard-e11000-t32-s42-player.pth',
                './results/DSAC_Easy_easy-e10000-t32-s42-player.pth',
                './results/DSAC_multiple_opponents_alternating_self_play-e17000-t32-s42-player2.pth']
    results_file = "results.csv"
    columns=['filename','weak_win','weak_loss','weak_tie','strong_win','strong_loss','strong_tie']
    df = pd.DataFrame(columns=columns)
    for file_name in all_SAC:
        stats = evaluate(file_name,False)
        stats.insert(0,file_name)
        new_row_df = pd.DataFrame([stats], columns=columns)
        df = pd.concat([df,new_row_df],ignore_index=True)
        

    for file_name in all_DSAC:
        stats = evaluate(file_name,True)
        stats.insert(0,file_name)
        new_row_df = pd.DataFrame([stats], columns=columns)
        df = pd.concat([df,new_row_df],ignore_index=True)
    
    df = df.reset_index(drop=True)
    df.to_csv(results_file)

if __name__ == '__main__':
    main()