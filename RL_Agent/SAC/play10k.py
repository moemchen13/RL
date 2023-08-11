import time

import laserhockey.hockey_env as h_env
import matplotlib.pyplot as plt
import numpy as np
import torch
from dsac import DSAC_Agent
from sac import SAC_Agent

torch.manual_seed(42)
np.random.seed(42)

env = h_env.HockeyEnv()
agent = DSAC_Agent(env.observation_space,env.action_space)
agent.load_network_states(torch.load("./results/best/DSAC_easy-e24000-t32-s42-player.pth", map_location=torch.device('cpu')))
agent.eval()
#agent.load_network_states(torch.load("SAC_run_easy-e1500-t32-s42-player.pth"))
opponent = h_env.BasicOpponent(weak=False)

loss = 0
win = 0
tie = 0

for run in range(0,10000):
    ob, info = env.reset()
    ob_opponent = env.obs_agent_two()
    for t in range(1,400):
        action = agent.act(ob)
        opponent_action = opponent.act(ob_opponent)
        #opponent_action = np.zeros_like(opponent_action)
        ob_new,reward,done,trunc,info = env.step(np.hstack([action,opponent_action]))
        info_opponent = env.get_info_agent_two()
        #if info["reward_touch_puck"]!=0:
        #    print(info)
        #    print("touched")
        #time.sleep(0.05)
        #env.render()
        #print(info["reward_closeness_to_puck"])
        if done:
            if env.winner ==1:
                win +=1
                #if reward > 0:
                #    print(reward)
            elif env.winner == -1: 
                loss +=1
            else:
                tie +=1
            break
        #if reward > 0:
        #    print(reward)

        ob = ob_new
        ob_opponent = env.obs_agent_two()
print(info)
env.close()

print(f"weak : win(left_side): {win},loss: {loss}, tie: {tie}")


opponent = h_env.BasicOpponent(weak=True)
loss = 0
win = 0
tie = 0

for run in range(0,10000):
    ob, info = env.reset()
    ob_opponent = env.obs_agent_two()
    for t in range(1,400):
        action = agent.act(ob)
        opponent_action = opponent.act(ob_opponent)
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
print(info)
env.close()

print(f"strong: win(left_side): {win},loss: {loss}, tie: {tie}")
