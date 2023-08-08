import laserhockey.hockey_env as h_env
import numpy as np
from client.backend.client import Client
from client.remoteControllerInterface import RemoteControllerInterface
from dsac import DSAC_Agent
from sac import SAC_Agent


class RemoteSACAgent(SAC_Agent, RemoteControllerInterface):
    def __init__(self):
        SAC_Agent.__init__(self,h_env.HockeyEnv().observation_space,h_env.HockeyEnv().action_space)
        file = './results/SAC_Easy_easy-e10000-t32-s42-player.pth'
        SAC_Agent.load_network_states_from_file(file,True)
        RemoteControllerInterface.__init__(self, identifier='SAC_Great_Descent')
    
    def remote_act(self, obs: np.ndarray) -> np.ndarray:
        return super().remote_act(obs)
    

class RemoteDSACAgent(DSAC_Agent, RemoteControllerInterface):
    def __init__(self):
        DSAC_Agent.__init__(self,h_env.HockeyEnv().observation_space,h_env.HockeyEnv().action_space)
        file = './results/DSAC_Easy_easy-e10000-t32-s42-player.pth'
        DSAC_Agent.load_network_states_from_file(file,True)
        RemoteControllerInterface.__init__(self, identifier='DSAC_Great_Descent')

    
    def remote_act(self, obs: np.ndarray) -> np.ndarray:
        return super().remote_act(obs)

if __name__ == '__main__':
    
    controller = RemoteDSACAgent()

    # Play n (None for an infinite amount) games and quit
    client = Client(username='yourusername',
                    password='1234',
                    controller=controller,
                    output_path='logs/SAC', # rollout buffer with finished games will be saved in here
                    interactive=False,
                    op='start_queuing',
                    # server_addr='localhost',
                    num_games=None)

    # Start interactive mode. Start playing by typing start_queuing. Stop playing by pressing escape and typing stop_queueing
    # client = Client(username='user0',
    #                 password='1234',
    #                 controller=controller,
    #                 output_path='logs/basic_opponents',
    #                )
