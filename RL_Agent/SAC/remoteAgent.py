import laserhockey.hockey_env as h_env
import numpy as np
from client.backend.client import Client
from client.remoteControllerInterface import RemoteControllerInterface
from dsac import DSAC_Agent
from sac import SAC_Agent


class RemoteSACAgent(SAC_Agent, RemoteControllerInterface):
    def __init__(self,file='',cuda=True):
        SAC_Agent.__init__(self,h_env.HockeyEnv().observation_space,h_env.HockeyEnv().action_space)
        self.eval()
        self.load_network_states_from_file(file,cuda)
        RemoteControllerInterface.__init__(self)
    
    def remote_act(self, obs):
        return self.act(obs)
    

class RemoteDSACAgent(DSAC_Agent, RemoteControllerInterface):

    def __init__(self,file="",cuda=""):
        DSAC_Agent.__init__(self,h_env.HockeyEnv().observation_space,h_env.HockeyEnv().action_space)
        self.eval()
        self.load_network_states_from_file(file,cuda)
        RemoteControllerInterface.__init__(self)

    def remote_act(self, obs):
        return self.act(obs)


if __name__ == '__main__':
    file = './results/DSAC_Easy_easy-e10000-t32-s42-player.pth'
    use_DSAC = True
    if use_DSAC:
        controller = RemoteDSACAgent(file=file,cuda=True)
    else:
        controller = RemoteSACAgent(file=file,cuda=True)

    # Play n (None for an infinite amount) games and quit
    client = Client(username='great descent',
                    password='',
                    controller=controller,
                    output_path='logs/stud3', # rollout buffer with finished games will be saved in here
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
