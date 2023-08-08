import argparse
import json
import gymnasium as gym
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
import laserhockey.hockey_env as h_env



# Remote Play
from client.remoteControllerInterface import RemoteControllerInterface
from client.backend.client import Client



# Preparations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(42)
np.random.seed(42)


class ReplayBuffer(object):
    """
    Class to represent the replay buffer.
    Stores environment transitions for experience replay.
    Provides storage management by functions for adding, sampling and filling the buffer.
    """
    

    def __init__(self, config):
        """
        Initialization function. Initializes the replay buffer with config.
        """
        
        self.capacity = config["Replay Buffer"]["capacity"]
        self.storage = []
        self.full = False
        self.ptr = 0


    def add(self, data):
        """
        Adds given transition to the replay buffer.
        """
        
        if not self.full:
            self.storage.append(data)
            if len(self.storage) == self.capacity:
                self.full = True
        else:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.capacity


    def random_fill(self, env):
        """
        Fills the buffer with random environment transitions.
        """
        
        state, info = env.reset()

        while not self.full:
  
            action = env.action_space.sample()
            agent_action = action[:4]
            opponent_action = action[-4:]
            
            next_state, reward, done, trunc, info = env.step(action)
            self.add((state, agent_action, reward, next_state, done)) #max(done, trunc)))

            if done or trunc:
                state, info = env.reset()
            else:
                state = next_state


    def game_fill(self, env, agent, opponent, exploration_noise=0.1):
        """
        Fills the buffer with transitions from a game.
        """

        # sample agent side
        #agent_side = np.random.choice(["left", "right"])
        agent_side = "left"

        if agent_side == "left":
            state, info = env.reset()
            opponent_state = env.obs_agent_two()


        elif agent_side == "right":
            opponent_state, _ = env.reset()
            state = env.obs_agent_two()


        while not self.full:
 
            # agent action with exploration noise
            agent_action = agent.select_action(state, exploration_noise)
                
            # opponent action
            opponent_action = opponent.act(opponent_state)

            # environment transition

            if agent_side == "left":

                next_state, reward, done, trunc, info = env.step(np.hstack([agent_action, opponent_action]))
                opponent_state = env.obs_agent_two()
                 
                # store transition in replay buffer
                self.add((state, agent_action, reward, next_state, done)) #max(done, trunc)))

            elif agent_side == "right":

                opponent_state, _, done, trunc, _ = env.step(np.hstack([opponent_action, agent_action]))
                next_state = env.obs_agent_two()

                agent_info = env.get_info_agent_two()
                reward = env.get_reward_agent_two(agent_info)

                # store transition in replay buffer
                self.add((state, agent_action, reward, next_state, done)) #max(done, trunc)))


            if done or trunc:
                break
            else:
                state = next_state




    def sample(self, batch_size):
        """
        Samples random transitions of given size from the buffer.
        """
        
        data_idx = np.random.randint(0, len(self.storage), size=batch_size)
        states, actions, rewards, next_states, dones = [], [], [], [], []

        for idx in data_idx: 
            state, action, reward, next_state, done = self.storage[idx]
            states.append(np.array(state, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(np.array(reward, copy=False))
            next_states.append(np.array(next_state, copy=False))
            dones.append(np.array(done, copy=False))

        return np.array(states), np.array(actions), np.array(rewards).reshape(-1, 1), np.array(next_states), np.array(dones).reshape(-1, 1)



class Actor(torch.nn.Module):
    """
    Class to represent actor networks for TD3.
    Defines network architecture and forward pass.
    """

    def __init__(self, state_dim, action_dim, max_action, config):
        """
        Initialization function. Defines network architecture from environment information and config.
        """    

        super(Actor, self).__init__()

        # Layers
        input_size = state_dim
        hidden_sizes  = config["Actor"]["hidden_sizes"]
        output_size  = action_dim

        self.layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.layers = torch.nn.ModuleList([torch.nn.Linear(in_size,out_size) for in_size,out_size in zip(self.layer_sizes[:-1], self.layer_sizes[1:])])

        # Activations
        activation = eval(config["Actor"]["activation"])
        hidden_activations = [activation for layer in self.layers[:-1]]
        output_activation = eval(config["Actor"]["output_activation"])
        
        self.activations =  hidden_activations + [output_activation]
        
        # Max Action
        self.max_action = max_action


    def forward(self, state):
        """
        Forward pass of the actor network.
        Takes a state and returns an action.
        """

        action = state
        
        for layer, activation in zip(self.layers, self.activations):
            action = activation(layer(action))
        
        action = self.max_action * action

        return action



class Critic(torch.nn.Module):
    """
    Class to represent critic networks for TD3.
    Defines network architecture and forward pass.
    One critic object already contains the two twin networks.
    """

    def __init__(self, state_dim, action_dim, config):
        """
        Initialization function. Defines network architecture from environment information and config.
        """    

        super(Critic, self).__init__()

        # Layers
        input_size = state_dim + action_dim
        hidden_sizes  = config["Critic"]["hidden_sizes"]
        output_size  = 1

        self.layer_sizes = [input_size] + hidden_sizes + [output_size]

        # Twin Networks
        self.layers_1 = torch.nn.ModuleList([torch.nn.Linear(in_size,out_size) for in_size,out_size in zip(self.layer_sizes[:-1], self.layer_sizes[1:])])
        self.layers_2 = torch.nn.ModuleList([torch.nn.Linear(in_size,out_size) for in_size,out_size in zip(self.layer_sizes[:-1], self.layer_sizes[1:])])

        # Activations
        activation = eval(config["Critic"]["activation"])
        hidden_activations = [activation for layer in self.layers_1[:-1]]
        output_activation = eval(config["Critic"]["output_activation"])

        self.activations =  hidden_activations + [output_activation]
        

    def forward(self, state, action):
        """
        Forward pass of the critic twin network.
        Takes state and action and returns two Q values.
        """

        Q1 = Q2 = torch.hstack([state, action])

        for layer_1, layer_2, activation in zip(self.layers_1, self.layers_2, self.activations):
            Q1 = activation(layer_1(Q1))
            Q2 = activation(layer_2(Q2))

        return Q1, Q2


    def Q1(self, state, action):
        """
        Forward pass of the main critic network.
        Takes state and action and returns only the main Q value.
        """

        Q1 = torch.hstack([state, action])

        for layer_1, activation in zip(self.layers_1, self.activations):
            Q1 = activation(layer_1(Q1))
        
        return Q1



class TD3(object):
    """
    Class to represent a TD3 agent. 
    Consists of actor and critic networks.
    Provides functions to select (noisy) actions and to train from a replay buffer.
    Agents can be stored or read from disk.
    """

    def __init__(self, agent_name, env, config, load=False):
        """
        Initialization function. Defines agent networks and training setup from environment information and config.
        """

        # Environment
        self.env = env
        self.state_dim =  self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0] // 2
        self.max_action = self.env.action_space.high[0]

        # Actor Networks
        self.actor = Actor(self.state_dim, self.action_dim, self.max_action, config).to(device)

        self.actor_target = Actor(self.state_dim, self.action_dim, self.max_action, config).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config["TD3"]["actor_lr"], weight_decay=config["TD3"]["actor_reg"])

        # Critic Networks
        self.critic = Critic(self.state_dim, self.action_dim, config).to(device)
        
        self.critic_target = Critic(self.state_dim, self.action_dim, config).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config["TD3"]["critic_lr"], weight_decay=config["TD3"]["critic_reg"])


        # Load Agent Instance
        if load:
            self.name = "_".join(agent_name.split("_")[:-1])
            self.load(agent_name)    
        self.name = agent_name


        # Training
        self.training_batch_size = config["TD3"]["training_batch_size"]
        self.gamma = config["TD3"]["gamma"]
        self.tau = config["TD3"]["tau"]
        self.training_noise = config["TD3"]["training_noise"]
        self.noise_clip = config["TD3"]["noise_clip"]
        self.update_frequency = config["TD3"]["update_frequency"]


    def select_action(self, state, noise, noise_clip=None):
        """
        Function to select an action from the agent policy for a given state.
        Can be perturbed by Gaussian noise. Noise can be clipped.
        """
        
        if not torch.is_tensor(state):
            state = torch.FloatTensor(state).to(device)

        action = self.actor(state).cpu().detach().numpy()
        
        noise = np.random.normal(0, noise, size=self.action_dim)

        if noise_clip is not None:
            noise = noise.clip(-noise_clip,noise_clip)

        action = action + noise

        return action.clip(self.env.action_space.low[:4], self.env.action_space.high[:4])


    def act(self, state):
        """
        Wrapper for select action without noise.
        """
        return self.select_action(state, noise=0.0, noise_clip=None)


    def train(self, buffer, train_iter):
        """
        Trains the agent for a given number of iterations with TD3 using transitions from a replay buffer.
        Hyperparameters are given by the agent's attributes. Returns actor and critic losses.
        """

        actor_losses = []
        critic_losses = []

        for i in range(train_iter):
            
            # sample batch of transitions
            states, actions, rewards, next_states, dones = buffer.sample(self.training_batch_size)

            # transfer data to device
            states = torch.FloatTensor(states).to(device)
            actions = torch.FloatTensor(actions).to(device)
            rewards = torch.FloatTensor(rewards).to(device)
            next_states = torch.FloatTensor(next_states).to(device)
            dones = torch.FloatTensor(dones).to(device)

            # target policy smoothing
            noisy_next_actions = torch.FloatTensor(self.select_action(states, self.training_noise, self.noise_clip)).to(device)

            # TD target
            target_Q1, target_Q2 = self.critic_target(next_states, noisy_next_actions)
            td_target = rewards + (1.0-dones) * self.gamma * torch.min(target_Q1, target_Q2)

            # critic loss
            current_Q1, current_Q2 = self.critic(states, actions)
            critic_loss = F.mse_loss(current_Q1, td_target) + F.mse_loss(current_Q2, td_target) 
            critic_losses.append(critic_loss.item())

            # update critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # delayed policy updates
            if i % self.update_frequency == 0:

                # actor loss
                actor_loss = -self.critic.Q1(states, self.actor(states)).mean()
                actor_losses.append(actor_loss.item())

                # update actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # update target networks
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


        return actor_losses, critic_losses


    def save(self, agent_instance, directory="./results"):
        """
        Function to save agent instance to disk.
        """
        if not self.name in os.listdir(directory):
            os.mkdir(f"{directory}/{self.name}")


        if not agent_instance in os.listdir(f"{directory}/{self.name}"):
            os.mkdir(f"{directory}/{self.name}/{agent_instance}")

        torch.save(self.actor.state_dict(), f'{directory}/{self.name}/{agent_instance}/{agent_instance}_actor.pth')
        torch.save(self.critic.state_dict(), f'{directory}/{self.name}/{agent_instance}/{agent_instance}_critic.pth')


    def load(self, agent_instance, directory="./results"):
        """
        Function to load agent instance from disk.
        """
        self.actor.load_state_dict(torch.load(f'{directory}/{self.name}/{agent_instance}/{agent_instance}_actor.pth', map_location=device),  strict=False)
        self.critic.load_state_dict(torch.load(f'{directory}/{self.name}/{agent_instance}/{agent_instance}_critic.pth', map_location=device), strict=False)




class RemoteTD3(TD3, RemoteControllerInterface):

    def __init__(self, agent_name, env, config):
        TD3.__init__(self, agent_name, env, config, load=True)
        RemoteControllerInterface.__init__(self, identifier='TD3')

    def remote_act(self,
            obs : np.ndarray,
           ) -> np.ndarray:

        return self.act(obs)





class Trainer(object):
    """
    Class to organize the training process of a TD3 agent.
    Manages environment transitions and replay buffer housekeeping.
    """
    
    def __init__(self, env, agent, buffer, config):
        """
        Inititalization function.
        Initializes trainer with configuration and logging setup.
        """
        
        # Configuration
        self.env = env
        self.agent = agent
        self.buffer = buffer

        self.opponents = []
        for opponent in config["Trainer"]["opponents"]:
            try:
                self.opponents.append(eval(opponent))
            except:
                self.opponents.append(TD3(opponent, env, config, load=True))
        self.opponent_probs = config["Trainer"]["opponent_probs"]

        self.train_episodes = config["Trainer"]["train_episodes"]
        self.train_iter = config["Trainer"]["train_iter"]

        self.exploration_noise = config["Trainer"]["exploration_noise"]

        # Logging
        self.episode_rewards = [] # length = train_episodes 
        self.actor_losses = [] # length  = train_episodes * (train_iter / update_frequency)
        self.critic_losses = [] # length = train_episodes * train_iter

        self.save_episodes = config["Trainer"]["save_episodes"]


    def run(self):
        """
        Trains a TD3 agent for the specified number of episodes and with respect to the specified configuration.
        Consists of two phases: In the 'Play Phase', the agent's current policy is unrolled to fill the replay buffer and to collect training rewards.
        In the 'Training Phase', the agent is asked to learn from the replay buffer for the specified number of iterations.
        The information is collected and stored in plots to enable further analysis.
        """

        print("\n##########################################")
        print("##             TRAINING MODE            ##")
        print("##########################################\n")

        print(f"Training agent '{self.agent.name}' on Hockey environment'...\n")


        start = time.time()


        for episode in range(1, self.train_episodes+1):   

            # sample opponent 
            opponent = np.random.choice(self.opponents, p=self.opponent_probs)

            # sample agent side
            #agent_side = np.random.choice(["left", "right"])
            agent_side = "left"

            if agent_side == "left":
                state, info = self.env.reset()
                opponent_state = self.env.obs_agent_two()


            elif agent_side == "right":
                opponent_state, _ = self.env.reset()
                state = self.env.obs_agent_two()


            ## Play Phase ##
            episode_reward = 0

            while True:

                # agent action with exploration noise
                agent_action = self.agent.select_action(state, self.exploration_noise)
                
                # opponent action
                opponent_action = opponent.act(opponent_state)

                # environment transition

                if agent_side == "left":

                    next_state, reward, done, trunc, info = self.env.step(np.hstack([agent_action, opponent_action]))
                    opponent_state = self.env.obs_agent_two()
                 
                    # store transition in replay buffer
                    self.buffer.add((state, agent_action, reward, next_state, done)) #max(done, trunc)))

                    episode_reward += reward

                    if done or trunc:
                        break
                    else:
                        state = next_state

                elif agent_side == "right":

                    opponent_state, _, done, trunc, _ = self.env.step(np.hstack([opponent_action, agent_action]))
                    next_state = self.env.obs_agent_two()

                    agent_info = self.env.get_info_agent_two()
                    reward = self.env.get_reward_agent_two(agent_info)

                    # store transition in replay buffer
                    self.buffer.add((state, agent_action, reward, next_state, done)) #max(done, trunc)))

                    episode_reward += reward

                    if done or trunc:
                        break
                    else:
                        state = next_state

            # Training Phase
            actor_losses, critic_losses =  self.agent.train(self.buffer, self.train_iter)

            # Logging
            if episode % 10 == 0:
                print(f"  Episode {episode} - Reward: {episode_reward}")

            self.episode_rewards.append(episode_reward)
            self.actor_losses += actor_losses
            self.critic_losses += critic_losses

            if episode % self.save_episodes == 0:
                self.agent.save(f"{self.agent.name}_{episode}")
                print("  Agent instance stored!")


        stop = time.time()
        print("\nDone!\n")
        print("----------------------------------------")
        print(f"Episodes: {self.train_episodes}")
        print(f"Time: {round(stop-start,2)}s\n")


    def store_results(self, directory="./results"):
        """
        Function to store plots of the training results.
        """

        # Result Plots
        plt.rcParams['figure.figsize'] = [15, 5]
        fig, (ax1, ax2, ax3) = plt.subplots(1,3)
        fig.suptitle(f'Training Results')# - {self.agent.name} - {self.env.spec.id}')

        # Training Rewards
        ax1.plot(range(1,self.train_episodes+1), self.episode_rewards, 'g') 
        ax1.set_title("Training Rewards")
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Reward")
        ax1.grid(True)

        # Actor Loss
        ax2.plot(range(1, len(self.actor_losses)+1), self.actor_losses, 'b')
        ax2.set_title("Actor Loss")
        ax2.set_xlabel("Actor Update")
        ax2.set_ylabel("Loss")
        ax2.grid(True)

        # Critic Loss
        ax3.plot(range(1, len(self.critic_losses)+1), self.critic_losses, 'r')
        ax3.set_title("Critic Loss")
        ax3.set_xlabel("Critic Update")
        ax3.set_ylabel("Loss")
        ax3.grid(True)

        fig.tight_layout()
        fig.savefig(f"{directory}/{self.agent.name}/training_results", format="png")



class Evaluator(object):
    """
    Class to organize the evaluation process of a TD3 agent.
    Evaluates all instances of the given agent, i.e. all intermediate agents.
    Manages environment transitions and collects rewards.
    """
    
    def __init__(self, env, agent_name, directory, config):
        """
        Inititalization function.
        Initializes evaluator with configuration and logging setup.
        Collects all agent instances to be evaluated from the given agent name.
        """
        
        # Configuration
        self.env = env
    
        self.agent_name = agent_name
        self.agent_dir  = f"{directory}/{self.agent_name}"
        self.eval_instances = [file for file in os.listdir(self.agent_dir) if self.agent_name in file]
        self.eval_instances = sorted(self.eval_instances, key=lambda x: int((x.split("_")[-1])))

        self.opponents = []
        self.oppenent_names = []
        for opponent in config["Evaluator"]["opponents"]:
            try:
                self.opponents.append(eval(opponent))
                if "weak=True" in opponent:
                    self.oppenent_names.append("Weak Baseline")
                elif "weak=False" in opponent:
                    self.oppenent_names.append("Strong Baseline")
            except:
                self.opponents.append(TD3(opponent, env, config, load=True))
                self.oppenent_names.append(opponent)


        self.eval_episodes = config["Evaluator"]["eval_episodes"]

        self.exploration_noise = config["Evaluator"]["exploration_noise"]

        self.config = config

        # Logging
        self.evaluation_data = []


    def run(self):
        """
        Evaluates all instances of the agent with specified name.
        Collects evaluation information for each agent.
        """

        print("\n##########################################")
        print("##           EVALUATION MODE            ##")
        print("##########################################\n")

        print(f"Evaluating agent '{self.agent_name}' on Hockey environment...\n")
        
        start = time.time()

        for agent_instance in self.eval_instances:

            print(f"  Evaluating instance '{agent_instance}'...")

            # load agent
            agent = TD3(self.agent_name, self.env, self.config)
            agent.load(agent_instance)

            # opponent selection
            for i in range(len(self.opponents)):

                opponent = self.opponents[i]
                opponent_name = self.oppenent_names[i]

                # preparations

                evaluated_episodes = 0
                wins = 0
                draws = 0
                losses = 0

                evaluated_episodes_left = 0
                wins_left = 0
                draws_left = 0
                losses_left = 0

                evaluated_episodes_right = 0
                wins_right = 0
                draws_right = 0
                losses_right = 0

                total_reward = 0
                min_reward = np.infty
                max_reward = -np.infty
                avg_reward = 0

                # evaluation

                # agent plays on left side
                for left_episode in range(self.eval_episodes // 2):
                
                    episode_reward = 0

                    state, info = self.env.reset()
                    opponent_state = self.env.obs_agent_two()
          
                    while True:

                        # agent action with exploration noise    
                        agent_action = agent.select_action(state, self.exploration_noise)
                    
                        # opponent action
                        opponent_action = opponent.act(opponent_state)

                        # environment transition
                        next_state, reward, done, trunc, info = self.env.step(np.hstack([agent_action, opponent_action]))
                        opponent_state = self.env.obs_agent_two()

                        episode_reward += reward

                        if done or trunc:
                            if info["winner"] == 1:
                                wins_left += 1
                                wins += 1
                            elif info["winner"] == 0:
                                draws_left += 1
                                draws += 1
                            elif info["winner"] == -1:
                                losses_left += 1
                                losses += 1
                            evaluated_episodes_left += 1
                            evaluated_episodes += 1
                            break
                        else:
                            state = next_state

                    # logging
                    total_reward += episode_reward
                    if episode_reward < min_reward:
                        min_reward = episode_reward
                    if episode_reward > max_reward:
                        max_reward = episode_reward


                # agent plays on right side
                for right_episode in range(self.eval_episodes // 2):

                    episode_reward = 0
                
                    opponent_state, _ = self.env.reset()
                    state = self.env.obs_agent_two()

                    while True:

                        # agent action with exploration noise    
                        agent_action = agent.select_action(state, self.exploration_noise)
                    
                        # opponent action
                        opponent_action = opponent.act(opponent_state)

                        # environment transition

                        opponent_state, _, done, trunc, _ = self.env.step(np.hstack([opponent_action, agent_action]))
                        next_state = self.env.obs_agent_two()

                        agent_info = self.env.get_info_agent_two()
                        reward = self.env.get_reward_agent_two(agent_info)

                        episode_reward += reward

                        if done or trunc:
                            if agent_info["winner"] == 1:
                                wins_right += 1
                                wins += 1
                            elif agent_info["winner"] == 0:
                                draws_right += 1
                                draws += 1
                            elif agent_info["winner"] == -1:
                                losses_right += 1
                                losses +=1
                            evaluated_episodes_right += 1
                            evaluated_episodes += 1
                            break
                        else:
                            state = next_state
            
                    # logging
                    total_reward += episode_reward
                    if episode_reward < min_reward:
                        min_reward = episode_reward
                    if episode_reward > max_reward:
                        max_reward = episode_reward


                win_rate_left = round(wins_left/evaluated_episodes_left, 2)
                win_rate_right = round(wins_right/evaluated_episodes_right, 2)
                win_rate = round(wins/evaluated_episodes, 2)

                avg_reward = total_reward/evaluated_episodes

                self.evaluation_data.append([agent_instance, opponent_name, evaluated_episodes, wins, draws, losses, win_rate, evaluated_episodes_left, wins_left, draws_left, losses_left, win_rate_left, evaluated_episodes_right, wins_right, draws_right, losses_right, win_rate_right, total_reward, min_reward, max_reward, avg_reward])


        stop = time.time()
        print("\nDone!\n")
        print("----------------------------------------")
        print(f"Evaluated Instances: {len(self.eval_instances)}")
        print(f"Evaluated Episodes: {self.eval_episodes}")
        print(f"Time: {round(stop-start,2)}s\n")


    def store_results(self, directory="./results"):
        """
        Function to store evaluation results in a csv file.
        """

        df_results = pd.DataFrame(data= self.evaluation_data, columns=["Agent Instance", "Opponent", "Total Episodes", "Total Wins", "Total Draws", "Total Losses", "Total Win Rate", "Left Episodes", "Left Wins", "Left Draws", "Left Losses", "Left Win Rate", "Right Episodes", "Right Wins", "Right Draws", "Right Losses", "Right Win Rate", "Total Reward", "Min Reward", "Max Reward", "Average Reward"])
        df_results.to_csv(f"{directory}/{self.agent_name}/evaluation_results.csv", index=False)



class Player(object):
    """
    Class to enable playing a particular instance of a TD3 ageto_csvnt.
    Manages environment transitions and collects rewards..
    """

    def __init__(self, env, agent_instance, config):
        """
        Inititalization function.
        Initializes player with configuration.
        """
        
        self.env = env
        self.agent_instance = agent_instance

        try:
            self.opponent = eval(config["Player"]["opponent"])
        except:
            self.opponent= TD3(config["Player"]["opponent"], env, config, load=True)

        self.play_episodes = config["Player"]["play_episodes"]
        self.exploration_noise = config["Player"]["exploration_noise"]

        self.render = config["Player"]["render"]


    def run(self):
        """
        Runs the specified agent instance in the given environment for the specified number of episodes.
        """

        print("\n##########################################")
        print("##             PLAYER MODE              ##")
        print("##########################################\n")

        print(f"Playing agent '{self.agent_instance.name}' in Hockey environment...\n")
        
        start = time.time()

        self.env.reset()
        
        if self.render:
            self.env.render(mode="human")

        # Logging
        played_episodes = 0
        wins = 0
        draws = 0
        losses = 0

        played_episodes_left = 0
        wins_left = 0
        draws_left = 0
        losses_left = 0

        played_episodes_right = 0
        wins_right = 0
        draws_right = 0
        losses_right = 0


        for episode in range(1, self.play_episodes+1):
            
            # sample agent side
            agent_side = np.random.choice(["left", "right"])

            episode_reward = 0

            print(f"  Episode {episode} - {agent_side} - ", end="", flush=True)

            if agent_side == "left":
                state, info = self.env.reset()
                opponent_state = self.env.obs_agent_two()

            elif agent_side == "right":
                opponent_state, _ = self.env.reset()
                state = self.env.obs_agent_two()


            while True:

                if self.render:
                    self.env.render()

                # agent action with exploration noise    
                agent_action = self.agent_instance.select_action(state, self.exploration_noise)
                    
                # opponent action
                opponent_action = self.opponent.act(opponent_state)

                # environment transition

                if agent_side == "left":

                    next_state, reward, done, trunc, info = self.env.step(np.hstack([agent_action, opponent_action]))
                    opponent_state = self.env.obs_agent_two()

                    episode_reward += reward

                    if done or trunc:
                        if info["winner"] == 1:
                            wins_left += 1
                            wins += 1
                        elif info["winner"] == 0:
                            draws_left += 1
                            draws += 1
                        elif info["winner"] == -1:
                            losses_left += 1
                            losses += 1
                        played_episodes_left += 1
                        played_episodes += 1
                        break
                    else:
                        state = next_state
   
                elif agent_side == "right":

                    opponent_state, _, done, trunc, _ = self.env.step(np.hstack([opponent_action, agent_action]))
                    next_state = self.env.obs_agent_two()

                    agent_info = self.env.get_info_agent_two()
                    reward = self.env.get_reward_agent_two(agent_info)

                    episode_reward += reward

                    if done or trunc:
                        if agent_info["winner"] == 1:
                            wins_right += 1
                            wins += 1
                        elif agent_info["winner"] == 0:
                            draws_right += 1
                            draws += 1
                        elif agent_info["winner"] == -1:
                            losses_right += 1
                            losses +=1
                        played_episodes_right += 1
                        played_episodes += 1
                        break
                    else:
                        state = next_state

            print(f"Reward: {episode_reward}")


        stop = time.time()
        print("\nDone!\n")
        print("----------------------------------------")
        print(f"Episodes: {played_episodes}")
        print(f"Wins: {wins}")
        print(f"Draws: {draws}")
        print(f"Losses: {losses}")
        print("----------------------------------------")
        print(f"Left Episodes: {played_episodes_left}")
        print(f"Left Wins: {wins_left}")
        print(f"Left Draws: {draws_left}")
        print(f"Left Losses: {losses_left}")
        print("----------------------------------------")
        print(f"Right Episodes: {played_episodes_right}")
        print(f"Right Wins: {wins_right}")
        print(f"Right Draws: {draws_right}")
        print(f"Right Losses: {losses_right}")
        print("----------------------------------------")
        print(f"Time: {round(stop-start,2)}s\n")


def main():


    ## Argument Parser ##

    parser = argparse.ArgumentParser(
                    prog='TD3',
                    description='Training and Evaluation of the TD3 Algorithm on the Hockey Environment')

    parser.add_argument('-c', '--config', action='store', dest='config_file', default='config.json', help='Configuration File')
    parser.add_argument('-m', '--mode', action='store', dest='mode', choices=['train','eval', 'play', 'remote'], default='train', help='Training or Evaluation')
    parser.add_argument('-a', '--agent', action='store', dest='agent_name', default='TD3', help='Agent Name')
    parser.add_argument('-l', '--load', action='store', dest='load', default=False, help='Load Agent for Training')

    args = parser.parse_args()

    config_file = args.config_file
    mode = args.mode
    agent_name = args.agent_name
    load = args.load

    with open(config_file, 'r') as f:
        config = json.load(f)


    ## Training Mode ##

    if mode == 'train':

        # create environment
        env = h_env.HockeyEnv()

        # create agent
        agent = TD3(agent_name, env, config)
        if load:
            agent.load(agent_name)
            print(f"\n{agent_name} loaded!")

        # create replay buffer
        buffer = ReplayBuffer(config)
        buffer.game_fill(env, agent, eval(config["Trainer"]["opponents"][0]))

        # create trainer
        trainer = Trainer(env, agent, buffer, config)

        # train agent
        trainer.run()

        # store results
        trainer.store_results()

        env.close()


    ## Evaluation Mode ##

    elif mode == 'eval':
 
        # create environment
        env = h_env.HockeyEnv()

        # create evaluator
        evaluator = Evaluator(env, agent_name, "./results", config)

        # evaluate agent
        evaluator.run()

        # store results
        evaluator.store_results()

        env.close()
 

    ## Play Mode ##

    elif mode == "play":

        # create environment
        env = h_env.HockeyEnv()
  
        # load TD3 agent
        agent = TD3(agent_name, env, config, load=True)

        # create player
        player = Player(env, agent, config)

        # run player
        player.run()

        env.close()


    ## Remote Mode ##
    elif mode == "remote":
        
        # create environment
        env = h_env.HockeyEnv()
  
        # create remote TD3 agent
        controller = RemoteTD3(agent_name, env, config)

        # Play n (None for an infinite amount) games and quit
        client = Client(username='great descent',
                    password='lo2beisaeK',
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







if __name__ == "__main__":
    main()


## IDEAS ##


# Twin Polcies? => Slide!
# Prioritized Replay Buffer?
