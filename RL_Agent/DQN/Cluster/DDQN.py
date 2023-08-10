import torch
import numpy as np
import laserhockey.hockey_env as h_env
from importlib import reload
import gymnasium as gym
from gymnasium import spaces
import pylab as plt
import time
# %matplotlib inline
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import argparse


class Feedforward(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_sizes  = hidden_sizes
        self.output_size  = output_size

        print(self.input_size, self.hidden_sizes[0], self.hidden_sizes[1], self.output_size)

        self.input_connector = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, self.hidden_sizes[0])
        )

        self.value_layer = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_sizes[0], self.hidden_sizes[1]),
            torch.nn.Tanh(),
            torch.nn.Linear(self.hidden_sizes[1],1)
        )

        self.advantage_layer = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_sizes[0], self.hidden_sizes[1]),
            torch.nn.Tanh(),
            torch.nn.Linear(self.hidden_sizes[1], self.output_size)
        )

    def forward(self, x):
        x = self.input_connector(x)
        values = self.value_layer(x)
        advantages = self.advantage_layer(x)
        q_values = values + (advantages - advantages.mean())
        return q_values
    
    def predict(self, x):
        with torch.no_grad():
            return self.forward(torch.from_numpy(x.astype(np.float32))).numpy()

# class to store transitions
class Memory():
    def __init__(self, max_size=100000):
        self.transitions = np.asarray([])
        self.size = 0
        self.current_idx = 0
        self.max_size=max_size

    def add_transition(self, transitions_new):
        if self.size == 0:
            blank_buffer = [np.asarray(transitions_new, dtype=object)] * self.max_size
            self.transitions = np.asarray(blank_buffer)

        self.transitions[self.current_idx,:] = np.asarray(transitions_new, dtype=object)
        self.size = min(self.size + 1, self.max_size)
        self.current_idx = (self.current_idx + 1) % self.max_size

    def sample(self, batch=1):
        if batch > self.size:
            batch = self.size
        self.inds=np.random.choice(range(self.size), size=batch, replace=False)
        return self.transitions[self.inds,:]

    def get_all_transitions(self):
        return self.transitions[0:self.size]


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)



class QFunction(Feedforward):
    def __init__(self, observation_dim, action_dim, hidden_sizes=[100,100], 
                 learning_rate = 0.0002):
        super().__init__(input_size=observation_dim, hidden_sizes=hidden_sizes, 
                         output_size=action_dim)
        self.optimizer=torch.optim.Adam(self.parameters(), 
                                        lr=learning_rate, 
                                        eps=0.000001)
        self.loss = torch.nn.SmoothL1Loss() # MSELoss()
    
    def fit(self, observations, actions, targets):
        self.train() # put model in training mode
        self.optimizer.zero_grad()
        # Forward pass
        acts = torch.from_numpy(actions)
        pred = self.Q_value(torch.from_numpy(observations).float(), acts)
        # Compute Loss
        loss = self.loss(pred, torch.from_numpy(targets).float())
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def Q_value(self, observations, actions):
        return self.forward(observations).gather(1, actions[:,None])        
    
    def maxQ(self, observations):
        return np.max(self.predict(observations), axis=-1, keepdims=True)
        
    def greedyAction(self, observations):
        return np.argmax(self.predict(observations), axis=-1)


class DQNAgent(object):
    """
    Agent implementing Q-learning with NN function approximation.    
    """
    def __init__(self, observation_space, action_space, **userconfig):
        
        if not isinstance(observation_space, spaces.box.Box):
            raise UnsupportedSpace('Observation space {} incompatible ' \
                                   'with {}. (Require: Box)'.format(observation_space, self))
        if not isinstance(action_space, spaces.discrete.Discrete):
            raise UnsupportedSpace('Action space {} incompatible with {}.' \
                                   ' (Reqire Discrete.)'.format(action_space, self))
        
        self._observation_space = observation_space
        self._action_space = action_space
        self._action_n = action_space.n
        self._config = {
            "eps": 0.05,            # Epsilon in epsilon greedy policies                        
            "discount": 0.95,
            "buffer_size": int(5e5),
            "batch_size": 128,
            "learning_rate": 0.0002,
            "update_target_every": 20,
            "use_target_net":True
        }
        self._config.update(userconfig)        
        self._eps = self._config['eps']
        
        self.buffer = Memory(max_size=self._config["buffer_size"])
                
        # Q Network
        self.Q = QFunction(observation_dim=self._observation_space.shape[0], 
                           action_dim=self._action_n,
                           learning_rate = self._config["learning_rate"])
        # Q Network
        self.Q_target = QFunction(observation_dim=self._observation_space.shape[0], 
                                  action_dim=self._action_n,
                                  learning_rate = 0)
        self._update_target_net()
        self.train_iter = 0

    def save_state_dict(self, path):
        torch.save(self.Q.state_dict(), path)

    def load_net(self, path):
        self.Q.load_state_dict(torch.load(path))
            
    def _update_target_net(self):        
        self.Q_target.load_state_dict(self.Q.state_dict())

    def act(self, observation, eps=None):
        if eps is None:
            eps = self._eps
        # epsilon greedy
        if np.random.random() > eps:
            action = self.Q.greedyAction(observation)
        else: 
            action = self._action_space.sample()        
        return action
    
    def store_transition(self, transition):
        self.buffer.add_transition(transition)
            
    def train(self, iter_fit=32):
        losses = []
        self.train_iter+=1
        if self._config["use_target_net"] and self.train_iter % self._config["update_target_every"] == 0:
            self._update_target_net()                
        for i in range(iter_fit):

            # sample from the replay buffer
            data=self.buffer.sample(batch=self._config['batch_size'])
            s = np.stack(data[:,0]) # s_t
            a = np.stack(data[:,1]) # a_t
            rew = np.stack(data[:,2])[:,None] # rew  (batchsize,1)
            s_prime = np.stack(data[:,3]) # s_t+1
            done = np.stack(data[:,4])[:,None] # done signal  (batchsize,1)
            
            if self._config["use_target_net"]:
                v_prime = self.Q_target.maxQ(s_prime)
            else:
                v_prime = self.Q.maxQ(s_prime)
            # target
            gamma=self._config['discount']                                                
            td_target = rew + gamma * (1.0-done) * v_prime
            
            # optimize the lsq objective
            fit_loss = self.Q.fit(s, a, td_target)
            
            losses.append(fit_loss)
                
        return losses
    

def train_agent(seed, mode, episodes, episode_len, max_eps, log_interval, load_path):

    #init env
    np.set_printoptions(suppress=True)
    reload(h_env)
    env = h_env.HockeyEnv()
    print('Begin Training')

    obs,info = env.reset()
    obs_agent2 = env.obs_agent_two()
    ac_space = env.discrete_action_space
    o_space = env.observation_space
    layer_sizes = [ac_space] + [300,300]
    layer_construction_list = [(i, o) for i,o in zip(layer_sizes[:-1], layer_sizes[1:])]
    
    use_target = True
    target_update = 20
    learning_rate = 0.00001
    epsilon = 0.1
    disc = 0.975
    q_agent = DQNAgent(o_space, ac_space, discount=disc, eps=epsilon, 
                    use_target_net=use_target, update_target_every= target_update, learning_rate=learning_rate)
    stats = []
    losses = []
    if mode == 'defense':
        env = h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_DEFENSE)
    elif mode == 'weak':
        env = h_env.HockeyEnv(mode=h_env.HockeyEnv.NORMAL)
        opponent = opponent = h_env.BasicOpponent()
    elif mode == 'strong':
        env = h_env.HockeyEnv(mode=h_env.HockeyEnv.NORMAL)
        opponent = h_env.BasicOpponent(weak=False)
    else: return "Error"
    
    max_episodes=episodes
    max_steps=episode_len
    flag = False
    max_epsilon = max_eps
    min_epsilon = 0.05
    epsilon_desc = (max_epsilon - min_epsilon)/max_episodes
    epsilon = max_epsilon

    for i in range(max_episodes):
        env.reset()
        total_reward = 0
        ob, _info = env.reset()

        for t in range(max_steps):
            #env.render()
            done = False        
            a1_discrete = q_agent.act(ob, eps=epsilon)   # changed for laserhockey , eps=0.0
            a1_continuous = env.discrete_to_continous_action(a1_discrete) # changed for laserhockey
            if mode == 'defense':
                a2 = [0,0.,0,0 ]
            else:
                a2 = opponent.act(obs_agent2)
            (ob_new, reward, done, trunc, _info) = env.step(np.hstack([a1_continuous, a2]))
            total_reward += reward
            q_agent.store_transition((ob, a1_discrete, reward, ob_new, done))
            ob = ob_new
            obs_agent2 = env.obs_agent_two()

            if done: break

        losses.extend(q_agent.train(32))
        stats.append([i,total_reward,t+1])  
        epsilon = epsilon - epsilon_desc
    
        if ((i-1)%log_interval==0):
            if not flag: log(seed, "Learning Log:\n")
            log(seed, "{}: Done after {} steps. Reward: {} epsilon: {}".format(i, t+1, total_reward, epsilon))
            if flag:
                log(seed, "     mean reward over last 20 eisodes: {}\n".format(sum([x[1] for x in stats[i-log_interval:i]])/log_interval))# sum(stats[(i-20):i][1])/20))
                print("{}'%' done".format(((i-1)/max_episodes)*100))
            flag = True
    q_agent.save_state_dict(f"{seed}.pth")
    save_stats(seed, stats, losses)



def evaluate_agent(seed, mode, episodes, episode_len, load_path, render):

    np.set_printoptions(suppress=True)
    reload(h_env)
    env = h_env.HockeyEnv()
    print('Begin Evaluation')

    obs,info = env.reset()
    obs_agent2 = env.obs_agent_two()
    ac_space = env.discrete_action_space
    o_space = env.observation_space
    layer_sizes = [ac_space] + [300,300]
    layer_construction_list = [(i, o) for i,o in zip(layer_sizes[:-1], layer_sizes[1:])]
    
    use_target = True
    target_update = 20
    learning_rate = 0.00001
    epsilon = 0.1
    disc = 0.975
    q_agent = DQNAgent(o_space, ac_space, discount=disc, eps=epsilon, 
                    use_target_net=use_target, update_target_every= target_update, learning_rate=learning_rate)

    if mode == 'weak':
        env = h_env.HockeyEnv(mode=h_env.HockeyEnv.NORMAL)
        opponent = opponent = h_env.BasicOpponent()
    elif mode == 'strong':
        env = h_env.HockeyEnv(mode=h_env.HockeyEnv.NORMAL)
        opponent = h_env.BasicOpponent(weak=False)
    else:
        log(seed, "Invalid mode fo revaluation")
        print("Invalid mode fo revaluation")

    
    test_stats = []
    episodes=episodes
    max_steps=episode_len
    mode = mode
    if load_path != None:
        q_agent.load_net(load_path)


    for i in range(episodes):
        total_reward = 0
        ob, _info = env.reset()
        for t in range(max_steps):
            if render: env.render()
            done = False        
            a = env.discrete_to_continous_action(q_agent.act(ob, eps=0.0))
            a2 = opponent.act(obs_agent2)
            (ob_new, reward, done, trunc, _info) = env.step(np.hstack([a, a2]))
            total_reward+= reward
            ob=ob_new        
            obs_agent2 = env.obs_agent_two()
            if done: break    
        test_stats.append([i,total_reward,t+1]) 
        
    test_stats_np = np.array(test_stats)
    print(np.mean(test_stats_np[:,1]), "+-", np.std(test_stats_np[:,1]))


def log(seed, log_str):
    with open(f'./logs/{seed}-log.txt', 'a') as f:
        f.write(f'{log_str} \n')

def save_stats(seed, stats, losses):
    np.save(f'./stats/{seed}-rewards', stats)
    np.save(f'./stats/{seed}-losses', losses)
 

#def log_stats(seed, )

def main():

    parser = argparse.ArgumentParser()# Add an argument
    parser.add_argument('-s', '--seed', type=int, default = 1337, dest='seed')
    parser.add_argument('-n', '--num-ep', type=int, default = 1000, dest='num_ep')
    parser.add_argument('-l', '--len-ep', type=int, default = 300, dest='len_ep')
    parser.add_argument('-m', '--mode', choices=('defense', 'weak', 'strong'), default='weak', dest='mode')
    parser.add_argument('-e', '--evaluate', type=bool, default=False, dest='eval')
    parser.add_argument('--load-path', type=str, required=False, dest='load_path')
    parser.add_argument('--log-interval', type = int, default=50, dest='log_interval')
    parser.add_argument('--render', type=bool, default=False, dest='render')
    args = parser.parse_args()

    
    main_seed = args.seed
    num_ep = args.num_ep
    len_ep = args.len_ep
    mode = args.mode
    eval = args.eval
    load_path = args.load_path
    log_interval = args.log_interval
    render = args.render

    log(main_seed,
        f'Configuration:\n  seed: {main_seed}\n  num-ep: {num_ep}\n  ep-len: {len_ep}\n  mode: {mode}\n  evaluate: {eval}\n  loaded network: {load_path}\n\nLog:\n')

    if eval == False:
        train_agent(main_seed, mode, num_ep, len_ep, 0.4, log_interval, load_path)
    elif eval == True:
        evaluate_agent(main_seed, mode, num_ep, len_ep,load_path, render=True)
    else:
        log(main_seed, "ERROR: evaluation arg")
        
if __name__ == "__main__":
    main()

