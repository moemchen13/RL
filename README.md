# RL Agents
***
This is a Github Repository for Reinforcement Learning agents.
These are programmed by:
* Torsten Foehr
* Moritz Christ
* August Berger

They comprise the State of the art Agents:
* SAC [Soft-Actor-Critic](https://doi.org/10.48550/arXiv.1801.01290)
* TD3 [Twin Delayed DDPG 3](https://doi.org/10.48550/arXiv.1802.09477)
* DDQN [Dueling Deep Q Networks](https://doi.org/10.48550/arXiv.1509.06461)

All of the algorithms are programmed for the Contest at the Reinforcement class
at the *University of Tübingen* in the Sommersemester of 2023.
(Given by Prof. George Martinus)

Change for all Networks come from the [Rainbow Paper](https://doi.org/10.48550/arXiv.1710.02298).

We hope to achieve good results at a laser-hockey playground.
Therefore our agent has to qualify by beating the baseline method in a match.
After that it will play against multiple different other agents from the course.

***
### Folderstructure
Our code is seperated into the different Reinforcement learning Agents.
Here the implemented algorithms can be found.\
Including a *main* file in which the agent is evaluated on a common environment. (default: *Pendulum-v1-Environment*).\
In the folder **Basic** non original files from the course can be found.
These files should only comprise files necessary to run the agent.\
Not yet correctly implemented but soon coming, is the folder **Playground**, comprising the environment of the *Laser-hockey*.\
And the files to let agents train in the laserhockey environment or play the game against the given bot or an implemented agent.\


### Implemented Algorithms
[SAC](https://doi.org/10.48550/arXiv.1801.01290)
Implementation with 4 Critic Networks Entropy autotuning and 1 Actor with exploration based on the entropy
[DSAC](https://arxiv.org/abs/2001.02811)
Implementation of an distributional representation of SAC Q_value function to mitigate the overestimation bias
[DR3 regularization](https://arxiv.org/abs/2112.04716)
Implemented regularization in the Q value function to mitigate effect of growing dot product for consecutive state action pairs

All agents in this repository were tested on the hockey environment of Prof. Martius [hockey_env](https://github.com/martius-lab/hockey-env).
Our results can be found in the finalized report.


### Comments
Write Extension generic and modular if possible so other people can also use them (e.g. Prioritized ReplayBuffer)
