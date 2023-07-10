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

***
### Commit Messages Workflow
This is a hierarchical guide.
If the baseline of an Agent is up and works correctly please commit with the message: "DONE: Baseline {agent name} implemented".\
If extended a common structure of the agent and working, please write "EXTENSION: {Extension_name}".\
Furthermore update the section **TODO** in the ReadMe file and link the paper of extension and also note the commit code (look at the example in the TODO section).\
If something new is started please start the commit with "STARTED: {Extension_name}".\
If simple changes are done nothing works describe the changes.

***
### TODO
- [x] Write Example TODO\
  [Example](https://doi.org/10.1016/j.jml.2015.09.001)\
  Description:\
    This is an example on how we should note TODOs.
  - [ ] means the task is still open
  - [x] means the task has already finished \
  :eyes: This emoji behind a task notes that it was already started\
  Dont forget to check the box.\
  commit: \
  f5ba072
- [ ] Implement DDQN
- [ ] Implement DDPG
- [x] Implement SAC
- [ ] DQN Extension
- [ ] DDPG Extension
- [ ] SAC Extension :eyes:
  [SAC](https://doi.org/10.48550/arXiv.1801.01290)
  Implementation with 4 Critic Networks Entropy autotuning and 1 Actor with exploration based on the entropy
- [ ] Testing DQN with Extensions
- [ ] Testing DDPG with Extensions
- [ ] Testing SAC with Extensions
- [ ] Report August
- [ ] Report Moritz
- [ ] Report Torsten
- [ ] Submitted Agent evaluation
- [ ] Clean Repository
- [ ] Create detailed ReadME

### Comments
Write Extension generic and modular if possible so other people can also use them (e.g. Prioritized ReplayBuffer)
