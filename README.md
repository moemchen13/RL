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
* DDQN [Double Doueling QNetworks](https://doi.org/10.48550/arXiv.1509.06461)

All of the algorithms are programmed for the Contest at the Reinforcement class
at the *University of Tübingen* in the Sommersemester of 2023.
(Given by Prof. George Martinus)

Change for all Networks come from the [Rainbow Paper](https://doi.org/10.48550/arXiv.1710.02298).

We hope to achieve good results at a laser-hockey playground.
Therefore our agent has to qualify by beating the baseline method in a match.
After that it will play against multiple different other agents from the course.

***
### Folderstructure
Our code is seperated into the different Reinforcement learning Agents here the implemented algorithms can be found.\
Including *main* file to prove that the agent is working we evaluated him on the *Pendulum-v1-Environment*.\
In the folder **Basic** non original files from the course can be found reused for the agents.\
Not yet implemented but soon coming, is the folder **Playground**, comprising the environment of the *Laser-hockey*.\
And methods to let two agents combat each other.
