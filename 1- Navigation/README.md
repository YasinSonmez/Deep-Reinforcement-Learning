# Project : Navigation

## Description 
For this project, we train an agent to navigate and collect bananas in a large, 
square world.

![](https://github.com/YasinSonmez/Deep-Reinforcement-Learning/blob/master/1-%20Navigation/media/navigation.gif)

## Problem statement 
A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided 
for collecting a blue banana. Thus, the goal of the agent is to collect 
as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions, and contains the agent's velocity, along
with ray-based perception of objects around the agent's forward
direction. Given this information, the agent has to learn how to best select 
actions. 
Four discrete actions are available, corresponding to: 
- `0` - move forward
- `1` - move backward
- `2` - turn left
- `3` - turn right

The task is episodic, and in order to solve the environment, the 
agent must get an average score of +13 over 100 consecutive episodes.

## Files
- `Navigation.ipynb`: Notebook used to control and train the agent 
- `dqn_agent.py`: Create an Agent class that interacts with and learns from the environment 
- `model.py`: Q-network class used to map state to action values 
- `checkpoinnt.pth`: Saved weights of the neural network 
- `report.md`: Technical report 

## Getting started

1. Activate the environment:

    Please follow the instructions in the [DRLND GitHub repository](https://github.com/udacity/deep-reinforcement-learning#dependencies) to set up your Python environment. These instructions can be found in README.md at the root of the repository. By following these instructions, you will install PyTorch, the ML-Agents toolkit, and a few more Python packages required to complete the project.

    (For Windows users) The ML-Agents toolkit supports Windows 10. While it might be possible to run the ML-Agents toolkit using other versions of Windows, it has not been tested on other versions. Furthermore, the ML-Agents toolkit has not been tested on a Windows VM such as Bootcamp or Parallels. 

2. Download the environment from one of the links below. You need only select the environment that matches your operating system:

    * Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    * Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    * Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    * Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

    For Windows users) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (For AWS) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md) to obtain the environment.

2. Place the file in the DRLND GitHub repository, in the p1_navigation/ folder, and unzip (or decompress) the file.


## Instructions
Use Navigation.ipynb to start training the agent.
