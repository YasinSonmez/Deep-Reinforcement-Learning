# Project 1 : Navigation

Yasin Sönmez

In this project Deep q learning is used to train an agent to learn collect yellow bananas and avoid blue bananas in unity environment.
A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions, and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions. 

## Learning Algorithm

In the project deep Q networks are used since the state space is continuous 
learning algorithm is as follows:

![](https://github.com/YasinSonmez/Deep-Reinforcement-Learning/blob/master/1-%20Navigation/media/dqn_algo.png)

Model architecture is as follows:

        self.fc1 = nn.Linear(state_size,64)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 128)
        self.relu2 = nn.ReLU()  
        self.bn2 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(p=0.5)
        self.fc5 = nn.Linear(128, 64)
        self.relu5 = nn.ReLU() 
        self.bn5 = nn.BatchNorm1d(64)
        self.drop5 = nn.Dropout(p=0.5)
        self.fc6 = nn.Linear(64, action_size)
        self.softmax1 = nn.Softmax() 

Used hyperparameters are as follows:

Hyperparameter | Value
---------------|-------------
epsilon_start|1
epsilon_min|0.01
epsilon_decay|0.995
replay buffer size| 1e5
Batch size|64
gamma(discount factor)|0.99
learning rate|5e-4
Update interval|4

## Results

Trained agent can be seen below

![](https://github.com/YasinSonmez/Deep-Reinforcement-Learning/blob/master/1-%20Navigation/media/navigation.gif)

and the learning graph can be seen here:

<img src="https://github.com/YasinSonmez/Deep-Reinforcement-Learning/blob/master/1-%20Navigation/media/learning_graph.png" height="440">

the task is solved in *456* episodes.

## Ideas for improvement

The methods to improve deeq q networks can be implemented such as: 
- Double DQN
- Dueling DQN
- Prioritized Replay
- Distributional DQN
- Noisy DQN
