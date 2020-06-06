# Project 2: Reacher

Yasin Sonmez

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

The barrier for solving the second version of the environment is slightly different, to take into account the presence of many agents. In particular, your agents must get an average score of +30 (over 100 consecutive episodes, and over all agents). Specifically,

After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 20 (potentially different) scores. We then take the average of these 20 scores.
This yields an average score for each episode (where the average is over all 20 agents).
The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30.

## Learning Algorithm

In the project DDPG is used since the action space is continuous and state space is high dimensional.
Algorithm is as follows:

<img src="https://github.com/YasinSonmez/Deep-Reinforcement-Learning/blob/master/2-%20Continuous%20Control/media/Algorithm.png" height="440">

Architectures of models are as follows:

Actor: 

    fc1_units=256, fc2_units=128
        
    self.fc1 = nn.Linear(state_size, fc1_units)
    self.bn1 = nn.BatchNorm1d(fc1_units)
    self.fc2 = nn.Linear(fc1_units, fc2_units)
    self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = F.relu(self.bn1(self.fc1(state)))
        x = F.relu(self.fc2(x))
        return F.tanh(self.fc3(x))
        
Critic:

    fcs1_units=256, fc2_units=256, fc3_units=128
       
    self.fcs1 = nn.Linear(state_size, fcs1_units)
    self.bn1 = nn.BatchNorm1d(fcs1_units)
    self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units)
    self.fc3 = nn.Linear(fc2_units, fc3_units)
    self.fc4 = nn.Linear(fc3_units, 1)
        
    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = F.leaky_relu(self.bn1(self.fcs1(state)))
        x = torch.cat((xs, action), dim=1)
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        return self.fc4(x)        
        
        
Hyperparameters are as follows:

Hyperparameter | Value
---------------|-------------
BUFFER_SIZE | 1e6 # replay buffer size
BATCH_SIZE |128  # minibatch size
GAMMA | 0.99  # discount factor
TAU | 1e-3  # for soft update of target parameters
LR_ACTOR | 1e-3  # learning rate of the actor
LR_CRITIC | 1e-3  # learning rate of the critic
WEIGHT_DECAY | 0  # L2 weight decay
UPDATE_EVERY | 20      # how many timesteps before update     
UPDATE_TIMES | 5      # how many times to update the network each time
EPSILON | 1.0         # for epsilon in the noise process (act step)
EPSILON_DECAY | 1e-6
GRAD_CLIPPING | 1.5         # gradient clipping 

## Results

Trained and untrained agents can be seen below:

| Random agent             |  Trained agent |
:-------------------------:|:-------------------------:
![Random Agent](https://github.com/YasinSonmez/Deep-Reinforcement-Learning/blob/master/2-%20Continuous%20Control/media/untrained.gif)  |  ![Trained Agent](https://github.com/YasinSonmez/Deep-Reinforcement-Learning/blob/master/2-%20Continuous%20Control/media/trained.gif)

and the learning graph can be seen here:

![](https://github.com/YasinSonmez/Deep-Reinforcement-Learning/blob/master/2-%20Continuous%20Control/media/learning_graph.png)

Average reward of +30 points is achieved after 53 episodes!

## Ideas for improvement

Some of the below techniques can be used to improve the model

- PPO
- A3C, A2C
- Prioritized Replay
- Q-prop
