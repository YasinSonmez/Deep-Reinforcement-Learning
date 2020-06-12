# Project 3: Colloboration and Competition

Yasin Sonmez

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.


## Learning Algorithm

In the project DDPG is used since the action space is continuous and state space is high dimensional. Both players are using same networks to update since the system is symmetric
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
