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

Deep Deterministic Policy Gradient (DDPG) is an algorithm which concurrently learns a Q-function and a policy. It uses off-policy data and the Bellman equation to learn the Q-function, and uses the Q-function to learn the policy.

This approach is closely connected to Q-learning, and is motivated the same way: if you know the optimal action-value function Q^*(s,a), then in any given state, the optimal action a^*(s) can be found by solving

a^*(s) = \arg \max_a Q^*(s,a).

DDPG interleaves learning an approximator to Q^*(s,a) with learning an approximator to a^*(s), and it does so in a way which is specifically adapted for environments with continuous action spaces. But what does it mean that DDPG is adapted specifically for environments with continuous action spaces? It relates to how we compute the max over actions in \max_a Q^*(s,a).

When there are a finite number of discrete actions, the max poses no problem, because we can just compute the Q-values for each action separately and directly compare them. (This also immediately gives us the action which maximizes the Q-value.) But when the action space is continuous, we can’t exhaustively evaluate the space, and solving the optimization problem is highly non-trivial. Using a normal optimization algorithm would make calculating \max_a Q^*(s,a) a painfully expensive subroutine. And since it would need to be run every time the agent wants to take an action in the environment, this is unacceptable.

Because the action space is continuous, the function Q^*(s,a) is presumed to be differentiable with respect to the action argument. This allows us to set up an efficient, gradient-based learning rule for a policy \mu(s) which exploits that fact. Then, instead of running an expensive optimization subroutine each time we wish to compute \max_a Q(s,a), we can approximate it with \max_a Q(s,a) \approx Q(s,\mu(s)).

- DDPG is an off-policy algorithm.
- DDPG can only be used for environments with continuous action spaces.
- DDPG can be thought of as being deep Q-learning for continuous action spaces.
- The Spinning Up implementation of DDPG does not support parallelization

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

The learning graph can be seen here:

![](https://github.com/YasinSonmez/Deep-Reinforcement-Learning/blob/master/3-%20Collaboration%20and%20Competition/learning.png)

Average reward of +.5 points is achieved after 489 episodes!

## Ideas for improvement

Some of the below techniques can be used to improve the model

- PPO
- A3C, A2C
- Prioritized Replay
- Q-prop
