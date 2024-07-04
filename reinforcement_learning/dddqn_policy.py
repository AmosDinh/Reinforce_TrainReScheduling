import copy
import os
import pickle
import random
from collections import namedtuple, deque

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from reinforcement_learning.model import DuelingQNetwork
from reinforcement_learning.policy import Policy

from reinforcement_learning.ReplayBuffer import ReplayBuffer


class DDDQNPolicy(Policy):
    """Dueling Double DQN policy"""

    def __init__(self, state_size, action_size, parameters, evaluation_mode=False):
        self.evaluation_mode = evaluation_mode

        self.state_size = state_size
        self.action_size = action_size
        self.double_dueling_dqn = True
        self.dueling_dqn = False
        self.hidsize = 1

        if not evaluation_mode:
            self.hidsize = parameters.hidden_size
            self.buffer_size = parameters.buffer_size
            self.batch_size = parameters.batch_size
            self.update_every = parameters.update_every
            self.learning_rate = parameters.learning_rate
            self.tau = parameters.tau
            self.gamma = parameters.gamma
            self.buffer_min_size = parameters.buffer_min_size

        # Device
        if torch.backends.mps.is_available() and False:
            self.device = torch.device("mps")
            print("🔥 Using MPS (multi-process service) for PyTorch on GPU")
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            print("🔥 Using GPU for PyTorch")
        else:
            self.device = torch.device("cpu")
            print("🐌 Using CPU for PyTorch")

        # Q-Network
        self.qnetwork_local = DuelingQNetwork(state_size, action_size, hidsize1=self.hidsize, hidsize2=self.hidsize).to(self.device)

        if not evaluation_mode:
            self.qnetwork_target = copy.deepcopy(self.qnetwork_local)
            self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.learning_rate)
            self.memory = ReplayBuffer(action_size, self.buffer_size, self.batch_size, self.device)

            self.t_step = 0
            self.loss = 0.0

    def act(self, state, eps=0.):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def step(self, state, action, reward, next_state, done):
        assert not self.evaluation_mode, "Policy has been initialized for evaluation only."

        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.buffer_min_size and len(self.memory) > self.batch_size:
                self._learn()

    def _learn(self):
        experiences = self.memory.sample()
        states, actions, rewards, next_states, dones = experiences
        
        # Get expected Q values from local model
        # Q network gives back (batch_size, action_size) tensor
        # We get the Q values for the actions taken
        q_expected = self.qnetwork_local(states).gather(1, actions)

        assert sum([self.double_dueling_dqn, self.dueling_dqn]) == 1, "Exactly one of double_dueling_dqn, dqn, must be True"
        if self.double_dueling_dqn:
            # off-policy
            # Double DQN
            # Loss = E[(r + γ * Q_target(s', argmax_{a'}(Q_local(s',a')) - Q_local(s, a))**2]
            # 1. Get the best action for the next state from the local model
            # 2. Get the Q values for the best action (selected by local model) of the target model
            q_best_action = self.qnetwork_local(next_states).max(1)[1]
            q_targets_next = self.qnetwork_target(next_states).gather(1, q_best_action.unsqueeze(-1))
        elif self.dueling_dqn:
            # off-policy
            # DQN
            # Get the Q value for the best action from the target model
            q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(-1)
        # elif self.sarsa:
        #     # on policy
        #     # AMOS SARSA  # have to add next action for sarsa, use different 

        # Compute Q targets for current states
        # Only update if not done
        q_targets = rewards + (self.gamma * q_targets_next * (1 - dones))

        # Compute loss
        # Learn bellmans equation with mse: Q(s,a) = r + γ * max_{a'}(Q(s', a')) -> minimize Q(s,a) - (r + γ * max_{a'}(Q(s', a'))) 
        # This is equivalent to saying Q(s,a) = Q(s,a) + (⍺) (r + γ * max_{a'}(Q(s', a')) - Q(s,a))
        self.loss = F.mse_loss(q_expected, q_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

        # Update target network
        self._soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

    def _soft_update(self, local_model, target_model, tau):
        # Soft update model parameters.
        # θ_target = τ*θ_local + (1 - τ)*θ_target
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            # interpolation, idk where it comes from
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def save(self, filename):
        torch.save(self.qnetwork_local.state_dict(), filename + ".local")
        torch.save(self.qnetwork_target.state_dict(), filename + ".target")

    def load(self, filename):
        if os.path.exists(filename + ".local"):
            self.qnetwork_local.load_state_dict(torch.load(filename + ".local"))
        if os.path.exists(filename + ".target"):
            self.qnetwork_target.load_state_dict(torch.load(filename + ".target"))

    def save_replay_buffer(self, filename):
        memory = self.memory.memory
        with open(filename, 'wb') as f:
            pickle.dump(list(memory)[-500000:], f)

    def load_replay_buffer(self, filename):
        with open(filename, 'rb') as f:
            self.memory.memory = pickle.load(f)

    def test(self):
        self.act(np.array([[0] * self.state_size]))
        self._learn()


class DoubleDQN_policy(DDDQNPolicy):
    def __init__(self, state_size, action_size, parameters, evaluation_mode=False):
        super().__init__(state_size, action_size, parameters, evaluation_mode)
        self.dueling_dqn = True
        self.double_dueling_dqn = False
