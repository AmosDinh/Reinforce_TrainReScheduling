import copy
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import namedtuple, deque
from collections.abc import Iterable

import os
import pickle

Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "next_action", "done"])


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, device):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.device = device

    def add(self, state, action, reward, next_state, next_action, done, use_graph_observator=False):
        """Add a new experience to memory."""
        if not use_graph_observator:
            e = Experience(np.expand_dims(state, 0), action, reward, np.expand_dims(next_state, 0), next_action, done)
        else:
            e = Experience(state, action, reward, next_state, next_action, done)
        self.memory.append(e)


    def _sample_fast(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(self.__v_stack_impr([e.state for e in experiences if e is not None])) \
            .float().to(self.device)
        actions = torch.from_numpy(self.__v_stack_impr([e.action for e in experiences if e is not None])) \
            .long().to(self.device)
        rewards = torch.from_numpy(self.__v_stack_impr([e.reward for e in experiences if e is not None])) \
            .float().to(self.device)
        next_states = torch.from_numpy(self.__v_stack_impr([e.next_state for e in experiences if e is not None])).float().to(self.device)\
        
        next_action = torch.from_numpy(self.__v_stack_impr([e.next_action for e in experiences if e is not None])) \
            .long().to(self.device)
        dones = torch.from_numpy(self.__v_stack_impr([e.done for e in experiences if e is not None]).astype(np.uint8)) \
            .float().to(self.device)
        
        return states, actions, rewards, next_states, next_action, dones

    def sample(self, n=1, gamma=1, use_graph_observator=False):
        """Randomly sample a batch of experiences from memory."""
        if use_graph_observator:
            assert n == 1, "Graph observator only works with n=1"
            experiences = random.sample(self.memory, k=self.batch_size)
            states = [e.state for e in experiences if e is not None]
            actions = torch.from_numpy(self.__v_stack_impr([e.action for e in experiences if e is not None])) \
                .long().to(self.device)
            rewards = torch.from_numpy(self.__v_stack_impr([e.reward for e in experiences if e is not None])) \
                .float().to(self.device)
            next_states = [e.next_state for e in experiences if e is not None]
            next_action = torch.from_numpy(self.__v_stack_impr([e.next_action for e in experiences if e is not None])) \
                .long().to(self.device)
            dones = torch.from_numpy(self.__v_stack_impr([e.done for e in experiences if e is not None]).astype(np.uint8)) \
                .float().to(self.device)
            return states, actions, rewards, next_states, next_action, dones
        else:
            if n == 1:
                return self._sample_fast()
        
            len_ = len(self.memory)
            rewards = torch.zeros((self.batch_size,1)).to(self.device).float()
            idx = random.sample(range(0,len_-n), self.batch_size)
            # 2
            dones = torch.zeros((self.batch_size,1)).to(self.device).float()
            for i in range(n):
                experiences = [self.memory[j+i] for j in idx]
                    
                reward_step_n = torch.from_numpy(self.__v_stack_impr([e.reward for e in experiences if e is not None])) \
                    .float().to(self.device)
                rewards += reward_step_n*(gamma**i)
                if i == n-1:
                    states = torch.from_numpy(self.__v_stack_impr([e.state for e in experiences if e is not None])) \
                        .float().to(self.device)
                    actions = torch.from_numpy(self.__v_stack_impr([e.action for e in experiences if e is not None])) \
                        .long().to(self.device)
                    next_states = torch.from_numpy(self.__v_stack_impr([e.next_state for e in experiences if e is not None])).float().to(self.device)
                    next_action = torch.from_numpy(self.__v_stack_impr([e.next_action for e in experiences if e is not None])) \
                        .long().to(self.device)
                    
                dones_temp = torch.from_numpy(self.__v_stack_impr([e.done for e in experiences if e is not None]).astype(np.uint8)) \
                    .float().to(self.device)
                dones = torch.logical_or(dones, dones_temp)  # only learn from experiences where none of the steps are done
                    
            dones = dones.float()
            return states, actions, rewards, next_states, next_action, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

    def __v_stack_impr(self, states):
        sub_dim = len(states[0][0]) if isinstance(states[0], Iterable) else 1
        np_states = np.reshape(np.array(states), (len(states), sub_dim))
        return np_states
