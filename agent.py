
import torch
import torch.optim as optim
import numpy as np
import random
from model import DQN
from collections import deque
import torch.nn.functional as f


class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate, gamma, epsilon, epsilon_decay, min_epsilon, memory_size,
                 batch_size, update_frequency, GPU):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.update_frequency = update_frequency
        self.step_count = 0
        self.device = torch.device("cuda" if GPU else "cpu")

        # Initialize the DQN model
        self.q_network = DQN(state_size, action_size).to(torch.float32).to(self.device)
        self.target_network = DQN(state_size, action_size).to(torch.float32).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Replay memory
        self.memory = deque(maxlen=memory_size)

    def initialize_weights_from_model_path(self, path): # TODO: raise error when we try to reach a file that isn't there
        self.q_network.load_state_dict(torch.load(path))
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

    def choose_action(self, state, exploration=True):

        if exploration and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            state = torch.from_numpy(state).to(self.device)
            q_values = self.q_network(state)
            return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        actions, dones, next_states, rewards, states = self.sample_from_memory()

        current_q_values = self.q_network(states)
        next_q_values = self.target_network(next_states)
        target_q_values = rewards + (1 - dones) * self.gamma * torch.max(next_q_values, dim=1)[0]
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze()

        loss = f.mse_loss(current_q_values, target_q_values.detach())

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step_count += 1
        # Update the target network periodically
        if self.step_count % self.update_frequency == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss.item()

    def sample_from_memory(self):

        minibatch = random.sample(self.memory, self.batch_size)

        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.stack([torch.from_numpy(state) for state in states]).to(self.device)
        actions = torch.stack([torch.tensor(action, dtype=torch.int64) for action in actions]).to(self.device)
        rewards = torch.stack([torch.tensor(reward, dtype=torch.float32) for reward in rewards]).to(self.device)
        next_states = torch.stack([torch.tensor(next_state, dtype=torch.float32) for next_state in next_states]).to(self.device)
        dones = torch.stack([torch.tensor(done, dtype=torch.float32) for done in dones]).to(self.device)

        return actions, dones, next_states, rewards, states
