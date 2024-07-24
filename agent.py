
import torch
import torch.optim as optim
import numpy as np
import random
from model import DQN, DQN_IM
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
        self.q_network = DQN_IM(state_size, action_size).to(torch.float32).to(self.device)
        self.target_network = DQN_IM(state_size, action_size).to(torch.float32).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Replay memory
        self.memory = deque(maxlen=memory_size)

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            state_vec, state_image = state
            state_vec = torch.tensor(state_vec, dtype=torch.float32)
            state_image = torch.from_numpy(state_image)

            q_values = self.q_network(state_vec, state_image)
            return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        actions, dones, next_states, rewards, states = self.sample_from_memory()

        states_vec, states_image = self.prep_im_states(states)
        next_states_vec, next_states_image = self.prep_im_states(next_states)

        current_q_values = self.q_network(states_vec, states_image)
        next_q_values = self.target_network(next_states_vec, next_states_image)
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

        # states = torch.stack([torch.from_numpy(state) for state in states])
        # next_states = torch.stack([torch.tensor(next_state, dtype=torch.float32) for next_state in next_states])
        states = [(torch.tensor(state_vec, dtype=torch.float32).to(self.device), torch.from_numpy(state_image).to(torch.float32).to(self.device))
                              for state_vec, state_image in states]

        actions = torch.stack([torch.tensor(action, dtype=torch.int64).to(self.device) for action in actions])
        rewards = torch.stack([torch.tensor(reward, dtype=torch.float32).to(self.device) for reward in rewards])

        next_states = [(torch.tensor(state_vec, dtype=torch.float32).to(self.device), torch.from_numpy(state_image).to(torch.float32).to(self.device))
                   for state_vec, state_image in next_states]


        dones = torch.stack([torch.tensor(done, dtype=torch.float32).to(self.device) for done in dones])

        return actions, dones, next_states, rewards, states

    @staticmethod
    def prep_im_states(states):
        vec, im = zip(*states)
        return torch.stack(vec), torch.stack(im)