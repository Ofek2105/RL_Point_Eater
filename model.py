import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DQN_IM(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN_IM, self).__init__()
        # Define convolutional layers with max pooling for image processing
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)  # Reduce image size after conv1
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)  # Reduce image size after conv2
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=0)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)  # Reduce image size after conv3

        # Define fully connected layers for processing state features (x)
        self.fc1_state = nn.Linear(input_size, 64)
        self.fc2_state = nn.Linear(64, 64)
        self.fc3_state = nn.Linear(64, 32)
        self.fc4_state = nn.Linear(32, 32)
        self.fc5_state = nn.Linear(32, 32)  # Five fully connected layers for state features

        # Define fully connected layers for processing concatenated vector
        # Adjust input size based on the final image size after pooling
        self.fc1_conv = nn.Linear((1280 + 32), 64)  # Assuming 3 pooling layers, final image size is 75x75
        self.fc2_conv = nn.Linear(64, 64)
        self.fc3_conv = nn.Linear(64, 32)
        self.fc4_conv = nn.Linear(32, output_size)  # Four fully connected layers for concatenated vector

    def forward(self, x, image):
        # Process image through convolutional layers with max pooling
        x2 = F.relu(self.conv1(image))
        x2 = self.pool1(x2)  # Reduce image size
        x2 = F.relu(self.conv2(x2))
        x2 = self.pool2(x2)  # Reduce image size
        x2 = F.relu(self.conv3(x2))
        x2 = self.pool3(x2)  # Reduce image size
        x2 = x2.view(x2.size(0), -1)  # Flatten the final convolutional output

        # Process state features (x) through fully connected layers
        state_features = F.relu(self.fc1_state(x))
        state_features = F.relu(self.fc2_state(state_features))
        state_features = F.relu(self.fc3_state(state_features))
        state_features = F.relu(self.fc4_state(state_features))
        state_features = F.relu(self.fc5_state(state_features))

        # Concatenate features from image and state
        combined_features = torch.cat((state_features, x2), 1)

        # Process combined features through fully connected layers
        ans = F.relu(self.fc1_conv(combined_features))
        ans = F.relu(self.fc2_conv(ans))
        ans = F.relu(self.fc3_conv(ans))
        ans = self.fc4_conv(ans)  # Final layer without ReLU for Q-value output
        return ans