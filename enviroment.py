import time
import torch
import pygame
import random
import math
import numpy as np


class ArrowGameEnv:
    def __init__(self, width=600, height=400, num_dots=20, max_dots=50, arrow_speed=5, plot=False):
        pygame.init()
        self.width = width
        self.height = height
        self.num_dots = num_dots
        self.arrow_speed = arrow_speed

        self.plot = plot

        display_flag = pygame.SHOWN if plot else pygame.HIDDEN
        self.screen = pygame.display.set_mode((width, height), flags=display_flag)
        pygame.display.set_caption("Arrow Game")

        # params
        self.min_collision_dist = 10
        self.max_dots = max_dots

        # Arrow properties
        self.arrow_length = 30
        self.arrow_angle = 0  # Initial angle in degrees
        self.arrow_pos = [width // 2, height // 2]

        # Dot properties
        self.dots = []
        for _ in range(num_dots):
            self.dots.append([random.randint(0, width), random.randint(0, height)])

    def update_screen_render_mode(self, is_shown=True):
        self.screen = pygame.display.set_mode((self.width, self.height), flags=is_shown)

    def reset(self):
        # Reset environment to initial state
        self.arrow_angle = 0
        self.arrow_pos = [self.width // 2, self.height // 2]
        self.dots = []
        for _ in range(self.num_dots):
            self.dots.append([random.randint(0, self.width), random.randint(0, self.height)])
        self.render()

        return self._get_state()

    def step(self, action):
        # Action is either 0 (turn left) or 1 (turn right) or 2 (do nothing)
        if action == 0:
            self.arrow_angle -= 5
        elif action == 1:
            self.arrow_angle += 5

        # Calculate new arrow position based on angle and speed
        dx = self.arrow_speed * math.cos(math.radians(self.arrow_angle))
        dy = self.arrow_speed * math.sin(math.radians(self.arrow_angle))

        if 0 < self.arrow_pos[0] + dx < self.width:
            self.arrow_pos[0] += dx

        if 0 < self.arrow_pos[1] + dy < self.height:
            self.arrow_pos[1] += dy

        self.render()

        # Check for dot collisions
        reward = 0
        for i, dot in enumerate(self.dots):
            if math.dist(self.arrow_pos, dot) < self.min_collision_dist:
                self.dots.pop(i)
                reward += 1

        # Handle boundary collision (e.g., bounce, stop, or wrap around)
        if self.arrow_pos[0] < 0 or self.arrow_pos[0] > self.width:
            reward -= 0.1
        if self.arrow_pos[1] < 0 or self.arrow_pos[1] > self.height:
            reward -= 0.1

        done = len(self.dots) == 0

        return self._get_state(), reward, done

    def render(self):
        self.screen.fill((0, 0, 0))

        # Draw arrow
        arrow_end_x = self.arrow_pos[0] + self.arrow_length * math.cos(math.radians(self.arrow_angle))
        arrow_end_y = self.arrow_pos[1] + self.arrow_length * math.sin(math.radians(self.arrow_angle))
        pygame.draw.line(self.screen, (255, 0, 0), self.arrow_pos, (arrow_end_x, arrow_end_y), 3)

        # Draw dots
        for dot in self.dots:
            pygame.draw.circle(self.screen, (255, 255, 255), dot, 5)

        pygame.display.flip()
        # time.sleep(0.001)

    def _get_state(self):

        arrow_x, arrow_y = self.arrow_pos
        state = np.array([
            arrow_x / self.width,  # Normalized arrow x position
            arrow_y / self.height,  # Normalized arrow y position
            math.cos(math.radians(self.arrow_angle)),  # Arrow direction cosine
            math.sin(math.radians(self.arrow_angle))  # Arrow direction sine
        ], dtype=np.float32)

        # Pad with zeros for missing dots
        dot_info = np.zeros(self.max_dots * 2, dtype=np.float32)
        for i, dot in enumerate(self.dots):
            dot_x, dot_y = dot
            dot_info[i * 2] = (dot_x - arrow_x) / self.width
            dot_info[i * 2 + 1] = (dot_y - arrow_y) / self.height
        state = np.concatenate((state, dot_info))
        image_ = np.transpose(pygame.surfarray.array3d(self.screen) / 255, (2, 0, 1))
        return state, image_
