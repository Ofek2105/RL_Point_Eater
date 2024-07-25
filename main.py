from enviroment import ArrowGameEnv
from agent import DQNAgent
from tensorboardX import SummaryWriter
import os
import torch


def create_dirs():
    """
    Creates "runs" and "saved_models" directories if they do not exist in the project's root directory.
    """
    dirs = ["runs", "saved_models"]

    for dir_ in dirs:
        if not os.path.exists(dir_):
            os.makedirs(dir_)


def train(env, agent, num_episodes, max_steps, log_dir='runs', save_dir='saved_models'):
    writer = SummaryWriter(log_dir=log_dir)
    best_loss = float('inf')

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0

        for step in range(max_steps):
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            loss = agent.replay()
            state = next_state
            total_reward += reward

        print(f"Episode: {episode}, Total reward: {total_reward}")
        writer.add_scalar('reward', total_reward, episode)

        if loss < best_loss:
            best_loss = loss
            torch.save(agent.q_network.state_dict(), os.path.join(save_dir, 'best.pth'))

        # Update epsilon
        agent.epsilon = max(agent.min_epsilon, agent.epsilon * agent.epsilon_decay)

    writer.close()


def evaluate(env, agent, num_episodes, max_steps, model_path, render=True):
    env.update_screen_render_mode(is_shown=render)
    agent.q_network.load_state_dict(torch.load(model_path))
    agent.q_network.eval()

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0

        for step in range(max_steps):
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            state = next_state
            total_reward += reward

            if render:
                env.render()

            if done:
                print(f"Evaluation Episode: {episode}, Total reward: {total_reward}")
                break

def main():
    # Environment parameters
    env_width = 600
    env_height = 400
    num_dots = 20
    max_dots = 50
    arrow_speed = 1

    # bools
    show_screen = True
    use_gpu = True

    # Calculate state size based on your environment's state representation
    state_size = 4 + max_dots * 2  # Example: arrow position (2), angle (2), relative dot positions (2 * num_dots)

    # Agent parameters
    agent_params = {
        'state_size': state_size,
        'action_size': 3,  # turn left, turn right, do nothing
        'learning_rate': 0.001,
        'gamma': 0.95,
        'epsilon': 1.0,
        'epsilon_decay': 0.995,
        'min_epsilon': 0.01,
        'memory_size': 10000,
        'batch_size': 32,
        'update_frequency': 4,
        'GPU': use_gpu
    }
    agent = DQNAgent(**agent_params)
    env = ArrowGameEnv(env_width, env_height, num_dots, max_dots, arrow_speed, show_screen)

    # Training
    num_episodes = 50000
    max_steps = 500
    train(env, agent, num_episodes, max_steps)
    evaluate(env, agent, 10, 200, 'saved_models/best.pth')


if __name__ == "__main__":
    main()
