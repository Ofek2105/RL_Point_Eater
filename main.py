from enviroment import ArrowGameEnv
from agent import DQNAgent
import os
import torch

'''
    This Project was run on cuda version:
    Cuda compilation tools, release 12.0, V12.0.76
    Build cuda_12.0.r12.0/compiler.31968024_0

    and python version 3.8
'''


def create_dirs():
    """
    Creates "runs" and "saved_models" directories if they do not exist in the project's root directory.
    """
    dirs = ["runs", "saved_models"]

    for dir_ in dirs:
        if not os.path.exists(dir_):
            os.makedirs(dir_)


def train(env, agent, num_episodes, max_steps, log_dir='runs/', save_dir='saved_models', continue_model_path=None):
    # writer = SummaryWriter(log_dir=log_dir)
    best_loss = float('inf')
    if continue_model_path is not None:
        agent.initialize_weights_from_model_path(continue_model_path)
        print("continue training with given model")

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

        # writer.add_scalar('reward', total_reward, episode)

        saved_str = ""
        if loss < best_loss:
            torch.save(agent.q_network.state_dict(), os.path.join(save_dir, 'best.pth'))
            saved_str = "Saved"
        best_loss = loss

        print(f"Episode: {episode}, Total reward: {total_reward} randomness {agent.epsilon:.3f} {saved_str}")

        # Update epsilon
        agent.epsilon = max(agent.min_epsilon, agent.epsilon * agent.epsilon_decay)

    # writer.close()


def evaluate(env, agent, num_episodes, max_steps, model_path, render=True):
    print("\nStarting visual Evaluating")
    env.update_screen_render_mode(is_shown=render)
    agent.q_network.load_state_dict(torch.load(model_path))
    agent.q_network.eval()

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0

        for step in range(max_steps):
            action = agent.choose_action(state, exploration=False)
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
    num_dots = 15
    max_dots = 20
    arrow_speed = 1

    # bools
    show_screen = False
    use_gpu = True

    # Calculate state size based on your environment's state representation
    state_size = 4 + max_dots * 3  # Example: arrow position (2), angle (2), relative dot positions (3 * num_dots)

    # Agent parameters
    agent_params = {
        'state_size': state_size,
        'action_size': 3,  # turn left, turn right, do nothing
        'learning_rate': 0.0001,
        'gamma': 0.98,
        'epsilon': 1.0,
        'epsilon_decay': 0.995,
        'min_epsilon': 0.01,
        'memory_size': 5000,
        'batch_size': 32,
        'update_frequency': 50,
        'GPU': use_gpu
    }
    agent = DQNAgent(**agent_params)
    env = ArrowGameEnv(env_width, env_height, num_dots, max_dots, arrow_speed, show_screen)

    # Training
    num_episodes = 5000
    max_steps = 300

    train(env, agent, num_episodes, max_steps, continue_model_path=None)

    # eval
    evaluate(env, agent, 10, 200, 'saved_models/best.pth')


if __name__ == "__main__":
    main()
