import gymnasium as gym
import yaml
import os

def run_multiple_episodes(num_episodes=5):
    # Check if config.yaml exists
    config_path = '../config.yaml'
    use_config = os.path.exists(config_path)
    
    if use_config:
        # Load configuration from config.yaml
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        env = gym.make('LiteSocNavGym-v0', **config)
    else:
        # Initialize environment with default parameters
        env = gym.make('LiteSocNavGym-v0')
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        step = 0
        total_reward = 0
        print(f"\n--- Episode {episode + 1} ---")
        
        while not done and not truncated:
            action = env.action_space.sample()  # Random action
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            step += 1
            env.render()
        
        print(f"Episode {episode + 1} finished after {step} steps with total reward {total_reward:.2f}")
    
    env.close()

if __name__ == "__main__":
    run_multiple_episodes()


Explanation of the Script:

    Import Statements:
        gymnasium as gym: For interacting with the Gymnasium environment.
        yaml: For parsing config.yaml if it exists.
        os: To check for the existence of config.yaml.
    Function run_multiple_episodes:
        Parameters: num_episodes=5 specifies the number of episodes to run.
        Configuration Handling:
            Checks if config.yaml exists in the root directory.
            If it exists, loads configuration parameters from it.
            If not, initializes the environment with default parameters.
        Episode Loop: Runs the specified number of episodes.
            Reset Environment: Resets at the start of each episode.
            Step Loop: Continues taking random actions until the episode is done or truncated.
                Sample Action: Chooses a random action from the action space.
                Step Environment: Applies the action.
                Render: Visualizes the environment.
                Accumulate Rewards and Steps: Tracks total rewards and steps taken.
        Close Environment: Cleans up after all episodes.
    Main Block:
        Calls run_multiple_episodes when the script is executed.
