from socnav_env import LiteSocNavGym

def run_multiple_episodes(num_episodes=5):
    try:
        # Initialize the environment
        env = LiteSocNavGym(render_mode='rgb_array')  # 'rgb_array' for visualization
    except Exception as e:
        print(f"Failed to initialize the environment: {e}")
        return

    try:
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
    except Exception as e:
        print(f"An error occurred during the simulation: {e}")
    finally:
        env.close()

if __name__ == "__main__":
    run_multiple_episodes()
