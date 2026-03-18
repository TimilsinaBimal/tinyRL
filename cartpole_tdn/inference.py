import gymnasium as gym
from .model import CartpolePolicy
import torch
from pathlib import Path

current_file_path = Path(__file__).resolve()
current_dir_path = current_file_path.parent
# load model
nn_policy = CartpolePolicy(4, 2)

nn_policy.load_state_dict(torch.load(f"{current_dir_path}/actor.pth", weights_only=True))
nn_policy.eval()

env = gym.make("CartPole-v1", render_mode="human")
state, info = env.reset()


total_reward = 0
done = False

while not done:
    action, log_probs = nn_policy.get_action(state)
    state, reward, terminated, truncated, info = env.step(action)

    done = terminated or truncated  # TODO: What is truncated. When its terminates??

    total_reward += reward

    print(f"State: {state[:3]}, Action: {action}, Reward: {reward}")

print(f"total Reward: {total_reward}")
env.close()
