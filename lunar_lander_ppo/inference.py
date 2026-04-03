import gymnasium as gym
from .model import PPOPolicy
import torch
from pathlib import Path

env = gym.make("LunarLander-v3", render_mode="human")
state, info = env.reset()

# load model
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
nn_policy = PPOPolicy(obs_dim, action_dim)

current_dir = Path(__file__).resolve().parent

nn_policy.load_state_dict(torch.load(f"{current_dir}/model.pth", weights_only=True))
nn_policy.eval()


total_reward = 0
done = False

while not done:
    action, _, _, _ = nn_policy.act(state)
    state, reward, terminated, truncated, info = env.step(action.item())

    done = terminated or truncated

    total_reward += reward

    print(f"State: {state[:3]}, Action: {action}, Reward: {reward}")

print(f"Total Reward: {total_reward}")

env.close()
