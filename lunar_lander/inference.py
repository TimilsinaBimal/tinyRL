import gymnasium as gym
from .model import LunarLanderPolicy
import torch

# load model
nn_policy = LunarLanderPolicy(8, 4)

nn_policy.load_state_dict(torch.load("./lunar_lander/policy.pth", weights_only=True))
nn_policy.eval()

env = gym.make("LunarLander-v3", render_mode="human")
state, info = env.reset()


total_reward = 0
done = False

while not done:
    action, log_probs = nn_policy.get_action(state)
    state, reward, terminated, truncated, info = env.step(action)

    done = terminated or truncated  # TODO: What is truncated. When its terminates??

    total_reward += reward

    print(f"State: {state[:3]}, Action: {action}, Reward: {reward}")


env.close()
