import gymnasium as gym
import numpy as np
from enum import Enum
import math
from .nn_policy import NNRewardPolicy
import torch

nn_policy = NNRewardPolicy(4, 2)

nn_policy.load_state_dict(torch.load("./policy.pth", weights_only=True))
nn_policy.eval()


class PolicyType(Enum):
    RANDOM = "Random"
    RULE_BASED = "RuleBased"
    NEURAL_NETWORK = "NeuralNetwork"


def random_policy(**kwargs):
    return np.random.randint(0, 2)


def rule_based_policy(angle, **kwargs):
    if angle > 0:
        return 1
    else:
        return 0


def neural_network_policy(state, **kwargs):
    return nn_policy.get_action(state)


def get_policy(type: PolicyType, **kwargs) -> np.int64:
    policy_mapping = {
        PolicyType.RANDOM: random_policy,
        PolicyType.RULE_BASED: rule_based_policy,
        PolicyType.NEURAL_NETWORK: neural_network_policy,
    }

    policy = policy_mapping.get(type)

    if policy:
        return policy(**kwargs)
    raise ValueError(f"Invalid policy type: {type}")


env = gym.make("CartPole-v1", render_mode="human")
state, info = env.reset()

total_reward = 0
done = False

while not done:
    pole_angle = state[2]  # (position, velocity, angle, angular_velocity)
    action, log_probs = get_policy(PolicyType.NEURAL_NETWORK, angle=pole_angle, state=state)
    state, reward, terminated, truncated, info = env.step(action)

    done = terminated or truncated  # TODO: What is truncated. When its terminates??
    # The episode ends if any one of the following occurs:
    # Termination: Pole Angle is greater than ±12°
    # Termination: Cart Position is greater than ±2.4 (center of the cart reaches the edge of the display)
    # Truncation: Episode length is greater than 500 (200 for v0)

    total_reward += reward

    print(f"State: {state}, Action: {action}, Reward: {reward}")

    # check why it terminated when did
    if done:
        print(f"Total Reward: {total_reward}")
        position, velocity, angle, angular_velocity = state
        if position > 2.4 or position < -2.4:
            print(f"Terminated due to position: {position}")

        # convert angle to degree
        angle_degree = math.degrees(angle)
        if angle_degree > 12 or angle_degree < -12:
            print(f"Terminated due to angle: {angle_degree} degrees")


env.close()
