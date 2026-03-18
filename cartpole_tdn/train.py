import gymnasium as gym
from .model import CartpolePolicy, ValueNetwork
import torch
from shared.figure import plot_and_save_reward_graph
import torch.nn.functional as F
from pathlib import Path
from math import pow

GAMMA = 0.99
N = 10


def compute_returns(rewards, gamma=0.99):
    returns = []
    return_ = 0
    for reward in rewards[::-1]:  # reverse and loop
        return_ = reward + gamma * return_
        returns.append(return_)

    return returns[::-1]  # reverse


def calculate_advantage(returns: list, predicted_returns: torch.Tensor):
    # convert returns and value function to tensor
    returns = torch.tensor(returns)
    # detach tensor from gradients
    predicted_returns = predicted_returns.detach()
    advantage = returns - predicted_returns
    # now normalize advantage to solve learning stability issue
    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
    # 1e-8 for numerical stability as in std can be zero
    return advantage


def update_policy(policy, log_probs, optimizer, advantage):
    try:
        log_probs = torch.stack(log_probs)
    except Exception:
        pass
    objective = log_probs * advantage
    objective = objective.mean()

    # print(f"Advantage mean: {advantage.mean().item()}, std: {advantage.std().item()}")
    # print(f"Log probs mean: {log_probs.mean().item()}")

    # use average loss
    loss = -objective  # use negative sign for gradient ascent

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
    optimizer.step()
    return loss.item()


def update_critic(critic: ValueNetwork, returns: list, predicted_returns: torch.Tensor, optimizer):
    returns = torch.tensor(returns, dtype=torch.float32)

    # calculate loss
    loss = F.mse_loss(predicted_returns, returns)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=0.5)
    optimizer.step()
    return loss.item()


def calculate_td_target(reward, next_estimate=0):
    if not isinstance(reward, list):
        reward = [reward]

    total_reward = 0

    for i in range(len(reward)):
        total_reward += pow(GAMMA, i) * reward[i]

    # r + gamma r_t+1, gamma^2rt+2...+ gamma^n V_st+n
    return total_reward + pow(GAMMA, len(reward)) * next_estimate


def calculate_advantage_td(reward, current_estimate, next_estimate):
    td_target = calculate_td_target(reward, next_estimate)
    # r + gamma r_t+1, gamma^2rt+2...+ gamma^n V_st+n - Vst
    advantage = td_target - current_estimate.item()  # detach from graph
    return advantage


def main():
    env = gym.make("CartPole-v1")

    actor = CartpolePolicy(4, 2)  # states, action
    critic = ValueNetwork(4, 1)  # states, return

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)

    num_episodes = 10000

    all_losses = []
    all_rewards = []

    for episode in range(num_episodes):
        episode_reward = 0
        n = 1
        current_state, info = env.reset()
        done = False

        n_timestep_rewards = []
        n_step_log_probs = []
        n_step_start_state = current_state

        while not done:
            action, log_prob = actor.get_action(current_state, train=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            n_timestep_rewards.append(reward)
            n_step_log_probs.append(log_prob)
            done = terminated or truncated
            episode_reward += reward
            current_state = next_state

            if n != N and not done:
                n += 1
                continue

            # calculate td target
            if done:
                td_target = calculate_td_target(n_timestep_rewards)
            else:
                td_target = calculate_td_target(n_timestep_rewards, critic.get_return(current_state).detach())

            # V(s) at the start of the n-step window
            start_state_estimate = critic.get_return(n_step_start_state)
            advantage = td_target - start_state_estimate.item()

            critic_loss = update_critic(critic, [td_target], start_state_estimate, critic_optimizer)
            loss = update_policy(actor, n_step_log_probs, actor_optimizer, advantage)

            all_losses.append(loss)
            n = 1
            n_timestep_rewards = []
            n_step_log_probs = []
            n_step_start_state = current_state

        all_rewards.append(episode_reward)
        if episode % 20 == 0:  # Print every 20 episodes
            print(
                f"Episode {episode}, Total Reward: {episode_reward}, Actor Loss: {loss}, Critic Loss:"
                f" {critic_loss}"
            )

            # Get the Path object for the current file
            current_file_path = Path(__file__).resolve()

            # Get the directory containing the current file
            current_dir_path = current_file_path.parent

            plot_and_save_reward_graph(all_rewards, file_path=f"{current_dir_path}/reward_graph.png")

            torch.save(actor.state_dict(), f"{current_dir_path}/actor.pth")

    env.close()  #


# if __name__ == "__main__":
#     main()
