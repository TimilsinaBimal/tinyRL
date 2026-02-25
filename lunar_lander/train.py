import gymnasium as gym
from .model import LunarLanderPolicy
import torch
from shared.figure import plot_and_save_graph


def collect_episode(env, policy):
    states = []
    actions = []
    rewards = []
    log_probs = []

    state, info = env.reset()

    done = False

    while not done:
        action, log_prob = policy.get_action(state, train=True)
        states.append(state)
        state, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated  # TODO: What is truncated. When its terminates??

        actions.append(action)
        rewards.append(reward)
        log_probs.append(log_prob)

    return states, actions, rewards, log_probs


def compute_returns(rewards, gamma=0.99):
    returns = []
    return_ = 0
    for reward in rewards[::-1]:  # reverse and loop
        return_ = reward + gamma * return_
        returns.append(return_)

    return returns[::-1]  # reverse


def update_policy(log_probs, returns, optimizer, average_baseline):
    log_probs = torch.stack(log_probs)

    returns = torch.tensor(returns)
    # substract returns from average return
    returns = returns - average_baseline
    objective = log_probs * returns
    objective = objective.mean()

    # use average loss
    loss = -objective  # use negative sign for gradient ascent

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def main():
    env = gym.make("LunarLander-v3", render_mode="human")

    policy = LunarLanderPolicy(8, 4)

    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    num_episodes = 1000

    average_baseline = 100  # random starting point
    batch_returns = []

    all_losses = []
    all_rewards = []
    average_returns = []

    for episode in range(num_episodes):
        states, actions, rewards, log_probs = collect_episode(env, policy)

        returns = compute_returns(rewards)
        batch_returns.extend(returns)
        # calculate average for first episode
        if episode == 0:
            average_baseline = sum(batch_returns) / len(batch_returns)

        loss = update_policy(log_probs, returns, optimizer, average_baseline)

        total_reward = sum(rewards)
        all_losses.append(loss)
        all_rewards.append(total_reward)
        average_returns.append(sum(batch_returns) / len(batch_returns))

        if episode % 20 == 0:  # Print every 20 episodes
            # calculate average baseline every 20 episodes
            average_baseline = sum(batch_returns) / len(batch_returns)
            batch_returns = []
            print(f"Episode {episode}, Total Reward: {total_reward}, Loss: {loss}, Average: {average_baseline}")
            plot_and_save_graph(all_losses, all_rewards, average_returns, file_path="cartpole/figs/chart.png")

            torch.save(policy.state_dict(), "./policy.pth")
    env.close()


# if __name__ == "__main__":
#     main()
