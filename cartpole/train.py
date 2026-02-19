import gymnasium as gym
from .nn_policy import NNRewardPolicy
import torch
import matplotlib.pyplot as plt


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


def update_policy(policy, log_probs, returns, optimizer, average_baseline):
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
    env = gym.make("CartPole-v1", render_mode="human")

    policy = NNRewardPolicy(4, 2)

    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2)

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

        loss = update_policy(policy, log_probs, returns, optimizer, average_baseline)

        total_reward = sum(rewards)
        all_losses.append(loss)
        all_rewards.append(total_reward)
        average_returns.append(sum(batch_returns) / len(batch_returns))

        if episode % 20 == 0:  # Print every 20 episodes
            # calculate average baseline every 20 episodes
            average_baseline = sum(batch_returns) / len(batch_returns)
            batch_returns = []
            print(f"Episode {episode}, Total Reward: {total_reward}, Loss: {loss}, Average: {average_baseline}")
            # plot chart and save
            plt.figure(figsize=(10, 6))
            x_losses = range(1, len(all_losses) + 1)
            x_rewards = range(1, len(all_rewards) + 1)
            x_avg = range(1, len(average_returns) + 1)

            plt.plot(x_losses, all_losses, color="tab:blue", linewidth=1, label="Loss")
            plt.plot(
                x_rewards,
                all_rewards,
                color="tab:orange",
                linewidth=1,
                marker="o",
                markersize=4,
                markerfacecolor="tab:orange",
                markeredgecolor="k",
                label="Total Reward",
            )
            plt.plot(
                x_avg,
                average_returns,
                color="tab:green",
                linewidth=1,
                marker="o",
                markersize=4,
                markerfacecolor="tab:green",
                markeredgecolor="k",
                label="Average Return",
            )

            plt.xlabel("Episode")
            plt.ylabel("Value")
            plt.title("Training Metrics per Episode")
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig("training.png", dpi=150)
            plt.close()
            torch.save(policy.state_dict(), "./policy.pth")
    env.close()


# if __name__ == "__main__":
#     main()
