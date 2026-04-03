import gymnasium as gym
from .model import CartpolePolicy, ValueNetwork
import torch
from shared.figure import plot_and_save_reward_graph
import torch.nn.functional as F
from pathlib import Path

GAMMA = 0.99


def collect_episode(env, policy: CartpolePolicy):
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
    # objective = objective.mean()

    # print(f"Advantage mean: {advantage.mean().item()}, std: {advantage.std().item()}")
    # print(f"Log probs mean: {log_probs.mean().item()}")

    # use average loss
    loss = -objective  # use negative sign for gradient ascent

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.1)
    optimizer.step()
    return loss.item()


def update_critic(critic: ValueNetwork, returns: list, predicted_returns: torch.Tensor, optimizer):
    returns = torch.tensor(returns, dtype=torch.float32)

    # calculate loss
    loss = F.mse_loss(predicted_returns, returns)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=0.1)
    optimizer.step()
    return loss.item()


def calculate_advantage_td(reward, current_estimate, next_estimate):
    current_estimate = current_estimate.detach()
    next_estimate = next_estimate.detach()
    return reward + GAMMA * next_estimate - current_estimate  # also called td error


def main():
    env = gym.make("CartPole-v1", render_mode="human")

    actor = CartpolePolicy(4, 2)  # states, action
    critic = ValueNetwork(4, 1)  # states, return

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-4)

    num_episodes = 1000

    all_losses = []
    all_rewards = []

    for episode in range(num_episodes):
        episode_reward = 0
        # states, actions, rewards, log_probs = collect_episode(env, actor)
        current_state, info = env.reset()
        done = False
        while not done:
            action, log_prob = actor.get_action(current_state, train=True)

            # current critic estimtate
            current_critic_estimate = critic.get_return(current_state)  # V(s_t)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward

            # calculate td target
            if terminated:
                # when it gets terminated, that means no next reward
                td_target = reward
            else:
                td_target = reward + GAMMA * critic.get_return(next_state).detach()

            # calculate estimate return
            for _ in range(2):
                current_critic_estimate = critic.get_return(current_state)
                # create predicted-returns from critic model now
                next_state_critic_estimate = critic.get_return(next_state)  # V(s_t+1)
                critic_loss = update_critic(critic, [td_target], current_critic_estimate, critic_optimizer)

            advantage = calculate_advantage_td(reward, current_critic_estimate, next_state_critic_estimate)

            # returns = compute_returns(rewards)

            # batch_returns.extend(returns)

            loss = update_policy(actor, log_prob, actor_optimizer, advantage)

            current_state = next_state  # for next time step, current next state will be its current state

        all_losses.append(loss)
        all_rewards.append(episode_reward)
        # average_returns.append(sum(batch_returns) / len(batch_returns))

        if episode % 20 == 0:  # Print every 20 episodes
            # batch_returns = []
            print(
                f"Episode {episode}, Total Reward: {episode_reward}, Actor Loss: {loss}, Critic Loss:"
                f" {critic_loss}, Predicted Returns: {next_state_critic_estimate.item()}"
            )

            # Get the Path object for the current file
            current_file_path = Path(__file__).resolve()

            # Get the directory containing the current file
            current_dir_path = current_file_path.parent

            plot_and_save_reward_graph(all_rewards, file_path=f"{current_dir_path}/reward_graph.png")

            torch.save(actor.state_dict(), f"{current_dir_path}/actor_td.pth")
            torch.save(critic.state_dict(), f"{current_dir_path}/critic_td.pth")
    env.close()


# if __name__ == "__main__":
#     main()
