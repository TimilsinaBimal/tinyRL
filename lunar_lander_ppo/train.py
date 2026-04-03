import gymnasium as gym
from .model import PPOPolicy
import torch
from shared.figure import plot_and_save_reward_graph
import torch.nn.functional as F
from pathlib import Path

GAMMA = 0.99
EPSILON = 0.2  # PPO clip range
C1 = 0.5  # value loss coefficient
C2 = 0.005  # entropy coefficient
LR = 2.5e-4
MAX_GRAD_NORM = 0.5

TOTAL_UPDATES = 1500  # number of policy update iterations
ROLLOUT_STEPS = 2048  # steps collected per update
N_EPOCHS = 10  # passes over rollout data per update
MINI_BATCH = 64  # mini-batch size
GAE_LAMBDA = 0.95


def calculate_gae(rewards, terminates, values, next_value):
    advantages = []
    gae = 0.0

    # values is a list of V(s_t), next_value is V(s_T+1)
    values_t = torch.stack(values)  # [T]
    values_next = torch.cat([values_t[1:], next_value.unsqueeze(0)])  # V(s_{t+1})

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + GAMMA * values_next[t] * (1 - terminates[t]) - values_t[t]
        gae = delta + GAMMA * GAE_LAMBDA * (1 - terminates[t]) * gae
        advantages.append(gae)

    advantages = torch.stack(advantages[::-1])  # [T]
    returns = advantages + values_t  # value loss target

    return advantages.detach(), returns.detach()


def calculate_return(rewards: list, terminates: list, next_value: torch.Tensor) -> torch.Tensor:
    returns = []
    G = next_value.clone()
    for reward, term in zip(reversed(rewards), reversed(terminates)):  # reverse and loop
        G = reward + GAMMA * G * (
            1 - term
        )  # why terminate? because if episode ends, we will use zero as next estimate# doesn't include truncated since its artificial end
        returns.append(G)

    return torch.stack(returns[::-1])  # reverse


def calculate_policy_loss(ratio: float, advantage: torch.Tensor, epsilon=EPSILON) -> torch.Tensor:
    # min(r*A, clip(r, 1-epsilon, 1+ epsilon) * A)
    surr1 = ratio * advantage
    surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantage
    policy_loss: torch.Tensor = -torch.min(surr1, surr2).mean()
    return policy_loss


def main():
    env = gym.make("LunarLander-v3")
    obs_dim = env.observation_space.shape[0]  # 8
    action_dim = env.action_space.n  # 4
    ppo_model: PPOPolicy = PPOPolicy(obs_dim, action_dim)  # states, action
    optimizer = torch.optim.Adam(ppo_model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.3, total_iters=TOTAL_UPDATES
    )
    episode_rewards: list[float] = []
    current_state, _ = env.reset()
    episode_reward = 0.0
    total_steps = 0

    done = False
    current_state, info = env.reset()
    episode_reward = 0

    for update_ in range(TOTAL_UPDATES):
        rollout = dict(states=[], actions=[], rewards=[], log_probs=[], values=[], terminates=[])
        for t in range(ROLLOUT_STEPS):
            # act for current state
            action, value_estimate, log_prob, entropy = ppo_model.act(current_state, train=True)

            # take action using predicted action above
            next_state, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated

            # store all intermediate values
            rollout["states"].append(current_state)  # tensor
            rollout["actions"].append(action)  # tensor
            rollout["rewards"].append(reward)  # float
            rollout["log_probs"].append(log_prob)  # tensor
            rollout["values"].append(value_estimate)  # tensor
            rollout["terminates"].append(terminated)  # boolean

            current_state = next_state
            episode_reward += reward
            total_steps += 1

            if done:
                episode_rewards.append(episode_reward)
                current_state, _ = env.reset()
                episode_reward = 0

        with torch.no_grad():
            # TODO: check here
            _, next_value_estimate, _, _ = ppo_model.act(current_state)

        # returns: torch.Tensor = calculate_return(rollout["rewards"], rollout["terminates"], next_value_estimate)

        # values_estimate = torch.stack(rollout["values"])

        # advantage: torch.Tensor = returns - values_estimate

        # advantage = advantage.detach()

        advantage, returns = calculate_gae(
            rollout["rewards"], rollout["terminates"], rollout["values"], next_value_estimate
        )
        advantage = advantage.detach()
        # normalize advantage
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        # old log probs
        old_log_probs = torch.stack(rollout["log_probs"]).detach()  # [T] # detach from graph

        # for new log probabilities, we need to calculate log prob from all previous states with current model
        states_t = torch.as_tensor(rollout["states"], dtype=torch.float32)  # [T, obs]
        actions_t = torch.stack(rollout["actions"]).squeeze()  # [T]
        returns_t = returns.detach()  # [T]

        T = states_t.shape[0]

        policy_losses, value_losses, entropy_vals = [], [], []

        for _ in range(N_EPOCHS):
            # minibatch
            perm = torch.randperm(T)
            for start in range(0, T, MINI_BATCH):
                idx = perm[start : start + MINI_BATCH]

                mb_states = states_t[idx]
                mb_actions = actions_t[idx]
                mb_old_lp = old_log_probs[idx]

                mb_returns = returns_t[idx]
                mb_adv = advantage[idx]

                logits, new_values = ppo_model(mb_states)
                dist = torch.distributions.Categorical(logits=logits)

                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                ratio_r: float = torch.exp(new_log_probs - mb_old_lp)

                policy_loss: torch.Tensor = calculate_policy_loss(ratio_r, mb_adv)

                value_loss: torch.Tensor = F.mse_loss(new_values.squeeze(), mb_returns)

                loss = policy_loss + C1 * value_loss - C2 * entropy

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(ppo_model.parameters(), max_norm=0.5)
                optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_vals.append(entropy.item())

        scheduler.step()

        if update_ % 10 == 0:  # Print every 20 upadates

            recent = episode_rewards[-50:] if len(episode_rewards) >= 50 else episode_rewards
            avg_r = sum(recent) / len(recent) if recent else 0.0
            print(
                f"Update {update_:4d}/{TOTAL_UPDATES} | "
                f"Steps {total_steps:7d} | "
                f"Episodes {len(episode_rewards):5d} | "
                f"Avg(50) {avg_r:8.1f} | "
                f"P-loss {sum(policy_losses)/len(policy_losses):.4f} | "
                f"V-loss {sum(value_losses)/len(value_losses):.4f} | "
                f"Entropy {sum(entropy_vals)/len(entropy_vals):.4f}"
            )

            # Get the Path object for the current file
            current_dir_path = Path(__file__).resolve().parent

            plot_and_save_reward_graph(episode_rewards, file_path=f"{current_dir_path}/reward_graph.png")

            torch.save(ppo_model.state_dict(), f"{current_dir_path}/model.pth")

    env.close()  #
