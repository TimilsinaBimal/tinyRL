import matplotlib.pyplot as plt
import numpy as np


def plot_and_save_graph(losses, rewards, returns, file_path="training.png"):
    # plot chart and save
    plt.figure(figsize=(10, 6))
    x_losses = range(1, len(losses) + 1)
    x_rewards = range(1, len(rewards) + 1)
    x_avg = range(1, len(returns) + 1)

    plt.plot(x_losses, losses, color="tab:blue", linewidth=1, label="Loss", marker="o")
    plt.plot(x_rewards, rewards, color="tab:orange", linewidth=1, marker="o", label="Total Reward")
    plt.plot(x_avg, returns, color="tab:green", linewidth=1, marker="o", label="Average Return")

    plt.xlabel("Episode")
    plt.ylabel("Value")
    plt.title("Training Metrics per Episode")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(file_path, dpi=150)
    plt.close()


def plot_and_save_reward_graph(rewards, file_path="training.png", window=50):
    """Plot per-episode rewards with a smoothed running average."""

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(rewards, alpha=0.3, color="#4C9BE8", linewidth=0.8, label="Episode reward")

    # Smoothed curve (window=50)
    if len(rewards) >= 50:
        window = 50
        smoothed = [
            sum(rewards[max(0, i - window) : i + 1]) / len(rewards[max(0, i - window) : i + 1])
            for i in range(len(rewards))
        ]
        ax.plot(smoothed, color="#E84C4C", linewidth=1.5, label=f"Rolling avg (w={window})")

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Reward", fontsize=12)
    ax.set_title("PPO — LunarLander-v3", fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(file_path, dpi=150)
    plt.close(fig)
