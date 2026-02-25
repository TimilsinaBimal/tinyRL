import matplotlib.pyplot as plt


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
