import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from gameEnvironment import EXPORT_FOLDER, EXPORT_LOSSES_FILENAME

# === CONFIG ===
MODEL_NUMBER = 40001
loss_file_path = os.path.join(EXPORT_FOLDER, f"{EXPORT_LOSSES_FILENAME}{MODEL_NUMBER}.pkl")
SAVE_DIR = f"{MODEL_NUMBER}_losses"
os.makedirs(SAVE_DIR, exist_ok=True)


# === LOAD DATA ===
with open(loss_file_path, 'rb') as f:
    data = pickle.load(f)

policy_losses = data['policy_losses']
value_losses = data['value_losses']
transition_losses = data['transition_losses']


def moving_average(data, window_size=1000):
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')


def save_plot(raw_data, title, ylabel, color, filename):
    x_raw = list(range(len(raw_data)))
    y_avg = moving_average(raw_data)

    plt.figure(figsize=(10, 6))
    plt.plot(x_raw, raw_data, label='Raw', color=color, alpha=0.3)
    plt.plot(range(len(y_avg)), y_avg, label=f'Moving Avg (1000)', color=color)
    plt.xlabel("Game")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, filename))
    plt.close()


# === SAVE EACH PLOT ===
save_plot(policy_losses, "Policy Loss", "Loss", 'blue', "policy_loss.png")
save_plot(value_losses, "Value Loss", "Loss", 'green', "value_loss.png")
save_plot(transition_losses, "Transition Loss", "Loss", 'red', "transition_loss.png")

# === SAVE COMBINED PLOT ===
x_policy = list(range(len(policy_losses)))
x_value = list(range(len(value_losses)))
x_transition = list(range(len(transition_losses)))

plt.figure(figsize=(10, 6))
plt.plot(x_policy, policy_losses, label="Policy Loss (Raw)", color='blue', alpha=0.3)
plt.plot(range(len(moving_average(policy_losses))), moving_average(policy_losses), label="Policy Loss (Avg)", color='blue')

plt.plot(x_value, value_losses, label="Value Loss (Raw)", color='green', alpha=0.3)
plt.plot(range(len(moving_average(value_losses))), moving_average(value_losses), label="Value Loss (Avg)", color='green')

plt.plot(x_transition, transition_losses, label="Transition Loss (Raw)", color='red', alpha=0.3)
plt.plot(range(len(moving_average(transition_losses))), moving_average(transition_losses), label="Transition Loss (Avg)", color='red')

plt.xlabel("Game")
plt.ylabel("Loss")
plt.title("Combined Losses")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "combined_loss.png"))
plt.close()
