import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import optuna
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from time import time
from itertools import count
import matplotlib.pyplot as plt
import matplotlib
import json
import os

matplotlib.use('TkAgg')  # Use a compatible backend like 'TkAgg' or 'Agg'

# Constants
MAX_STATE_DIM = 6
MAX_ACTION_DIM = 3


def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj


def save_model_and_params(model: nn.Module, hyperparameters: Dict[str, Any], env_name: str, save_dir: str = "models"):
    """Save the trained model and its hyperparameters."""
    os.makedirs(save_dir, exist_ok=True)

    # Save the model
    model_path = os.path.join(save_dir, f"{env_name}_progressive.pth")
    torch.save(model.state_dict(), model_path)

    # Convert numpy types to native Python types
    converted_params = convert_numpy_types(hyperparameters)

    # Save hyperparameters
    params_path = os.path.join(save_dir, f"{env_name}_progressive_params.json")
    with open(params_path, 'w') as f:
        json.dump(converted_params, f, indent=4)

    print(f"Saved model to {model_path}")
    print(f"Saved hyperparameters to {params_path}")


def plot_training_metrics(rewards_history, name, losses=None, save_dir="plots"):
    import os
    os.makedirs(save_dir, exist_ok=True)

    # Plot rewards
    plt.figure(figsize=(12, 6))
    plt.plot(rewards_history, label="Rewards")
    plt.title("Rewards During Training")
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_dir, f"{name}_prognets_rewards_plot.png"))
    plt.close()

    # Plot losses if provided
    if losses:
        plt.figure(figsize=(12, 6))
        plt.plot(losses, label="Loss")
        plt.title("Loss During Training")
        plt.xlabel("Episodes")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(save_dir, f"{name}_prognets_loss_plot.png"))
        plt.close()


@dataclass
class EnvConfig:
    """Configuration for different environments"""
    raw_state_dim: int
    raw_action_dim: int
    is_continuous: bool
    max_episodes: int
    convergence_reward: float
    optuna_config: Dict[str, Dict[str, Any]]


# Environment configurations
ENV_CONFIGS = {
    "CartPole-v1": EnvConfig(
        raw_state_dim=4,
        raw_action_dim=2,
        is_continuous=False,
        max_episodes=500,
        convergence_reward=475,
        optuna_config={
            "learning_rate": {
                "low": 1e-4,
                "high": 1e-2,
                "log": True
            },
            "hidden_size": {
                "low": 64,
                "high": 256,
                "step": 32
            },
            "gamma": {
                "low": 0.95,
                "high": 0.995,
                "step": 0.005
            }
        }
    ),
    "Acrobot-v1": EnvConfig(
        raw_state_dim=6,
        raw_action_dim=3,
        is_continuous=False,
        max_episodes=500,
        convergence_reward=-100,
        optuna_config={
            "learning_rate": {
                "low": 5e-5,
                "high": 5e-3,
                "log": True
            },
            "hidden_size": {
                "low": 128,
                "high": 512,
                "step": 64
            },
            "gamma": {
                "low": 0.97,
                "high": 0.999,
                "step": 0.002
            }
        }
    ),
    "MountainCarContinuous-v0": EnvConfig(
        raw_state_dim=2,
        raw_action_dim=1,
        is_continuous=True,
        max_episodes=1000,
        convergence_reward=90,
        optuna_config={
            "learning_rate": {
                "low": 1e-5,
                "high": 1e-3,
                "log": True
            },
            "hidden_size": {
                "low": 256,
                "high": 1024,
                "step": 128
            },
            "gamma": {
                "low": 0.98,
                "high": 0.999,
                "step": 0.001
            }
        }
    )
}


class UnifiedPolicyNetwork(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(MAX_STATE_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, MAX_ACTION_DIM)  # Output dimension large enough for all envs
        )

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x)
        return self.network(x)


class UnifiedValueNetwork(nn.Module):
    """Basic value network for critic."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(MAX_STATE_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x)
        return self.network(x)


class ProgressiveNetwork(nn.Module):
    def __init__(self, source_actor_networks: List[nn.Module], source_critic_networks: List[nn.Module],
                 target_actor_network: nn.Module, target_critic_network: nn.Module,
                 hidden_dim: int, output_dim: int, is_continuous: bool):
        super().__init__()
        self.source_actor_networks = source_actor_networks
        self.source_critic_networks = source_critic_networks
        self.target_actor_network = target_actor_network
        self.target_critic_network = target_critic_network
        self.is_continuous = is_continuous

        self.actor_adapters = nn.ModuleList([
            nn.Linear(source_actor.network[-1].out_features, hidden_dim)
            for source_actor in source_actor_networks
        ])

        self.critic_adapters = nn.ModuleList([
            nn.Linear(source_critic.network[-1].out_features, hidden_dim)
            for source_critic in source_critic_networks
        ])

        self.final_actor_layer = nn.Linear(MAX_ACTION_DIM + hidden_dim * len(source_actor_networks), output_dim)
        self.final_critic_layer = nn.Linear(1 + hidden_dim * len(source_critic_networks), 1)

    def forward(self, x):
        target_actor_output = self.target_actor_network(x)
        adapted_actor_outputs = [adapter(actor(x))
                               for actor, adapter in zip(self.source_actor_networks, self.actor_adapters)]

        target_critic_output = self.target_critic_network(x)
        adapted_critic_outputs = [adapter(critic(x))
                                for critic, adapter in zip(self.source_critic_networks, self.critic_adapters)]

        combined_actor_output = torch.cat([target_actor_output] + adapted_actor_outputs, dim=-1)
        combined_critic_output = torch.cat([target_critic_output] + adapted_critic_outputs, dim=-1)

        if self.is_continuous:
            # For continuous actions, output mean of action distribution
            final_actor_output = torch.tanh(self.final_actor_layer(combined_actor_output))
        else:
            # For discrete actions, output action probabilities
            final_actor_output = F.softmax(self.final_actor_layer(combined_actor_output), dim=-1)

        final_critic_output = self.final_critic_layer(combined_critic_output)

        return final_actor_output, final_critic_output


def load_model(path: str, model_class, hidden_dim: int, new_output_size: int = None) -> nn.Module:
    """Load and freeze a pre-trained model."""
    model = model_class(hidden_dim)
    saved_state = torch.load(path, weights_only=True, map_location=torch.device('cpu'))

    for name, param in saved_state.items():
        if name in model.state_dict():
            if model.state_dict()[name].size() == param.size():
                model.state_dict()[name].copy_(param)

    if new_output_size:
        model.network[-1] = nn.Linear(hidden_dim, new_output_size)

    model.eval()
    return model


def pad_state(state: np.ndarray, max_size: int = MAX_STATE_DIM) -> np.ndarray:
    """Pad state vector to fixed size."""
    padded = np.zeros(max_size)
    padded[:len(state)] = state
    return padded


def train_progressive_network(
        model: ProgressiveNetwork,
        env_name: str,
        num_episodes: int = 500,
        learning_rate: float = 1e-4,
        gamma: float = 0.99
):
    env = gym.make(env_name)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    is_continuous = isinstance(env.action_space, gym.spaces.Box)

    rewards_history = []
    losses = []
    start_time = time()

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        log_probs = []
        values = []
        rewards = []

        for t in count():
            state_tensor = torch.FloatTensor(pad_state(state)).unsqueeze(0)
            action_output, value = model(state_tensor)
            value = value.squeeze(-1)

            if torch.isnan(action_output).any() or torch.isinf(action_output).any():
                break

            if is_continuous:
                action_numpy = action_output.detach().numpy()
                noise = np.random.normal(0, 0.1, size=action_numpy.shape)
                action = np.clip(action_numpy + noise, env.action_space.low, env.action_space.high)
                log_prob = -0.5 * ((action - action_numpy) ** 2).sum()  # Simplified Gaussian log prob
                log_prob = torch.tensor(log_prob, dtype=torch.float32)  # Ensure it's a PyTorch tensor
            else:
                action_probs = action_output
                action = torch.multinomial(action_probs, 1).item()
                log_prob = torch.log(action_probs[0, action])

            if is_continuous:
                next_state, reward, done, truncated, _ = env.step(action[0])
            else:
                next_state, reward, done, truncated, _ = env.step(action)

            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value)
            episode_reward += reward

            if done or truncated:
                break
            state = next_state

        # Compute returns and advantages
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32)
        values = torch.cat([v.view(1) for v in values]).squeeze()
        advantages = returns - values.detach()

        # Update policy and value function
        actor_loss = -(torch.stack(log_probs) * advantages).mean()
        critic_loss = nn.MSELoss()(values, returns)
        loss = actor_loss + critic_loss
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        rewards_history.append(episode_reward)

        if episode % 100 == 0:
            print(f"Episode {episode}, Average Reward: {np.mean(rewards_history[-10:]):.2f}")

        if len(rewards_history) >= 10 and np.mean(rewards_history[-10:]) > ENV_CONFIGS[env_name].convergence_reward:
            break

    plot_training_metrics(rewards_history, env_name, losses)
    return rewards_history, time() - start_time


class OptunaOptimizer:
    """Handles hyperparameter optimization using Optuna."""

    def __init__(self, env_name: str, source_actor_networks: List[nn.Module], source_critic_networks: List[nn.Module]):
        self.env_name = env_name
        self.source_actor_networks = source_actor_networks
        self.source_critic_networks = source_critic_networks
        self.env = gym.make(env_name)
        self.config = ENV_CONFIGS[env_name]

    def objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function for hyperparameter optimization."""
        # Get hyperparameters from trial
        learning_rate = trial.suggest_float(
            'learning_rate',
            self.config.optuna_config['learning_rate']['low'],
            self.config.optuna_config['learning_rate']['high'],
            log=self.config.optuna_config['learning_rate']['log']
        )
        hidden_size = trial.suggest_int(
            'hidden_size',
            self.config.optuna_config['hidden_size']['low'],
            self.config.optuna_config['hidden_size']['high'],
            step=self.config.optuna_config['hidden_size']['step']
        )
        gamma = trial.suggest_float(
            'gamma',
            self.config.optuna_config['gamma']['low'],
            self.config.optuna_config['gamma']['high'],
            step=self.config.optuna_config['gamma']['step']
        )

        # Create target networks
        target_actor_network = UnifiedPolicyNetwork(hidden_size)
        target_critic_network = UnifiedValueNetwork(hidden_size)

        if isinstance(self.env.action_space, gym.spaces.Discrete):
            output_dim = self.env.action_space.n
            is_continuous = False
        elif isinstance(self.env.action_space, gym.spaces.Box):
            output_dim = self.env.action_space.shape[0]
            is_continuous = True
        else:
            raise ValueError("Unsupported action space type.")

        # Create progressive model
        progressive_model = ProgressiveNetwork(
            source_actor_networks=self.source_actor_networks,
            source_critic_networks=self.source_critic_networks,
            target_actor_network=target_actor_network,
            target_critic_network=target_critic_network,
            hidden_dim=hidden_size,
            output_dim=output_dim,
            is_continuous=is_continuous
        )

        # Train and evaluate
        rewards_history, _ = train_progressive_network(
            progressive_model,
            self.env_name,
            num_episodes=self.config.max_episodes,
            learning_rate=learning_rate,
            gamma=gamma
        )

        return np.mean(rewards_history[-10:])

    def optimize(self, n_trials: int = 30) -> Dict[str, Any]:
        """Run Optuna optimization."""
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=n_trials)
        return study.best_params


def train(target_env_name, hidden_dim, source_actor_paths, source_critic_paths, n_trials=20):
    env = gym.make(target_env_name)
    is_continuous = isinstance(env.action_space, gym.spaces.Box)

    # Load source networks
    source_actor_networks = [load_model(path, UnifiedPolicyNetwork, hidden_dim) for path in source_actor_paths]
    source_critic_networks = [load_model(path, UnifiedValueNetwork, hidden_dim) for path in source_critic_paths]

    # Optimize hyperparameters
    optimizer = OptunaOptimizer(target_env_name, source_actor_networks, source_critic_networks)
    best_params = optimizer.optimize(n_trials=n_trials)
    print(f"Best hyperparameters for {target_env_name}: {best_params}")

    # Create target networks with best parameters
    target_actor_network = UnifiedPolicyNetwork(best_params['hidden_size'])
    target_critic_network = UnifiedValueNetwork(best_params['hidden_size'])

    # Set output dimension based on action space
    if is_continuous:
        output_dim = env.action_space.shape[0]
    else:
        output_dim = env.action_space.n

    # Create progressive model
    progressive_model = ProgressiveNetwork(
        source_actor_networks=source_actor_networks,
        source_critic_networks=source_critic_networks,
        target_actor_network=target_actor_network,
        target_critic_network=target_critic_network,
        hidden_dim=best_params['hidden_size'],
        output_dim=output_dim,
        is_continuous=is_continuous
    )

    # Train the progressive model
    rewards, training_time = train_progressive_network(
        progressive_model,
        target_env_name,
        learning_rate=best_params['learning_rate'],
        gamma=best_params['gamma']
    )

    # Save the trained model and hyperparameters
    full_params = {
        **best_params,
        'training_time': training_time,
        'final_average_reward': float(np.mean(rewards[-10:])),
        'is_continuous': is_continuous,
        'hidden_dim': best_params['hidden_size'],
        'output_dim': output_dim,
        'source_actor_paths': source_actor_paths,
        'source_critic_paths': source_critic_paths
    }
    save_model_and_params(progressive_model, full_params, target_env_name)

    print(f"Training completed for {target_env_name} in {training_time:.2f} seconds")
    print(f"Final average reward: {np.mean(rewards[-10:]):.2f}")


def main():
    # Configuration
    hidden_dim = 128
    target_env_names = ["MountainCarContinuous-v0", "CartPole-v1"]
    source_actor_paths = [
        [
            'models/CartPole-v1_actor.pth',
            'models/Acrobot-v1_actor.pth'
        ],
        [
            'models/Acrobot-v1_actor.pth',
            'models/MountainCarContinuous-v0_actor.pth'
        ]
    ]
    source_critic_paths = [
        [
            'models/CartPole-v1_critic.pth',
            'models/Acrobot-v1_critic.pth'
        ],
        [
            'models/Acrobot-v1_critic.pth',
            'models/MountainCarContinuous-v0_critic.pth'
        ]
    ]

    # Loop through each environment
    for target_env_name, actor_paths, critic_paths in zip(target_env_names, source_actor_paths, source_critic_paths):
        print(f"Training for {target_env_name}...")
        train(target_env_name, hidden_dim, actor_paths, critic_paths, n_trials=20)


if __name__ == "__main__":
    main()

