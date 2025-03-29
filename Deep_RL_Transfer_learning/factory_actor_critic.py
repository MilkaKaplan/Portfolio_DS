import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from time import time
from itertools import count
import matplotlib.pyplot as plt
import optuna
import os
from dataclasses import dataclass
from typing import Dict, Any, Tuple

# Create directories for saving models and plots
os.makedirs("models2", exist_ok=True)
os.makedirs("plots2", exist_ok=True)


@dataclass
class EnvConfig:
    raw_state_dim: int  # Original state dimension
    raw_action_dim: int  # Original action dimension
    is_continuous: bool
    max_episodes: int
    convergence_reward: float
    optuna_config: Dict[str, Dict[str, Any]]


# Constants for unified input/output sizes
MAX_STATE_DIM = 6  # Maximum state dimension across all environments (Acrobot)
MAX_ACTION_DIM = 3  # Maximum action dimension across all environments (Acrobot)


class UnifiedPolicyNetwork(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(MAX_STATE_DIM, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, MAX_ACTION_DIM)
        )

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x)
        out = self.network(x)
        return torch.softmax(out, dim=-1)


class UnifiedValueNetwork(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(MAX_STATE_DIM, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x)
        return self.network(x)


class ActorCritic:
    def __init__(self, env_name: str, learning_rate: float = 3e-4, hidden_size: int = 128, gamma: float = 0.99):
        self.env = gym.make(env_name)
        self.env_name = env_name
        self.gamma = gamma

        # Get environment configuration
        self.config = ENV_CONFIGS[env_name]

        # Create networks
        self.actor = UnifiedPolicyNetwork(hidden_size)
        self.critic = UnifiedValueNetwork(hidden_size)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)

        self.rewards_history = []
        self.training_time = 0

    def pad_state(self, state):
        """Pad state with zeros to match MAX_STATE_DIM"""
        padded_state = np.zeros(MAX_STATE_DIM)
        padded_state[:self.config.raw_state_dim] = state
        return padded_state

    def process_action(self, action_probs):
        """Process network output to get valid action for the environment"""
        if self.config.is_continuous:
            # For MountainCar, use first output as continuous action
            return np.clip(action_probs[0].item() * 2 - 1, -1, 1)  # Scale to [-1, 1]
        else:
            # For discrete actions, only consider valid actions
            valid_probs = action_probs[:self.config.raw_action_dim]
            valid_probs = valid_probs / valid_probs.sum()  # Renormalize
            return torch.multinomial(valid_probs, 1).item()

    def select_action(self, state):
        with torch.no_grad():
            padded_state = self.pad_state(state)
            action_probs = self.actor(padded_state)
            return self.process_action(action_probs)

    def train(self, num_episodes: int = None):
        if num_episodes is None:
            num_episodes = self.config.max_episodes

        if self.env_name == "MountainCarContinuous-v0":
            num_goal_reached = 0

        start_time = time()

        for episode in range(num_episodes):
            state, _ = self.env.reset()
            episode_rewards = 0
            values = []
            rewards = []
            log_probs = []

            if self.env_name == "MountainCarContinuous-v0":
                max_left = max_right = state[0]

            for t in count():
                padded_state = self.pad_state(state)
                state_tensor = torch.FloatTensor(padded_state)

                value = self.critic(state_tensor)
                values.append(value)

                action_probs = self.actor(state_tensor)

                if self.config.is_continuous:
                    action_mean = action_probs[0] * 2 - 1  # Scale to [-1, 1]
                    action = torch.normal(action_mean, 0.1)
                    log_prob = -0.5 * ((action - action_mean) ** 2)
                    action = action.detach().numpy()
                    action = np.array([action])  # Wrap scalar action in an array

                else:
                    # For discrete actions
                    valid_probs = action_probs[:self.config.raw_action_dim]
                    valid_probs = valid_probs / valid_probs.sum()
                    action = torch.multinomial(valid_probs, 1)
                    log_prob = torch.log(valid_probs[action])
                    action = action.item()

                log_probs.append(log_prob)
                next_state, reward, done, truncated, _ = self.env.step(action)
                rewards.append(reward)
                episode_rewards += reward

                # Auxiliary rewards per height achieved by the car
                if self.env_name == "MountainCarContinuous-v0":
                    if num_goal_reached < 20:
                        if reward <= 0:
                            if next_state[0] < max_left:
                                reward = (2 + next_state[0]) ** 2
                                max_left = next_state[0]

                            if next_state[0] > max_right:
                                reward = (2 + next_state[0]) ** 2
                                max_right = next_state[0]
                        else:
                            num_goal_reached += 1
                            reward += 100
                            print(f'goal reached {num_goal_reached} times')

                if self.env_name == "Acrobot-v1":
                    reward = (reward - 50) / 500

                if self.env_name == "CartPole-v1":
                    reward = 1 - reward / 500

                if done or truncated:
                    break

                state = next_state

            returns = []
            R = 0
            for r in reversed(rewards):
                R = r + self.gamma * R
                returns.insert(0, R)
            returns = torch.tensor(returns)

            values = torch.cat(values).squeeze()
            advantages = returns - values.detach()

            actor_loss = -(torch.stack(log_probs) * advantages).mean()
            critic_loss = nn.MSELoss()(values, returns)

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            self.rewards_history.append(episode_rewards)

            # Check for convergence
            if len(self.rewards_history) >= 10:
                avg_reward = np.mean(self.rewards_history[-10:])
                if avg_reward > self.config.convergence_reward:
                    break

        self.training_time = time() - start_time
        return np.mean(self.rewards_history[-10:])

    def save_model(self):
        torch.save(self.actor.state_dict(), f"models2/{self.env_name}_actor.pth")
        torch.save(self.critic.state_dict(), f"models2/{self.env_name}_critic.pth")

    def load_model(self):
        self.actor.load_state_dict(torch.load(f"models2/{self.env_name}_actor.pth"))
        self.critic.load_state_dict(torch.load(f"models2/{self.env_name}_critic.pth"))


# Environment-specific configurations
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


def optimize_agent(env_name: str, n_trials: int = 20):
    config = ENV_CONFIGS[env_name]

    def objective(trial):
        # Get environment-specific parameter ranges
        lr_config = config.optuna_config["learning_rate"]
        hs_config = config.optuna_config["hidden_size"]
        gamma_config = config.optuna_config["gamma"]

        # Suggest parameters within environment-specific ranges
        learning_rate = trial.suggest_float(
            "learning_rate",
            lr_config["low"],
            lr_config["high"],
            log=lr_config.get("log", False)
        )
        hidden_size = trial.suggest_int(
            "hidden_size",
            hs_config["low"],
            hs_config["high"],
            step=hs_config.get("step", 1)
        )
        gamma = trial.suggest_float(
            "gamma",
            gamma_config["low"],
            gamma_config["high"],
            step=gamma_config.get("step", None)
        )

        agent = ActorCritic(
            env_name=env_name,
            learning_rate=learning_rate,
            hidden_size=hidden_size,
            gamma=gamma
        )
        return agent.train()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


def visualize_agent(agent: ActorCritic, num_episodes: int = 3):
    env = gym.make(agent.env_name, render_mode="human")

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        print(f"Episode {episode + 1} Reward: {episode_reward}")

    env.close()


if __name__ == "__main__":
    # Train and evaluate on all environments
    environments = ["MountainCarContinuous-v0", "Acrobot-v1", "CartPole-v1"]
    results = {}

    for env_name in environments:
        print(f"\nOptimizing {env_name}...")
        best_params = optimize_agent(env_name)
        print("Best parameters:", best_params)

        agent = ActorCritic(env_name, **best_params)
        agent.train()
        agent.save_model()

        plt.figure()
        plt.plot(agent.rewards_history)
        plt.title(f"{env_name} Learning Curve")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.savefig(f"plots2/{env_name}_learning_curve.png")

        results[env_name] = {
            "training_time": agent.training_time,
            "episodes": len(agent.rewards_history),
            "final_reward": np.mean(agent.rewards_history[-10:]),
            "best_params": best_params
        }

        print(f"\n{env_name} Statistics:")
        print(f"Training Time: {agent.training_time:.2f} seconds")
        print(f"Episodes: {len(agent.rewards_history)}")
        print(f"Final Average Reward: {np.mean(agent.rewards_history[-10:]):.2f}")

        print("\nVisualizing trained agent...")
        visualize_agent(agent)

    # Print final comparative statistics
    print("\nFinal Statistics Summary:")
    for env_name, stats in results.items():
        print(f"\n{env_name}:")
        print(f"Training Time: {stats['training_time']:.2f} seconds")
        print(f"Episodes to Converge: {stats['episodes']}")
        print(f"Final Average Reward: {stats['final_reward']:.2f}")