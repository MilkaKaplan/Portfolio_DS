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

# Before saving models
os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)


class UnifiedPolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(UnifiedPolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x)
        out = self.network(x)
        if self.network[-1].out_features == 1:
            return torch.tanh(out)
        return torch.softmax(out, dim=-1)


class UnifiedValueNetwork(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(UnifiedValueNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x)
        return self.network(x)


class UnifiedActorCritic:
    def __init__(self, env_name, max_input_size=6, learning_rate=3e-4, hidden_size=128, gamma=0.99):
        self.env = gym.make(env_name)
        self.env_name = env_name
        self.gamma = gamma

        self.state_dim = self.env.observation_space.shape[0]
        self.padded_state_dim = max_input_size

        if isinstance(self.env.action_space, gym.spaces.Discrete):
            self.action_dim = self.env.action_space.n
            self.continuous = False
        else:
            self.action_dim = 1
            self.continuous = True

        self.actor = UnifiedPolicyNetwork(self.padded_state_dim, self.action_dim, hidden_size)
        self.critic = UnifiedValueNetwork(self.padded_state_dim, hidden_size)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)

        self.rewards_history = []
        self.training_time = 0

    def pad_state(self, state):
        padded = np.zeros(self.padded_state_dim)
        padded[:self.state_dim] = state
        return padded

    def select_action(self, state):
        state = self.pad_state(state)
        with torch.no_grad():
            if self.continuous:
                action = self.actor(state).numpy()
                return np.clip(action, -1, 1)
            else:
                probs = self.actor(state)
                action = torch.multinomial(probs, 1).item()
                return action

    def train(self, num_episodes=500):
        start_time = time()

        for episode in range(num_episodes):
            state, _ = self.env.reset()
            episode_rewards = 0
            values = []
            rewards = []
            log_probs = []

            for t in count():
                state_tensor = torch.FloatTensor(self.pad_state(state))
                value = self.critic(state_tensor)
                values.append(value)

                if self.continuous:
                    action_mean = self.actor(state_tensor)
                    action = torch.normal(action_mean, 0.1)
                    log_prob = -0.5 * ((action - action_mean) ** 2)
                    action = action.detach().numpy()
                else:
                    probs = self.actor(state_tensor)
                    action = torch.multinomial(probs, 1)
                    log_prob = torch.log(probs[action])
                    action = action.item()

                log_probs.append(log_prob)

                next_state, reward, done, truncated, _ = self.env.step(action)
                rewards.append(reward)
                episode_rewards += reward

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

            if len(self.rewards_history) >= 10:
                avg_reward = np.mean(self.rewards_history[-10:])
                if (self.env_name == "CartPole-v1" and avg_reward > 475) or \
                        (self.env_name == "Acrobot-v1" and avg_reward > -100) or \
                        (self.env_name == "MountainCarContinuous-v0" and avg_reward > 90):
                    break

        self.training_time = time() - start_time
        return np.mean(self.rewards_history[-10:])

    def save_model(self, actor_path, critic_path):
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    def load_model(self, actor_path, critic_path):
        self.actor.load_state_dict(torch.load(actor_path, weights_only=True))
        self.critic.load_state_dict(torch.load(critic_path, weights_only=True))


def objective(trial, env_name):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    hidden_size = trial.suggest_int("hidden_size", 32, 256)
    gamma = trial.suggest_float("gamma", 0.9, 0.99)

    agent = UnifiedActorCritic(
        env_name=env_name,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        gamma=gamma
    )
    if env_name == "MountainCarContinuous-v0":
        num_episodes = 1000
    else:
        num_episodes = 500
    return agent.train(num_episodes)


def optimize_and_train():
    envs = ["Acrobot-v1", "MountainCarContinuous-v0", "CartPole-v1"]
    results = {}

    for env_name in envs:
        print(f"\nOptimizing hyperparameters for {env_name}")
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, env_name), n_trials=15)

        best_params = study.best_params
        print(f"Best parameters for {env_name}:", best_params)

        agent = UnifiedActorCritic(
            env_name=env_name,
            learning_rate=best_params["learning_rate"],
            hidden_size=best_params["hidden_size"],
            gamma=best_params["gamma"]
        )

        if env_name == "MountainCarContinuous-v0":
            num_episodes = 1000
        else:
            num_episodes = 500
        rewards = agent.train(num_episodes)

        actor_path = f"models/{env_name}_actor.pth"
        critic_path = f"models/{env_name}_critic.pth"
        agent.save_model(actor_path, critic_path)

        results[env_name] = {
            "best_params": best_params,
            "training_time": agent.training_time,
            "episodes": len(agent.rewards_history),
            "final_avg_reward": np.mean(agent.rewards_history[-10:]),
            "actor_path": actor_path,
            "critic_path": critic_path
        }

        plt.figure()
        plt.plot(agent.rewards_history)
        plt.title(f"Learning Curve - {env_name}")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.savefig(f"plots/{env_name}_learning_curve.png")

    return results


def visualize_agent(agent, num_episodes=3):
    """Visualize trained agent's performance"""
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


def visualize_agent_with_saved_models(env_name, actor_path, critic_path, hidden_size=99, max_input_size=6, gamma=0.99, num_episodes=5):
    """Visualize trained agent using saved models."""
    env = gym.make(env_name, render_mode="human")

    # Initialize the agent with the same architecture as the saved models
    agent = UnifiedActorCritic(env_name, max_input_size=max_input_size, hidden_size=hidden_size, gamma=gamma)
    agent.load_model(actor_path, critic_path)

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


# if __name__ == "__main__":
#     env_name = "CartPole-v1"
#     print(f"\nOptimizing hyperparameters for {env_name}")
#
#     study = optuna.create_study(direction="maximize")
#     study.optimize(lambda trial: objective(trial, env_name), n_trials=20)
#
#     best_params = study.best_params
#     print(f"Best parameters: {best_params}")
#
#     agent = UnifiedActorCritic(
#         env_name=env_name,
#         learning_rate=best_params["learning_rate"],
#         hidden_size=best_params["hidden_size"],
#         gamma=best_params["gamma"]
#     )
#
#     agent.train()
#     print(f"\nTraining Time: {agent.training_time:.2f} seconds")
#     print(f"Episodes: {len(agent.rewards_history)}")
#     print(f"Final Average Reward: {np.mean(agent.rewards_history[-10:]):.2f}")
#
#     plt.figure()
#     plt.plot(agent.rewards_history)
#     plt.title("CartPole-v1 Learning Curve")
#     plt.xlabel("Episode")
#     plt.ylabel("Total Reward")
#     plt.savefig("cartpole_learning_curve.png")
#
#     print("\nVisualizing trained agent...")
#     visualize_agent(agent)

if __name__ == "__main__":
    # Uncomment for optimizing
    # results = optimize_and_train()
    #
    # for env_name, stats in results.items():
    #     print(f"\n{env_name} Statistics:")
    #     print(f"Best Hyperparameters: {stats['best_params']}")
    #     print(f"Training Time: {stats['training_time']:.2f} seconds")
    #     print(f"Episodes to Converge: {stats['episodes']}")
    #     print(f"Final Average Reward: {stats['final_avg_reward']:.2f}")

    # Visualize the best agent for each environment using saved models
    environments = [
        {"env_name": "Acrobot-v1", "hidden_size": 99},# fixed  size from saved model
        {"env_name": "CartPole-v1", "hidden_size": 228},
        {"env_name": "MountainCarContinuous-v0", "hidden_size": 138},
    ]

    for env in environments:
        env_name = env["env_name"]
        hidden_size = env["hidden_size"]

        actor_path = f"models/{env_name}_actor.pth"
        critic_path = f"models/{env_name}_critic.pth"

        print(f"\nVisualizing {env_name}")
        visualize_agent_with_saved_models(env_name, actor_path, critic_path, hidden_size=hidden_size)

    # # Visualize the best agent for each environment
    # for env_name, stats in results.items():
    #     print(f"\nVisualizing {env_name}")
    #     best_agent = UnifiedActorCritic(
    #         env_name=env_name,
    #         learning_rate=stats['best_params']["learning_rate"],
    #         hidden_size=stats['best_params']["hidden_size"],
    #         gamma=stats['best_params']["gamma"]
    #     )
    #     if env_name == "MountainCarContinuous-v0":
    #         num_episodes = 1000
    #     else:
    #         num_episodes = 500
    #     best_agent.train(num_episodes)  # Retrain with best parameters
    #     visualize_agent(best_agent)
