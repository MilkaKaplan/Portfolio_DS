import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
from time import time
import matplotlib.pyplot as plt
import os

from assignment_3.ActorCritic import UnifiedPolicyNetwork


class ProgressiveNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128, max_input_size=6):
        super(ProgressiveNetwork, self).__init__()
        self.input_size = input_size
        self.max_input_size = max_input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Target network layers
        self.target_fc1 = nn.Linear(max_input_size, hidden_size)
        self.target_fc2 = nn.Linear(hidden_size, hidden_size)
        self.target_output = nn.Linear(hidden_size * (len(self.source_networks) + 1), output_size)

        # Source networks will be loaded externally
        self.source_networks = nn.ModuleList()

    def add_source_network(self, source_network):
        """Add and freeze a source network"""
        for param in source_network.parameters():
            param.requires_grad = False
        self.source_networks.append(source_network)

    def pad_state(self, state):
        padded = torch.zeros(self.max_input_size)
        padded[:len(state)] = torch.FloatTensor(state)
        return padded

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = self.pad_state(x)

        # Get activations from source networks
        source_activations = []
        for source in self.source_networks:
            h1 = torch.relu(source.network[0](x))
            source_activations.append(h1)

        # Target network forward pass
        target_h1 = torch.relu(self.target_fc1(x))
        target_h2 = torch.relu(self.target_fc2(target_h1))

        # Concatenate all activations
        combined = torch.cat([target_h2] + source_activations, dim=-1)

        # Final output layer
        if self.output_size == 1:  # For continuous action space
            return torch.tanh(self.target_output(combined))
        return torch.softmax(self.target_output(combined), dim=-1)  # For discrete action space


class ProgressiveAgent:
    def __init__(self, target_env_name, source_env_names, models_dir="models"):
        self.target_env = gym.make(target_env_name)
        self.target_env_name = target_env_name
        self.state_dim = self.target_env.observation_space.shape[0]

        if isinstance(self.target_env.action_space, gym.spaces.Discrete):
            self.action_dim = self.target_env.action_space.n
            self.continuous = False
        else:
            self.action_dim = 1
            self.continuous = True

        # Load and add source networks
        self.source_networks = []
        hidden_size = None
        for env_name in source_env_names:
            # Initialize the source network manually
            source_network = UnifiedPolicyNetwork(
                input_size=6,
                output_size=2 if env_name != "MountainCarContinuous-v0" else 1,
                hidden_size=128
            )

            # Load only the weights for the source network
            state_dict = torch.load(f"{models_dir}/{env_name}_actor.pth", map_location=torch.device('cpu'))
            source_network.load_state_dict(state_dict)  # Load only the weights into the initialized network

            source_hidden_size = source_network.network[0].weight.shape[0]
            if hidden_size is None:
                hidden_size = source_hidden_size
            assert hidden_size == source_hidden_size, f"Hidden size mismatch: {hidden_size} vs {source_hidden_size}"

            self.source_networks.append(source_network)

        # Initialize the progressive network
        self.network = ProgressiveNetwork(self.state_dim, self.action_dim, hidden_size)

        # Add the source networks to the progressive network
        for source in self.source_networks:
            self.network.add_source_network(source)

        # Define the optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=3e-4)
        self.rewards_history = []


    def select_action(self, state):
        with torch.no_grad():
            if self.continuous:
                action = self.network(state).numpy()
                return np.clip(action, -1, 1)
            else:
                probs = self.network(state)
                action = torch.multinomial(probs, 1).item()
                return action

    def train(self, num_episodes=500):
        start_time = time()

        for episode in range(num_episodes):
            state, _ = self.target_env.reset()
            episode_rewards = 0
            done = False

            while not done:
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.target_env.step(action)
                episode_rewards += reward

                state_tensor = self.network.pad_state(state)
                action_probs = self.network(state_tensor)

                if self.continuous:
                    loss = -torch.mean(action_probs * reward)
                else:
                    loss = -torch.log(action_probs[action]) * reward

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if terminated or truncated:
                    done = True
                else:
                    state = next_state

            self.rewards_history.append(episode_rewards)

            if len(self.rewards_history) >= 10:
                avg_reward = np.mean(self.rewards_history[-10:])
                if (self.target_env_name == "CartPole-v1" and avg_reward > 475) or \
                        (self.target_env_name == "MountainCarContinuous-v0" and avg_reward > 90):
                    break

        training_time = time() - start_time
        return training_time, len(self.rewards_history), np.mean(self.rewards_history[-10:])


def run_experiments():
    sources1 = ["Acrobot-v1", "MountainCarContinuous-v0"]
    target1 = "CartPole-v1"

    sources2 = ["CartPole-v1", "Acrobot-v1"]
    target2 = "MountainCarContinuous-v0"

    experiments = [
        (sources1, target1),
        (sources2, target2)
    ]

    results = {}

    for sources, target in experiments:
        print(f"\nExperiment: {sources} -> {target}")

        progressive_agent = ProgressiveAgent(target, sources)
        prog_time, prog_episodes, prog_reward = progressive_agent.train()

        plt.figure(figsize=(10, 5))
        plt.plot(progressive_agent.rewards_history, label="Progressive")
        plt.title(f"Learning Curves for {target}")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.legend()
        plt.savefig(f"plots/progressive_{target.split('-')[0]}_comparison.png")

    return results


if __name__ == "__main__":
    results = run_experiments()

    for target, stats in results.items():
        print(f"\nResults for {target}:")
        print("Progressive Network:")
        print(f"  Training time: {stats['progressive']['time']:.2f} seconds")
        print(f"  Episodes to converge: {stats['progressive']['episodes']}")
        print(f"  Final average reward: {stats['progressive']['final_reward']:.2f}")
