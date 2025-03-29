import ActorCritic as ac
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import time


def fine_tune_model(source_env, target_env, source_actor_weights, source_critic_weights, source_hidden_size,
                    target_hidden_size, max_input_size=6, learning_rate=3e-4, gamma=0.99, num_episodes=1000):
    """
    Fine-tune model between environments by reinitializing only the output layer.
    """
    # Initialize source agent
    source_agent = ac.UnifiedActorCritic(source_env, max_input_size, learning_rate, source_hidden_size, gamma)
    source_agent.load_model(source_actor_weights, source_critic_weights)

    # Initialize target agent
    target_agent = ac.UnifiedActorCritic(target_env, max_input_size, learning_rate, target_hidden_size, gamma)

    # Transfer weights for all layers except the output layer
    with torch.no_grad():
        # Transfer actor weights (only input and hidden layers)
        for source_layer, target_layer in zip(source_agent.actor.network[:-1], target_agent.actor.network[:-1]):
            if isinstance(source_layer, nn.Linear):
                min_in_dim = min(source_layer.weight.size(1), target_layer.weight.size(1))
                min_out_dim = min(source_layer.weight.size(0), target_layer.weight.size(0))
                target_layer.weight[:min_out_dim, :min_in_dim] = source_layer.weight[:min_out_dim, :min_in_dim]
                target_layer.bias[:min_out_dim] = source_layer.bias[:min_out_dim]

        # Transfer critic weights (only input and hidden layers)
        for source_layer, target_layer in zip(source_agent.critic.network[:-1], target_agent.critic.network[:-1]):
            if isinstance(source_layer, nn.Linear):
                min_in_dim = min(source_layer.weight.size(1), target_layer.weight.size(1))
                min_out_dim = min(source_layer.weight.size(0), target_layer.weight.size(0))
                target_layer.weight[:min_out_dim, :min_in_dim] = source_layer.weight[:min_out_dim, :min_in_dim]
                target_layer.bias[:min_out_dim] = source_layer.bias[:min_out_dim]

    # Reinitialize output layers
    nn.init.xavier_uniform_(target_agent.actor.network[-1].weight)
    nn.init.zeros_(target_agent.actor.network[-1].bias)
    nn.init.xavier_uniform_(target_agent.critic.network[-1].weight)
    nn.init.zeros_(target_agent.critic.network[-1].bias)

    # Record start time for statistics
    start_time = time.time()

    # Train the target agent
    final_avg_reward = target_agent.train(num_episodes)

    # Calculate training time
    training_time = time.time() - start_time

    # Plot learning curve using the rewards history from the agent
    plt.figure(figsize=(10, 6))
    plt.plot(target_agent.rewards_history)
    plt.title(f"Learning Curve - {source_env} to {target_env}")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.savefig(f"plots/{source_env}_to_{target_env}_learning_curve.png")
    plt.close()

    return target_agent, target_agent.rewards_history, training_time


if __name__ == '__main__':
    env_pairs = [
        {
            "source_env": "Acrobot-v1",
            "target_env": "CartPole-v1",
            "source_hidden_size": 99,
            "target_hidden_size": 228
        },
        {
            "source_env": "CartPole-v1",
            "target_env": "MountainCarContinuous-v0",
            "source_hidden_size": 228,
            "target_hidden_size": 138
        }
    ]

    results = {}

    for env_pair in env_pairs:
        source_env = env_pair["source_env"]
        target_env = env_pair["target_env"]
        source_hidden_size = env_pair["source_hidden_size"]
        target_hidden_size = env_pair["target_hidden_size"]

        source_actor_path = f"models/{source_env}_actor.pth"
        source_critic_path = f"models/{source_env}_critic.pth"

        print(f"\nFine-tuning from {source_env} to {target_env}")
        target_agent, rewards_history, training_time = fine_tune_model(
            source_env=source_env,
            target_env=target_env,
            source_actor_weights=source_actor_path,
            source_critic_weights=source_critic_path,
            source_hidden_size=source_hidden_size,
            target_hidden_size=target_hidden_size
        )

        # Save statistics
        results[f"{source_env}->{target_env}"] = {
            "training_time": training_time,
            "num_episodes": len(target_agent.rewards_history),
            "final_avg_reward": np.mean(target_agent.rewards_history[-10:])
        }

        # Save fine-tuned model
        target_actor_path = f"models/{target_env}_fine_tuned_actor.pth"
        target_critic_path = f"models/{target_env}_fine_tuned_critic.pth"
        target_agent.save_model(target_actor_path, target_critic_path)

        print(f"Fine-tuning completed for {source_env} → {target_env}")
        print(f"Training Time: {training_time:.2f} seconds")
        print(f"Number of Episodes: {len(target_agent.rewards_history)}")
        print(f"Final Average Reward: {np.mean(target_agent.rewards_history[-10:]):.2f}")

        # Visualize the fine-tuned agent
        print("\nVisualizing fine-tuned agent performance:")
        ac.visualize_agent_with_saved_models(target_env, target_actor_path, target_critic_path,
                                             hidden_size=target_hidden_size)

    # Print comparative results
    print("\nSummary of Results:")
    for pair, stats in results.items():
        print(f"\n{pair}:")
        print(f"Training Time: {stats['training_time']:.2f} seconds")
        print(f"Number of Episodes: {stats['num_episodes']}")
        print(f"Final Average Reward: {stats['final_avg_reward']:.2f}")