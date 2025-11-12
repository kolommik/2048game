"""
Training Script for DQN Agent on 2048
Main script to train and evaluate DQN agent
"""

import os
import argparse
from datetime import datetime
import numpy as np

# Game and agents
from game2048_engine import Game2048
from dqn.dqn_agent import DQNAgent

# For comparison with baseline agents
try:
    from agents_2048 import RandomAgent, MonotonicAgent

    BASELINE_AGENTS_AVAILABLE = True
except ImportError:
    BASELINE_AGENTS_AVAILABLE = False
    print("Warning: Baseline agents not available for comparison")


def train_dqn(
    # Network
    hidden_sizes=[256, 256],
    dropout=0.0,
    # Training
    learning_rate=1e-4,
    gamma=0.99,
    batch_size=64,
    buffer_capacity=10000,
    # Exploration
    epsilon_start=1.0,
    epsilon_end=0.1,
    epsilon_decay_steps=100000,
    # Schedule
    target_update_freq=1000,
    train_freq=4,
    min_replay_size=1000,
    # Training loop
    num_episodes=1000,
    max_steps_per_episode=10000,
    eval_freq=100,
    eval_games=20,
    save_freq=500,
    # Reward
    reward_type="simple",
    # Other
    device="cpu",
    save_dir="./models",
    run_name=None,
    resume_from=None,
):
    """
    Train DQN agent with specified hyperparameters

    Args:
        All hyperparameters for DQNAgent and training
        resume_from: Path to checkpoint to resume from (optional)
    """
    # Create run directory
    if run_name is None:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_dir = os.path.join(save_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"DQN Training Run: {run_name}")
    print(f"{'='*70}")
    print(f"Save directory: {run_dir}")
    print(f"Device: {device}")

    # Check if resuming from checkpoint
    if resume_from:
        print(f"\n🔄 Resuming from checkpoint: {resume_from}")

    print("\nHyperparameters:")
    print(f"  Network: hidden_sizes={hidden_sizes}, dropout={dropout}")
    print(f"  Learning: lr={learning_rate}, gamma={gamma}, batch_size={batch_size}")
    print(f"  Buffer: capacity={buffer_capacity}, min_size={min_replay_size}")
    print(
        f"  Exploration: ε={epsilon_start}→{epsilon_end} over {epsilon_decay_steps} steps"
    )
    print(
        f"  Training: {num_episodes} episodes, eval every {eval_freq}, save every {save_freq}"
    )
    print(f"  Reward: {reward_type}")
    print(f"{'='*70}\n")

    # Create agent
    agent = DQNAgent(
        name=f"DQN_{run_name}",
        hidden_sizes=hidden_sizes,
        dropout=dropout,
        learning_rate=learning_rate,
        gamma=gamma,
        batch_size=batch_size,
        buffer_capacity=buffer_capacity,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay_steps=epsilon_decay_steps,
        target_update_freq=target_update_freq,
        train_freq=train_freq,
        min_replay_size=min_replay_size,
        reward_type=reward_type,
        device=device,
    )

    # Load checkpoint if resuming
    if resume_from:
        try:
            agent.load(resume_from)
            print("✅ Checkpoint loaded successfully!")
            print(f"   Total steps: {agent.total_steps}")
            print(f"   Training steps: {agent.training_steps}")
            print(f"   Games played: {agent.games_played}")
            print(f"   Current epsilon: {agent.epsilon_scheduler.get_epsilon():.4f}")
            print(f"   Best score so far: {agent.best_score}")
            print(f"   Best tile so far: {agent.best_tile}\n")
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
            print("Starting from scratch instead...\n")

    # Save hyperparameters
    import json

    hyperparams = {
        "hidden_sizes": hidden_sizes,
        "dropout": dropout,
        "learning_rate": learning_rate,
        "gamma": gamma,
        "batch_size": batch_size,
        "buffer_capacity": buffer_capacity,
        "epsilon_start": epsilon_start,
        "epsilon_end": epsilon_end,
        "epsilon_decay_steps": epsilon_decay_steps,
        "target_update_freq": target_update_freq,
        "train_freq": train_freq,
        "min_replay_size": min_replay_size,
        "num_episodes": num_episodes,
        "reward_type": reward_type,
        "device": device,
        "resumed_from": resume_from,  # ← записываем откуда продолжили
    }

    with open(os.path.join(run_dir, "hyperparameters.json"), "w") as f:
        json.dump(hyperparams, f, indent=2)

    # Train
    agent.train(
        num_episodes=num_episodes,
        max_steps_per_episode=max_steps_per_episode,
        eval_freq=eval_freq,
        eval_games=eval_games,
        save_freq=save_freq,
        save_dir=run_dir,
        verbose=True,
    )

    # Final evaluation
    print(f"\n{'='*70}")
    print("Final Evaluation")
    print(f"{'='*70}")

    final_eval = agent.evaluate(num_games=100, verbose=False)

    print("Results over 100 games:")
    print(f"  Average Score: {final_eval['avg_score']:.1f}")
    print(f"  Best Score: {final_eval['best_score']}")
    print(f"  Average Max Tile: {final_eval['avg_max_tile']:.0f}")
    print(f"  Best Tile: {final_eval['best_tile']}")
    print(f"  Average Moves: {final_eval['avg_moves']:.1f}")

    # Save final evaluation
    with open(os.path.join(run_dir, "final_evaluation.json"), "w") as f:
        json.dump(final_eval, f, indent=2)

    # Compare with baseline if available
    if BASELINE_AGENTS_AVAILABLE:
        print(f"\n{'='*70}")
        print("Comparison with Baseline Agents")
        print(f"{'='*70}")

        baseline_agents = [
            RandomAgent(),
            MonotonicAgent(),
        ]

        comparison_results = {}

        for baseline in baseline_agents:
            print(f"\nEvaluating {baseline.name}...")
            baseline_scores = []
            baseline_tiles = []

            for i in range(100):
                result = baseline.play_game(game=Game2048(seed=i), verbose=False)
                baseline_scores.append(result["final_score"])
                baseline_tiles.append(result["max_tile"])

            comparison_results[baseline.name] = {
                "avg_score": float(np.mean(baseline_scores)),
                "best_score": int(np.max(baseline_scores)),
                "avg_max_tile": float(np.mean(baseline_tiles)),
                "best_tile": int(np.max(baseline_tiles)),
            }

            print(f"  Average Score: {np.mean(baseline_scores):.1f}")
            print(f"  Best Score: {np.max(baseline_scores)}")

        # Print comparison table
        print(f"\n{'='*70}")
        print("Performance Comparison")
        print(f"{'='*70}")
        print(f"{'Agent':<25} {'Avg Score':>12} {'Best Score':>12} {'Avg Tile':>12}")
        print(f"{'-'*70}")

        # DQN
        print(
            f"{agent.name:<25}"
            + f" {final_eval['avg_score']:>12.1f} {final_eval['best_score']:>12} {final_eval['avg_max_tile']:>12.0f}"
        )

        # Baselines
        for name, stats in comparison_results.items():
            print(
                f"{name:<25} {stats['avg_score']:>12.1f} {stats['best_score']:>12} {stats['avg_max_tile']:>12.0f}"
            )

        print(f"{'='*70}\n")

        # Save comparison
        comparison_results["DQN"] = final_eval
        with open(os.path.join(run_dir, "comparison.json"), "w") as f:
            json.dump(comparison_results, f, indent=2)

    return agent, run_dir


def evaluate_saved_model(model_path: str, num_games: int = 100):
    """
    Evaluate a saved DQN model

    Args:
        model_path: Path to saved model file
        num_games: Number of games to evaluate
    """
    print(f"\n{'='*70}")
    print(f"Evaluating saved model: {model_path}")
    print(f"{'='*70}\n")

    # Create agent (hyperparameters should be loaded from saved file)
    agent = DQNAgent(name="Loaded DQN Agent", device="cpu")

    # Load model
    agent.load(model_path)

    print("Model loaded successfully")
    print(f"Total training steps: {agent.total_steps}")
    print(f"Games played: {agent.games_played}")

    # Evaluate
    print(f"\nEvaluating over {num_games} games...")
    eval_results = agent.evaluate(num_games=num_games, verbose=False)

    print("\nResults:")
    print(f"  Average Score: {eval_results['avg_score']:.1f}")
    print(f"  Best Score: {eval_results['best_score']}")
    print(f"  Average Max Tile: {eval_results['avg_max_tile']:.0f}")
    print(f"  Best Tile: {eval_results['best_tile']}")
    print(f"  Average Moves: {eval_results['avg_moves']:.1f}")

    return agent, eval_results


def quick_test():
    """Quick test with minimal settings for debugging"""
    print("\n" + "=" * 70)
    print("QUICK TEST MODE")
    print("=" * 70 + "\n")

    agent = DQNAgent(
        name="Quick Test Agent",
        hidden_sizes=[64, 64],
        buffer_capacity=1000,
        epsilon_decay_steps=5000,
        min_replay_size=100,
        device="cpu",
    )

    agent.train(
        num_episodes=50,
        eval_freq=25,
        eval_games=5,
        save_freq=50,
        save_dir="./test_models",
        verbose=True,
    )

    print("\nQuick test complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN Agent for 2048")

    # Mode
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "eval", "test"],
        help="Mode: train, eval (existing model), or test (quick test)",
    )

    # Network architecture
    parser.add_argument(
        "--hidden-sizes",
        type=int,
        nargs="+",
        default=[256, 256],
        help="Hidden layer sizes",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.0, help="Dropout probability"
    )

    # Training hyperparameters
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--buffer-capacity", type=int, default=10000, help="Replay buffer capacity"
    )

    # Exploration
    parser.add_argument(
        "--epsilon-start", type=float, default=1.0, help="Initial epsilon"
    )
    parser.add_argument("--epsilon-end", type=float, default=0.1, help="Final epsilon")
    parser.add_argument(
        "--epsilon-decay-steps", type=int, default=100000, help="Epsilon decay steps"
    )

    # Training schedule
    parser.add_argument(
        "--episodes", type=int, default=1000, help="Number of training episodes"
    )
    parser.add_argument(
        "--target-update-freq",
        type=int,
        default=1000,
        help="Target network update frequency",
    )
    parser.add_argument("--train-freq", type=int, default=4, help="Training frequency")
    parser.add_argument(
        "--min-replay-size",
        type=int,
        default=1000,
        help="Minimum replay buffer size before training",
    )
    parser.add_argument(
        "--eval-freq", type=int, default=100, help="Evaluation frequency"
    )
    parser.add_argument(
        "--eval-games", type=int, default=20, help="Number of games for evaluation"
    )
    parser.add_argument(
        "--save-freq", type=int, default=500, help="Model save frequency"
    )

    # Reward
    parser.add_argument(
        "--reward-type",
        type=str,
        default="simple",
        choices=["simple", "shaped", "log"],
        help="Reward function type",
    )

    # Other
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use",
    )
    parser.add_argument(
        "--save-dir", type=str, default="./models", help="Directory to save models"
    )
    parser.add_argument("--run-name", type=str, default=None, help="Name for this run")
    parser.add_argument(
        "--model-path", type=str, default=None, help="Path to model for evaluation"
    )
    parser.add_argument(
        "--eval-games-count",
        type=int,
        default=100,
        help="Number of games for model evaluation",
    )

    # Resume training
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )

    args = parser.parse_args()

    if args.mode == "test":
        # Quick test
        quick_test()

    elif args.mode == "train":
        # Training
        agent, run_dir = train_dqn(
            hidden_sizes=args.hidden_sizes,
            dropout=args.dropout,
            learning_rate=args.lr,
            gamma=args.gamma,
            batch_size=args.batch_size,
            buffer_capacity=args.buffer_capacity,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            epsilon_decay_steps=args.epsilon_decay_steps,
            target_update_freq=args.target_update_freq,
            train_freq=args.train_freq,
            min_replay_size=args.min_replay_size,
            num_episodes=args.episodes,
            eval_freq=args.eval_freq,
            eval_games=args.eval_games,
            save_freq=args.save_freq,
            reward_type=args.reward_type,
            device=args.device,
            save_dir=args.save_dir,
            run_name=args.run_name,
            resume_from=args.resume_from,  # ← НОВЫЙ параметр
        )

        print(f"\nTraining complete! Models saved in: {run_dir}")

    elif args.mode == "eval":
        # Evaluation
        if args.model_path is None:
            print("Error: --model-path required for evaluation mode")
        else:
            agent, results = evaluate_saved_model(
                args.model_path, num_games=args.eval_games_count
            )

    else:
        print(f"Unknown mode: {args.mode}")
