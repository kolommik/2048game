"""
DQN Agent Implementation for 2048
Add this class to agents_2048.py
"""

import os
import time
import random
from typing import Dict, List, Optional
import numpy as np
from game2048_engine import Game2048, Direction

# Import DQN components
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from dqn.dqn_components import (
        QNetwork,
        ReplayBuffer,
        StatePreprocessor,
        EpsilonScheduler,
        TrainingMetrics,
        RewardShaper,
    )

    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("Warning: PyTorch or DQN components not available. DQNAgent will not work.")


class DQNAgent:
    """
    Deep Q-Network Agent for 2048

    Implements DQN algorithm with:
    - Experience replay
    - Target network
    - Epsilon-greedy exploration
    """

    def __init__(
        self,
        name: str = "DQN Agent",
        # Network architecture
        hidden_sizes: List[int] = [256, 256],
        dropout: float = 0.0,
        # Training hyperparameters
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        batch_size: int = 64,
        buffer_capacity: int = 10000,
        # Exploration
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.1,
        epsilon_decay_steps: int = 100000,
        # Training schedule
        target_update_freq: int = 1000,
        train_freq: int = 4,
        min_replay_size: int = 1000,
        # Reward function
        reward_type: str = "simple",  # 'simple', 'shaped'
        # Device
        device: str = "cpu",
    ):
        """
        Initialize DQN Agent

        Args:
            name: Agent name
            hidden_sizes: List of hidden layer sizes
            dropout: Dropout probability
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            batch_size: Mini-batch size for training
            buffer_capacity: Replay buffer capacity
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay_steps: Steps to decay epsilon
            target_update_freq: Frequency of target network updates
            train_freq: Frequency of training steps
            min_replay_size: Minimum buffer size before training
            reward_type: Type of reward function
            device: 'cpu' or 'cuda'
        """
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch is required for DQNAgent")

        self.name = name
        self.device = torch.device(device)

        # Statistics
        self.games_played = 0
        self.total_score = 0
        self.best_score = 0
        self.best_tile = 0

        # Hyperparameters
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.train_freq = train_freq
        self.min_replay_size = min_replay_size
        self.reward_type = reward_type

        # Networks
        self.q_network = QNetwork(
            input_size=16, hidden_sizes=hidden_sizes, output_size=4, dropout=dropout
        ).to(self.device)

        self.target_network = QNetwork(
            input_size=16, hidden_sizes=hidden_sizes, output_size=4, dropout=dropout
        ).to(self.device)

        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

        # Exploration
        self.epsilon_scheduler = EpsilonScheduler(
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay_steps=epsilon_decay_steps,
        )

        # Training metrics
        self.metrics = TrainingMetrics()
        self.total_steps = 0
        self.training_steps = 0

        # Preprocessor
        self.preprocessor = StatePreprocessor()

    def choose_action(
        self, game: Game2048, epsilon: Optional[float] = None, training: bool = True
    ) -> Optional[Direction]:
        """
        Choose action using epsilon-greedy policy

        Args:
            game: Current game state
            epsilon: Exploration rate (uses scheduler if None)
            training: Whether in training mode

        Returns:
            Direction to move
        """
        available_moves = game.get_available_moves()

        if not available_moves:
            return None

        # Get epsilon
        if epsilon is None:
            epsilon = self.epsilon_scheduler.get_epsilon() if training else 0.0

        # Epsilon-greedy action selection
        if training and random.random() < epsilon:
            # Explore: random action
            return random.choice(available_moves)
        else:
            # Exploit: best action according to Q-network
            return self._get_best_action(game, available_moves)

    def _get_best_action(
        self, game: Game2048, available_moves: List[Direction]
    ) -> Direction:
        """
        Get best action according to Q-network

        Args:
            game: Current game state
            available_moves: List of valid moves

        Returns:
            Best direction
        """
        state = self.preprocessor.board_to_tensor(game.board, device=self.device)

        with torch.no_grad():
            q_values = self.q_network(state).cpu().numpy()[0]

        # Mask invalid actions
        action_mask = np.full(4, -np.inf)
        for move in available_moves:
            action_mask[move.value] = q_values[move.value]

        # Get best valid action
        best_action_idx = np.argmax(action_mask)
        return Direction(best_action_idx)

    def _compute_reward(
        self,
        points_gained: int,
        board_before: np.ndarray,
        board_after: np.ndarray,
        game_over: bool,
    ) -> float:
        """Compute reward based on configured reward type"""
        if self.reward_type == "simple":
            return RewardShaper.simple_reward(points_gained, game_over)
        elif self.reward_type == "shaped":
            return RewardShaper.shaped_reward(
                points_gained, board_before, board_after, game_over
            )
        else:
            return float(points_gained)

    def train_step(self) -> Optional[Dict]:
        """
        Perform one training step on a mini-batch

        Returns:
            Dictionary with loss and Q-value stats, or None if not enough data
        """
        if len(self.replay_buffer) < self.min_replay_size:
            return None

        # Sample mini-batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Compute current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        # Compute target Q values (Double DQN)
        with torch.no_grad():
            # --------
            # Vanila DQN
            # next_q_values = self.target_network(next_states).max(1)[0]

            # Double DQN: select actions with Q-network, evaluate with target network
            # This reduces overestimation bias
            best_next_actions = self.q_network(next_states).argmax(1)
            next_q_values = (
                self.target_network(next_states)
                .gather(1, best_next_actions.unsqueeze(1))
                .squeeze()
            )
            # ---------
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(
            self.q_network.parameters(), max_norm=1.0
        )  # 10.0->1.0 more stable
        self.optimizer.step()

        self.training_steps += 1

        # Update target network
        if self.training_steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return {
            "loss": loss.item(),
            "avg_q_value": current_q_values.mean().item(),
        }

    def train(
        self,
        num_episodes: int = 1000,
        max_steps_per_episode: int = 10000,
        eval_freq: int = 100,
        eval_games: int = 10,
        save_freq: int = 500,
        save_dir: str = "./models",
        verbose: bool = True,
    ):
        """
        Train the DQN agent

        Args:
            num_episodes: Number of episodes to train
            max_steps_per_episode: Maximum steps per episode
            eval_freq: Frequency of evaluation
            eval_games: Number of games for evaluation
            save_freq: Frequency of model saving
            save_dir: Directory to save models
            verbose: Print training progress
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Training {self.name}")
            print(f"{'='*60}")
            print(f"Episodes: {num_episodes}")
            print(
                f"Replay buffer: {len(self.replay_buffer)}/{self.replay_buffer.capacity}"
            )
            print(f"Device: {self.device}")
            print(f"{'='*60}\n")

        # Create save directory
        os.makedirs(save_dir, exist_ok=True)

        start_time = time.time()

        for episode in range(num_episodes):
            # Play one episode
            game = Game2048()
            episode_reward = 0
            episode_steps = 0

            while not game.game_over and episode_steps < max_steps_per_episode:
                # Get current state
                state = self.preprocessor.board_to_state(game.board)
                board_before = game.board.copy()

                # Choose action
                action = self.choose_action(game, training=True)

                if action is None:
                    break

                # Execute action
                valid, points = game.move(action)

                if not valid:
                    continue

                # Compute reward
                reward = self._compute_reward(
                    points, board_before, game.board, game.game_over
                )
                episode_reward += reward

                # Get next state
                next_state = self.preprocessor.board_to_state(game.board)

                # Store transition
                self.replay_buffer.push(
                    state, action.value, reward, next_state, game.game_over
                )

                # Training step
                if self.total_steps % self.train_freq == 0:
                    train_stats = self.train_step()
                    if train_stats:
                        self.metrics.add_training_step(
                            train_stats["loss"], train_stats["avg_q_value"]
                        )

                self.total_steps += 1
                episode_steps += 1
                self.epsilon_scheduler.step()

            # Episode finished
            self.update_stats(game.score, game.get_max_tile())
            self.metrics.add_episode(
                episode=episode,
                score=game.score,
                max_tile=game.get_max_tile(),
                episode_length=episode_steps,
                epsilon=self.epsilon_scheduler.get_epsilon(),
            )

            # Logging
            if verbose and (episode + 1) % 10 == 0:
                stats = self.metrics.get_recent_stats(n=10)
                elapsed = time.time() - start_time
                print(
                    f"Episode {episode+1}/{num_episodes} | "
                    f"Avg Score: {stats['avg_score']:.1f} | "
                    f"Avg Tile: {stats['avg_max_tile']:.0f} | "
                    f"Epsilon: {self.epsilon_scheduler.get_epsilon():.3f} | "
                    f"Steps: {self.total_steps} | "
                    f"Time: {elapsed:.1f}s"
                )

            # Evaluation
            if (episode + 1) % eval_freq == 0:
                eval_stats = self.evaluate(num_games=eval_games, verbose=False)
                if verbose:
                    print(f"\n--- Evaluation (Episode {episode+1}) ---")
                    print(f"Avg Score: {eval_stats['avg_score']:.1f}")
                    print(f"Best Score: {eval_stats['best_score']}")
                    print(f"Avg Max Tile: {eval_stats['avg_max_tile']:.0f}")
                    print(f"Best Tile: {eval_stats['best_tile']}")
                    print(f"{'='*40}\n")
                self.metrics.add_evaluation(episode, eval_stats)

            # Save model
            if (episode + 1) % save_freq == 0:
                self.save(os.path.join(save_dir, f"dqn_episode_{episode+1}.pt"))
                self.metrics.save_to_file(
                    os.path.join(save_dir, f"metrics_episode_{episode+1}.json")
                )

        # Final save
        self.save(os.path.join(save_dir, "dqn_final.pt"))
        self.metrics.save_to_file(os.path.join(save_dir, "metrics_final.json"))

        if verbose:
            total_time = time.time() - start_time
            print(f"\n{'='*60}")
            print("Training Complete!")
            print(
                f"Total time: {total_time:.1f}s ({total_time/num_episodes:.2f}s per episode)"
            )
            print(f"Total steps: {self.total_steps}")
            print(f"Training steps: {self.training_steps}")
            print(f"{'='*60}\n")

    def evaluate(self, num_games: int = 10, verbose: bool = False) -> Dict:
        """
        Evaluate agent performance

        Args:
            num_games: Number of games to play
            verbose: Print game progress

        Returns:
            Statistics dictionary
        """
        scores = []
        max_tiles = []
        moves_counts = []

        for i in range(num_games):
            result = self.play_game(game=None, max_moves=10000, verbose=False)
            scores.append(result["final_score"])
            max_tiles.append(result["max_tile"])
            moves_counts.append(result["moves_count"])

            if verbose:
                print(
                    f"Game {i+1}: Score={result['final_score']}, Tile={result['max_tile']}"
                )

        return {
            "avg_score": float(np.mean(scores)),
            "best_score": int(np.max(scores)),
            "avg_max_tile": float(np.mean(max_tiles)),
            "best_tile": int(np.max(max_tiles)),
            "avg_moves": float(np.mean(moves_counts)),
        }

    def play_game(
        self,
        game: Optional[Game2048] = None,
        max_moves: int = 10000,
        verbose: bool = False,
    ) -> Dict:
        """
        Play a complete game (inference mode)

        Args:
            game: Game instance (creates new if None)
            max_moves: Maximum moves before stopping
            verbose: Print game progress

        Returns:
            Game statistics
        """
        if game is None:
            game = Game2048()
        else:
            game.reset()

        move_history = []

        while not game.game_over and game.moves_count < max_moves:
            action = self.choose_action(game, epsilon=0.0, training=False)

            if action is None:
                break

            valid, _ = game.move(action)

            if valid:
                move_history.append(
                    {
                        "move": action.name,
                        "score": game.score,
                        "max_tile": game.get_max_tile(),
                    }
                )

                if verbose and game.moves_count % 10 == 0:
                    print(
                        f"Move {game.moves_count}: Score={game.score}, "
                        f"Max={game.get_max_tile()}"
                    )

        self.update_stats(game.score, game.get_max_tile())

        return {
            "final_score": game.score,
            "moves_count": game.moves_count,
            "max_tile": game.get_max_tile(),
            "won": game.won,
            "history": move_history,
        }

    def save(self, filepath: str):
        """Save model to file"""
        torch.save(
            {
                "q_network_state_dict": self.q_network.state_dict(),
                "target_network_state_dict": self.target_network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "total_steps": self.total_steps,
                "training_steps": self.training_steps,
                "epsilon_scheduler_step": self.epsilon_scheduler.current_step,
                "games_played": self.games_played,
                "total_score": self.total_score,
                "best_score": self.best_score,
                "best_tile": self.best_tile,
            },
            filepath,
        )

    def load(self, filepath: str):
        """Load model from file"""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
        self.target_network.load_state_dict(checkpoint["target_network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.total_steps = checkpoint["total_steps"]
        self.training_steps = checkpoint["training_steps"]
        self.epsilon_scheduler.current_step = checkpoint["epsilon_scheduler_step"]
        self.games_played = checkpoint["games_played"]
        self.total_score = checkpoint["total_score"]
        self.best_score = checkpoint["best_score"]
        self.best_tile = checkpoint["best_tile"]

    def reset_stats(self):
        """Reset agent statistics"""
        self.games_played = 0
        self.total_score = 0
        self.best_score = 0
        self.best_tile = 0

    def update_stats(self, score: int, max_tile: int):
        """Update agent statistics after a game"""
        self.games_played += 1
        self.total_score += score
        self.best_score = max(self.best_score, score)
        self.best_tile = max(self.best_tile, max_tile)

    def get_stats(self) -> Dict:
        """Get agent statistics"""
        return {
            "name": self.name,
            "games_played": self.games_played,
            "average_score": self.total_score / max(1, self.games_played),
            "best_score": self.best_score,
            "best_tile": self.best_tile,
            "total_steps": self.total_steps,
            "training_steps": self.training_steps,
        }


if __name__ == "__main__":
    print("Testing DQN Agent\n")

    # Create agent with small network for testing
    agent = DQNAgent(
        name="Test DQN Agent",
        hidden_sizes=[64, 64],
        buffer_capacity=1000,
        epsilon_decay_steps=5000,
        device="cpu",
    )

    print(f"Agent created: {agent.name}")
    print(
        f"Q-Network parameters: {sum(p.numel() for p in agent.q_network.parameters())}"
    )

    # Test single game
    print("\nPlaying single test game...")
    result = agent.play_game(verbose=False)
    print(f"Score: {result['final_score']}, Max Tile: {result['max_tile']}")

    # Quick training test (just a few episodes)
    print("\nQuick training test (10 episodes)...")
    agent.train(
        num_episodes=10,
        eval_freq=5,
        eval_games=3,
        save_freq=10,
        save_dir="./test_models",
        verbose=True,
    )

    print("\nDQN Agent test complete!")
