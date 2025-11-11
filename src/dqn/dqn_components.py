"""
DQN Components for 2048
Core components for Deep Q-Network implementation
"""

import random
from collections import deque
from typing import Tuple, List
import numpy as np

# PyTorch imports - будут использоваться в вашем окружении
try:
    import torch
    import torch.nn as nn
except ImportError:
    print("Warning: PyTorch not installed. DQN components will not work.")


class QNetwork(nn.Module):
    """
    Q-Network for estimating action values

    Architecture:
    - Input: 16 features (4x4 board, log-transformed and normalized)
    - Hidden: 2-3 fully connected layers
    - Output: 4 Q-values (one for each direction)
    """

    def __init__(
        self,
        input_size: int = 16,
        hidden_sizes: List[int] = [256, 256],
        output_size: int = 4,
        dropout: float = 0.0,
    ):
        """
        Initialize Q-Network

        Args:
            input_size: Size of input state vector (16 for 4x4 board)
            hidden_sizes: List of hidden layer sizes
            output_size: Number of actions (4 directions)
            dropout: Dropout probability for regularization
        """
        super(QNetwork, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        # Build network layers
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, output_size))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch_size, input_size)

        Returns:
            Q-values tensor of shape (batch_size, output_size)
        """
        return self.network(x)


class ReplayBuffer:
    """
    Experience Replay Buffer for storing and sampling transitions

    Stores (state, action, reward, next_state, done) tuples
    """

    def __init__(self, capacity: int = 10000):
        """
        Initialize replay buffer

        Args:
            capacity: Maximum number of transitions to store
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """
        Add a transition to the buffer

        Args:
            state: Current state
            action: Action taken (0-3)
            reward: Reward received
            next_state: Next state after action
            done: Whether episode ended
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        """
        Sample a random batch of transitions

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Tuple of (states, actions, rewards, next_states, dones) as numpy arrays
        """
        batch = random.sample(self.buffer, batch_size)

        states = np.array([t[0] for t in batch], dtype=np.float32)
        actions = np.array([t[1] for t in batch], dtype=np.int64)
        rewards = np.array([t[2] for t in batch], dtype=np.float32)
        next_states = np.array([t[3] for t in batch], dtype=np.float32)
        dones = np.array([t[4] for t in batch], dtype=np.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        """Return current size of buffer"""
        return len(self.buffer)


class StatePreprocessor:
    """
    Preprocessor for converting game board to neural network input
    """

    @staticmethod
    def board_to_state(board: np.ndarray) -> np.ndarray:
        """
        Convert 4x4 board to normalized state vector

        Applies log2 transformation: 0→0, 2→1, 4→2, 8→3, ..., 2048→11
        Then normalizes by dividing by 11 (max value)

        Args:
            board: 4x4 numpy array with tile values

        Returns:
            Flattened and normalized state vector (16 values in [0, 1])
        """
        # Log2 transformation (0 stays 0, other values get log2)
        state = np.zeros_like(board, dtype=np.float32)
        mask = board > 0
        state[mask] = np.log2(board[mask])

        # Normalize by max possible value (2048 → 11)
        state = state / 11.0

        # Flatten to vector
        return state.flatten()

    @staticmethod
    def board_to_tensor(board: np.ndarray, device: str = "cpu") -> torch.Tensor:
        """
        Convert board to PyTorch tensor

        Args:
            board: 4x4 numpy array
            device: Device to place tensor on ('cpu' or 'cuda')

        Returns:
            PyTorch tensor of shape (1, 16)
        """
        state = StatePreprocessor.board_to_state(board)
        return torch.FloatTensor(state).unsqueeze(0).to(device)


class EpsilonScheduler:
    """
    Epsilon-greedy exploration scheduler
    """

    def __init__(
        self,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.1,
        epsilon_decay_steps: int = 100000,
    ):
        """
        Initialize epsilon scheduler

        Args:
            epsilon_start: Initial epsilon value
            epsilon_end: Final epsilon value
            epsilon_decay_steps: Number of steps to decay from start to end
        """
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.current_step = 0

    def get_epsilon(self) -> float:
        """Get current epsilon value"""
        if self.current_step >= self.epsilon_decay_steps:
            return self.epsilon_end

        # Linear decay
        decay_ratio = self.current_step / self.epsilon_decay_steps
        epsilon = (
            self.epsilon_start - (self.epsilon_start - self.epsilon_end) * decay_ratio
        )
        return epsilon

    def step(self):
        """Increment step counter"""
        self.current_step += 1


class TrainingMetrics:
    """
    Track and store training metrics
    """

    def __init__(self):
        """Initialize metrics tracker"""
        self.episodes = []
        self.scores = []
        self.max_tiles = []
        self.losses = []
        self.epsilons = []
        self.avg_q_values = []
        self.episode_lengths = []

    def add_episode(
        self,
        episode: int,
        score: int,
        max_tile: int,
        episode_length: int,
        epsilon: float,
    ):
        """Add episode metrics"""
        # Convert to Python native types to avoid JSON serialization issues
        self.episodes.append(int(episode))
        self.scores.append(int(score))
        self.max_tiles.append(int(max_tile))
        self.episode_lengths.append(int(episode_length))
        self.epsilons.append(float(epsilon))

    def add_training_step(self, loss: float, avg_q_value: float):
        """Add training step metrics"""
        # Convert to Python native types
        self.losses.append(float(loss))
        self.avg_q_values.append(float(avg_q_value))

    def get_recent_stats(self, n: int = 100) -> dict:
        """
        Get statistics over recent n episodes

        Args:
            n: Number of recent episodes to consider

        Returns:
            Dictionary with average statistics
        """
        if len(self.scores) < n:
            n = len(self.scores)

        if n == 0:
            return {
                "avg_score": 0.0,
                "avg_max_tile": 0.0,
                "avg_episode_length": 0.0,
            }

        recent_scores = self.scores[-n:]
        recent_tiles = self.max_tiles[-n:]
        recent_lengths = self.episode_lengths[-n:]

        return {
            "avg_score": float(np.mean(recent_scores)),
            "max_score": int(np.max(recent_scores)),
            "avg_max_tile": float(np.mean(recent_tiles)),
            "max_tile": int(np.max(recent_tiles)),
            "avg_episode_length": float(np.mean(recent_lengths)),
            "total_episodes": int(len(self.scores)),
        }

    def save_to_file(self, filepath: str):
        """Save metrics to file"""
        import json

        # Convert numpy types to Python types for JSON serialization
        def convert_to_python_type(obj):
            """Convert numpy types to Python native types recursively"""
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, list):
                return [convert_to_python_type(item) for item in obj]
            elif isinstance(obj, dict):
                return {
                    key: convert_to_python_type(value) for key, value in obj.items()
                }
            return obj

        # Convert all data
        data = {
            "episodes": [int(x) for x in self.episodes],
            "scores": [int(x) for x in self.scores],
            "max_tiles": [int(x) for x in self.max_tiles],
            "losses": [float(x) for x in self.losses],
            "epsilons": [float(x) for x in self.epsilons],
            "avg_q_values": [float(x) for x in self.avg_q_values],
            "episode_lengths": [int(x) for x in self.episode_lengths],
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def load_from_file(self, filepath: str):
        """Load metrics from file"""
        import json

        with open(filepath, "r") as f:
            data = json.load(f)

        self.episodes = data["episodes"]
        self.scores = data["scores"]
        self.max_tiles = data["max_tiles"]
        self.losses = data["losses"]
        self.epsilons = data["epsilons"]
        self.avg_q_values = data["avg_q_values"]
        self.episode_lengths = data["episode_lengths"]


class RewardShaper:
    """
    Reward shaping for 2048 game

    Provides different reward functions to experiment with
    """

    @staticmethod
    def simple_reward(points_gained: int, game_over: bool) -> float:
        """
        Simple reward: just the points from merging tiles

        Args:
            points_gained: Points from the move
            game_over: Whether game ended

        Returns:
            Reward value
        """
        reward = float(points_gained)

        # Small penalty for game over to encourage longer games
        if game_over:
            reward -= 10.0

        return reward

    @staticmethod
    def shaped_reward(
        points_gained: int,
        board_before: np.ndarray,
        board_after: np.ndarray,
        game_over: bool,
    ) -> float:
        """
        Shaped reward with additional heuristics

        Args:
            points_gained: Points from the move
            board_before: Board state before move
            board_after: Board state after move
            game_over: Whether game ended

        Returns:
            Reward value
        """
        # Base reward from points
        reward = float(points_gained)

        # Bonus for empty cells (encourage keeping board sparse)
        empty_before = np.sum(board_before == 0)
        empty_after = np.sum(board_after == 0)
        reward += (empty_after - empty_before) * 2.0

        # Bonus for max tile in corner
        max_tile = np.max(board_after)
        if board_after[0, 0] == max_tile:  # Top-left corner
            reward += 5.0

        # Penalty for game over
        if game_over:
            reward -= 50.0

        return reward

    @staticmethod
    def log_reward(points_gained: int, game_over: bool) -> float:
        """
        Logarithmic reward scaling

        Args:
            points_gained: Points from the move
            game_over: Whether game ended

        Returns:
            Reward value
        """
        if points_gained > 0:
            reward = float(np.log2(points_gained + 1))
        else:
            reward = 0.0

        if game_over:
            reward -= 5.0

        return reward


if __name__ == "__main__":
    print("Testing DQN Components\n")

    # Test QNetwork
    print("1. Testing QNetwork...")
    network = QNetwork(input_size=16, hidden_sizes=[256, 256], output_size=4)
    print(
        f"   Network created with {sum(p.numel() for p in network.parameters())} parameters"
    )

    # Test forward pass
    dummy_input = torch.randn(1, 16)
    output = network(dummy_input)
    print(f"   Output shape: {output.shape}")
    print(f"   Sample Q-values: {output.detach().numpy()[0]}")

    # Test ReplayBuffer
    print("\n2. Testing ReplayBuffer...")
    buffer = ReplayBuffer(capacity=1000)

    for i in range(10):
        state = np.random.rand(16)
        action = np.random.randint(0, 4)
        reward = np.random.rand()
        next_state = np.random.rand(16)
        done = np.random.rand() > 0.9
        buffer.push(state, action, reward, next_state, done)

    print(f"   Buffer size: {len(buffer)}")

    # Sample batch
    states, actions, rewards, next_states, dones = buffer.sample(5)
    print(f"   Sampled batch shapes: states={states.shape}, actions={actions.shape}")

    # Test StatePreprocessor
    print("\n3. Testing StatePreprocessor...")
    test_board = np.array(
        [[2, 4, 8, 16], [0, 0, 0, 0], [32, 64, 128, 256], [512, 1024, 2048, 0]]
    )

    state = StatePreprocessor.board_to_state(test_board)
    print(f"   State vector shape: {state.shape}")
    print(f"   State vector range: [{state.min():.3f}, {state.max():.3f}]")
    print(f"   Sample values: {state[:4]}")

    # Test EpsilonScheduler
    print("\n4. Testing EpsilonScheduler...")
    scheduler = EpsilonScheduler(
        epsilon_start=1.0, epsilon_end=0.1, epsilon_decay_steps=1000
    )

    epsilons = []
    for _ in range(1500):
        epsilons.append(scheduler.get_epsilon())
        scheduler.step()

    print(f"   Initial epsilon: {epsilons[0]:.3f}")
    print(f"   Epsilon at step 500: {epsilons[500]:.3f}")
    print(f"   Epsilon at step 1000: {epsilons[1000]:.3f}")
    print(f"   Final epsilon: {epsilons[-1]:.3f}")

    # Test TrainingMetrics
    print("\n5. Testing TrainingMetrics...")
    metrics = TrainingMetrics()

    for i in range(20):
        metrics.add_episode(
            episode=i,
            score=np.random.randint(500, 2000),
            max_tile=2 ** np.random.randint(7, 11),
            episode_length=np.random.randint(50, 200),
            epsilon=1.0 - i * 0.05,
        )

    stats = metrics.get_recent_stats(n=10)
    print("   Recent stats (last 10 episodes):")
    print(f"   - Avg score: {stats['avg_score']:.1f}")
    print(f"   - Avg max tile: {stats['avg_max_tile']:.0f}")
    print(f"   - Avg episode length: {stats['avg_episode_length']:.1f}")

    # Test RewardShaper
    print("\n6. Testing RewardShaper...")

    board_before = np.array([[2, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    board_after = np.array([[4, 0, 0, 0], [2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

    simple = RewardShaper.simple_reward(points_gained=4, game_over=False)
    shaped = RewardShaper.shaped_reward(4, board_before, board_after, False)
    log_r = RewardShaper.log_reward(points_gained=4, game_over=False)

    print(f"   Simple reward: {simple:.2f}")
    print(f"   Shaped reward: {shaped:.2f}")
    print(f"   Log reward: {log_r:.2f}")

    print("\nAll tests completed successfully!")
