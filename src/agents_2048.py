"""
2048 Game Agents
Different agent implementations for playing 2048
"""

import time
import random
from typing import Dict, List, Optional
from abc import ABC, abstractmethod
import numpy as np
from game2048_engine import Game2048, Direction


class Agent2048(ABC):
    """Base class for all 2048 agents"""

    def __init__(self, name: str = "Agent"):
        self.name = name
        self.games_played = 0
        self.total_score = 0
        self.best_score = 0
        self.best_tile = 0

    @abstractmethod
    def choose_action(self, game: Game2048) -> Optional[Direction]:
        """
        Choose an action given the current game state

        Args:
            game: Current game instance

        Returns:
            Direction to move or None if no move available
        """

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
        }

    def play_game(
        self,
        game: Optional[Game2048] = None,
        max_moves: int = 10000,
        verbose: bool = False,
    ) -> Dict:
        """
        Play a complete game

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
            action = self.choose_action(game)

            if action is None:
                break

            valid, _ = game.move(action)  # _ for points

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
                        f"Move {game.moves_count}: Score={game.score}, Max={game.get_max_tile()}"
                    )

        self.update_stats(game.score, game.get_max_tile())

        return {
            "final_score": game.score,
            "moves_count": game.moves_count,
            "max_tile": game.get_max_tile(),
            "won": game.won,
            "history": move_history,
        }


class RandomAgent(Agent2048):
    """Agent that plays random valid moves"""

    def __init__(self):
        super().__init__("Random Agent")

    def choose_action(self, game: Game2048) -> Optional[Direction]:
        """Choose a random valid move"""
        available_moves = game.get_available_moves()

        if not available_moves:
            return None

        return random.choice(available_moves)


class GreedyAgent(Agent2048):
    """Agent that chooses move with highest immediate reward"""

    def __init__(self):
        super().__init__("Greedy Agent")

    def choose_action(self, game: Game2048) -> Optional[Direction]:
        """Choose move that gives maximum immediate points"""
        available_moves = game.get_available_moves()

        if not available_moves:
            return None

        best_move = None
        best_points = -1

        for move in available_moves:
            # Simulate the move
            test_game = game.copy()
            valid, points = test_game.move(move)

            if valid and points > best_points:
                best_points = points
                best_move = move

        # If no move gives points, choose randomly
        if best_move is None:
            best_move = random.choice(available_moves)

        return best_move


class CornerAgent(Agent2048):
    """Agent that tries to keep the maximum tile in a corner"""

    def __init__(self, preferred_corner: str = "top-left"):
        super().__init__(f"Corner Agent ({preferred_corner})")
        self.preferred_corner = preferred_corner

        # Define move priorities for each corner strategy
        self.move_priorities = {
            "top-left": [Direction.UP, Direction.LEFT, Direction.DOWN, Direction.RIGHT],
            "top-right": [
                Direction.UP,
                Direction.RIGHT,
                Direction.DOWN,
                Direction.LEFT,
            ],
            "bottom-left": [
                Direction.DOWN,
                Direction.LEFT,
                Direction.UP,
                Direction.RIGHT,
            ],
            "bottom-right": [
                Direction.DOWN,
                Direction.RIGHT,
                Direction.UP,
                Direction.LEFT,
            ],
        }

    def choose_action(self, game: Game2048) -> Optional[Direction]:
        """Choose move based on corner strategy"""
        available_moves = game.get_available_moves()

        if not available_moves:
            return None

        # Try moves in priority order
        priorities = self.move_priorities[self.preferred_corner]

        for move in priorities:
            if move in available_moves:
                return move

        # Fallback to first available move
        return available_moves[0]


class MonotonicAgent(Agent2048):
    """Agent that tries to maintain monotonic rows and columns"""

    def __init__(self):
        super().__init__("Monotonic Agent")

    def _calculate_monotonicity(self, board: np.ndarray) -> float:
        """
        Calculate monotonicity score (higher is better)
        Rewards boards where values decrease/increase consistently
        """
        score = 0

        # Check rows
        for i in range(4):
            row = board[i, :]
            # Check if monotonically increasing or decreasing
            inc = all(row[j] <= row[j + 1] for j in range(3))
            dec = all(row[j] >= row[j + 1] for j in range(3))
            if inc or dec:
                score += 1.0

        # Check columns
        for j in range(4):
            col = board[:, j]
            inc = all(col[i] <= col[i + 1] for i in range(3))
            dec = all(col[i] >= col[i + 1] for i in range(3))
            if inc or dec:
                score += 1.0

        return score

    def _calculate_smoothness(self, board: np.ndarray) -> float:
        """Calculate smoothness (penalty for large differences between adjacent tiles)"""
        smoothness = 0

        for i in range(4):
            for j in range(4):
                if board[i, j] != 0:
                    # Check right neighbor
                    if j < 3 and board[i, j + 1] != 0:
                        smoothness -= abs(
                            np.log2(board[i, j]) - np.log2(board[i, j + 1])
                        )
                    # Check down neighbor
                    if i < 3 and board[i + 1, j] != 0:
                        smoothness -= abs(
                            np.log2(board[i, j]) - np.log2(board[i + 1, j])
                        )

        return smoothness

    def _evaluate_board(self, board: np.ndarray) -> float:
        """Evaluate board state"""
        # Combine multiple heuristics
        monotonicity = self._calculate_monotonicity(board)
        smoothness = self._calculate_smoothness(board)
        empty_cells = np.sum(board == 0)
        max_tile = np.max(board)

        # Weighted combination
        score = (
            monotonicity * 10.0
            + smoothness * 1.0
            + empty_cells * 2.0
            + np.log2(max_tile)
            if max_tile > 0
            else 0
        )

        return score

    def choose_action(self, game: Game2048) -> Optional[Direction]:
        """Choose move that maximizes board evaluation"""
        available_moves = game.get_available_moves()

        if not available_moves:
            return None

        best_move = None
        best_score = float("-inf")

        for move in available_moves:
            # Simulate the move
            test_game = game.copy()
            valid, _ = test_game.move(move)  # _ for points

            if valid:
                score = self._evaluate_board(test_game.board)

                if score > best_score:
                    best_score = score
                    best_move = move

        return best_move if best_move else random.choice(available_moves)


def compare_agents(
    agents: List[Agent2048], num_games: int = 10, seed_offset: int = 0
) -> None:
    """
    Compare performance of multiple agents

    Args:
        agents: List of agents to compare
        num_games: Number of games each agent plays
        seed_offset: Offset for random seeds
    """
    print(f"\n{'='*60}")
    print(f"Comparing {len(agents)} agents over {num_games} games each")
    print(f"{'='*60}\n")

    for agent in agents:
        agent.reset_stats()
        print(f"Testing {agent.name}...")

        start_time = time.time()

        for i in range(num_games):
            # Use deterministic seeds for fair comparison
            agent.play_game(Game2048(seed=seed_offset + i), verbose=False)

            if (i + 1) % 10 == 0:
                print(f"  Completed {i+1}/{num_games} games")

        elapsed = time.time() - start_time
        stats = agent.get_stats()

        print(f"\n{agent.name} Results:")
        print(f"  Average Score: {stats['average_score']:.1f}")
        print(f"  Best Score: {stats['best_score']}")
        print(f"  Best Tile: {stats['best_tile']}")
        print(f"  Time: {elapsed:.2f}s ({elapsed/num_games:.3f}s per game)")
        print()

    # Summary table
    print(f"\n{'='*60}")
    print("Summary Ranking (by average score):")
    print(f"{'='*60}")

    # Sort agents by average score
    sorted_agents = sorted(
        agents, key=lambda a: a.total_score / max(1, a.games_played), reverse=True
    )

    for i, agent in enumerate(sorted_agents, 1):
        avg_score = agent.total_score / max(1, agent.games_played)
        print(
            f"{i}. {agent.name:25s} - Avg: {avg_score:7.1f}, Best: {agent.best_score:5d}"
        )


if __name__ == "__main__":
    # Create different agents
    agents_list = [
        RandomAgent(),
        GreedyAgent(),
        CornerAgent("top-left"),
        CornerAgent("bottom-right"),
        MonotonicAgent(),
    ]

    # Test single game with verbose output
    print("\nSample game with Corner Agent (top-left):")
    print("-" * 40)
    corner_agent = CornerAgent("top-left")
    result = corner_agent.play_game(Game2048(seed=42), verbose=True)
    print(f"\nFinal Score: {result['final_score']}")
    print(f"Max Tile: {result['max_tile']}")

    # Compare all agents
    compare_agents(agents_list, num_games=50)
