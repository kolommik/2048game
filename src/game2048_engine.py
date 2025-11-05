"""
2048 Game Engine
Pure game logic implementation without any UI dependencies
"""

import random
from typing import Tuple, List, Optional, Dict
from enum import Enum
import numpy as np


class Direction(Enum):
    """Enum for movement directions"""

    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class Game2048:
    """
    Core 2048 game engine

    Features:
    - 4x4 grid
    - Deterministic mode for reproducibility
    - Full game state management
    - Efficient numpy-based operations
    """

    def __init__(self, seed: Optional[int] = None, initial_tiles: int = 2):
        """
        Initialize game

        Args:
            seed: Random seed for reproducibility
            initial_tiles: Number of initial tiles (usually 2)
        """
        self.size = 4
        self.seed = seed
        self.initial_tiles = initial_tiles
        self.reset()

    def reset(self) -> np.ndarray:
        """Reset game to initial state"""
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0
        self.moves_count = 0
        self.game_over = False
        self.won = False  # True when 2048 tile is reached

        # Add initial tiles
        for _ in range(self.initial_tiles):
            self._add_random_tile()

        return self.board.copy()

    def _add_random_tile(self) -> bool:
        """
        Add a random tile (90% chance of 2, 10% chance of 4)

        Returns:
            True if tile was added, False if board is full
        """
        empty_cells = list(zip(*np.where(self.board == 0)))

        if not empty_cells:
            return False

        row, col = random.choice(empty_cells)
        # 90% chance for 2, 10% for 4
        self.board[row, col] = 2 if random.random() < 0.9 else 4
        return True

    def _slide_row_left(self, row: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Slide and merge a single row to the left

        Args:
            row: A row from the board

        Returns:
            Tuple of (new_row, points_gained)
        """
        # Remove zeros
        non_zero = row[row != 0]

        if len(non_zero) == 0:
            return np.zeros_like(row), 0

        points = 0
        merged = []
        i = 0

        while i < len(non_zero):
            if i + 1 < len(non_zero) and non_zero[i] == non_zero[i + 1]:
                # Merge tiles
                merged_value = non_zero[i] * 2
                merged.append(merged_value)
                points += merged_value

                # Check for win condition
                if merged_value == 2048:
                    self.won = True

                i += 2
            else:
                merged.append(non_zero[i])
                i += 1

        # Pad with zeros
        new_row = np.zeros_like(row)
        new_row[: len(merged)] = merged

        return new_row, points

    def _move_left(self) -> Tuple[np.ndarray, int]:
        """Execute left move on the entire board"""
        new_board = np.zeros_like(self.board)
        total_points = 0

        for i in range(self.size):
            new_board[i], points = self._slide_row_left(self.board[i])
            total_points += points

        return new_board, total_points

    def _move_right(self) -> Tuple[np.ndarray, int]:
        """Execute right move on the entire board"""
        new_board = np.zeros_like(self.board)
        total_points = 0

        for i in range(self.size):
            # Flip, slide left, flip back
            flipped_row = self.board[i][::-1]
            new_row, points = self._slide_row_left(flipped_row)
            new_board[i] = new_row[::-1]
            total_points += points

        return new_board, total_points

    def _move_up(self) -> Tuple[np.ndarray, int]:
        """Execute up move on the entire board"""
        # Transpose, move left, transpose back
        transposed = self.board.T
        new_board = np.zeros_like(transposed)
        total_points = 0

        for i in range(self.size):
            new_board[i], points = self._slide_row_left(transposed[i])
            total_points += points

        return new_board.T, total_points

    def _move_down(self) -> Tuple[np.ndarray, int]:
        """Execute down move on the entire board"""
        # Transpose, move right, transpose back
        transposed = self.board.T
        new_board = np.zeros_like(transposed)
        total_points = 0

        for i in range(self.size):
            flipped_row = transposed[i][::-1]
            new_row, points = self._slide_row_left(flipped_row)
            new_board[i] = new_row[::-1]
            total_points += points

        return new_board.T, total_points

    def move(self, direction: Direction) -> Tuple[bool, int]:
        """
        Execute a move in the given direction

        Args:
            direction: Direction enum

        Returns:
            Tuple of (move_was_valid, points_gained)
        """
        if self.game_over:
            return False, 0

        # Get new board state based on direction
        move_funcs = {
            Direction.LEFT: self._move_left,
            Direction.RIGHT: self._move_right,
            Direction.UP: self._move_up,
            Direction.DOWN: self._move_down,
        }

        new_board, points = move_funcs[direction]()

        # Check if move is valid (board changed)
        if np.array_equal(self.board, new_board):
            return False, 0

        # Update game state
        self.board = new_board
        self.score += points
        self.moves_count += 1

        # Add new random tile
        self._add_random_tile()

        # Check for game over
        if self.is_game_over():
            self.game_over = True

        return True, points

    def is_game_over(self) -> bool:
        """Check if no more moves are possible"""
        # Check for empty cells
        if 0 in self.board:
            return False

        # Check for possible merges horizontally
        for i in range(self.size):
            for j in range(self.size - 1):
                if self.board[i, j] == self.board[i, j + 1]:
                    return False

        # Check for possible merges vertically
        for i in range(self.size - 1):
            for j in range(self.size):
                if self.board[i, j] == self.board[i + 1, j]:
                    return False

        return True

    def get_available_moves(self) -> List[Direction]:
        """Get list of valid moves from current state"""
        available = []

        for direction in Direction:
            # Simulate move without modifying state
            original_board = self.board.copy()

            move_funcs = {
                Direction.LEFT: self._move_left,
                Direction.RIGHT: self._move_right,
                Direction.UP: self._move_up,
                Direction.DOWN: self._move_down,
            }

            new_board, _ = move_funcs[direction]()

            if not np.array_equal(original_board, new_board):
                available.append(direction)

        return available

    def get_state(self) -> Dict:
        """Get complete game state"""
        return {
            "board": self.board.copy(),
            "score": self.score,
            "moves_count": self.moves_count,
            "game_over": self.game_over,
            "won": self.won,
            "max_tile": self.board.max(),
        }

    def set_state(self, state: Dict):
        """Restore game state from dictionary"""
        self.board = state["board"].copy()
        self.score = state["score"]
        self.moves_count = state["moves_count"]
        self.game_over = state["game_over"]
        self.won = state["won"]

    def get_max_tile(self) -> int:
        """Get the maximum tile value on the board"""
        return self.board.max()

    def copy(self):
        """Create a deep copy of the game"""
        new_game = Game2048(seed=self.seed)
        new_game.set_state(self.get_state())
        return new_game

    def __str__(self) -> str:
        """String representation for debugging"""
        s = f"Score: {self.score} | Moves: {self.moves_count} | Max: {self.get_max_tile()}\n"
        s += "-" * 25 + "\n"

        for row in self.board:
            s += "|"
            for val in row:
                if val == 0:
                    s += "    |"
                else:
                    s += f"{val:4d}|"
            s += "\n"
        s += "-" * 25

        return s


# Additional utility functions
def play_random_game(seed: Optional[int] = None, max_moves: int = 1000) -> Dict:
    """
    Play a complete game with random moves

    Returns:
        Game statistics
    """
    game = Game2048(seed=seed)
    moves_history = []

    while not game.game_over and game.moves_count < max_moves:
        available_moves = game.get_available_moves()

        if not available_moves:
            break

        move = random.choice(available_moves)
        valid, _ = game.move(move)  # _ for points

        if valid:
            moves_history.append(
                {
                    "move": move.name,
                    "score": game.score,
                    "max_tile": game.get_max_tile(),
                    "board": game.board.copy(),
                }
            )

    return {
        "final_score": game.score,
        "moves_count": game.moves_count,
        "max_tile": game.get_max_tile(),
        "won": game.won,
        "history": moves_history,
    }


if __name__ == "__main__":
    # Test the game engine
    print("Testing 2048 Game Engine\n")

    # Test deterministic game
    game1 = Game2048(seed=42)
    game2 = Game2048(seed=42)

    print("Initial board (seed=42):")
    print(game1)

    # Test moves
    print("\nTesting moves...")
    moves_sequence = [Direction.UP, Direction.LEFT, Direction.DOWN, Direction.RIGHT]

    for single_move in moves_sequence:
        valid1, points1 = game1.move(single_move)
        valid2, points2 = game2.move(single_move)

        assert valid1 == valid2, "Determinism check failed"
        assert points1 == points2, "Points mismatch"

        if valid1:
            print(f"Move {single_move.name}: +{points1} points")

    print("\nBoard after moves:")
    print(game1)

    # Play a random game
    print("\n" + "=" * 50)
    print("Playing random game...")
    stats = play_random_game(seed=123)

    print("\nGame Over!")
    print(f"Final Score: {stats['final_score']}")
    print(f"Total Moves: {stats['moves_count']}")
    print(f"Max Tile: {stats['max_tile']}")
    print(f"Won (reached 2048): {stats['won']}")

    # Test multiple random games for statistics
    print("\n" + "=" * 50)
    print("Running 100 random games for statistics...")

    scores = []
    max_tiles = []
    win_count = 0

    for i in range(100):
        stats = play_random_game(seed=i)
        scores.append(stats["final_score"])
        max_tiles.append(stats["max_tile"])
        if stats["won"]:
            win_count += 1

    print("\nStatistics over 100 random games:")
    print(f"Average Score: {np.mean(scores):.1f}")
    print(f"Max Score: {max(scores)}")
    print(f"Min Score: {min(scores)}")
    print(f"Games Won (2048 reached): {win_count}/100")

    # Count max tile distribution
    from collections import Counter

    tile_dist = Counter(max_tiles)
    print("\nMax tile distribution:")
    for tile in sorted(tile_dist.keys()):
        print(f"  {tile}: {tile_dist[tile]} games")
