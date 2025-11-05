"""
2048 Interactive Terminal Player
Allows playing the game manually or watching agents play with visualization
"""

import os
import sys
import time
from typing import Optional
from game2048_engine import Game2048, Direction
from agents_2048 import Agent2048, RandomAgent, GreedyAgent, CornerAgent, MonotonicAgent


class ColoredOutput:
    """ANSI color codes for terminal output"""

    # Color scheme for tiles
    COLORS = {
        0: "\033[90m",  # Dark gray for empty
        2: "\033[97m",  # White
        4: "\033[93m",  # Yellow
        8: "\033[91m",  # Light red
        16: "\033[95m",  # Magenta
        32: "\033[96m",  # Cyan
        64: "\033[92m",  # Green
        128: "\033[33m",  # Orange
        256: "\033[34m",  # Blue
        512: "\033[35m",  # Purple
        1024: "\033[31m",  # Red
        2048: "\033[32m",  # Bright green
        4096: "\033[36m",  # Bright cyan
        8192: "\033[37m",  # Bright white
    }

    RESET = "\033[0m"
    BOLD = "\033[1m"
    CLEAR = "\033[2J\033[H"  # Clear screen and move cursor to top

    @classmethod
    def get_color(cls, value: int) -> str:
        """Get color code for a tile value"""
        return cls.COLORS.get(value, cls.COLORS[2048])


class Terminal2048:
    """Terminal interface for playing 2048"""

    def __init__(self):
        self.game = Game2048()
        self.use_colors = self._check_color_support()

    def _check_color_support(self) -> bool:
        """Check if terminal supports colors"""
        return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

    def clear_screen(self):
        """Clear the terminal screen"""
        if os.name == "nt":  # Windows
            os.system("cls")
        else:  # Unix/Linux/Mac
            os.system("clear")

    def display_board(self):
        """Display the game board with colors"""
        self.clear_screen()

        # Header
        print("\n" + "=" * 50)
        print("                    2048 GAME")
        print("=" * 50)

        # Score and stats
        print(
            f"\n  Score: {self.game.score:6d}    "
            f"Moves: {self.game.moves_count:4d}    "
            f"Max: {self.game.get_max_tile():4d}"
        )

        # Board
        print("\n  +" + "------+" * 4)

        for row in self.game.board:
            print("  |", end="")
            for val in row:
                if val == 0:
                    cell = "     "
                else:
                    if self.use_colors:
                        color = ColoredOutput.get_color(val)
                        cell = f"{color}{val:^5d}{ColoredOutput.RESET}"
                    else:
                        cell = f"{val:^5d}"
                print(f"{cell}|", end="")
            print("\n  +" + "------+" * 4)

        # Status
        if self.game.won:
            print("\n  🎉 CONGRATULATIONS! You reached 2048! 🎉")
        elif self.game.game_over:
            print("\n  💀 GAME OVER 💀")

    def get_user_input(self) -> Optional[Direction]:
        """Get move input from user"""
        print("\n  Controls: W/↑ (Up), S/↓ (Down), A/← (Left), D/→ (Right)")
        print("           Q (Quit), R (Restart), U (Undo)")
        print("\n  Your move: ", end="", flush=True)

        try:
            # For simple terminal input (can be enhanced with getch for instant response)
            key = input().strip().lower()

            if key in ["w", "8"]:
                return Direction.UP
            elif key in ["s", "2"]:
                return Direction.DOWN
            elif key in ["a", "4"]:
                return Direction.LEFT
            elif key in ["d", "6"]:
                return Direction.RIGHT
            elif key == "q":
                return "QUIT"
            elif key == "r":
                return "RESTART"
            elif key == "u":
                return "UNDO"
            else:
                print("  Invalid input! Use W/A/S/D or arrow keys.")
                return None

        except KeyboardInterrupt:
            return "QUIT"

    def play_manual(self):
        """Play the game manually"""
        print("\n  Welcome to 2048!")
        print("  Try to reach the 2048 tile by merging tiles.")
        time.sleep(2)

        # Game history for undo
        history = []

        while True:
            self.display_board()

            if self.game.game_over:
                print("\n  Play again? (Y/N): ", end="", flush=True)
                if input().strip().lower() != "y":
                    break
                self.game.reset()
                history = []
                continue

            action = self.get_user_input()

            if action == "QUIT":
                print("\n  Thanks for playing! Goodbye!")
                break
            elif action == "RESTART":
                self.game.reset()
                history = []
                print("\n  Game restarted!")
                time.sleep(1)
                continue
            elif action == "UNDO":
                if history:
                    self.game.set_state(history.pop())
                    print("\n  Move undone!")
                else:
                    print("\n  No moves to undo!")
                time.sleep(1)
                continue
            elif action and isinstance(action, Direction):
                # Save state for undo
                history.append(self.game.get_state())
                if len(history) > 10:  # Keep only last 10 states
                    history.pop(0)

                valid, _ = self.game.move(action)  # _ for points

                if not valid:
                    print("\n  Invalid move! Try another direction.")
                    time.sleep(1)

    def watch_agent(self, agent: Agent2048, delay: float = 0.5):
        """Watch an agent play the game"""
        print(f"\n  Watching {agent.name} play...")
        print("  Press Ctrl+C to stop")
        time.sleep(2)

        self.game.reset()

        try:
            while not self.game.game_over:
                self.display_board()

                action = agent.choose_action(self.game)

                if action is None:
                    break

                print(f"\n  {agent.name} chooses: {action.name}")
                _, points = self.game.move(action)  # _ for valid

                if points > 0:
                    print(f"  +{points} points!")

                time.sleep(delay)

        except KeyboardInterrupt:
            print("\n\n  Stopped watching.")

        self.display_board()
        print(f"\n  Final Score: {self.game.score}")
        print(f"  Total Moves: {self.game.moves_count}")
        print(f"  Max Tile: {self.game.get_max_tile()}")


def main_menu():
    """Main menu for the terminal game"""
    terminal = Terminal2048()

    agents = {
        "1": RandomAgent(),
        "2": GreedyAgent(),
        "3": CornerAgent("top-left"),
        "4": MonotonicAgent(),
    }

    while True:
        terminal.clear_screen()
        print("\n" + "=" * 50)
        print("                2048 TERMINAL GAME")
        print("=" * 50)
        print("\n  MAIN MENU:")
        print("  ----------")
        print("  1. Play Manually")
        print("  2. Watch Random Agent")
        print("  3. Watch Greedy Agent")
        print("  4. Watch Corner Agent")
        print("  5. Watch Monotonic Agent")
        print("  6. Compare Agents (Quick)")
        print("  Q. Quit")
        print("\n  Choose option: ", end="", flush=True)

        try:
            choice = input().strip().lower()

            if choice == "1":
                terminal.play_manual()
            elif choice in ["2", "3", "4", "5"]:
                agent_key = str(int(choice) - 1)
                agent = agents[agent_key]

                print(
                    "\n  Speed (seconds between moves, 0 for instant): ",
                    end="",
                    flush=True,
                )
                try:
                    delay = float(input().strip() or "0.5")
                except ValueError:
                    delay = 0.5

                terminal.watch_agent(agent, delay)
                print("\n  Press Enter to continue...", end="", flush=True)
                input()
            elif choice == "6":
                quick_compare_agents()
                print("\n  Press Enter to continue...", end="", flush=True)
                input()
            elif choice == "q":
                print("\n  Thanks for playing! Goodbye!")
                break
            else:
                print("\n  Invalid option! Please try again.")
                time.sleep(1)

        except KeyboardInterrupt:
            print("\n\n  Thanks for playing! Goodbye!")
            break


def quick_compare_agents():
    """Quick comparison of agents (fewer games for demo)"""
    print("\n" + "=" * 50)
    print("  QUICK AGENT COMPARISON (10 games each)")
    print("=" * 50)

    agents = [RandomAgent(), GreedyAgent(), CornerAgent("top-left"), MonotonicAgent()]

    results = []

    for agent in agents:
        print(f"\n  Testing {agent.name}...", end="", flush=True)

        scores = []
        max_tiles = []

        for i in range(10):
            game = Game2048(seed=i)
            result = agent.play_game(game)
            scores.append(result["final_score"])
            max_tiles.append(result["max_tile"])
            print(".", end="", flush=True)

        avg_score = sum(scores) / len(scores)
        avg_max = sum(max_tiles) / len(max_tiles)

        results.append(
            {
                "name": agent.name,
                "avg_score": avg_score,
                "best_score": max(scores),
                "avg_max_tile": avg_max,
                "best_tile": max(max_tiles),
            }
        )

        print(" Done!")

    # Display results
    print("\n" + "=" * 50)
    print("  RESULTS:")
    print("=" * 50)

    # Sort by average score
    results.sort(key=lambda x: x["avg_score"], reverse=True)

    for i, res in enumerate(results, 1):
        print(f"\n  {i}. {res['name']}")
        print(f"     Average Score: {res['avg_score']:.0f}")
        print(f"     Best Score: {res['best_score']}")
        print(f"     Average Max Tile: {res['avg_max_tile']:.0f}")
        print(f"     Best Tile: {res['best_tile']}")


if __name__ == "__main__":
    # Check if running in interactive mode
    if len(sys.argv) > 1:
        if sys.argv[1] == "play":
            terminal = Terminal2048()
            terminal.play_manual()
        elif sys.argv[1] == "watch":
            terminal = Terminal2048()
            agent = MonotonicAgent()  # Best performing agent
            terminal.watch_agent(agent, delay=0.3)
        elif sys.argv[1] == "compare":
            quick_compare_agents()
    else:
        main_menu()
