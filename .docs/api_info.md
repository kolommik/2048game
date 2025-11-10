# API Documentation

Generated for modules: game2048_engine, agents_2048, terminal_2048, streamlit_2048

================================================================================

# Module: game2048_engine


2048 Game Engine
Pure game logic implementation without any UI dependencies



## Class: Direction
```python
class Direction:
```
Enum for movement directions


## Class: Game2048
```python
class Game2048:
```
Core 2048 game engine

Features:
- 4x4 grid
- Deterministic mode for reproducibility
- Full game state management
- Efficient numpy-based operations

### Game2048.copy(self)
Create a deep copy of the game

### Game2048.get_available_moves(self) -> List[game2048_engine.Direction]
Get list of valid moves from current state

### Game2048.get_max_tile(self) -> int
Get the maximum tile value on the board

### Game2048.get_state(self) -> Dict
Get complete game state

### Game2048.is_game_over(self) -> bool
Check if no more moves are possible

### Game2048.move(self, direction: game2048_engine.Direction) -> Tuple[bool, int]
Execute a move in the given direction

Args:
    direction: Direction enum

Returns:
    Tuple of (move_was_valid, points_gained)

### Game2048.reset(self) -> numpy.ndarray
Reset game to initial state

### Game2048.set_state(self, state: Dict)
Restore game state from dictionary


## Function: play_random_game(seed: Optional[int] = None, max_moves: int = 1000) -> Dict
Play a complete game with random moves

Returns:
    Game statistics


================================================================================

# Module: agents_2048


2048 Game Agents
Different agent implementations for playing 2048



## Class: Agent2048
```python
class Agent2048:
```
Base class for all 2048 agents

### Agent2048.choose_action(self, game: game2048_engine.Game2048) -> Optional[game2048_engine.Direction]
Choose an action given the current game state

Args:
    game: Current game instance

Returns:
    Direction to move or None if no move available

### Agent2048.get_stats(self) -> Dict
Get agent statistics

### Agent2048.play_game(self, game: Optional[game2048_engine.Game2048] = None, max_moves: int = 10000, verbose: bool = False) -> Dict
Play a complete game

Args:
    game: Game instance (creates new if None)
    max_moves: Maximum moves before stopping
    verbose: Print game progress

Returns:
    Game statistics

### Agent2048.reset_stats(self)
Reset agent statistics

### Agent2048.update_stats(self, score: int, max_tile: int)
Update agent statistics after a game


## Class: CornerAgent
```python
class CornerAgent:
```
Agent that tries to keep the maximum tile in a corner

### CornerAgent.choose_action(self, game: game2048_engine.Game2048) -> Optional[game2048_engine.Direction]
Choose move based on corner strategy

### CornerAgent.get_stats(self) -> Dict
Get agent statistics

### CornerAgent.play_game(self, game: Optional[game2048_engine.Game2048] = None, max_moves: int = 10000, verbose: bool = False) -> Dict
Play a complete game

Args:
    game: Game instance (creates new if None)
    max_moves: Maximum moves before stopping
    verbose: Print game progress

Returns:
    Game statistics

### CornerAgent.reset_stats(self)
Reset agent statistics

### CornerAgent.update_stats(self, score: int, max_tile: int)
Update agent statistics after a game


## Class: GreedyAgent
```python
class GreedyAgent:
```
Agent that chooses move with highest immediate reward

### GreedyAgent.choose_action(self, game: game2048_engine.Game2048) -> Optional[game2048_engine.Direction]
Choose move that gives maximum immediate points

### GreedyAgent.get_stats(self) -> Dict
Get agent statistics

### GreedyAgent.play_game(self, game: Optional[game2048_engine.Game2048] = None, max_moves: int = 10000, verbose: bool = False) -> Dict
Play a complete game

Args:
    game: Game instance (creates new if None)
    max_moves: Maximum moves before stopping
    verbose: Print game progress

Returns:
    Game statistics

### GreedyAgent.reset_stats(self)
Reset agent statistics

### GreedyAgent.update_stats(self, score: int, max_tile: int)
Update agent statistics after a game


## Class: MonotonicAgent
```python
class MonotonicAgent:
```
Agent that tries to maintain monotonic rows and columns

### MonotonicAgent.choose_action(self, game: game2048_engine.Game2048) -> Optional[game2048_engine.Direction]
Choose move that maximizes board evaluation

### MonotonicAgent.get_stats(self) -> Dict
Get agent statistics

### MonotonicAgent.play_game(self, game: Optional[game2048_engine.Game2048] = None, max_moves: int = 10000, verbose: bool = False) -> Dict
Play a complete game

Args:
    game: Game instance (creates new if None)
    max_moves: Maximum moves before stopping
    verbose: Print game progress

Returns:
    Game statistics

### MonotonicAgent.reset_stats(self)
Reset agent statistics

### MonotonicAgent.update_stats(self, score: int, max_tile: int)
Update agent statistics after a game


## Class: RandomAgent
```python
class RandomAgent:
```
Agent that plays random valid moves

### RandomAgent.choose_action(self, game: game2048_engine.Game2048) -> Optional[game2048_engine.Direction]
Choose a random valid move

### RandomAgent.get_stats(self) -> Dict
Get agent statistics

### RandomAgent.play_game(self, game: Optional[game2048_engine.Game2048] = None, max_moves: int = 10000, verbose: bool = False) -> Dict
Play a complete game

Args:
    game: Game instance (creates new if None)
    max_moves: Maximum moves before stopping
    verbose: Print game progress

Returns:
    Game statistics

### RandomAgent.reset_stats(self)
Reset agent statistics

### RandomAgent.update_stats(self, score: int, max_tile: int)
Update agent statistics after a game


## Function: compare_agents(agents: List[agents_2048.Agent2048], num_games: int = 10, seed_offset: int = 0) -> None
Compare performance of multiple agents

Args:
    agents: List of agents to compare
    num_games: Number of games each agent plays
    seed_offset: Offset for random seeds


================================================================================

# Module: terminal_2048


2048 Interactive Terminal Player
Allows playing the game manually or watching agents play with visualization



## Class: ColoredOutput
```python
class ColoredOutput:
```
ANSI color codes for terminal output


## Class: Terminal2048
```python
class Terminal2048:
```
Terminal interface for playing 2048

### Terminal2048.clear_screen(self)
Clear the terminal screen

### Terminal2048.display_board(self)
Display the game board with colors

### Terminal2048.get_user_input(self) -> Optional[game2048_engine.Direction]
Get move input from user

### Terminal2048.play_manual(self)
Play the game manually

### Terminal2048.watch_agent(self, agent: agents_2048.Agent2048, delay: float = 0.5)
Watch an agent play the game


## Function: main_menu()
Main menu for the terminal game


## Function: quick_compare_agents()
Quick comparison of agents (fewer games for demo)


================================================================================

# Module: streamlit_2048


2048 Web Interface using Streamlit
Interactive game with manual play, agent watching, and analytics



## Class: GameAnalytics
```python
class GameAnalytics:
```
Analytics and visualization for game data

### GameAnalytics.plot_agent_comparison(comparison_data: Dict)
Plot comparison between multiple agents

### GameAnalytics.plot_game_statistics(games_data: List[Dict])
Plot various game statistics

### GameAnalytics.plot_max_tile_distribution(games_data: List[Dict])
Plot distribution of maximum tiles achieved

### GameAnalytics.plot_score_progression(games_data: List[Dict])
Plot score progression across games


## Class: GameUI
```python
class GameUI:
```
UI components for the game

### GameUI.render_board(game: game2048_engine.Game2048, editable: bool = False)
Render the game board using Streamlit dataframe with styling

### GameUI.style_cell(val)
Style individual cells with colors


## Class: UIConfig
```python
class UIConfig:
```
Configuration constants for UI appearance


## Function: analytics_page()
Analytics and insights page


## Function: compare_agents_page()
Compare multiple agents


## Function: initialize_session_state()
Initialize session state variables


## Function: main()
Main application


## Function: manual_play_page()
Manual play interface


## Function: watch_agent_page()
Watch agent play interface


================================================================================

