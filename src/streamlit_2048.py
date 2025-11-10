"""
2048 Web Interface using Streamlit
Interactive game with manual play, agent watching, and analytics
"""

import streamlit as st
import numpy as np
import pandas as pd
import time
from typing import List, Dict
import plotly.graph_objects as go
from game2048_engine import Game2048, Direction
from agents_2048 import (
    RandomAgent,
    GreedyAgent,
    CornerAgent,
    MonotonicAgent,
)


# Page configuration
st.set_page_config(
    page_title="2048 Game",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)


# UI Configuration Constants
class UIConfig:
    """Configuration constants for UI appearance"""

    # Board display settings
    BOARD_HEIGHT = 280  # Height of the game board in pixels
    BOARD_CELL_FONT_SIZE = 20  # Font size for tile numbers
    BOARD_MAX_WIDTH = 500  # Maximum width of the board container

    # Column widths for board cells
    BOARD_COLUMN_WIDTH = "small"  # Options: "small", "medium", "large"


class GameUI:
    """UI components for the game"""

    # Color scheme for tiles
    TILE_COLORS = {
        0: "#cdc1b4",
        2: "#eee4da",
        4: "#ede0c8",
        8: "#f2b179",
        16: "#f59563",
        32: "#f67c5f",
        64: "#f65e3b",
        128: "#edcf72",
        256: "#edcc61",
        512: "#edc850",
        1024: "#edc53f",
        2048: "#edc22e",
        4096: "#3c3a32",
        8192: "#3c3a32",
    }

    TILE_TEXT_COLORS = {
        0: "#776e65",
        2: "#776e65",
        4: "#776e65",
        8: "#f9f6f2",
        16: "#f9f6f2",
        32: "#f9f6f2",
        64: "#f9f6f2",
        128: "#f9f6f2",
        256: "#f9f6f2",
        512: "#f9f6f2",
        1024: "#f9f6f2",
        2048: "#f9f6f2",
        4096: "#f9f6f2",
        8192: "#f9f6f2",
    }

    @staticmethod
    def style_cell(val):
        """Style individual cells with colors"""
        bg_color = GameUI.TILE_COLORS.get(val, GameUI.TILE_COLORS[2048])
        text_color = GameUI.TILE_TEXT_COLORS.get(val, GameUI.TILE_TEXT_COLORS[2048])
        return (
            f"background-color: {bg_color}; color: {text_color}; "
            f"font-weight: bold; font-size: {UIConfig.BOARD_CELL_FONT_SIZE}px; "
            f"text-align: center;"
        )

    @staticmethod
    def render_board(game: Game2048, editable: bool = False):
        """Render the game board using Streamlit dataframe with styling"""

        # Center the board with max width
        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            # Display game stats
            stat1, stat2, stat3 = st.columns(3)
            with stat1:
                st.metric("Score", game.score)
            with stat2:
                st.metric("Moves", game.moves_count)
            with stat3:
                st.metric("Max Tile", game.get_max_tile())

            # Convert board to DataFrame
            df = pd.DataFrame(game.board)

            # Apply styling using .map instead of deprecated .applymap
            styled_df = df.style.map(GameUI.style_cell)

            # Format display (empty cells as blank, numbers as is)
            styled_df = styled_df.format(lambda x: "" if x == 0 else str(x))

            if editable:
                # For editable mode, use data_editor
                st.info("🔧 Edit mode: You can modify cell values manually")
                edited_df = st.data_editor(
                    df,
                    width="stretch",
                    hide_index=True,
                    column_config={
                        i: st.column_config.NumberColumn(
                            label="",
                            min_value=0,
                            max_value=8192,
                            step=2,
                        )
                        for i in df.columns
                    },
                    key="board_editor",
                )
                return edited_df.values
            else:
                # For display mode, use styled dataframe
                st.dataframe(
                    styled_df,
                    width="stretch",
                    hide_index=True,
                    height=UIConfig.BOARD_HEIGHT,
                    column_config={
                        i: st.column_config.NumberColumn(
                            label="", width=UIConfig.BOARD_COLUMN_WIDTH
                        )
                        for i in df.columns
                    },
                )
                return None


class GameAnalytics:
    """Analytics and visualization for game data"""

    @staticmethod
    def plot_score_progression(games_data: List[Dict]):
        """Plot score progression across games"""
        if not games_data:
            return

        fig = go.Figure()

        for game in games_data:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(game["score_history"]))),
                    y=game["score_history"],
                    mode="lines",
                    name=f"Game {game['game_num']}",
                    hovertemplate="Move: %{x}<br>Score: %{y}<extra></extra>",
                )
            )

        fig.update_layout(
            title="Score Progression During Games",
            xaxis_title="Move Number",
            yaxis_title="Score",
            hovermode="x unified",
            height=400,
        )

        st.plotly_chart(fig, width="stretch")

    @staticmethod
    def plot_max_tile_distribution(games_data: List[Dict]):
        """Plot distribution of maximum tiles achieved"""
        if not games_data:
            return

        max_tiles = [game["max_tile"] for game in games_data]

        # Define tile categories (powers of 2 from 128 to 2048+)
        tile_categories = [128, 256, 512, 1024, 2048]
        tile_labels = ["128", "256", "512", "1024", "2048", ">2048"]

        # Count tiles in each category
        tile_counts = []
        for tile_val in tile_categories:
            count = sum(1 for t in max_tiles if t == tile_val)
            tile_counts.append(count)

        # Count tiles greater than 2048
        count_above = sum(1 for t in max_tiles if t > 2048)
        tile_counts.append(count_above)

        fig = go.Figure(
            data=[
                go.Bar(
                    x=tile_labels,
                    y=tile_counts,
                    text=tile_counts,
                    textposition="auto",
                    marker=dict(
                        color=[
                            "#edcf72",
                            "#edcc61",
                            "#edc850",
                            "#edc53f",
                            "#edc22e",
                            "#3c3a32",
                        ]
                    ),
                )
            ]
        )

        fig.update_layout(
            title="Distribution of Maximum Tiles Achieved",
            xaxis_title="Max Tile Value",
            yaxis_title="Number of Games",
            height=400,
            xaxis=dict(type="category"),  # Force categorical axis
        )

        st.plotly_chart(fig, width="stretch")

    @staticmethod
    def plot_agent_comparison(comparison_data: Dict):
        """Plot comparison between multiple agents"""
        if not comparison_data:
            return

        agents = list(comparison_data.keys())
        metrics = ["avg_score", "best_score", "avg_max_tile", "best_tile"]
        metric_names = [
            "Average Score",
            "Best Score",
            "Average Max Tile",
            "Best Tile",
        ]

        fig = go.Figure()

        for metric, name in zip(metrics, metric_names):
            values = [comparison_data[agent][metric] for agent in agents]
            fig.add_trace(go.Bar(name=name, x=agents, y=values))

        fig.update_layout(
            title="Agent Performance Comparison",
            xaxis_title="Agent",
            yaxis_title="Value",
            barmode="group",
            height=500,
        )

        st.plotly_chart(fig, width="stretch")

    @staticmethod
    def plot_game_statistics(games_data: List[Dict]):
        """Plot various game statistics"""
        if not games_data:
            return

        df = pd.DataFrame(games_data)

        # Show basic statistics as metrics instead of box plots
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Min Score", f"{df['final_score'].min():.0f}")
            st.metric("Median Score", f"{df['final_score'].median():.0f}")
            st.metric("Max Score", f"{df['final_score'].max():.0f}")

        with col2:
            st.metric("Min Moves", f"{df['total_moves'].min():.0f}")
            st.metric("Median Moves", f"{df['total_moves'].median():.0f}")
            st.metric("Max Moves", f"{df['total_moves'].max():.0f}")

        with col3:
            st.metric("Min Tile", f"{df['max_tile'].min():.0f}")
            st.metric("Median Tile", f"{df['max_tile'].median():.0f}")
            st.metric("Max Tile", f"{df['max_tile'].max():.0f}")


def initialize_session_state():
    """Initialize session state variables"""
    if "game" not in st.session_state:
        st.session_state.game = Game2048()

    if "game_history" not in st.session_state:
        st.session_state.game_history = []

    if "agent_games_data" not in st.session_state:
        st.session_state.agent_games_data = []

    if "comparison_data" not in st.session_state:
        st.session_state.comparison_data = {}

    if "move_counter" not in st.session_state:
        st.session_state.move_counter = 0


def manual_play_page():
    """Manual play interface"""
    st.title("🎮 Play 2048")

    game = st.session_state.game

    # Add toggle for edit mode
    edit_mode = st.toggle("🔧 Edit Mode (manually set board values)", value=False)

    if edit_mode:
        st.info(
            "In edit mode, you can manually set the values in any cell. Use powers of 2 (2, 4, 8, 16, etc.)"
        )

        # Render editable board
        new_board = GameUI.render_board(game, editable=True)

        if new_board is not None:
            # Update game board with edited values
            if st.button("✅ Apply Changes", width="stretch"):
                game.board = new_board.astype(np.int32)
                st.success("Board updated!")
                st.rerun()

        # In edit mode, show manual controls for score
        col1, col2 = st.columns(2)
        with col1:
            new_score = st.number_input(
                "Score", min_value=0, value=game.score, step=100
            )
            if new_score != game.score:
                game.score = new_score
        with col2:
            st.metric("Moves", game.moves_count)
    else:
        # Normal play mode - render board
        GameUI.render_board(game, editable=False)

    # Game status
    if game.game_over:
        st.error("🎭 Game Over!")
        if st.button("🔄 Restart Game", width="stretch"):
            st.session_state.game = Game2048()
            st.session_state.move_counter += 1
            st.rerun()
    elif game.won:
        st.success("🎉 Congratulations! You reached 2048!")
        if st.button("🔄 Continue Playing", width="stretch"):
            st.session_state.move_counter += 1
            st.rerun()

    if not edit_mode:
        # Controls - only in normal play mode
        st.markdown("### 🎯 Controls")

        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            if st.button("⬆️ Up", width="stretch", disabled=game.game_over):
                valid, _ = game.move(Direction.UP)
                if valid:
                    st.session_state.move_counter += 1
                    st.rerun()

        with col2:
            if st.button("⬇️ Down", width="stretch", disabled=game.game_over):
                valid, _ = game.move(Direction.DOWN)
                if valid:
                    st.session_state.move_counter += 1
                    st.rerun()

        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            if st.button("⬅️ Left", width="stretch", disabled=game.game_over):
                valid, _ = game.move(Direction.LEFT)
                if valid:
                    st.session_state.move_counter += 1
                    st.rerun()

        with col2:
            if st.button("➡️ Right", width="stretch", disabled=game.game_over):
                valid, _ = game.move(Direction.RIGHT)
                if valid:
                    st.session_state.move_counter += 1
                    st.rerun()

    # Additional controls
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔄 New Game", width="stretch"):
            st.session_state.game = Game2048()
            st.session_state.move_counter += 1
            st.rerun()

    with col2:
        if st.button("💾 Save Game State", width="stretch"):
            st.session_state.game_history.append(game.get_state())
            st.success("Game state saved!")

    # Keyboard instructions
    if not edit_mode:
        st.info(
            "💡 **Tip**: Use the buttons above to make moves, or enable Edit Mode to manually set board values!"
        )


def watch_agent_page():
    """Watch agent play interface"""
    st.title("🤖 Watch AI Agents Play")

    # Agent selection
    agent_options = {
        "Random Agent": RandomAgent(),
        "Greedy Agent": GreedyAgent(),
        "Corner Agent (Top-Left)": CornerAgent("top-left"),
        "Corner Agent (Top-Right)": CornerAgent("top-right"),
        "Monotonic Agent": MonotonicAgent(),
    }

    selected_agent_name = st.selectbox(
        "Select Agent", list(agent_options.keys()), index=4
    )
    agent = agent_options[selected_agent_name]

    # Game settings
    col1, col2, col3 = st.columns(3)

    with col1:
        num_games = st.number_input(
            "Number of Games", min_value=1, max_value=100, value=10
        )

    with col2:
        seed_start = st.number_input("Seed Start", min_value=0, value=0)

    with col3:
        show_visualization = st.checkbox("Show Game Visualization", value=False)

    if st.button("▶️ Start Simulation", width="stretch", type="primary"):
        games_data = []

        progress_bar = st.progress(0)
        status_text = st.empty()

        # Container for live board if visualization enabled
        if show_visualization:
            board_container = st.empty()
            stats_container = st.empty()

        for i in range(num_games):
            status_text.text(f"Playing game {i+1}/{num_games}...")

            game = Game2048(seed=seed_start + i)
            score_history = [0]

            while not game.game_over:
                if show_visualization and i == 0:  # Show only first game
                    with board_container.container():
                        # Center the board
                        viz_col1, viz_col2, viz_col3 = st.columns([1, 2, 1])

                        with viz_col2:
                            with stats_container.container():
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Score", game.score)
                                with col2:
                                    st.metric("Moves", game.moves_count)
                                with col3:
                                    st.metric("Max Tile", game.get_max_tile())

                            # Display board as dataframe
                            df = pd.DataFrame(game.board.astype(int))
                            styled_df = df.style.map(GameUI.style_cell).format(
                                lambda x: "" if x == 0 else str(x)
                            )

                            st.dataframe(
                                styled_df,
                                width="stretch",
                                hide_index=True,
                                height=UIConfig.BOARD_HEIGHT,
                            )

                        time.sleep(0.1)

                action = agent.choose_action(game)
                if action is None:
                    break

                game.move(action)
                score_history.append(game.score)

            games_data.append(
                {
                    "game_num": i + 1,
                    "final_score": game.score,
                    "max_tile": game.get_max_tile(),
                    "total_moves": game.moves_count,
                    "score_history": score_history,
                }
            )

            progress_bar.progress((i + 1) / num_games)

        status_text.text("✅ Simulation complete!")

        # Store results
        st.session_state.agent_games_data = games_data

        # Display summary statistics
        st.markdown("### 📊 Summary Statistics")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            avg_score = np.mean([g["final_score"] for g in games_data])
            st.metric("Average Score", f"{avg_score:.0f}")

        with col2:
            best_score = max([g["final_score"] for g in games_data])
            st.metric("Best Score", f"{best_score}")

        with col3:
            avg_tile = np.mean([g["max_tile"] for g in games_data])
            st.metric("Average Max Tile", f"{avg_tile:.0f}")

        with col4:
            best_tile = max([g["max_tile"] for g in games_data])
            st.metric("Best Tile", f"{best_tile}")

        # Display visualizations
        st.markdown("### 📈 Performance Visualization")

        GameAnalytics.plot_max_tile_distribution(games_data)
        GameAnalytics.plot_score_progression(games_data[:5])  # Show first 5 games
        GameAnalytics.plot_game_statistics(games_data)

        # Show detailed results table
        st.markdown("### 📋 Detailed Results")
        df = pd.DataFrame(games_data)
        df = df.drop(columns=["score_history"])
        st.dataframe(df, width="stretch")


def compare_agents_page():
    """Compare multiple agents"""
    st.title("⚔️ Compare AI Agents")

    st.markdown(
        """
    Compare the performance of different agents across multiple games.
    This will help you understand which strategies work best.
    """
    )

    # Settings
    col1, col2 = st.columns(2)

    with col1:
        num_games = st.number_input(
            "Games per Agent", min_value=5, max_value=100, value=20
        )

    with col2:
        seed_start = st.number_input("Random Seed", min_value=0, value=42)

    # Agent selection
    st.markdown("### Select Agents to Compare")

    col1, col2, col3 = st.columns(3)

    with col1:
        use_random = st.checkbox("Random Agent", value=True)
        use_greedy = st.checkbox("Greedy Agent", value=True)

    with col2:
        use_corner = st.checkbox("Corner Agent", value=True)
        corner_position = st.selectbox(
            "Corner Position", ["top-left", "top-right"], disabled=not use_corner
        )

    with col3:
        use_monotonic = st.checkbox("Monotonic Agent", value=True)

    if st.button("🚀 Start Comparison", width="stretch", type="primary"):
        agents_to_compare = []

        if use_random:
            agents_to_compare.append(("Random", RandomAgent()))
        if use_greedy:
            agents_to_compare.append(("Greedy", GreedyAgent()))
        if use_corner:
            agents_to_compare.append(
                (f"Corner ({corner_position})", CornerAgent(corner_position))
            )
        if use_monotonic:
            agents_to_compare.append(("Monotonic", MonotonicAgent()))

        if not agents_to_compare:
            st.error("Please select at least one agent!")
            return

        # Initialize fresh comparison data for this run
        new_comparison_data = {}
        progress_bar = st.progress(0)
        status_text = st.empty()

        total_games = len(agents_to_compare) * num_games
        completed_games = 0

        for agent_name, agent in agents_to_compare:
            status_text.text(f"Testing {agent_name}...")

            scores = []
            max_tiles = []
            moves = []

            for i in range(num_games):
                game = Game2048(seed=seed_start + i)
                result = agent.play_game(game)

                scores.append(result["final_score"])
                max_tiles.append(result["max_tile"])
                moves.append(game.moves_count)  # Get directly from game object

                completed_games += 1
                progress_bar.progress(completed_games / total_games)

            new_comparison_data[agent_name] = {
                "avg_score": np.mean(scores),
                "best_score": max(scores),
                "avg_max_tile": np.mean(max_tiles),
                "best_tile": max(max_tiles),
                "avg_moves": np.mean(moves),
                "games_count": num_games,
            }

        status_text.text("✅ Comparison complete!")

        # Store results in session state
        st.session_state.comparison_data = new_comparison_data

        # Display results
        st.markdown("### 🏆 Results")

        # Create summary table
        summary_df = pd.DataFrame(new_comparison_data).T
        summary_df = summary_df.round(1)

        # Sort by average score
        summary_df = summary_df.sort_values("avg_score", ascending=False)

        st.dataframe(summary_df, width="stretch")

        # Visualizations
        st.markdown("### 📊 Visual Comparison")
        GameAnalytics.plot_agent_comparison(new_comparison_data)

        # Winner announcement
        winner = summary_df.index[0]
        winner_score = summary_df.loc[winner, "avg_score"]

        st.success(
            f"🥇 **Winner**: {winner} with an average score of {winner_score:.0f}!"
        )


def analytics_page():
    """Analytics and insights page"""
    st.title("📊 Game Analytics")

    if not st.session_state.agent_games_data and not st.session_state.comparison_data:
        st.info(
            "No data available yet. Run some agent simulations or comparisons first!"
        )
        return

    tab1, tab2 = st.tabs(["Agent Games Data", "Agent Comparison"])

    with tab1:
        if st.session_state.agent_games_data:
            st.markdown("### Latest Agent Simulation Results")

            games_data = st.session_state.agent_games_data

            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Games", len(games_data))

            with col2:
                avg_score = np.mean([g["final_score"] for g in games_data])
                st.metric("Avg Score", f"{avg_score:.0f}")

            with col3:
                avg_moves = np.mean([g["total_moves"] for g in games_data])
                st.metric("Avg Moves", f"{avg_moves:.0f}")

            with col4:
                best_tile = max([g["max_tile"] for g in games_data])
                st.metric("Best Tile", best_tile)

            # Visualizations
            GameAnalytics.plot_score_progression(games_data[:10])
            GameAnalytics.plot_max_tile_distribution(games_data)
            GameAnalytics.plot_game_statistics(games_data)

            # Download data
            if st.button("💾 Download Data as CSV"):
                df = pd.DataFrame(games_data)
                df = df.drop(columns=["score_history"])
                csv = df.to_csv(index=False)
                st.download_button(
                    "Download CSV",
                    csv,
                    "agent_games_data.csv",
                    "text/csv",
                    width="stretch",
                )
        else:
            st.info("No agent simulation data available.")

    with tab2:
        if st.session_state.comparison_data:
            st.markdown("### Latest Agent Comparison Results")

            GameAnalytics.plot_agent_comparison(st.session_state.comparison_data)

            # Show detailed table
            df = pd.DataFrame(st.session_state.comparison_data).T
            st.dataframe(df, width="stretch")

            # Download comparison data
            if st.button("💾 Download Comparison Data"):
                csv = df.to_csv()
                st.download_button(
                    "Download CSV",
                    csv,
                    "agent_comparison.csv",
                    "text/csv",
                    width="stretch",
                )
        else:
            st.info("No comparison data available.")


def main():
    """Main application"""
    initialize_session_state()

    # Sidebar
    st.sidebar.title("🎮 2048 Game")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigation",
        ["🎮 Play Manually", "🤖 Watch Agents", "⚔️ Compare Agents", "📊 Analytics"],
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        """
        **2048 Game with AI Agents**

        Play the classic 2048 game or watch different AI agents play.
        Compare strategies and analyze performance!

        **Features:**
        - Manual gameplay
        - Multiple AI agents
        - Performance analytics
        - Visual comparisons
        """
    )

    # Route to appropriate page
    if page == "🎮 Play Manually":
        manual_play_page()
    elif page == "🤖 Watch Agents":
        watch_agent_page()
    elif page == "⚔️ Compare Agents":
        compare_agents_page()
    elif page == "📊 Analytics":
        analytics_page()


if __name__ == "__main__":
    main()
