"""
Streamlit App for DQN Metrics Visualization
Simple drag-and-drop interface for analyzing training metrics
"""

import streamlit as st
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

st.set_page_config(page_title="DQN Metrics Viewer", page_icon="📊", layout="wide")

st.title("📊 DQN Training Metrics Viewer")
st.markdown("Upload your `metrics_final.json` file to visualize training progress")

# File uploader
uploaded_file = st.file_uploader(
    "Choose a metrics file",
    type=["json"],
    help="Upload metrics_final.json from your training run",
)


def smooth_data(data, window_size=50):
    """Apply moving average smoothing"""
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size) / window_size, mode="valid")


def load_metrics(file):
    """Load metrics from JSON file"""
    try:
        content = file.read()
        metrics = json.loads(content)
        return metrics
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None


if uploaded_file is not None:
    metrics = load_metrics(uploaded_file)

    if metrics:
        # Show basic stats
        st.success(f"✅ Loaded metrics for {len(metrics['episodes'])} episodes")

        # Summary statistics
        st.subheader("📈 Summary Statistics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Episodes", len(metrics["episodes"]))
            st.metric("Avg Score", f"{np.mean(metrics['scores']):.0f}")

        with col2:
            st.metric("Best Score", int(np.max(metrics["scores"])))
            st.metric("Worst Score", int(np.min(metrics["scores"])))

        with col3:
            st.metric("Avg Max Tile", f"{np.mean(metrics['max_tiles']):.0f}")
            st.metric("Best Tile", int(np.max(metrics["max_tiles"])))

        with col4:
            reached_512 = sum(1 for t in metrics["max_tiles"] if t >= 512)
            reached_1024 = sum(1 for t in metrics["max_tiles"] if t >= 1024)
            reached_2048 = sum(1 for t in metrics["max_tiles"] if t >= 2048)
            st.metric("Reached 512", f"{reached_512}x")
            st.metric("Reached 1024", f"{reached_1024}x")

        # Smoothing slider
        st.subheader("⚙️ Visualization Settings")
        smooth_window = st.slider(
            "Smoothing Window (episodes)",
            min_value=1,
            max_value=min(200, len(metrics["episodes"]) // 2),
            value=min(50, len(metrics["episodes"]) // 4),
            help="Higher values = smoother curves",
        )

        # Main metrics plots
        st.subheader("📊 Training Progress")

        # Create 2x2 subplot
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Score Over Time",
                "Max Tile Progress",
                "Episode Length",
                "Epsilon Decay",
            ),
            vertical_spacing=0.12,
            horizontal_spacing=0.1,
        )

        episodes = metrics["episodes"]

        # 1. Scores
        scores = metrics["scores"]
        fig.add_trace(
            go.Scatter(
                x=episodes,
                y=scores,
                mode="lines",
                name="Score (raw)",
                line=dict(color="lightblue", width=1),
                opacity=0.3,
            ),
            row=1,
            col=1,
        )

        if len(scores) >= smooth_window:
            smoothed_scores = smooth_data(scores, smooth_window)
            fig.add_trace(
                go.Scatter(
                    x=episodes[smooth_window - 1 :],
                    y=smoothed_scores,
                    mode="lines",
                    name="Score (smoothed)",
                    line=dict(color="blue", width=2),
                ),
                row=1,
                col=1,
            )

        fig.update_xaxes(title_text="Episode", row=1, col=1)
        fig.update_yaxes(title_text="Score", row=1, col=1)

        # 2. Max Tiles
        max_tiles = metrics["max_tiles"]
        fig.add_trace(
            go.Scatter(
                x=episodes,
                y=max_tiles,
                mode="lines",
                name="Max Tile (raw)",
                line=dict(color="lightgreen", width=1),
                opacity=0.3,
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        if len(max_tiles) >= smooth_window:
            smoothed_tiles = smooth_data(max_tiles, smooth_window)
            fig.add_trace(
                go.Scatter(
                    x=episodes[smooth_window - 1 :],
                    y=smoothed_tiles,
                    mode="lines",
                    name="Max Tile (smoothed)",
                    line=dict(color="green", width=2),
                    showlegend=False,
                ),
                row=1,
                col=2,
            )

        fig.update_xaxes(title_text="Episode", row=1, col=2)
        fig.update_yaxes(title_text="Max Tile", row=1, col=2)

        # 3. Episode Lengths
        if "episode_lengths" in metrics and metrics["episode_lengths"]:
            lengths = metrics["episode_lengths"]
            fig.add_trace(
                go.Scatter(
                    x=episodes,
                    y=lengths,
                    mode="lines",
                    name="Length (raw)",
                    line=dict(color="lightsalmon", width=1),
                    opacity=0.3,
                    showlegend=False,
                ),
                row=2,
                col=1,
            )

            if len(lengths) >= smooth_window:
                smoothed_lengths = smooth_data(lengths, smooth_window)
                fig.add_trace(
                    go.Scatter(
                        x=episodes[smooth_window - 1 :],
                        y=smoothed_lengths,
                        mode="lines",
                        name="Length (smoothed)",
                        line=dict(color="red", width=2),
                        showlegend=False,
                    ),
                    row=2,
                    col=1,
                )

        fig.update_xaxes(title_text="Episode", row=2, col=1)
        fig.update_yaxes(title_text="Moves per Game", row=2, col=1)

        # 4. Epsilon
        if "epsilons" in metrics and metrics["epsilons"]:
            epsilons = metrics["epsilons"]
            fig.add_trace(
                go.Scatter(
                    x=episodes,
                    y=epsilons,
                    mode="lines",
                    name="Epsilon",
                    line=dict(color="purple", width=2),
                    showlegend=False,
                ),
                row=2,
                col=2,
            )

        fig.update_xaxes(title_text="Episode", row=2, col=2)
        fig.update_yaxes(title_text="Epsilon", row=2, col=2)

        # Update layout
        fig.update_layout(
            height=700,
            showlegend=True,
            legend=dict(x=0.01, y=0.99),
            hovermode="x unified",
        )

        st.plotly_chart(fig, width="stretch")

        # Training metrics (Loss and Q-values)
        if "losses" in metrics and metrics["losses"] and len(metrics["losses"]) > 0:
            st.subheader("🎓 Training Metrics")

            fig2 = make_subplots(
                rows=1, cols=2, subplot_titles=("Training Loss", "Average Q-Values")
            )

            # Loss
            losses = metrics["losses"]
            training_steps = list(range(len(losses)))

            fig2.add_trace(
                go.Scatter(
                    x=training_steps,
                    y=losses,
                    mode="lines",
                    name="Loss",
                    line=dict(color="orange", width=1),
                    opacity=0.6,
                ),
                row=1,
                col=1,
            )

            if len(losses) >= smooth_window:
                smoothed_loss = smooth_data(losses, smooth_window)
                fig2.add_trace(
                    go.Scatter(
                        x=training_steps[smooth_window - 1 :],
                        y=smoothed_loss,
                        mode="lines",
                        name="Loss (smoothed)",
                        line=dict(color="darkorange", width=2),
                    ),
                    row=1,
                    col=1,
                )

            fig2.update_xaxes(title_text="Training Step", row=1, col=1)
            fig2.update_yaxes(title_text="Loss", type="log", row=1, col=1)

            # Q-values
            if "avg_q_values" in metrics and metrics["avg_q_values"]:
                q_values = metrics["avg_q_values"]

                fig2.add_trace(
                    go.Scatter(
                        x=training_steps[: len(q_values)],
                        y=q_values,
                        mode="lines",
                        name="Q-Value",
                        line=dict(color="teal", width=1),
                        opacity=0.6,
                        showlegend=False,
                    ),
                    row=1,
                    col=2,
                )

                if len(q_values) >= smooth_window:
                    smoothed_q = smooth_data(q_values, smooth_window)
                    fig2.add_trace(
                        go.Scatter(
                            x=training_steps[smooth_window - 1 : len(q_values)],
                            y=smoothed_q,
                            mode="lines",
                            name="Q-Value (smoothed)",
                            line=dict(color="darkcyan", width=2),
                            showlegend=False,
                        ),
                        row=1,
                        col=2,
                    )

            fig2.update_xaxes(title_text="Training Step", row=1, col=2)
            fig2.update_yaxes(title_text="Avg Q-Value", row=1, col=2)

            fig2.update_layout(height=400, hovermode="x unified")
            st.plotly_chart(fig2, width="stretch")

        # Distribution plots
        st.subheader("📊 Distributions")

        col1, col2 = st.columns(2)

        with col1:
            # Score distribution
            fig_dist1 = go.Figure()
            fig_dist1.add_trace(
                go.Histogram(
                    x=metrics["scores"],
                    nbinsx=50,
                    name="Score Distribution",
                    marker_color="blue",
                    opacity=0.7,
                )
            )
            fig_dist1.add_vline(
                x=np.mean(metrics["scores"]),
                line_dash="dash",
                line_color="red",
                annotation_text=f"Mean: {np.mean(metrics['scores']):.0f}",
            )
            fig_dist1.update_layout(
                title="Score Distribution",
                xaxis_title="Score",
                yaxis_title="Count",
                height=400,
            )
            st.plotly_chart(fig_dist1, width="stretch")

        with col2:
            # Max tile distribution
            from collections import Counter

            tile_counts = Counter(metrics["max_tiles"])
            tiles = sorted(tile_counts.keys())
            counts = [tile_counts[t] for t in tiles]

            fig_dist2 = go.Figure()
            fig_dist2.add_trace(
                go.Bar(
                    x=[str(t) for t in tiles],
                    y=counts,
                    name="Max Tile Distribution",
                    marker_color="green",
                    text=counts,
                    textposition="auto",
                )
            )
            fig_dist2.update_layout(
                title="Max Tile Distribution",
                xaxis_title="Max Tile",
                yaxis_title="Count",
                height=400,
            )
            st.plotly_chart(fig_dist2, width="stretch")

        # Progress over time (binned)
        st.subheader("📈 Progress Analysis")

        # Bin episodes into chunks
        num_bins = min(10, len(episodes) // 100)
        if num_bins > 0:
            bin_size = len(episodes) // num_bins

            bin_labels = []
            bin_avg_scores = []
            bin_avg_tiles = []

            for i in range(num_bins):
                start_idx = i * bin_size
                end_idx = (i + 1) * bin_size if i < num_bins - 1 else len(episodes)

                bin_labels.append(f"{episodes[start_idx]}-{episodes[end_idx-1]}")
                bin_avg_scores.append(np.mean(scores[start_idx:end_idx]))
                bin_avg_tiles.append(np.mean(max_tiles[start_idx:end_idx]))

            fig3 = make_subplots(
                rows=1,
                cols=2,
                subplot_titles=("Average Score per Bin", "Average Max Tile per Bin"),
            )

            fig3.add_trace(
                go.Bar(
                    x=bin_labels,
                    y=bin_avg_scores,
                    name="Avg Score",
                    marker_color="blue",
                    text=[f"{s:.0f}" for s in bin_avg_scores],
                    textposition="auto",
                ),
                row=1,
                col=1,
            )

            fig3.add_trace(
                go.Bar(
                    x=bin_labels,
                    y=bin_avg_tiles,
                    name="Avg Max Tile",
                    marker_color="green",
                    text=[f"{t:.0f}" for t in bin_avg_tiles],
                    textposition="auto",
                    showlegend=False,
                ),
                row=1,
                col=2,
            )

            fig3.update_xaxes(title_text="Episode Range", row=1, col=1)
            fig3.update_yaxes(title_text="Avg Score", row=1, col=1)
            fig3.update_xaxes(title_text="Episode Range", row=1, col=2)
            fig3.update_yaxes(title_text="Avg Max Tile", row=1, col=2)

            fig3.update_layout(height=400)
            st.plotly_chart(fig3, width="stretch")

        # Raw data viewer (expandable)
        with st.expander("🔍 View Raw Data"):
            st.json(metrics, expanded=False)

else:
    st.info("👆 Upload a metrics_final.json file to get started")

    # Show example
    st.markdown("---")
    st.markdown("### 📁 Example File Structure")
    st.code(
        """
{
  "episodes": [0, 1, 2, ..., 999],
  "scores": [1200, 1450, 890, ...],
  "max_tiles": [128, 128, 64, ...],
  "losses": [0.145, 0.132, ...],
  "epsilons": [1.0, 0.999, ...],
  "avg_q_values": [2.3, 2.5, ...],
  "episode_lengths": [120, 145, ...]
}
    """,
        language="json",
    )

    st.markdown("### 💡 Tips")
    st.markdown(
        """
    - Upload your `metrics_final.json` from the models directory
    - Use the smoothing slider to reduce noise in the plots
    - Hover over plots to see detailed values
    - Look for upward trends in Score and Max Tile
    - Loss should decrease over time
    - Q-values should increase as the agent learns
    """
    )

# Footer
st.markdown("---")
st.markdown("🎮 **DQN Metrics Viewer** | Made for 2048 RL Training Analysis")
