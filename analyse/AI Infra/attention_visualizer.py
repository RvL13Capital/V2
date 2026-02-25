"""
Attention Visualization Module
Visualizes attention weights from LSTM model to understand which days are important
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import logging

# Try plotly for interactive visualizations
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Plotly not available. Install with: pip install plotly")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AttentionVisualizer:
    """
    Visualize attention weights from LSTM model

    Shows which days in the 30-day window the model focuses on
    when making predictions.
    """

    def __init__(self, style: str = 'seaborn-v0_8-darkgrid'):
        """
        Initialize visualizer

        Args:
            style: Matplotlib style
        """
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')

        self.figsize_default = (14, 6)
        self.dpi = 100

    def plot_single_attention(self,
                             attention_weights: np.ndarray,
                             dates: Optional[List] = None,
                             prediction: Optional[float] = None,
                             actual_label: Optional[int] = None,
                             title: str = "Attention Weights Over Time",
                             save_path: Optional[str] = None,
                             show: bool = True) -> plt.Figure:
        """
        Visualize attention weights for a single prediction

        Args:
            attention_weights: Attention weights (window_size,)
            dates: Optional list of dates
            prediction: Model prediction probability
            actual_label: Actual label (0 or 1)
            title: Plot title
            save_path: Path to save figure
            show: Whether to display figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize_default, dpi=self.dpi)

        # Prepare x-axis
        if dates is not None:
            x = list(range(len(dates)))
            x_labels = [d.strftime('%Y-%m-%d') if isinstance(d, datetime) else str(d)
                       for d in dates]
        else:
            x = list(range(len(attention_weights)))
            x_labels = [f"Day {i}" for i in x]

        # Plot attention weights as bars
        colors = plt.cm.RdYlGn(attention_weights / attention_weights.max())
        bars = ax.bar(x, attention_weights, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)

        # Highlight top-k most important days
        top_k = 5
        top_indices = np.argsort(attention_weights)[-top_k:]
        for idx in top_indices:
            ax.bar(x[idx], attention_weights[idx], color='red', alpha=0.3, edgecolor='red', linewidth=2)

        # Add value labels on bars
        for i, (xi, weight) in enumerate(zip(x, attention_weights)):
            if i in top_indices:
                ax.text(xi, weight + 0.001, f'{weight:.3f}',
                       ha='center', va='bottom', fontsize=8, fontweight='bold')

        # Styling
        ax.set_xlabel('Day in Sequence', fontsize=12, fontweight='bold')
        ax.set_ylabel('Attention Weight', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

        # X-axis labels
        ax.set_xticks(x[::max(1, len(x)//15)])  # Show ~15 labels max
        ax.set_xticklabels(x_labels[::max(1, len(x)//15)], rotation=45, ha='right')

        # Grid
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

        # Add prediction info if available
        info_text = []
        if prediction is not None:
            pred_class = 'WINNER' if prediction > 0.5 else 'NOT WINNER'
            confidence = abs(prediction - 0.5) * 2 * 100
            info_text.append(f'Prediction: {pred_class} ({confidence:.1f}% confidence)')

        if actual_label is not None:
            actual_class = 'WINNER' if actual_label == 1 else 'NOT WINNER'
            info_text.append(f'Actual: {actual_class}')

        if info_text:
            ax.text(0.02, 0.98, '\n'.join(info_text),
                   transform=ax.transAxes,
                   va='top', ha='left',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                   fontsize=10, fontweight='bold')

        # Add mean attention line
        mean_attention = attention_weights.mean()
        ax.axhline(y=mean_attention, color='blue', linestyle='--', linewidth=2,
                  label=f'Mean: {mean_attention:.4f}')
        ax.legend(loc='upper right')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Figure saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return fig

    def plot_multiple_attentions(self,
                                attention_matrix: np.ndarray,
                                n_samples: int = 10,
                                dates: Optional[List] = None,
                                predictions: Optional[np.ndarray] = None,
                                labels: Optional[np.ndarray] = None,
                                title: str = "Attention Patterns Across Multiple Predictions",
                                save_path: Optional[str] = None,
                                show: bool = True) -> plt.Figure:
        """
        Visualize attention weights for multiple predictions as heatmap

        Args:
            attention_matrix: Attention weights (n_samples, window_size)
            n_samples: Number of samples to show
            dates: Optional list of dates
            predictions: Optional predictions array
            labels: Optional labels array
            title: Plot title
            save_path: Path to save figure
            show: Whether to display figure

        Returns:
            Matplotlib figure
        """
        # Select random samples if more than n_samples
        if len(attention_matrix) > n_samples:
            indices = np.random.choice(len(attention_matrix), n_samples, replace=False)
            attention_data = attention_matrix[indices]
            pred_data = predictions[indices] if predictions is not None else None
            label_data = labels[indices] if labels is not None else None
        else:
            attention_data = attention_matrix
            pred_data = predictions
            label_data = labels

        fig, ax = plt.subplots(figsize=(16, max(8, n_samples * 0.6)), dpi=self.dpi)

        # Create heatmap
        im = ax.imshow(attention_data, aspect='auto', cmap='RdYlGn', interpolation='nearest')

        # Set ticks
        ax.set_yticks(range(len(attention_data)))
        y_labels = []
        for i in range(len(attention_data)):
            label_parts = [f"Sample {i}"]
            if pred_data is not None:
                pred_class = 'WIN' if pred_data[i] > 0.5 else 'LOSE'
                label_parts.append(f"Pred: {pred_class}")
            if label_data is not None:
                actual_class = 'WIN' if label_data[i] == 1 else 'LOSE'
                label_parts.append(f"Actual: {actual_class}")
            y_labels.append(" | ".join(label_parts))

        ax.set_yticklabels(y_labels, fontsize=9)

        # X-axis
        window_size = attention_data.shape[1]
        if dates and len(dates[0]) == window_size:
            x_labels = [dates[0][i].strftime('%m-%d') if i % 3 == 0 else ''
                       for i in range(window_size)]
        else:
            x_labels = [f"D{i}" if i % 3 == 0 else '' for i in range(window_size)]

        ax.set_xticks(range(window_size))
        ax.set_xticklabels(x_labels, rotation=45, ha='right')

        # Labels
        ax.set_xlabel('Day in Sequence', fontsize=12, fontweight='bold')
        ax.set_ylabel('Sample', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Attention Weight', rotation=270, labelpad=20, fontsize=11, fontweight='bold')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Figure saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return fig

    def plot_attention_distribution(self,
                                   attention_weights: np.ndarray,
                                   title: str = "Attention Weight Distribution",
                                   save_path: Optional[str] = None,
                                   show: bool = True) -> plt.Figure:
        """
        Plot distribution of attention weights across time steps

        Args:
            attention_weights: Attention weights (window_size,) or (n_samples, window_size)
            title: Plot title
            save_path: Path to save figure
            show: Whether to display figure

        Returns:
            Matplotlib figure
        """
        if len(attention_weights.shape) == 1:
            # Single sample
            weights = attention_weights
        else:
            # Multiple samples - compute mean and std
            weights = attention_weights.mean(axis=0)
            std = attention_weights.std(axis=0)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5), dpi=self.dpi)

        # Plot 1: Line plot with confidence bands
        x = list(range(len(weights)))
        ax1.plot(x, weights, 'b-', linewidth=2, label='Mean Attention')
        if len(attention_weights.shape) > 1:
            ax1.fill_between(x, weights - std, weights + std, alpha=0.3, label='±1 Std Dev')

        ax1.set_xlabel('Day in Sequence', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Attention Weight', fontsize=12, fontweight='bold')
        ax1.set_title(f'{title} - Time Series', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Plot 2: Box plot by time period
        if len(attention_weights.shape) > 1:
            # Group days into early, mid, late periods
            third = len(weights) // 3
            periods = {
                'Early (Days 0-{})'.format(third-1): attention_weights[:, :third].flatten(),
                'Mid (Days {}-{})'.format(third, 2*third-1): attention_weights[:, third:2*third].flatten(),
                'Late (Days {}-{})'.format(2*third, len(weights)-1): attention_weights[:, 2*third:].flatten()
            }

            bp = ax2.boxplot([periods[k] for k in periods.keys()],
                            labels=list(periods.keys()),
                            patch_artist=True)

            # Color boxes
            colors = ['lightblue', 'lightgreen', 'lightcoral']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)

            ax2.set_ylabel('Attention Weight', fontsize=12, fontweight='bold')
            ax2.set_title(f'{title} - By Period', fontsize=13, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')
        else:
            # Single sample - histogram
            ax2.hist(weights, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
            ax2.set_xlabel('Attention Weight', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
            ax2.set_title(f'{title} - Histogram', fontsize=13, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Figure saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return fig

    def plot_attention_with_features(self,
                                    attention_weights: np.ndarray,
                                    feature_data: np.ndarray,
                                    feature_names: List[str],
                                    dates: Optional[List] = None,
                                    top_k_features: int = 5,
                                    title: str = "Attention Weights with Feature Values",
                                    save_path: Optional[str] = None,
                                    show: bool = True) -> plt.Figure:
        """
        Visualize attention weights alongside feature values

        Shows which days are important AND what the feature values were

        Args:
            attention_weights: Attention weights (window_size,)
            feature_data: Feature values (window_size, n_features)
            feature_names: List of feature names
            dates: Optional list of dates
            top_k_features: Number of top features to show
            title: Plot title
            save_path: Path to save figure
            show: Whether to display figure

        Returns:
            Matplotlib figure
        """
        n_days = len(attention_weights)

        # Select top-k most important features
        feature_importance = np.abs(feature_data).mean(axis=0)
        top_features_idx = np.argsort(feature_importance)[-top_k_features:]

        fig, axes = plt.subplots(top_k_features + 1, 1,
                                figsize=(14, 3 + 2*top_k_features),
                                dpi=self.dpi)

        # Prepare x-axis
        if dates is not None:
            x = list(range(len(dates)))
            x_labels = [d.strftime('%m-%d') if isinstance(d, datetime) else str(d)
                       for d in dates]
        else:
            x = list(range(n_days))
            x_labels = [f"D{i}" for i in x]

        # Plot attention weights (top subplot)
        ax = axes[0]
        colors = plt.cm.RdYlGn(attention_weights / attention_weights.max())
        ax.bar(x, attention_weights, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.set_ylabel('Attention\nWeight', fontsize=10, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xticklabels([])

        # Plot top features (subsequent subplots)
        for i, feat_idx in enumerate(top_features_idx):
            ax = axes[i + 1]
            feature_values = feature_data[:, feat_idx]

            # Normalize for visualization
            if feature_values.std() > 0:
                feature_norm = (feature_values - feature_values.mean()) / feature_values.std()
            else:
                feature_norm = feature_values

            # Plot as line
            ax.plot(x, feature_norm, 'o-', linewidth=2, markersize=4,
                   label=feature_names[feat_idx])

            # Highlight high-attention days
            top_attention_days = np.argsort(attention_weights)[-3:]
            ax.scatter([x[j] for j in top_attention_days],
                      [feature_norm[j] for j in top_attention_days],
                      color='red', s=100, zorder=5, alpha=0.6)

            ax.set_ylabel(feature_names[feat_idx][:15] + '...' if len(feature_names[feat_idx]) > 15
                         else feature_names[feat_idx],
                         fontsize=9, fontweight='bold')
            ax.grid(True, alpha=0.3)

            if i < len(top_features_idx) - 1:
                ax.set_xticklabels([])

        # X-axis labels on bottom subplot only
        axes[-1].set_xticks(x[::max(1, len(x)//15)])
        axes[-1].set_xticklabels(x_labels[::max(1, len(x)//15)], rotation=45, ha='right')
        axes[-1].set_xlabel('Day in Sequence', fontsize=11, fontweight='bold')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Figure saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return fig

    def create_interactive_attention_plot(self,
                                         attention_weights: np.ndarray,
                                         dates: Optional[List] = None,
                                         prediction: Optional[float] = None,
                                         feature_data: Optional[np.ndarray] = None,
                                         feature_names: Optional[List[str]] = None,
                                         title: str = "Interactive Attention Visualization",
                                         save_path: Optional[str] = None) -> Optional[go.Figure]:
        """
        Create interactive plotly visualization

        Args:
            attention_weights: Attention weights (window_size,)
            dates: Optional list of dates
            prediction: Model prediction
            feature_data: Optional feature values (window_size, n_features)
            feature_names: Optional feature names
            title: Plot title
            save_path: Path to save HTML file

        Returns:
            Plotly figure or None if plotly not available
        """
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available. Install with: pip install plotly")
            return None

        # Prepare x-axis
        if dates is not None:
            x = [d.strftime('%Y-%m-%d') if isinstance(d, datetime) else str(d)
                for d in dates]
        else:
            x = [f"Day {i}" for i in range(len(attention_weights))]

        # Create figure with subplots
        if feature_data is not None and feature_names is not None:
            # 2 subplots: attention + features
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Attention Weights', 'Top Feature Values'),
                row_heights=[0.5, 0.5],
                vertical_spacing=0.12
            )
        else:
            # Single plot: attention only
            fig = go.Figure()

        # Add attention weights as bar chart
        colors = ['rgb({},{},{})'.format(
            int(255 * (1 - w/attention_weights.max())),
            int(255 * w/attention_weights.max()),
            50
        ) for w in attention_weights]

        attention_trace = go.Bar(
            x=x,
            y=attention_weights,
            marker=dict(color=colors, line=dict(color='black', width=1)),
            name='Attention Weight',
            hovertemplate='%{x}<br>Weight: %{y:.4f}<extra></extra>'
        )

        if feature_data is not None:
            fig.add_trace(attention_trace, row=1, col=1)
        else:
            fig.add_trace(attention_trace)

        # Add mean line
        mean_attention = attention_weights.mean()
        fig.add_hline(
            y=mean_attention,
            line_dash='dash',
            line_color='blue',
            annotation_text=f'Mean: {mean_attention:.4f}',
            row=1 if feature_data is not None else None,
            col=1 if feature_data is not None else None
        )

        # Add feature data if provided
        if feature_data is not None and feature_names is not None:
            # Show top 3 most important features
            feature_importance = np.abs(feature_data).mean(axis=0)
            top_features_idx = np.argsort(feature_importance)[-3:]

            for idx in top_features_idx:
                feature_values = feature_data[:, idx]
                # Normalize
                if feature_values.std() > 0:
                    feature_norm = (feature_values - feature_values.mean()) / feature_values.std()
                else:
                    feature_norm = feature_values

                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=feature_norm,
                        mode='lines+markers',
                        name=feature_names[idx][:30],
                        hovertemplate='%{x}<br>Value: %{y:.3f}<extra></extra>'
                    ),
                    row=2, col=1
                )

        # Update layout
        fig.update_layout(
            title_text=title,
            title_font_size=16,
            showlegend=True,
            height=600 if feature_data is not None else 400,
            hovermode='x unified'
        )

        if feature_data is not None:
            fig.update_xaxes(title_text="Day", row=2, col=1)
            fig.update_yaxes(title_text="Attention Weight", row=1, col=1)
            fig.update_yaxes(title_text="Normalized Value", row=2, col=1)
        else:
            fig.update_xaxes(title_text="Day")
            fig.update_yaxes(title_text="Attention Weight")

        # Add prediction info as annotation
        if prediction is not None:
            pred_class = 'WINNER' if prediction > 0.5 else 'NOT WINNER'
            confidence = abs(prediction - 0.5) * 2 * 100
            fig.add_annotation(
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                text=f"<b>Prediction:</b> {pred_class}<br><b>Confidence:</b> {confidence:.1f}%",
                showarrow=False,
                bgcolor="wheat",
                bordercolor="black",
                borderwidth=1
            )

        if save_path:
            fig.write_html(save_path)
            logger.info(f"Interactive plot saved to {save_path}")

        return fig


if __name__ == "__main__":
    logger.info("Attention Visualizer Module")
    logger.info("\nFeatures:")
    logger.info("  • Single attention weight visualization")
    logger.info("  • Multiple predictions heatmap")
    logger.info("  • Attention distribution analysis")
    logger.info("  • Attention + feature values overlay")
    logger.info("  • Interactive plotly visualizations")
    logger.info("\nReady to visualize attention patterns!")
