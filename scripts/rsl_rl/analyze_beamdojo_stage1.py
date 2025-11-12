#!/usr/bin/env python3
"""
BeamDojo Stage 1 Training Analysis and Visualization

Analyzes training metrics from TensorBoard logs and creates comprehensive visualizations.
Automatically finds the latest checkpoint and training run.

Usage:
    python analyze_beamdojo_stage1.py --log_dir logs/rsl_rl/beamdojo_stage1
    python analyze_beamdojo_stage1.py --log_dir logs/rsl_rl/beamdojo_stage1 --run_name 2025-01-15_10-30-45
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator


def parse_tf_logs(log_dir: str) -> dict:
    """Parse TensorBoard event files and return log data as dictionary.
    
    Args:
        log_dir: Directory containing TensorBoard event files
        
    Returns:
        Dictionary with metric names as keys and lists of values as values
    """
    # Search for event files
    list_of_files = glob.glob(os.path.join(log_dir, "events*"))
    if not list_of_files:
        raise FileNotFoundError(f"No TensorBoard event files found in {log_dir}")
    
    # Get the latest event file
    latest_file = max(list_of_files, key=os.path.getctime)
    print(f"[INFO] Loading TensorBoard logs from: {latest_file}")
    
    # Parse event file
    log_data = {}
    ea = event_accumulator.EventAccumulator(latest_file)
    ea.Reload()
    
    tags = ea.Tags()["scalars"]
    for tag in tags:
        log_data[tag] = []
        for event in ea.Scalars(tag):
            log_data[tag].append((event.step, event.value))
    
    return log_data


def find_latest_checkpoint(log_dir: str) -> str | None:
    """Find the latest checkpoint file in the log directory.
    
    Args:
        log_dir: Directory containing checkpoints
        
    Returns:
        Path to latest checkpoint or None if not found
    """
    checkpoint_pattern = os.path.join(log_dir, "model_*.pt")
    checkpoints = glob.glob(checkpoint_pattern)
    
    if not checkpoints:
        return None
    
    # Sort by iteration number (extract from filename)
    def get_iteration(path):
        filename = os.path.basename(path)
        # Extract number from "model_<number>.pt"
        try:
            return int(filename.replace("model_", "").replace(".pt", ""))
        except ValueError:
            return 0
    
    latest = max(checkpoints, key=get_iteration)
    return latest


def plot_training_overview(log_data: dict, save_path: str | None = None):
    """Create overview plot with main training metrics.
    
    Args:
        log_data: Dictionary of training metrics
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("BeamDojo Stage 1 Training Overview", fontsize=16, fontweight="bold")
    
    # Plot 1: Mean Episode Reward
    if "Train/mean_reward" in log_data:
        steps, values = zip(*log_data["Train/mean_reward"])
        axes[0, 0].plot(steps, values, "b-", linewidth=2, alpha=0.8, label="Mean Reward")
        # Add moving average
        if len(values) > 10:
            window = min(50, len(values) // 10)
            ma = np.convolve(values, np.ones(window) / window, mode="valid")
            ma_steps = steps[window - 1 :]
            axes[0, 0].plot(ma_steps, ma, "r--", linewidth=2, alpha=0.7, label=f"MA({window})")
        axes[0, 0].set_title("Episode Reward", fontsize=12, fontweight="bold")
        axes[0, 0].set_xlabel("Iteration")
        axes[0, 0].set_ylabel("Reward")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
    
    # Plot 2: Mean Episode Length
    if "Train/mean_episode_length" in log_data:
        steps, values = zip(*log_data["Train/mean_episode_length"])
        axes[0, 1].plot(steps, values, "g-", linewidth=2, alpha=0.8)
        axes[0, 1].set_title("Episode Length", fontsize=12, fontweight="bold")
        axes[0, 1].set_xlabel("Iteration")
        axes[0, 1].set_ylabel("Steps")
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Value Function Loss
    if "Loss/value_function" in log_data:
        steps, values = zip(*log_data["Loss/value_function"])
        axes[0, 2].plot(steps, values, "r-", linewidth=2, alpha=0.8)
        axes[0, 2].set_title("Value Function Loss", fontsize=12, fontweight="bold")
        axes[0, 2].set_xlabel("Iteration")
        axes[0, 2].set_ylabel("Loss")
        axes[0, 2].set_yscale("log")
        axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Surrogate Loss (Policy Loss)
    if "Loss/surrogate" in log_data:
        steps, values = zip(*log_data["Loss/surrogate"])
        axes[1, 0].plot(steps, values, "m-", linewidth=2, alpha=0.8)
        axes[1, 0].set_title("Policy Loss (Surrogate)", fontsize=12, fontweight="bold")
        axes[1, 0].set_xlabel("Iteration")
        axes[1, 0].set_ylabel("Loss")
        axes[1, 0].set_yscale("log")
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Learning Rate
    if "Params/learning_rate" in log_data:
        steps, values = zip(*log_data["Params/learning_rate"])
        axes[1, 1].plot(steps, values, "c-", linewidth=2, alpha=0.8)
        axes[1, 1].set_title("Learning Rate", fontsize=12, fontweight="bold")
        axes[1, 1].set_xlabel("Iteration")
        axes[1, 1].set_ylabel("LR")
        axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Policy Entropy
    if "Train/entropy_loss" in log_data:
        steps, values = zip(*log_data["Train/entropy_loss"])
        axes[1, 2].plot(steps, values, "orange", linewidth=2, alpha=0.8)
        axes[1, 2].set_title("Policy Entropy", fontsize=12, fontweight="bold")
        axes[1, 2].set_xlabel("Iteration")
        axes[1, 2].set_ylabel("Entropy")
        axes[1, 2].grid(True, alpha=0.3)
    elif "Params/entropy_coef" in log_data:
        # If entropy loss not available, show entropy coefficient
        steps, values = zip(*log_data["Params/entropy_coef"])
        axes[1, 2].plot(steps, values, "orange", linewidth=2, alpha=0.8)
        axes[1, 2].set_title("Entropy Coefficient", fontsize=12, fontweight="bold")
        axes[1, 2].set_xlabel("Iteration")
        axes[1, 2].set_ylabel("Coefficient")
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] Saved overview plot to: {save_path}")
    else:
        plt.show()


def plot_reward_components(log_data: dict, save_path: str | None = None):
    """Plot individual reward components.
    
    Args:
        log_data: Dictionary of training metrics
        save_path: Optional path to save the figure
    """
    # Find all reward-related metrics
    reward_metrics = [k for k in log_data.keys() if "Episode/" in k or "Reward/" in k]
    
    if not reward_metrics:
        print("[WARNING] No reward component metrics found")
        return
    
    # Create subplots
    n_metrics = len(reward_metrics)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle("Reward Components Breakdown", fontsize=16, fontweight="bold")
    
    for idx, metric in enumerate(reward_metrics):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        
        steps, values = zip(*log_data[metric])
        ax.plot(steps, values, linewidth=2, alpha=0.8)
        
        # Clean metric name for display
        display_name = metric.replace("Episode/", "").replace("Reward/", "")
        ax.set_title(display_name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_metrics, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        ax.axis("off")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] Saved reward components plot to: {save_path}")
    else:
        plt.show()


def plot_performance_metrics(log_data: dict, save_path: str | None = None):
    """Plot performance-related metrics (FPS, timing, etc.).
    
    Args:
        log_data: Dictionary of training metrics
        save_path: Optional path to save the figure
    """
    perf_metrics = [k for k in log_data.keys() if "Perf/" in k or "FPS" in k or "Time" in k]
    
    if not perf_metrics:
        print("[WARNING] No performance metrics found")
        return
    
    n_metrics = len(perf_metrics)
    n_cols = 2
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle("Performance Metrics", fontsize=16, fontweight="bold")
    
    for idx, metric in enumerate(perf_metrics):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        
        steps, values = zip(*log_data[metric])
        ax.plot(steps, values, linewidth=2, alpha=0.8)
        
        display_name = metric.replace("Perf/", "")
        ax.set_title(display_name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_metrics, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        ax.axis("off")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] Saved performance metrics plot to: {save_path}")
    else:
        plt.show()


def print_training_summary(log_data: dict, checkpoint_path: str | None = None):
    """Print summary statistics of training.
    
    Args:
        log_data: Dictionary of training metrics
        checkpoint_path: Optional path to latest checkpoint
    """
    print("\n" + "=" * 80)
    print("BEAMDOJO STAGE 1 TRAINING SUMMARY")
    print("=" * 80)
    
    # Key metrics
    key_metrics = {
        "Train/mean_reward": "Mean Episode Reward",
        "Train/mean_episode_length": "Mean Episode Length",
        "Loss/value_function": "Value Function Loss",
        "Loss/surrogate": "Policy Loss",
    }
    
    print("\n📊 Key Metrics (Final Values):")
    print("-" * 80)
    for metric_key, display_name in key_metrics.items():
        if metric_key in log_data:
            steps, values = zip(*log_data[metric_key])
            final_value = values[-1]
            max_value = max(values)
            min_value = min(values)
            print(f"  {display_name:30s}: {final_value:10.4f} (min: {min_value:.4f}, max: {max_value:.4f})")
    
    # Training progress
    if "Train/mean_reward" in log_data:
        steps, values = zip(*log_data["Train/mean_reward"])
        initial_reward = values[0] if len(values) > 0 else 0
        final_reward = values[-1] if len(values) > 0 else 0
        improvement = final_reward - initial_reward
        improvement_pct = (improvement / abs(initial_reward) * 100) if initial_reward != 0 else 0
        
        print(f"\n📈 Training Progress:")
        print("-" * 80)
        print(f"  Initial Reward: {initial_reward:.4f}")
        print(f"  Final Reward:   {final_reward:.4f}")
        print(f"  Improvement:    {improvement:+.4f} ({improvement_pct:+.2f}%)")
        print(f"  Total Iterations: {steps[-1] if steps else 0}")
    
    # Checkpoint info
    if checkpoint_path:
        print(f"\n💾 Latest Checkpoint:")
        print("-" * 80)
        print(f"  {checkpoint_path}")
        # Extract iteration number
        filename = os.path.basename(checkpoint_path)
        try:
            iteration = int(filename.replace("model_", "").replace(".pt", ""))
            print(f"  Iteration: {iteration}")
        except ValueError:
            pass
    
    # Available metrics
    print(f"\n📋 Available Metrics ({len(log_data)} total):")
    print("-" * 80)
    for metric in sorted(log_data.keys()):
        n_points = len(log_data[metric])
        print(f"  {metric:40s}: {n_points:5d} data points")
    
    print("\n" + "=" * 80)


def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(description="Analyze BeamDojo Stage 1 training metrics")
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs/rsl_rl/beamdojo_stage1",
        help="Base log directory (default: logs/rsl_rl/beamdojo_stage1)",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Specific run directory name (e.g., 2025-01-15_10-30-45). If not provided, uses latest.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save plots (default: same as log_dir)",
    )
    parser.add_argument(
        "--no_show",
        action="store_true",
        help="Don't display plots interactively (only save)",
    )
    
    args = parser.parse_args()
    
    # Find the specific run directory
    log_root = Path(args.log_dir)
    if args.run_name:
        run_dir = log_root / args.run_name
    else:
        # Find latest run directory
        run_dirs = sorted([d for d in log_root.iterdir() if d.is_dir()], key=os.path.getmtime, reverse=True)
        if not run_dirs:
            raise FileNotFoundError(f"No run directories found in {log_root}")
        run_dir = run_dirs[0]
        print(f"[INFO] Using latest run: {run_dir.name}")
    
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    
    # Set output directory
    output_dir = Path(args.output_dir) if args.output_dir else run_dir
    
    print(f"[INFO] Analyzing training run: {run_dir}")
    print(f"[INFO] Output directory: {output_dir}")
    
    # Parse TensorBoard logs
    try:
        log_data = parse_tf_logs(str(run_dir))
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return
    
    # Find latest checkpoint
    checkpoint_path = find_latest_checkpoint(str(run_dir))
    if checkpoint_path:
        print(f"[INFO] Latest checkpoint: {checkpoint_path}")
    
    # Print summary
    print_training_summary(log_data, checkpoint_path)
    
    # Create visualizations
    print("\n[INFO] Generating visualizations...")
    
    # Overview plot
    overview_path = output_dir / "training_overview.png"
    plot_training_overview(log_data, save_path=str(overview_path) if args.no_show or args.output_dir else None)
    
    # Reward components plot
    reward_path = output_dir / "reward_components.png"
    plot_reward_components(log_data, save_path=str(reward_path) if args.no_show or args.output_dir else None)
    
    # Performance metrics plot
    perf_path = output_dir / "performance_metrics.png"
    plot_performance_metrics(log_data, save_path=str(perf_path) if args.no_show or args.output_dir else None)
    
    print("\n[INFO] Analysis complete!")
    if args.output_dir or args.no_show:
        print(f"[INFO] Plots saved to: {output_dir}")


if __name__ == "__main__":
    main()

