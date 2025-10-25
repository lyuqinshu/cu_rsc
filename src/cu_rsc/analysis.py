"""
GPU version of get_n_distribution with per-axis plot control
------------------------------------------------------------
Compute and optionally plot the vibrational quantum number distributions (n_x, n_y, n_z)
from a CuPy (N,6) molecule array:
    [nx, ny, nz, state, spin, is_lost]

Plots can be selectively enabled for each axis with `plot=[bool_x, bool_y, bool_z]`.
"""

from __future__ import annotations
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from typing import Sequence


def get_n_distribution_gpu(
    mol_dev: cp.ndarray,
    plot: Sequence[bool] = (True, True, True),
    scatter: bool = True,
    max_bins: int = 50,
):
    """
    Parameters
    ----------
    mol_dev : cupy.ndarray
        (N,6) array on device: [nx, ny, nz, state, spin, is_lost]
    plot : Sequence[bool], optional
        A list or tuple of 3 booleans [plot_x, plot_y, plot_z].
        Each entry controls whether the histogram for that axis is plotted.
        Default: (True, True, True)
    scatter : bool, optional
        If True, plot a 3D scatter plot of (nx, ny, nz) (default: True).
    max_bins : int
        Maximum number of tick labels for each axis.

    Returns
    -------
    counts_x, counts_y, counts_z : dict[int, int]
        Frequency counts for each vibrational number (on CPU).
    """
    if mol_dev.ndim != 2 or mol_dev.shape[1] != 6:
        raise ValueError("mol_dev must be shape (N,6)")

    if not isinstance(plot, (list, tuple)) or len(plot) != 3:
        raise ValueError("plot must be a list or tuple of 3 booleans [x, y, z]")

    # Mask for survivors: spin == 0, state == +1, not lost
    spin = mol_dev[:, 4]
    state = mol_dev[:, 3]
    lost = mol_dev[:, 5]
    valid_mask = (spin == 0) & (state == 1) & (lost == 0)

    # Extract vibrational numbers on GPU
    n_xyz = mol_dev[valid_mask, 0:3]
    mol_num = int(n_xyz.shape[0])

    # Move to host for counting/plotting
    n_xyz_cpu = cp.asnumpy(n_xyz).astype(int)
    n_x, n_y, n_z = n_xyz_cpu[:, 0], n_xyz_cpu[:, 1], n_xyz_cpu[:, 2]

    # Frequency counts
    counts_x = Counter(n_x)
    counts_y = Counter(n_y)
    counts_z = Counter(n_z)

    # --- Plotting ---
    any_plot = any(plot)
    if any_plot and mol_num > 0:
        fig, axes = plt.subplots(1, sum(plot), figsize=(5 * sum(plot), 5), sharey=True)
        if sum(plot) == 1:
            axes = [axes]

        plot_axes = ['X', 'Y', 'Z']
        colors = ['salmon', 'mediumseagreen', 'cornflowerblue']
        count_dicts = [counts_x, counts_y, counts_z]
        for i, do_plot in enumerate(plot):
            if not do_plot:
                continue
            ax = axes[0] if sum(plot) == 1 else axes[plot[:i].count(True)]
            counts = count_dicts[i]
            ax.bar(counts.keys(), counts.values(),
                   color=colors[i], edgecolor='black')
            ax.set_title(f"n Distribution ({plot_axes[i]} axis)")
            max_n = max(counts.keys()) if counts else 0
            ax.set_xticks(np.linspace(0, max_n, min(max_n+1, max_bins), dtype=int))
            ax.set_xlabel("n")
            ax.grid(True, linestyle="--", alpha=0.5)
        axes[0].set_ylabel("Count")
        fig.suptitle(f"{mol_num} molecules survived")
        plt.tight_layout()
        plt.show()

    if scatter and mol_num > 0:
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(n_x, n_y, n_z, c="purple", alpha=0.7, edgecolor="k", s=10)
        ax.set_xlabel("n_x")
        ax.set_ylabel("n_y")
        ax.set_zlabel("n_z")
        ax.set_title(f"3D Scatter ({mol_num} molecules)")
        plt.show()

    return dict(counts_x), dict(counts_y), dict(counts_z)
