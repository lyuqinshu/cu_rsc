from __future__ import annotations
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from typing import Sequence
from pathlib import Path

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



def bootstrap_stats_from_molecules(
    molecules: cp.ndarray,
    *,
    B: int = 1000,
    rng=None,
) -> dict[str, cp.ndarray]:
    """
    Compute bootstrap statistics directly on GPU (no synchronization).
    cols: [n_x, n_y, n_z, mN, spin, is_lost]

    Returns a dict of cp.ndarray:
      - survival_rate_mean (scalar cp array)
      - survival_rate_sem  (scalar cp array)
      - mot_mean (3,)
      - mot_sem  (3,)
      - ground_state_rate_mean (scalar)
      - ground_state_rate_sem  (scalar)

    Survival: (mN == 1) & (spin == 0) & (is_lost == 0)
    Ground-state (not conditioned on survive): (n_x==0)&(n_y==0)&(n_z==0)&(is_lost==0)
    """

    if rng is None:
        rng = cp.random

    if molecules.ndim != 2 or molecules.shape[1] != 6:
        raise ValueError("molecules must be (N,6)")

    N = molecules.shape[0]
    if N == 0:
        raise ValueError("molecules is empty")

    n_x = molecules[:, 0]
    n_y = molecules[:, 1]
    n_z = molecules[:, 2]
    mN  = molecules[:, 3]
    spin = molecules[:, 4]
    is_lost = molecules[:, 5]

    # Boolean masks
    survived = (mN == 1) & (spin == 0) & (is_lost == 0)
    non_lost = (is_lost == 0)
    in_gnd   = (n_x == 0) & (n_y == 0) & (n_z == 0) & non_lost

    def _stats_on_indices(idx: cp.ndarray):
        idx = idx.astype(cp.int64, copy=False)
        surv = survived[idx]
        nl   = non_lost[idx]
        gnd  = in_gnd[idx]

        surv_rate = surv.mean() if surv.size > 0 else cp.asarray(0.0)

        # Ground-state rate among non-lost
        denom = nl.sum()
        g_rate = gnd[nl].mean() if denom > 0 else cp.asarray(0.0)

        # Motional means conditioned on survive
        if surv.any():
            nx_mean = n_x[idx][surv].mean()
            ny_mean = n_y[idx][surv].mean()
            nz_mean = n_z[idx][surv].mean()
        else:
            nx_mean = ny_mean = nz_mean = cp.asarray(0.0)

        mot_vec = cp.stack([nx_mean, ny_mean, nz_mean])
        return surv_rate, mot_vec, g_rate

    # Point estimate (full cohort)
    full_idx = cp.arange(N)
    surv_rate_0, mot_vec_0, g_rate_0 = _stats_on_indices(full_idx)

    # Bootstrap resampling
    surv_rates = cp.empty(B, dtype=cp.float64)
    mot_vecs   = cp.empty((B, 3), dtype=cp.float64)
    g_rates    = cp.empty(B, dtype=cp.float64)

    for b in range(B):
        idx_b = rng.randint(0, N, size=N)
        sr, mv, gr = _stats_on_indices(idx_b)
        surv_rates[b] = sr
        mot_vecs[b, :] = mv
        g_rates[b] = gr

    # Standard errors (on GPU)
    Bf = cp.asarray(float(B))
    inv_sqrtB = 1.0 / cp.sqrt(Bf)

    sem_surv = surv_rates.std(ddof=1) * inv_sqrtB
    sem_mot  = mot_vecs.std(axis=0, ddof=1) * inv_sqrtB
    sem_gnd  = g_rates.std(ddof=1) * inv_sqrtB

    # Pack results — all GPU arrays
    out = {
        "survival_rate_mean": surv_rate_0,
        "survival_rate_sem": sem_surv,
        "mot_mean": mot_vec_0,
        "mot_sem": sem_mot,
        "ground_state_rate_mean": g_rate_0,
        "ground_state_rate_sem": sem_gnd,
        "N": cp.asarray(N, dtype=cp.int32),
        "B": cp.asarray(B, dtype=cp.int32),
    }
    return out

def save_molecules(molecules, filename: str | Path) -> None:
    """
    Save a (N,6) molecules array to a .npy file.
    If input is a CuPy array, it will be transferred to host memory first.

    Args
    ----
    molecules : np.ndarray or cp.ndarray
        Molecule data array, typically shape (N, 6)
    filename : str or Path
        Output file path (should end with .npy)
    """
    # Ensure valid filename
    filename = Path(filename)
    if not filename.suffix:
        filename = filename.with_suffix(".npy")

    # Convert CuPy → NumPy if needed
    if isinstance(molecules, cp.ndarray):
        mols_host = cp.asnumpy(molecules)
    elif isinstance(molecules, np.ndarray):
        mols_host = molecules
    else:
        raise TypeError("molecules must be a NumPy or CuPy array")

    # Save as .npy
    np.save(filename, mols_host)
    print(f"[✓] Molecules saved to: {filename.resolve()}  (shape={mols_host.shape})")