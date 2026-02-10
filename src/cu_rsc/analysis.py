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
    if mol_dev.ndim != 2 or mol_dev.shape[1] != 7:
        raise ValueError("mol_dev must be shape (N,7)")

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

    return dict(counts_x), dict(counts_y), dict(counts_z)



def bootstrap_stats_from_molecules(
    molecules: cp.ndarray,
    *,
    B: int = 1000,
    rng=None,
) -> dict[str, cp.ndarray]:
    """
    Compute bootstrap statistics directly on GPU (no synchronization).
    cols: [n_x, n_y, n_z, mN, spin, is_lost, (optional extra col...)]

    Returns a dict of cp.ndarray:
      - survival_rate_mean (scalar cp array)
      - survival_rate_sem  (scalar cp array)
      - mot_mean (3,)                     <-- conditioned on survival
      - mot_sem  (3,)                     <-- conditioned on survival
      - ground_state_rate_mean (scalar)
      - ground_state_rate_sem  (scalar)

    Survival: (mN == 1) & (spin == 0) & (is_lost == 0)
    Ground-state (conditioned on survive): (n_x==0)&(n_y==0)&(n_z==0)&(is_lost==0)
    Motional stats: conditioned on survival (computed ONLY over survivors).
    """

    if rng is None:
        rng = cp.random

    if molecules.ndim != 2 or molecules.shape[1] != 7:
        raise ValueError("molecules must be (N,7)")

    N = int(molecules.shape[0])
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

    # -------------------------
    # Point estimates
    # -------------------------
    surv_rate_0 = survived.mean()

    denom = non_lost.sum()
    g_rate_0 = in_gnd[non_lost].mean() if denom > 0 else cp.asarray(0.0)

    # Motional mean conditioned on survival (use survivors only)
    surv_idx_all = cp.where(survived)[0]
    Ns = int(surv_idx_all.size)

    if Ns > 0:
        mot_vec_0 = cp.stack([
            n_x[surv_idx_all].mean(),
            n_y[surv_idx_all].mean(),
            n_z[surv_idx_all].mean(),
        ]).astype(cp.float64)
    else:
        mot_vec_0 = cp.zeros((3,), dtype=cp.float64)

    # -------------------------
    # Bootstrap resampling
    # -------------------------
    surv_rates = cp.empty(B, dtype=cp.float64)
    g_rates    = cp.empty(B, dtype=cp.float64)
    mot_vecs   = cp.empty((B, 3), dtype=cp.float64)

    for b in range(B):
        # Resample full cohort for survival + ground-state stats
        idx_b = rng.randint(0, N, size=N).astype(cp.int64, copy=False)

        surv_b = survived[idx_b]
        nl_b   = non_lost[idx_b]
        gnd_b  = in_gnd[idx_b]

        surv_rates[b] = surv_b.mean() if surv_b.size > 0 else 0.0

        denom_b = nl_b.sum()
        g_rates[b] = gnd_b[nl_b].mean() if denom_b > 0 else 0.0

        # Resample survivors ONLY for motional stats (conditional on survival)
        if Ns > 0:
            # sample from survivor index list with replacement
            pick = rng.randint(0, Ns, size=Ns)
            sidx = surv_idx_all[pick]

            mot_vecs[b, 0] = n_x[sidx].mean()
            mot_vecs[b, 1] = n_y[sidx].mean()
            mot_vecs[b, 2] = n_z[sidx].mean()
        else:
            mot_vecs[b, :] = 0.0

    # Standard errors (keep your existing convention)
    Bf = cp.asarray(float(B))
    inv_sqrtB = 1.0 / cp.sqrt(Bf)

    sem_surv = surv_rates.std(ddof=1) * inv_sqrtB
    sem_gnd  = g_rates.std(ddof=1) * inv_sqrtB
    sem_mot  = mot_vecs.std(axis=0, ddof=1) * inv_sqrtB

    return {
        "survival_rate_mean": surv_rate_0,
        "survival_rate_sem": sem_surv,
        "mot_mean": mot_vec_0,
        "mot_sem": sem_mot,
        "ground_state_rate_mean": g_rate_0,
        "ground_state_rate_sem": sem_gnd,
        "N": cp.asarray(N, dtype=cp.int32),
        "Ns_surv": cp.asarray(Ns, dtype=cp.int32),
        "B": cp.asarray(B, dtype=cp.int32),
    }


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

def visualize_sequence(seq, show_time_color: bool = False, title: str | None = None) -> None:
    """
    Visualize a Raman sideband cooling sequence from a sequence matrix.

    Parameters
    ----------
    seq : np.ndarray or cp.ndarray
        (P, 3) array representing the sequence: columns [axis, delta_n, time].
    show_time_color : bool, optional
        If True, color-code pulses by duration (time); otherwise by axis (default: False).
    title : str, optional
        Title for the plot.

    Example
    -------
    >>> seq = np.load("top1_sequence.npy")
    >>> visualize_sequence_matrix(seq)
    >>> visualize_sequence_matrix(seq, show_time_color=True)
    """
    # Convert CuPy → NumPy if needed
    if not isinstance(seq, np.ndarray):
        try:
            import cupy as cp
            if isinstance(seq, cp.ndarray):
                seq = cp.asnumpy(seq)
            else:
                raise TypeError("seq must be a NumPy or CuPy array.")
        except ImportError:
            raise TypeError("seq must be a NumPy array or CuPy array (CuPy not installed).")

    if seq.ndim != 2 or seq.shape[1] != 3:
        raise ValueError(f"Invalid sequence shape {seq.shape}, expected (P, 3).")

    axes = seq[:, 0].astype(int)
    delta_n = seq[:, 1].astype(int)
    times = seq[:, 2].astype(float)
    pulse_idx = np.arange(len(seq))

    plt.figure(figsize=(10, 5))
    if show_time_color:
        sc = plt.scatter(pulse_idx, delta_n, c=times, cmap='plasma', s=40, alpha=0.9, edgecolor='k')
        cbar = plt.colorbar(sc, label="Pulse duration (s)")
    else:
        # Distinct color per axis
        colors = {0: "red", 1: "green", 2: "blue"}
        labels = {0: "X", 1: "Y", 2: "Z"}
        for ax in np.unique(axes):
            mask = axes == ax
            plt.scatter(pulse_idx[mask], delta_n[mask],
                        label=f"Axis {labels.get(ax, ax)}",
                        color=colors.get(ax, "gray"),
                        s=40, alpha=0.9, edgecolor='k')

    plt.xlabel("Pulse index")
    plt.ylabel("Δn (vibrational quantum number change)")
    plt.grid(True, linestyle='--', alpha=0.4)
    if title is not None:
        plt.title(title)
    else:
        plt.title("Raman Sideband Cooling Sequence")
    if not show_time_color:
        plt.legend()
    plt.tight_layout()
    plt.show()