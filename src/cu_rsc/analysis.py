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
        (N,8) array on device: [nx, ny, nz, state, spin, is_lost, trap_det, carrier_det]
    plot : Sequence[bool]
        3 booleans [plot_x, plot_y, plot_z]
    max_bins : int
        Maximum number of tick labels for each axis.

    Returns
    -------
    counts_x, counts_y, counts_z : dict[int, int]
        Frequency counts for each vibrational number (on CPU) among survivors.
    """
    if mol_dev.ndim != 2 or mol_dev.shape[1] != 8:
        raise ValueError("mol_dev must be shape (N,8)")

    if not isinstance(plot, (list, tuple)) or len(plot) != 3:
        raise ValueError("plot must be a list or tuple of 3 booleans [x, y, z]")

    # Survivors: spin==0, state==+1, not lost
    valid_mask = (mol_dev[:, 4] == 0) & (mol_dev[:, 3] == 1) & (mol_dev[:, 5] == 0)

    n_xyz = mol_dev[valid_mask, 0:3]
    mol_num = int(n_xyz.shape[0])

    # Move to host for counting/plotting
    n_xyz_cpu = cp.asnumpy(n_xyz).astype(int)
    if mol_num > 0:
        n_x, n_y, n_z = n_xyz_cpu[:, 0], n_xyz_cpu[:, 1], n_xyz_cpu[:, 2]
    else:
        n_x = n_y = n_z = np.array([], dtype=int)

    counts_x = Counter(n_x)
    counts_y = Counter(n_y)
    counts_z = Counter(n_z)

    # --- Plotting ---
    any_plot = any(plot)
    if any_plot and mol_num > 0:
        nplots = sum(bool(x) for x in plot)
        fig, axes = plt.subplots(1, nplots, figsize=(5 * nplots, 5), sharey=True)
        if nplots == 1:
            axes = [axes]

        plot_axes = ["X", "Y", "Z"]
        colors = ["salmon", "mediumseagreen", "cornflowerblue"]
        count_dicts = [counts_x, counts_y, counts_z]

        out_j = 0
        for i, do_plot in enumerate(plot):
            if not do_plot:
                continue
            ax = axes[out_j]
            out_j += 1

            counts = count_dicts[i]
            ax.bar(list(counts.keys()), list(counts.values()),
                   color=colors[i], edgecolor="black")

            ax.set_title(f"n Distribution ({plot_axes[i]} axis)")
            max_n = max(counts.keys()) if counts else 0
            xt = np.linspace(0, max_n, min(max_n + 1, max_bins), dtype=int)
            ax.set_xticks(xt)
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
    cols: [n_x, n_y, n_z, mN, spin, is_lost, trap_det, carrier_det]

    Returns dict of cp.ndarray:
      - survival_rate_mean (scalar)
      - survival_rate_sem  (scalar)
      - mot_mean (3,)                     <-- conditioned on survival
      - mot_sem  (3,)                     <-- conditioned on survival
      - ground_state_rate_mean (scalar)   <-- conditioned on survival
      - ground_state_rate_sem  (scalar)

    Survival: (mN == 1) & (spin == 0) & (is_lost == 0)
    Ground-state (conditioned on survive): survived & (nx==0)&(ny==0)&(nz==0)
    """
    if rng is None:
        rng = cp.random

    if molecules.ndim != 2 or molecules.shape[1] != 8:
        raise ValueError("molecules must be (N,8)")

    N = int(molecules.shape[0])
    if N == 0:
        raise ValueError("molecules is empty")

    n_x = molecules[:, 0]
    n_y = molecules[:, 1]
    n_z = molecules[:, 2]
    mN  = molecules[:, 3]
    spin = molecules[:, 4]
    is_lost = molecules[:, 5]

    survived = (mN == 1) & (spin == 0) & (is_lost == 0)  # (N,) bool
    in_gnd_surv = survived & (n_x == 0) & (n_y == 0) & (n_z == 0)

    # -------------------------
    # Point estimates (no sync)
    # -------------------------
    surv_rate_0 = survived.mean(dtype=cp.float64)

    Ns0 = survived.sum(dtype=cp.int32).astype(cp.float64)
    gnd0 = in_gnd_surv.sum(dtype=cp.int32).astype(cp.float64)
    g_rate_0 = cp.where(Ns0 > 0, gnd0 / Ns0, cp.asarray(0.0, dtype=cp.float64))

    # Motional mean conditioned on survival (use sums/denom)
    sx = cp.where(survived, n_x, 0).sum(dtype=cp.float64)
    sy = cp.where(survived, n_y, 0).sum(dtype=cp.float64)
    sz = cp.where(survived, n_z, 0).sum(dtype=cp.float64)
    mot_vec_0 = cp.where(
        Ns0 > 0,
        cp.stack([sx / Ns0, sy / Ns0, sz / Ns0]),
        cp.zeros((3,), dtype=cp.float64),
    )

    # Precompute survivor indices for conditional bootstrap motional stats
    surv_idx_all = cp.where(survived)[0]
    Ns = int(surv_idx_all.size)

    # -------------------------
    # Bootstrap resampling
    # -------------------------
    surv_rates = cp.empty(B, dtype=cp.float64)
    g_rates    = cp.empty(B, dtype=cp.float64)
    mot_vecs   = cp.empty((B, 3), dtype=cp.float64)

    for b in range(B):
        idx_b = rng.randint(0, N, size=N).astype(cp.int64, copy=False)

        surv_b = survived[idx_b]
        Ns_b = surv_b.sum(dtype=cp.int32).astype(cp.float64)

        surv_rates[b] = surv_b.mean(dtype=cp.float64)

        gnd_b = in_gnd_surv[idx_b].sum(dtype=cp.int32).astype(cp.float64)
        g_rates[b] = cp.where(Ns_b > 0, gnd_b / Ns_b, 0.0)

        # Motional stats: resample survivors only (conditional on survival)
        if Ns > 0:
            pick = rng.randint(0, Ns, size=Ns)
            sidx = surv_idx_all[pick]
            mot_vecs[b, 0] = n_x[sidx].mean(dtype=cp.float64)
            mot_vecs[b, 1] = n_y[sidx].mean(dtype=cp.float64)
            mot_vecs[b, 2] = n_z[sidx].mean(dtype=cp.float64)
        else:
            mot_vecs[b, :] = 0.0

    inv_sqrtB = 1.0 / cp.sqrt(cp.asarray(float(B), dtype=cp.float64))

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
    Save a molecules array (typically (N,8)) to a .npy file.
    If input is a CuPy array, it will be transferred to host memory first.
    """
    filename = Path(filename)
    if not filename.suffix:
        filename = filename.with_suffix(".npy")

    if isinstance(molecules, cp.ndarray):
        mols_host = cp.asnumpy(molecules)
    elif isinstance(molecules, np.ndarray):
        mols_host = molecules
    else:
        raise TypeError("molecules must be a NumPy or CuPy array")

    np.save(filename, mols_host)
    print(f"[✓] Molecules saved to: {filename.resolve()}  (shape={mols_host.shape})")


def visualize_sequence(seq, show_time_color: bool = False, title: str | None = None) -> None:
    """
    Visualize a Raman sequence.

    Supported seq shapes:
      - (P,3): [axis, delta_n, time]
      - (P,4): [axis, delta_n, omega, time]
      - (P,5): [axis, delta_n, omega, time, detuning]

    Plots delta_n vs pulse index.
    If show_time_color=True, color by pulse duration (time column).
    Else color by axis.
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

    if seq.ndim != 2 or seq.shape[1] not in (3, 4, 5):
        raise ValueError(f"Invalid sequence shape {seq.shape}, expected (P,3), (P,4), or (P,5).")

    P, C = seq.shape
    axes = seq[:, 0].astype(int)
    delta_n = seq[:, 1].astype(int)

    if C == 3:
        times = seq[:, 2].astype(float)
    elif C == 4:
        times = seq[:, 3].astype(float)
    else:  # C == 5
        times = seq[:, 3].astype(float)

    pulse_idx = np.arange(P)

    plt.figure(figsize=(10, 5))
    if show_time_color:
        sc = plt.scatter(pulse_idx, delta_n, c=times, cmap="plasma", s=40, alpha=0.9, edgecolor="k")
        plt.colorbar(sc, label="Pulse duration (s)")
    else:
        colors = {0: "red", 1: "green", 2: "blue"}
        labels = {0: "X", 1: "Y", 2: "Z"}
        for axv in np.unique(axes):
            mask = axes == axv
            plt.scatter(
                pulse_idx[mask], delta_n[mask],
                label=f"Axis {labels.get(axv, axv)}",
                color=colors.get(axv, "gray"),
                s=40, alpha=0.9, edgecolor="k",
            )
        plt.legend()

    plt.xlabel("Pulse index")
    plt.ylabel("Δn")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.title(title if title is not None else "Raman Sequence")
    plt.tight_layout()
    plt.show()
