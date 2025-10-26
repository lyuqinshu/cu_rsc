"""
GPU-accelerated Raman sideband cooling primitives using CuPy.

State layout (SoA-in-a-matrix):
    molecules: (N, 6) int32 array on **device** with columns
        [nx, ny, nz, state, spin, is_lost]
      - nx, ny, nz: vibrational quanta (>=0)
      - state: mN value in {-1, 0, 1}
      - spin: 0 (good manifold) or 1 (other)
      - is_lost: 0 (active) or 1 (lost)

Pulse layout:
    pulses: (P, 3) array with rows [axis, delta_n, time]
      - axis ∈ {0,1,2}
      - delta_n: int
      - time: float (in units of 1/Ω0)

Notes:
    - All experiment parameters are loaded from config.json at import.
    - Heavy ops stay on the GPU via CuPy.
    - Random draws use cupy.random; seed once via set_seed() for reproducibility.
    - M_FACTOR_TABLE is expected on host as a NumPy array with shape
      (nmax, nmax, ld_bins). Upload it once with to_device_m_table().
    - LD discretization: ld_index = round(|eta| / LD_RES), clamped to [0, ld_bins-1].
    - Optical pumping uses a fixed maximum number of cycles K to keep GPU-friendly control flow.
"""
from __future__ import annotations

import json
from importlib.resources import files, as_file
from pathlib import Path
from typing import Tuple
from tqdm import tqdm

import cupy as cp
import numpy as np
import scipy.constants as cts

# -------------------------
# Load experiment config (CPU-side)
# -------------------------

_pkg = "cu_rsc"
_rel = "data/M_FACTOR_TABLE.npy"

with files(_pkg).joinpath("config.json").open("r") as _f:
    _cfg = json.load(_f)

# Physical constants & experiment parameters
amu = 1.660_539_066_60e-27  # kg
MASS = float(_cfg["mass"]) * amu
TRAP_FREQ = np.array(_cfg["trap_freq"], dtype=float) * 2 * np.pi       # ω (rad/s), shape (3,)
K_VEC = float(2 * np.pi / _cfg["lambda"])                               # |k| = 2π/λ
DECAY_RATIO = tuple(map(float, _cfg["decay_ratio"]))                    # P(mN=-1,0,1)
BRANCH_RATIO = float(_cfg["branch_ratio"])                              # spin branch prob
TRAP_DEPTH_J = float(_cfg["trap_depth"]) * cts.k                        # J
MAX_N = tuple(map(int, _cfg["max_n"]))                                  # per-axis basis cap
LD_RES = float(_cfg["LD_RES"])                                          # η resolution (lookup)
LD_RAMAN = tuple(map(float, _cfg["LD_raman"]))                          # η per axis for Raman
ANGLE_PUMP_SIGMA = tuple(map(float, _cfg["angle_pump_sigma"]))          # (theta, phi)
ANGLE_PUMP_PI    = tuple(map(float, _cfg["angle_pump_pi"]))             # (theta, phi)

# n_limit from trap depth (consistent with CPU impl: int(trap_depth / (h * f)))
def _n_limit_for_axis(omega_rad_s: float) -> int:
    f_hz = omega_rad_s / (2 * np.pi)
    return int(TRAP_DEPTH_J / (cts.h * f_hz))

N_LIMIT = tuple(_n_limit_for_axis(w) for w in TRAP_FREQ)

def setup_tables(force: bool = False) -> Path:
    """
    Ensure M_FACTOR_TABLE.npy exists under cu_rsc/data/. If force, recompute.
    Returns the on-disk path.
    """
    from .generate_M import precompute_M_factors_gpu
    with as_file(files(_pkg).joinpath(_rel)) as target:
        target = Path(target)
        if force or not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            precompute_M_factors_gpu(save_path=target)
        return target

def load_m_table_host(allow_compute: bool = False) -> np.ndarray:
    """
    Load cu_rsc/data/M_FACTOR_TABLE.npy from the installed package.
    If missing and allow_compute=True, compute it on the GPU and save it there.
    """
    from .generate_M import precompute_M_factors_gpu  # lazy import to avoid heavy deps at import time

    with as_file(files(_pkg).joinpath(_rel)) as target:
        target = Path(target)
        if target.exists():
            return np.load(target, allow_pickle=False)

        if not allow_compute:
            raise FileNotFoundError(
                f"M-factor table not found at {target}. "
                f"Run `cu_rsc.setup_tables()` or call load_m_table_host(allow_compute=True)."
            )

        # Compute and save into the package data path
        target.parent.mkdir(parents=True, exist_ok=True)
        M_host = precompute_M_factors_gpu(save_path=target)
        return M_host


def load_m_table_device(dtype=cp.float32, allow_compute: bool = False) -> cp.ndarray:
    """
    Load the M table from the package (host) and upload to device.
    """
    M_host = load_m_table_host(allow_compute=allow_compute)
    return cp.asarray(M_host, dtype=dtype, order="C")

# -------------------------
# Seeding / utils
# -------------------------

def set_seed(seed: int | None) -> None:
    """Seed CuPy RNG (and NumPy for any host prep)."""
    if seed is not None:
        cp.random.seed(int(seed))
        np.random.seed(int(seed ^ 0xA5A5_1234))

# -------------------------
# Device resources
# -------------------------

class GPUResources:
    """Container for GPU-resident constants and tables."""
    def __init__(
        self,
        M_FACTOR_TABLE_dev: cp.ndarray,
        *,
        ld_res: float,
        trap_freq: np.ndarray | cp.ndarray,  # rad/s, shape (3,)
        k_vec: float,
        max_n: tuple[int, int, int],
        n_limit: tuple[int, int, int],
    ) -> None:
        self.M = M_FACTOR_TABLE_dev
        self.LD_RES = float(ld_res)
        self.trap_freq = cp.asarray(trap_freq, dtype=cp.float64)
        self.k_vec = float(k_vec)
        self.max_n = tuple(int(x) for x in max_n)
        self.n_limit = cp.asarray(n_limit, dtype=cp.int32)

    @property
    def ld_bins(self) -> int:
        return int(self.M.shape[2])

def resources_from_config(M_dev: cp.ndarray) -> GPUResources:
    """Convenience builder that wires GPUResources from the loaded config.json."""
    return GPUResources(
        M_FACTOR_TABLE_dev=M_dev,
        ld_res=LD_RES,
        trap_freq=TRAP_FREQ,
        k_vec=K_VEC,
        max_n=MAX_N,
        n_limit=N_LIMIT,
    )

# -------------------------
# Upload helpers
# -------------------------

def to_device_m_table(M_FACTOR_TABLE_np: np.ndarray, dtype=cp.float32) -> cp.ndarray:
    """Upload host M-factor lookup (n_i, n_f, ld_idx) to device."""
    if M_FACTOR_TABLE_np.ndim != 3:
        raise ValueError("M_FACTOR_TABLE must have shape (nmax, nmax, ld_bins)")
    return cp.asarray(M_FACTOR_TABLE_np, dtype=dtype, order="C")

# -------------------------
# LD & geometry (vectorized)
# -------------------------

def convert_to_LD(dK_axis_abs: cp.ndarray, trap_w_axis: float) -> cp.ndarray:
    """eta = |Δk| * x0 ; x0 = sqrt(hbar/(2 m ω)). trap_w_axis is ω (rad/s)."""
    hbar = cts.hbar
    x0 = cp.sqrt(hbar / (2.0 * MASS * float(trap_w_axis)))
    return dK_axis_abs * x0

def delta_k(k_vec: float, angle_pump: Tuple[float, float],
            theta_s: cp.ndarray, phi_s: cp.ndarray) -> cp.ndarray:
    """Return Δk vectors (N,3) for pump (theta,phi) against scattered (theta_s,phi_s)."""
    theta_p, phi_p = angle_pump
    k_p = cp.asarray([
        cp.sin(theta_p) * cp.cos(phi_p),
        cp.sin(theta_p) * cp.sin(phi_p),
        cp.cos(theta_p),
    ], dtype=cp.float64)

    sin_ts = cp.sin(theta_s)
    k_s = cp.stack([
        sin_ts * cp.cos(phi_s),
        sin_ts * cp.sin(phi_s),
        cp.cos(theta_s),
    ], axis=1)

    delta = k_p[None, :] - k_s  # (N,3)
    return k_vec * delta

# -------------------------
# Row-wise categorical sampling on GPU
# -------------------------

def rowwise_categorical_sample(prob: cp.ndarray) -> cp.ndarray:
    """Sample one index per row from a row-wise categorical distribution.

    prob: (N, K) nonnegative (need not be normalized).
    Returns: (N,) int32 indices.
    """
    if prob.ndim != 2:
        raise ValueError("prob must be 2D (N, K)")

    N, K = prob.shape
    if K == 0:
        # No categories; return zeros to be safe (caller should ensure K>0)
        return cp.zeros((N,), dtype=cp.int32)

    # Normalize per row; handle all-zero rows safely
    sums = prob.sum(axis=1, keepdims=True)
    safe = cp.where(sums > 0, sums, 1.0)
    cdf = cp.cumsum(prob / safe, axis=1)
    cdf[:, -1] = 1.0  # ensure each row sums to 1 exactly

    # Uniform draws, one per row
    u = cp.random.random((N,), dtype=prob.dtype)

    # For each row, idx = count of bins with cdf < u (i.e., first index where cdf >= u)
    # Equivalent to searchsorted along axis=1, but works for 2D.
    idx = (u[:, None] > cdf).sum(axis=1).astype(cp.int32)
    return idx

# -------------------------
# M lookups (vectorized gathers)
# -------------------------

def _ld_index_from_eta(eta_abs: cp.ndarray, ld_res: float, ld_bins: int) -> cp.ndarray:
    ld_idx = cp.rint(eta_abs / ld_res).astype(cp.int32)
    return cp.minimum(ld_idx, ld_bins - 1)

def m_lookup_rows(M: cp.ndarray, n_i: cp.ndarray, ld_idx: cp.ndarray, k_final: int) -> cp.ndarray:
    """Return a (N, k_final) matrix where rows m[i, nf] = M[n_i[i], nf, ld_idx[i]]."""
    N = n_i.shape[0]
    nf = cp.arange(k_final, dtype=cp.int32)
    n_i_cols = cp.repeat(n_i[:, None], k_final, axis=1)
    ld_cols = cp.repeat(ld_idx[:, None], k_final, axis=1)
    nf_rows = cp.repeat(nf[None, :], N, axis=0)
    return M[n_i_cols, nf_rows, ld_cols]

# -------------------------
# Public API (reads config.json values)
# -------------------------

def raman_apply(
    molecules_dev: cp.ndarray,
    pulses_dev: cp.ndarray,
    res: GPUResources,
) -> None:
    """Apply a full Raman pulse sequence on GPU (in-place), using LD_RAMAN from config."""
    if molecules_dev.ndim != 2 or molecules_dev.shape[1] != 6:
        raise ValueError("molecules_dev must be (N,6)")
    if pulses_dev.ndim != 2 or pulses_dev.shape[1] != 3:
        raise ValueError("pulses_dev must be (P,3)")

    N = molecules_dev.shape[0]
    if N == 0:
        return

    LD_raman_vec = cp.asarray(LD_RAMAN, dtype=cp.float64)

    for p in range(pulses_dev.shape[0]):
        axis   = int(pulses_dev[p, 0].item())
        d_n    = int(pulses_dev[p, 1].item())
        t_puls = float(pulses_dev[p, 2].item())

        ok = (
            (molecules_dev[:, 5] == 0) &   # not lost
            (molecules_dev[:, 3] == 1) &   # state == +1
            (molecules_dev[:, 4] == 0)     # spin == 0
        )
        if not bool(ok.any()):
            continue

        n_i = molecules_dev[:, axis]
        n_f = n_i + d_n
        valid = ok & (n_f >= 0)
        if not bool(valid.any()):
            continue

        # Discrete LD index from config LD_RAMAN
        eta = cp.abs(LD_raman_vec[axis])
        ld_idx = _ld_index_from_eta(cp.full((N,), eta, dtype=cp.float64), res.LD_RES, res.ld_bins)

        sel = cp.where(valid)[0]
        m_vals = cp.zeros((N,), dtype=cp.float32)
        m_vals[sel] = res.M[n_i[sel], n_f[sel], ld_idx[sel]]

        prob = cp.sin(m_vals * (t_puls * 0.5)) ** 2
        u = cp.random.random((N,), dtype=prob.dtype)
        success = valid & (u < prob)

        molecules_dev[success, axis] = n_f[success]
        molecules_dev[success, 3] = -1  # moved to mN = -1 after Raman
        molecules_dev[:, axis] = cp.maximum(molecules_dev[:, axis], 0)

def optical_pumping(
    molecules_dev: cp.ndarray,
    res: GPUResources,
    K_max: int = 12,
) -> cp.ndarray:
    if molecules_dev.ndim != 2 or molecules_dev.shape[1] != 6:
        raise ValueError("molecules_dev must be (N,6)")

    N = molecules_dev.shape[0]
    cycles_used = cp.zeros((N,), dtype=cp.int32)
    max_n = res.max_n

    decay_probs = cp.asarray(DECAY_RATIO, dtype=cp.float32)
    decay_probs = decay_probs / decay_probs.sum()
    cdf_decay = cp.cumsum(decay_probs)

    angle_sigma = ANGLE_PUMP_SIGMA
    angle_pi    = ANGLE_PUMP_PI

    for k in range(int(K_max)):
        # apply only to state -1 or 0, good spin, not lost
        is_m_minus1_or_0 = (molecules_dev[:, 3] == -1) | (molecules_dev[:, 3] == 0)
        active = (molecules_dev[:, 5] == 0) & (molecules_dev[:, 4] == 0) & is_m_minus1_or_0
        if not bool(active.any()):
            break

        use_sigma = active & (molecules_dev[:, 3] == -1)

        # Random scattering angles
        theta_s = cp.pi * cp.random.random((N,), dtype=cp.float64)
        phi_s   = 2 * cp.pi * cp.random.random((N,), dtype=cp.float64)

        dK_sigma = delta_k(res.k_vec, angle_sigma, theta_s, phi_s)
        dK_pi    = delta_k(res.k_vec, angle_pi,    theta_s, phi_s)
        dK = cp.where(use_sigma[:, None], dK_sigma, dK_pi)

        for axis in range(3):
            n_i = molecules_dev[:, axis]
            eta = cp.abs(convert_to_LD(cp.abs(dK[:, axis]), res.trap_freq[axis]))
            ld_idx = _ld_index_from_eta(eta, res.LD_RES, res.ld_bins)

            Kf = int(max_n[axis])
            # Build probabilities but zero them for inactive rows
            Mrows = m_lookup_rows(res.M, cp.clip(n_i, 0, Kf - 1), ld_idx, Kf)
            prob = (Mrows ** 2).astype(cp.float32)
            prob = cp.where(active[:, None], prob, 0.0)

            nf_idx = rowwise_categorical_sample(prob)
            # WRITE BACK ONLY FOR ACTIVE ROWS
            molecules_dev[active, axis] = nf_idx[active]

            lost_axis = (nf_idx >= res.n_limit[axis]) & active
            molecules_dev[lost_axis, 5] = 1

        # decay to mN ∈ {-1,0,1} only for active rows
        u = cp.random.random((N,), dtype=cp.float32)
        new_state = cp.where(u < cdf_decay[0], -1, cp.where(u < cdf_decay[1], 0, 1)).astype(cp.int32)
        molecules_dev[active, 3] = new_state[active]

        # branching to other spin only for active rows
        u2 = cp.random.random((N,), dtype=cp.float32)
        spun = active & (u2 < BRANCH_RATIO)
        molecules_dev[spun, 4] = 1

        cycles_used = cp.where(active, cycles_used + 1, cycles_used)

    return cycles_used


def raman_cool_with_pumping(
    molecules_dev: cp.ndarray,
    pulses_dev: cp.ndarray,
    res: GPUResources,
    *,
    K_max: int = 12,
    show_progress: bool = False,
) -> None:
    """
    GPU Raman + optical pumping simulation.
    Added optional tqdm progress bar for pulse loop.
    """
    if molecules_dev.ndim != 2 or molecules_dev.shape[1] != 6:
        raise ValueError("molecules_dev must be (N,6)")
    if pulses_dev.ndim != 2 or pulses_dev.shape[1] != 3:
        raise ValueError("pulses_dev must be (P,3)")

    N = molecules_dev.shape[0]
    if N == 0:
        return

    LD_raman_vec = cp.asarray(LD_RAMAN, dtype=cp.float64)

    # Precompute CDF for decays
    decay_probs = cp.asarray(DECAY_RATIO, dtype=cp.float32)
    decay_probs /= decay_probs.sum()
    cdf_decay = cp.cumsum(decay_probs)

    angle_sigma = ANGLE_PUMP_SIGMA
    angle_pi = ANGLE_PUMP_PI

    # -------------------------------
    # Main Raman pulse loop
    # -------------------------------
    pulse_iter = tqdm(range(pulses_dev.shape[0]), desc="Raman pulses", disable=not show_progress)

    for p in pulse_iter:
        axis = int(pulses_dev[p, 0].item())
        d_n = int(pulses_dev[p, 1].item())
        t_puls = float(pulses_dev[p, 2].item())

        ok = (molecules_dev[:, 5] == 0) & (molecules_dev[:, 3] == 1) & (molecules_dev[:, 4] == 0)
        n_i = molecules_dev[:, axis]
        n_f = n_i + d_n
        valid = ok & (n_f >= 0)

        if bool(valid.any()):
            eta_axis = cp.abs(LD_raman_vec[axis])
            ld_idx = _ld_index_from_eta(cp.full((N,), eta_axis, dtype=cp.float64), res.LD_RES, res.ld_bins)
            sel = cp.where(valid)[0]
            m_vals = cp.zeros((N,), dtype=cp.float32)
            m_vals[sel] = res.M[n_i[sel], n_f[sel], ld_idx[sel]]
            prob = cp.sin(m_vals * (t_puls * 0.5)) ** 2
            u = cp.random.random((N,), dtype=prob.dtype)
            success = valid & (u < prob)
            molecules_dev[success, axis] = n_f[success]
            molecules_dev[success, 3] = -1
            molecules_dev[:, axis] = cp.maximum(molecules_dev[:, axis], 0)

        # -------------------------------
        # Optical pumping cycles
        # -------------------------------
        for k in range(int(K_max)):
            is_m_minus1_or_0 = (molecules_dev[:, 3] == -1) | (molecules_dev[:, 3] == 0)
            active = (molecules_dev[:, 5] == 0) & (molecules_dev[:, 4] == 0) & is_m_minus1_or_0
            if not bool(active.any()):
                break

            use_sigma = active & (molecules_dev[:, 3] == -1)

            theta_s = cp.pi * cp.random.random((N,), dtype=cp.float64)
            phi_s = 2 * cp.pi * cp.random.random((N,), dtype=cp.float64)

            dK_sigma = delta_k(res.k_vec, angle_sigma, theta_s, phi_s)
            dK_pi = delta_k(res.k_vec, angle_pi, theta_s, phi_s)
            dK = cp.where(use_sigma[:, None], dK_sigma, dK_pi)

            for ax in range(3):
                n_i_ax = molecules_dev[:, ax]
                eta = cp.abs(convert_to_LD(cp.abs(dK[:, ax]), res.trap_freq[ax]))
                ld_idx = _ld_index_from_eta(eta, res.LD_RES, res.ld_bins)
                Kf = int(res.max_n[ax])
                Mrows = m_lookup_rows(res.M, cp.clip(n_i_ax, 0, Kf - 1), ld_idx, Kf)
                prob = (Mrows ** 2).astype(cp.float32)
                prob = cp.where(active[:, None], prob, 0.0)

                nf_idx = rowwise_categorical_sample(prob)
                molecules_dev[active, ax] = nf_idx[active]

                lost_axis = (nf_idx >= res.n_limit[ax]) & active
                molecules_dev[lost_axis, 5] = 1

            # decay & branching (active only)
            u = cp.random.random((N,), dtype=cp.float32)
            new_state = cp.where(u < cdf_decay[0], -1, cp.where(u < cdf_decay[1], 0, 1)).astype(cp.int32)
            molecules_dev[active, 3] = new_state[active]

            u2 = cp.random.random((N,), dtype=cp.float32)
            spun = active & (u2 < BRANCH_RATIO)
            molecules_dev[spun, 4] = 1


# -------------------------
# Convenience
# -------------------------

def make_device_molecules(mols_host_n_by_6: np.ndarray) -> cp.ndarray:
    if mols_host_n_by_6.ndim != 2 or mols_host_n_by_6.shape[1] != 6:
        raise ValueError("Input must be (N,6)")
    return cp.asarray(mols_host_n_by_6, dtype=cp.int32)

def make_device_pulses(pulses_host: np.ndarray) -> cp.ndarray:
    if pulses_host.ndim != 2 or pulses_host.shape[1] != 3:
        raise ValueError("pulses must be (P,3)")
    arr = np.asarray(pulses_host, dtype=float)
    arr[:, 0] = np.rint(arr[:, 0])
    arr[:, 1] = np.rint(arr[:, 1])
    return cp.asarray(arr, dtype=cp.float64)

def count_survivors(molecules_dev: cp.ndarray) -> int:
    return int((molecules_dev[:, 5] == 0).sum().get())

def ground_state_rate(molecules_dev: cp.ndarray) -> float:
    ok = (molecules_dev[:, 5] == 0) & (molecules_dev[:, 4] == 0) & (molecules_dev[:, 3] == 1)
    gnd = ok & (molecules_dev[:, 0] == 0) & (molecules_dev[:, 1] == 0) & (molecules_dev[:, 2] == 0)
    denom = int(ok.sum().get())
    if denom == 0:
        return 0.0
    return float(gnd.sum().get() / denom)
