
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

Requirements:
    pip install cupy-cuda11x   # or appropriate CUDA wheel for your system

Notes:
    - This module keeps all heavy ops on the GPU via CuPy.
    - Random draws use cupy.random; seed once via set_seed() for reproducibility.
    - M_FACTOR_TABLE is expected on host as a NumPy array with shape
      (nmax, nmax, ld_bins). We upload it once with to_device_m_table().
    - LD discretization: ld_index = round(|ld| / LD_RES), clamped to [0, ld_bins-1].
    - Optical pumping uses a fixed maximum number of cycles K to avoid unbounded
      loops and keep GPU-friendly control flow.
"""
from __future__ import annotations

import cupy as cp
import numpy as np

# -------------------------
# Seeding / utils
# -------------------------

def set_seed(seed: int | None) -> None:
    """Seed CuPy RNG (and optionally NumPy for host-side prep)."""
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
        LD_RES: float,
        trap_freq: np.ndarray | cp.ndarray,  # rad/s, shape (3,)
        k_vec: float,
        max_n: tuple[int, int, int],
        n_limit: tuple[int, int, int],
    ) -> None:
        self.M = M_FACTOR_TABLE_dev
        self.LD_RES = float(LD_RES)
        self.trap_freq = cp.asarray(trap_freq, dtype=cp.float64)
        self.k_vec = float(k_vec)
        self.max_n = tuple(int(x) for x in max_n)
        self.n_limit = cp.asarray(n_limit, dtype=cp.int32)

    @property
    def ld_bins(self) -> int:
        return int(self.M.shape[2])

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

def convert_to_LD(dK_axis: cp.ndarray, trap_f_axis: float) -> cp.ndarray:
    """eta = |Δk| * x0 ; x0 = sqrt(hbar/(2 m ω)).  Here trap_f_axis is ω (rad/s).
    NOTE: mass may be folded into Δk scaling externally if desired.
    """
    hbar = 1.054_571_817e-34
    x0 = cp.sqrt(hbar / (2.0 * trap_f_axis))
    return dK_axis * x0


def delta_k(k_vec: float, angle_pump: tuple[float, float], theta_s: cp.ndarray, phi_s: cp.ndarray) -> cp.ndarray:
    """Return Δk vectors (N,3) for pump (theta,phi) against scattered (theta_s,phi_s).
    k_vec is scalar 2π/λ.
    """
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
    """Sample one index per row from a row-normalized probability matrix.
    prob: (N, K) nonnegative, not necessarily normalized.
    Returns: idx (N,) int32
    """
    sums = prob.sum(axis=1, keepdims=True)
    safe = cp.where(sums > 0, sums, 1.0)
    cdf = cp.cumsum(prob / safe, axis=1)
    cdf[:, -1] = 1.0
    u = cp.random.random((prob.shape[0],), dtype=prob.dtype)
    idx = cp.searchsorted(cdf, u[:, None], side="left").astype(cp.int32).ravel()
    return idx

# -------------------------
# M lookups (vectorized gathers)
# -------------------------

def _ld_index_from_eta(eta_abs: cp.ndarray, LD_RES: float, ld_bins: int) -> cp.ndarray:
    ld_idx = cp.rint(eta_abs / LD_RES).astype(cp.int32)
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
# Public API
# -------------------------

def raman_apply(
    molecules_dev: cp.ndarray,
    pulses_dev: cp.ndarray,
    LD_raman: tuple[float, float, float],
    res: GPUResources,
) -> None:
    """Apply a full Raman pulse sequence on GPU (in-place mutate molecules_dev)."""
    if molecules_dev.ndim != 2 or molecules_dev.shape[1] != 6:
        raise ValueError("molecules_dev must be (N,6)")

    N = molecules_dev.shape[0]
    if N == 0:
        return

    LD_raman = cp.asarray(LD_raman, dtype=cp.float64)

    for p in range(pulses_dev.shape[0]):
        axis   = int(pulses_dev[p, 0].item())
        d_n    = int(pulses_dev[p, 1].item())
        t_puls = float(pulses_dev[p, 2].item())

        ok = (
            (molecules_dev[:, 5] == 0) &
            (molecules_dev[:, 3] == 1) &
            (molecules_dev[:, 4] == 0)
        )
        if not bool(ok.any()):
            continue

        n_i = molecules_dev[:, axis]
        n_f = n_i + d_n
        valid = ok & (n_f >= 0)
        if not bool(valid.any()):
            continue

        eta = cp.abs(LD_raman[axis])
        ld_idx = _ld_index_from_eta(cp.full((N,), eta, dtype=cp.float64), res.LD_RES, res.ld_bins)

        sel = cp.where(valid)[0]
        m_vals = cp.zeros((N,), dtype=cp.float32)
        m_vals[sel] = res.M[n_i[sel], n_f[sel], ld_idx[sel]]

        prob = cp.sin(m_vals * (t_puls * 0.5)) ** 2
        u = cp.random.random((N,), dtype=prob.dtype)
        success = valid & (u < prob)

        molecules_dev[success, axis] = n_f[success]
        molecules_dev[success, 3] = -1
        molecules_dev[:, axis] = cp.maximum(molecules_dev[:, axis], 0)


def optical_pumping(
    molecules_dev: cp.ndarray,
    res: GPUResources,
    angle_pump_sigma: tuple[float, float],
    angle_pump_pi: tuple[float, float],
    decay_ratio: tuple[float, float, float],
    branch_ratio: float,
    K_max: int = 12,
) -> cp.ndarray:
    """Vectorized optical pumping with fixed maximum cycles (GPU in-place)."""
    if molecules_dev.ndim != 2 or molecules_dev.shape[1] != 6:
        raise ValueError("molecules_dev must be (N,6)")

    N = molecules_dev.shape[0]
    cycles_used = cp.zeros((N,), dtype=cp.int32)

    max_n = res.max_n

    for k in range(int(K_max)):
        active = (molecules_dev[:, 5] == 0) & (molecules_dev[:, 4] == 0) & (molecules_dev[:, 3] != 1)
        if not bool(active.any()):
            break

        use_sigma = active & (molecules_dev[:, 3] == -1)
        theta_s = cp.pi * cp.random.random((N,), dtype=cp.float64)
        phi_s   = 2 * cp.pi * cp.random.random((N,), dtype=cp.float64)

        dK_sigma = delta_k(res.k_vec, angle_pump_sigma, theta_s, phi_s)
        dK_pi    = delta_k(res.k_vec, angle_pump_pi,    theta_s, phi_s)
        dK = cp.where(use_sigma[:, None], dK_sigma, dK_pi)

        for axis in range(3):
            n_i = molecules_dev[:, axis]
            eta = cp.abs(convert_to_LD(cp.abs(dK[:, axis]), res.trap_freq[axis]))
            ld_idx = _ld_index_from_eta(eta, res.LD_RES, res.ld_bins)

            Kf = int(max_n[axis])
            Mrows = m_lookup_rows(res.M, cp.clip(n_i, 0, Kf - 1), ld_idx, Kf)
            prob = (Mrows ** 2).astype(cp.float32)
            prob = cp.where(active[:, None], prob, 0.0)
            nf_idx = rowwise_categorical_sample(prob)
            molecules_dev[:, axis] = nf_idx
            lost_axis = nf_idx >= res.n_limit[axis]
            molecules_dev[lost_axis, 5] = 1

        probs = cp.asarray(decay_ratio, dtype=cp.float32)
        probs = probs / probs.sum()
        cdf = cp.cumsum(probs)
        u = cp.random.random((N,), dtype=cp.float32)
        new_state = cp.where(u < cdf[0], -1, cp.where(u < cdf[1], 0, 1)).astype(cp.int32)
        molecules_dev[active, 3] = new_state[active]

        u2 = cp.random.random((N,), dtype=cp.float32)
        spun = active & (u2 < branch_ratio)
        molecules_dev[spun, 4] = 1

        cycles_used = cp.where(active, cycles_used + 1, cycles_used)

    return cycles_used

# -------------------------
# Combined: Raman + per-pulse optical pumping (GPU, no sync)
# -------------------------

def raman_cool_with_pumping(
    molecules_dev: cp.ndarray,
    pulses_dev: cp.ndarray,
    LD_raman: tuple[float, float, float],
    res: GPUResources,
    *,
    angle_pump_sigma: tuple[float, float],
    angle_pump_pi: tuple[float, float],
    decay_ratio: tuple[float, float, float],
    branch_ratio: float,
    K_max: int = 12,
) -> None:
    """Run a full Raman sequence where each pulse is followed by up to K_max
    optical-pumping cycles. GPU-only, no explicit synchronizations.
    """
    if molecules_dev.ndim != 2 or molecules_dev.shape[1] != 6:
        raise ValueError("molecules_dev must be (N,6)")
    if pulses_dev.ndim != 2 or pulses_dev.shape[1] != 3:
        raise ValueError("pulses_dev must be (P,3)")

    N = molecules_dev.shape[0]
    if N == 0:
        return

    LD_raman = cp.asarray(LD_raman, dtype=cp.float64)

    probs_decay = cp.asarray(decay_ratio, dtype=cp.float32)
    probs_decay = probs_decay / probs_decay.sum()
    cdf_decay = cp.cumsum(probs_decay)

    for p in range(pulses_dev.shape[0]):
        # Raman
        axis   = int(pulses_dev[p, 0].item())
        d_n    = int(pulses_dev[p, 1].item())
        t_puls = float(pulses_dev[p, 2].item())

        ok = (
            (molecules_dev[:, 5] == 0) &
            (molecules_dev[:, 3] == 1) &
            (molecules_dev[:, 4] == 0)
        )
        n_i = molecules_dev[:, axis]
        n_f = n_i + d_n
        valid = ok & (n_f >= 0)
        if bool(valid.any()):
            eta_axis = cp.abs(LD_raman[axis])
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

        # Pumping cycles
        for k in range(int(K_max)):
            active = (molecules_dev[:, 5] == 0) & (molecules_dev[:, 4] == 0) & (molecules_dev[:, 3] != 1)
            if not bool(active.any()):
                break

            use_sigma = active & (molecules_dev[:, 3] == -1)
            theta_s = cp.pi * cp.random.random((N,), dtype=cp.float64)
            phi_s   = 2 * cp.pi * cp.random.random((N,), dtype=cp.float64)

            dK_sigma = delta_k(res.k_vec, angle_pump_sigma, theta_s, phi_s)
            dK_pi    = delta_k(res.k_vec, angle_pump_pi,    theta_s, phi_s)
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
                molecules_dev[:, ax] = nf_idx
                lost_axis = nf_idx >= res.n_limit[ax]
                molecules_dev[lost_axis, 5] = 1

            u = cp.random.random((N,), dtype=cp.float32)
            new_state = cp.where(u < cdf_decay[0], -1, cp.where(u < cdf_decay[1], 0, 1)).astype(cp.int32)
            molecules_dev[active, 3] = new_state[active]

            u2 = cp.random.random((N,), dtype=cp.float32)
            spun = active & (u2 < branch_ratio)
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
    arr = np.asarray(pulses_host)
    arr[:, 0] = np.rint(arr[:, 0])
    arr[:, 1] = np.rint(arr[:, 1])
    return cp.asarray(arr)

def count_survivors(molecules_dev: cp.ndarray) -> int:
    return int((molecules_dev[:, 5] == 0).sum().get())

def ground_state_rate(molecules_dev: cp.ndarray) -> float:
    ok = (molecules_dev[:, 5] == 0) & (molecules_dev[:, 4] == 0) & (molecules_dev[:, 3] == 1)
    gnd = ok & (molecules_dev[:, 0] == 0) & (molecules_dev[:, 1] == 0) & (molecules_dev[:, 2] == 0)
    denom = int(ok.sum().get())
    if denom == 0:
        return 0.0
    return float(gnd.sum().get() / denom)
