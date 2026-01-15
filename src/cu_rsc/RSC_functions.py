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
from tqdm import tqdm

import cupy as cp
import numpy as np
import scipy.constants as cts
from typing import Tuple, Sequence

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
RABI_FREQ = float(_cfg["rabi_freq"]) * 2 * np.pi

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

def convert_to_LD(dK_axis_abs: cp.ndarray, trap_w_axis) -> cp.ndarray:
    """
    eta = |Δk| * x0 ; x0 = sqrt(hbar/(2 m ω)).
    trap_w_axis can be a float or a CuPy scalar; everything stays on device.
    """
    hbar = cts.hbar
    trap_w_axis_cp = cp.asarray(trap_w_axis, dtype=cp.float64)
    x0 = cp.sqrt(hbar / (2.0 * MASS * trap_w_axis_cp))
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

def ld_index_dithered(eta_abs: cp.ndarray, ld_res: float, ld_bins: int) -> cp.ndarray:
    x = eta_abs / ld_res
    i0 = cp.floor(x).astype(cp.int32)
    i0 = cp.clip(i0, 0, ld_bins - 1)
    i1 = cp.minimum(i0 + 1, ld_bins - 1)

    w = (x - i0.astype(cp.float64)).astype(cp.float32)  # fractional part in [0,1)
    u = cp.random.random(eta_abs.shape, dtype=cp.float32)
    return cp.where(u < w, i1, i0).astype(cp.int32)


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

def optical_pumping(
    molecules_dev: cp.ndarray,
    res: GPUResources,
    K_max: int = 12,
    delta_lim: int = None,
) -> cp.ndarray:
    if molecules_dev.ndim != 2 or molecules_dev.shape[1] != 6:
        raise ValueError("molecules_dev must be (N,6)")

    if delta_lim is not None and int(delta_lim) < 0:
        raise ValueError("delta_lim must be None or a nonnegative int")

    N = molecules_dev.shape[0]
    cycles_used = cp.zeros((N,), dtype=cp.int32)
    max_n = res.max_n

    decay_probs = cp.asarray(DECAY_RATIO, dtype=cp.float32)
    decay_probs = decay_probs / decay_probs.sum()
    cdf_decay = cp.cumsum(decay_probs)

    angle_sigma = ANGLE_PUMP_SIGMA
    angle_pi    = ANGLE_PUMP_PI

    # Precompute nf grid once (used for windowing)
    # Note: Kf differs by axis, so we build per-axis inside the loop.

    dlim = None if delta_lim is None else int(delta_lim)

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
            n_i_raw = molecules_dev[:, axis].astype(cp.int32)

            eta = cp.abs(convert_to_LD(cp.abs(dK[:, axis]), res.trap_freq[axis]))
            ld_idx = ld_index_dithered(eta, res.LD_RES, res.ld_bins)

            Kf = int(max_n[axis])
            nf_grid = cp.arange(Kf, dtype=cp.int32)  # (Kf,)

            # M lookup needs indices within [0, Kf-1]
            n_i_clip = cp.clip(n_i_raw, 0, Kf - 1)

            Mrows = m_lookup_rows(res.M, n_i_clip, ld_idx, Kf)  # (N, Kf)
            prob = (Mrows ** 2).astype(cp.float32)

            # Optional: enforce |Δn| <= delta_lim
            if dlim is not None:
                # |nf - n_i_raw| <= dlim  (broadcast nf over rows)
                dn_ok = (cp.abs(nf_grid[None, :] - n_i_raw[:, None]) <= dlim)
                prob = cp.where(dn_ok, prob, 0.0)

            # Zero for inactive rows
            prob = cp.where(active[:, None], prob, 0.0)

            nf_idx = rowwise_categorical_sample(prob)

            # WRITE BACK ONLY FOR ACTIVE ROWS
            molecules_dev[active, axis] = nf_idx[active]

            lost_axis = (nf_idx >= res.n_limit[axis]) & active
            molecules_dev[lost_axis, 5] = 1

        # decay to mN ∈ {-1,0,1} only for active rows
        u = cp.random.random((N,), dtype=cp.float32)
        new_state = cp.where(
            u < cdf_decay[0],
            -1,
            cp.where(u < cdf_decay[1], 0, 1)
        ).astype(cp.int32)
        molecules_dev[active, 3] = new_state[active]

        # branching to other spin only for active rows
        u2 = cp.random.random((N,), dtype=cp.float32)
        spun = active & (u2 < BRANCH_RATIO)
        molecules_dev[spun, 4] = 1

        cycles_used = cp.where(active, cycles_used + 1, cycles_used)

    return cycles_used


def rabi_transition_prob(omega: cp.ndarray, delta: cp.ndarray, t: cp.ndarray) -> cp.ndarray:
    """
    Detuned Rabi transition probability:

        P = (Ω^2 / (Ω^2 + Δ^2)) * sin^2( 0.5 * t * sqrt(Ω^2 + Δ^2) )

    omega, delta: angular frequencies (rad/s)
    t: time (s)
    """
    omega2 = omega ** 2
    delta2 = delta ** 2
    om_eff2 = omega2 + delta2
    om_eff = cp.sqrt(om_eff2)

    safe_denom = cp.where(om_eff2 > 0, om_eff2, 1.0)
    base = omega2 / safe_denom

    phase = 0.5 * om_eff * t
    return base * cp.sin(phase) ** 2


def excecute_single_raman_pulse(
    molecules_dev: cp.ndarray,
    *,
    axis: int,
    d_n,                  # int OR CuPy scalar
    Omega_lin: float,     # Hz (linear)
    t_sec: float,         # seconds
    res: GPUResources,
    LD_raman_vec: cp.ndarray,
    detuning_ang: cp.ndarray,  # (3,) rad/s (CuPy)
    Rabi_scale: float,
    k_max: int = 3,       # <-- NEW: include ±1 ... ±k_max
) -> None:
    """
    Execute a single (P,4) Raman pulse on-GPU (no sync).

    Channels:
      main: n -> n + d_n
      off-res: n -> n + d_n ± j,  j = 1 ... k_max
      detuning: Δ = Δ_axis ± j*ω_trap
    """
    d_n_dev = cp.asarray(d_n, dtype=cp.int32)
    N = molecules_dev.shape[0]

    ok = (
        (molecules_dev[:, 5] == 0) &
        (molecules_dev[:, 3] == 1) &
        (molecules_dev[:, 4] == 0)
    )

    n_i = molecules_dev[:, axis]

    # LD index
    eta_axis = cp.abs(LD_raman_vec[axis]).astype(cp.float64)
    eta_arr = cp.full((N,), eta_axis, dtype=cp.float64)
    ld_idx = _ld_index_from_eta(eta_arr, res.LD_RES, res.ld_bins)

    delta_axis_ang = detuning_ang[axis].astype(cp.float64)

    trap_axis = res.trap_freq[axis]
    if not isinstance(trap_axis, cp.ndarray):
        trap_axis = cp.asarray(trap_axis, dtype=cp.float64)
    trap_axis = trap_axis.astype(cp.float64)

    Omega_ang_base = (2.0 * np.pi) * float(Omega_lin) * float(Rabi_scale)
    t_arr = cp.full((N,), float(t_sec), dtype=cp.float64)

    nlim = int(res.n_limit[axis])
    
    def broadcast_detuning(det):
        return cp.full((N,), 1.0, dtype=cp.float64) * det

    # --------------------------------------------------
    # Build transition channels
    # Order:
    #   0: main
    #   1: +1, 2: -1, 3: +2, 4: -2, ..., 2*k_max
    # --------------------------------------------------
    P_list = []
    valid_list = []
    n_f_list = []

    # --- main (j = 0) ---
    n_f0 = n_i + d_n_dev
    valid0 = ok & (n_f0 >= 0)
    sel = cp.where(valid0)[0]

    m0 = cp.zeros((N,), dtype=cp.float32)
    if sel.size > 0:
        m0[sel] = res.M[n_i[sel], n_f0[sel], ld_idx[sel]]

    Omega0 = Omega_ang_base * m0.astype(cp.float64)
    Delta0 = broadcast_detuning(delta_axis_ang)
    P0 = rabi_transition_prob(Omega0, Delta0, t_arr)
    P0 = cp.where(valid0, P0, 0.0)

    P_list.append(P0.astype(cp.float32))
    valid_list.append(valid0)
    n_f_list.append(n_f0)

    # --- off-resonant ±j ---
    for j in range(1, k_max + 1):
        for sign in (+1, -1):
            j_dev = cp.asarray(sign * j, dtype=cp.int32)

            n_fj = n_i + d_n_dev + j_dev
            validj = ok & (n_fj >= 0)
            sel = cp.where(validj)[0]

            mj = cp.zeros((N,), dtype=cp.float32)
            if sel.size > 0:
                mj[sel] = res.M[n_i[sel], n_fj[sel], ld_idx[sel]]

            Omegaj = Omega_ang_base * mj.astype(cp.float64)
            Deltaj = broadcast_detuning(delta_axis_ang + trap_axis * j_dev.astype(cp.float64))
            Pj = rabi_transition_prob(Omegaj, Deltaj, t_arr)
            Pj = cp.where(validj, Pj, 0.0)

            P_list.append(Pj.astype(cp.float32))
            valid_list.append(validj)
            n_f_list.append(n_fj)

    # --------------------------------------------------
    # Probabilistic selection
    # --------------------------------------------------
    total_trans = P_list[0]
    for w in P_list[1:]:
        total_trans = total_trans + w

    w_none = cp.maximum(1.0 - total_trans, 0.0)

    scale = cp.where(
        total_trans > 1.0,
        1.0 / cp.maximum(total_trans, 1e-12),
        1.0,
    ).astype(cp.float32)

    w_none = cp.where(total_trans > 1.0, 0.0, w_none)
    P_list = [w * scale for w in P_list]

    weights = cp.stack([w_none] + P_list, axis=1)
    choice = rowwise_categorical_sample(weights)

    # --------------------------------------------------
    # Apply transitions
    # --------------------------------------------------
    for idx, (validj, n_fj) in enumerate(zip(valid_list, n_f_list), start=1):
        apply = validj & (choice == idx)
        molecules_dev[apply, axis] = n_fj[apply]
        molecules_dev[apply, 3] = -1
        # Mark lost if motional number exceeds trap depth on this axis
        lost_now = apply & (n_fj >= nlim)
        molecules_dev[lost_now, 5] = 1

    molecules_dev[:, axis] = cp.maximum(molecules_dev[:, axis], 0)


def raman_cool_with_pumping(
    molecules_dev: cp.ndarray,
    pulses_dev,
    res: GPUResources,
    *,
    K_max: int = 12,
    show_progress: bool = False,
    Rabi_scale: float = 1.0,
    Detuning: Sequence[float] = (0.0, 0.0, 0.0),  # per-axis detuning in linear Hz
    k_max: int = 3,
    delta_lim: int = None,
) -> None:
    """
    GPU Raman + optical pumping simulation (no GPU sync inside).

    Pulse layout:
        - Legacy (P,3):
            pulses[p] = [axis, delta_n, Omega_t]
            Omega_t = Omega_0 * t  (dimensionless pulse area)
            -> ONLY resonant transition n -> n + delta_n in the original model.

            With detuning:
                We interpret Omega_0 as the base angular Rabi frequency from config
                (RABI_FREQ), so:
                    t = Omega_t / RABI_FREQ
                    Omega_eff = RABI_FREQ * Rabi_scale * M
                and use:
                    P = Rabi( Omega_eff, Δ_axis, t )

        - New (P,4):
            pulses[p] = [axis, delta_n, Omega_lin, t_sec]
            Omega_lin: Rabi frequency in Hz (linear)
            t_sec:     pulse duration in seconds

            Internally:
                Omega_ang = 2π * Omega_lin * Rabi_scale   (rad/s)
                Δ_axis    = 2π * Detuning[axis]           (rad/s)

            Channels:
                main: n -> n + delta_n, detuning Δ = Δ_axis
                up:   n -> n + delta_n + 1, detuning Δ = Δ_axis + ω_trap
                down: n -> n + delta_n - 1, detuning Δ = Δ_axis - ω_trap

    Detuning:
        Detuning = (delta_x, delta_y, delta_z) in linear Hz.
        Detuning = (0,0,0) reproduces the old behavior (downwards compatible).
    """
    if molecules_dev.ndim != 2 or molecules_dev.shape[1] != 6:
        raise ValueError("molecules_dev must be (N,6)")

    pulses = np.asarray(pulses_dev, dtype=float)
    if pulses.ndim != 2 or pulses.shape[1] not in (3, 4):
        raise ValueError("pulses must be (P,3) or (P,4)")

    N = molecules_dev.shape[0]
    if N == 0:
        return

    LD_raman_vec = cp.asarray(LD_RAMAN, dtype=cp.float64)

    # Per-axis detuning in angular frequency (rad/s)
    detuning_ang = cp.asarray(
        2.0 * np.pi * np.asarray(Detuning, dtype=float),
        dtype=cp.float64,
    )

    pulse_iter = tqdm(range(pulses.shape[0]), desc="Raman pulses", disable=not show_progress)
    n_cols = pulses.shape[1]

    for p in pulse_iter:
        axis = int(pulses[p, 0])
        d_n = int(pulses[p, 1])

        # -----------------------------------
        # Legacy (P,3) behavior unchanged
        # -----------------------------------
        if n_cols == 3:
            # -----------------------------------
            # Common masks for Raman
            # -----------------------------------
            ok = (
                (molecules_dev[:, 5] == 0) &   # not lost
                (molecules_dev[:, 3] == 1) &   # state == +1
                (molecules_dev[:, 4] == 0)     # spin == 0
            )
            n_i = molecules_dev[:, axis]
            n_f = n_i + d_n
            valid = ok & (n_f >= 0)

            # LD index for this axis (constant per pulse)
            eta_axis = cp.abs(LD_raman_vec[axis])      # CuPy scalar
            eta_arr = cp.ones((N,), dtype=cp.float64) * eta_axis
            ld_idx = _ld_index_from_eta(eta_arr, res.LD_RES, res.ld_bins)

            # Axis detuning (rad/s)
            delta_axis_ang = detuning_ang[axis]

            Omega_t = float(pulses[p, 2])  # dimensionless pulse area

            sel = cp.where(valid)[0]
            m_vals = cp.zeros((N,), dtype=cp.float32)
            if sel.size > 0:
                m_vals[sel] = res.M[n_i[sel], n_f[sel], ld_idx[sel]]

            # Base angular Rabi frequency from config (rad/s)
            # RABI_FREQ is already in rad/s (loaded as 2π * rabi_freq[Hz])
            Omega0 = cp.float64(RABI_FREQ * Rabi_scale)

            # Interpret Omega_t = Omega0 * t  =>  t = Omega_t / RABI_FREQ
            t_sec = Omega_t / float(RABI_FREQ)
            t_arr = cp.full((N,), t_sec, dtype=cp.float64)

            # Effective Rabi frequency including M-factor and Rabi_scale
            Omega_eff = Omega0 * m_vals.astype(cp.float64)

            # Constant detuning for this axis
            Delta_arr = cp.full((N,), float(delta_axis_ang), dtype=cp.float64)

            prob = rabi_transition_prob(Omega_eff, Delta_arr, t_arr)
            prob = cp.where(valid, prob.astype(cp.float32), 0.0)

            u = cp.random.random((N,), dtype=prob.dtype)
            success = valid & (u < prob)

            molecules_dev[success, axis] = n_f[success]
            molecules_dev[success, 3] = -1
            molecules_dev[:, axis] = cp.maximum(molecules_dev[:, axis], 0)

        # -----------------------------------
        # New (P,4) behavior moved into excecute_single_raman_pulse (no sync)
        # -----------------------------------
        else:
            Omega_lin = float(pulses[p, 2])  # Hz
            t_sec = float(pulses[p, 3])      # s

            excecute_single_raman_pulse(
                molecules_dev,
                axis=axis,
                d_n=d_n,
                Omega_lin=Omega_lin,
                t_sec=t_sec,
                res=res,
                LD_raman_vec=LD_raman_vec,
                detuning_ang=detuning_ang,
                Rabi_scale=Rabi_scale,
                k_max=k_max,
            )

        optical_pumping(
            molecules_dev=molecules_dev,
            res=res,
            K_max=K_max,
            delta_lim=delta_lim,
        )


def raman_sideband_thermometry(
    molecules_dev: cp.ndarray,
    axis: int,
    frequencys: cp.ndarray,
    rabi_freq: float,
    pulse_time: float,
    res: GPUResources,
    show_progress: bool = True,
    k_max: int = 3,
) -> Tuple[cp.ndarray, cp.ndarray]:
    """
    Simulate Raman sideband thermometry without GPU synchronization.
    Parameters:
        molecules_dev: (N,6) array of molecules on device
        axis: 0, 1, or 2 for the trap axis to probe
        frequencys: (F,) array of Raman probe frequencies (linear Hz) on device
        rabi_freq: Rabi frequency (linear Hz) of the Raman probe
        pulse_time: pulse duration (seconds)
        res: GPUResources container
        show_progress: whether to show a progress bar
        k_max: maximum off-resonant sideband order to include
    """
    if molecules_dev.ndim != 2 or molecules_dev.shape[1] != 6:
        raise ValueError("molecules_dev must be (N,6)")
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2")
    if frequencys.ndim != 1:
        raise ValueError("frequencys must be a 1D array")

    polarizations = cp.zeros(frequencys.shape, dtype=cp.float64)

    # Precompute constants/buffers (device-side)
    LD_raman_vec = cp.asarray(LD_RAMAN, dtype=cp.float64)

    # Trap freq on device (rad/s) -> linear Hz on device
    trap_axis = res.trap_freq[axis]
    if not isinstance(trap_axis, cp.ndarray):
        trap_axis = cp.asarray(trap_axis, dtype=cp.float64)
    trap_axis = trap_axis.astype(cp.float64)
    trap_hz = trap_axis / (2.0 * np.pi)  # device scalar

    # Define the population we normalize over ONCE (device-side, constant across scan)
    # Include those starting in mN = +1 or -1, but only if not lost and correct spin.
    surv0 = (molecules_dev[:, 5] == 0) & (molecules_dev[:, 4] == 0)
    start_addr = surv0 & ((molecules_dev[:, 3] == 1) | (molecules_dev[:, 3] == -1))

    denom = start_addr.sum(dtype=cp.int32)         # device scalar
    denom_f = denom.astype(cp.float64)             # device scalar float

    # Work buffer so each frequency uses the same initial ensemble (no host sync)
    work = molecules_dev.copy()

    for i in tqdm(range(int(frequencys.size)), desc="Raman thermometry", disable=not show_progress):
        cp.copyto(work, molecules_dev)  # device->device reset, no sync

        freq = frequencys[i].astype(cp.float64)

        # Choose nearest sideband order (device scalar)
        d_n = cp.rint(freq / trap_hz).astype(cp.int32)

        # Residual detuning to nearest transition (linear Hz -> angular rad/s), on device
        detuning_ang = cp.zeros((3,), dtype=cp.float64)
        detuning_ang[axis] = (2.0 * np.pi) * (freq - d_n.astype(cp.float64) * trap_hz)

        excecute_single_raman_pulse(
            molecules_dev=work,
            axis=axis,
            d_n=d_n,                 # device scalar
            Omega_lin=rabi_freq,     # host float OK
            t_sec=pulse_time,        # host float OK
            res=res,
            LD_raman_vec=LD_raman_vec,
            detuning_ang=detuning_ang,
            Rabi_scale=1.0,
            k_max=k_max,
        )

        # Count mN == -1 AFTER the pulse, restricted to the initial addressable subset
        numer = (start_addr & (work[:, 3] == -1)).sum(dtype=cp.int32)  # device scalar
        numer_f = numer.astype(cp.float64)

        # Polarization on device; define behavior if denom==0 without syncing
        polarizations[i] = cp.where(
            denom > 0,
            # (2.0 * numer_f / denom_f) - 1.0,
            numer_f,
            cp.nan,   # or -1.0 if you prefer
        )

    return frequencys, polarizations



# -------------------------
# Convenience
# -------------------------

def make_device_molecules(mols_host_n_by_6: np.ndarray) -> cp.ndarray:
    if mols_host_n_by_6.ndim != 2 or mols_host_n_by_6.shape[1] != 6:
        raise ValueError("Input must be (N,6)")
    return cp.asarray(mols_host_n_by_6, dtype=cp.int32)

def make_device_pulses(pulses_host: np.ndarray) -> cp.ndarray:
    """
    Convert host pulses to device array.

    Accepts:
        - (P,3): [axis, delta_n, Omega_t]
        - (P,4): [axis, delta_n, Omega_lin (Hz), t_sec]

    axis and delta_n are rounded to integers; other columns are left as floats.
    """
    if pulses_host.ndim != 2 or pulses_host.shape[1] not in (3, 4):
        raise ValueError("pulses must be (P,3) or (P,4)")

    arr = np.asarray(pulses_host, dtype=float)

    # Discrete columns
    arr[:, 0] = np.rint(arr[:, 0])  # axis
    arr[:, 1] = np.rint(arr[:, 1])  # delta_n

    # Remaining columns are continuous (Omega_t or Omega_lin, t_sec)
    return cp.asarray(arr, dtype=cp.float64)


def count_survivors(molecules_dev: cp.ndarray) -> int:
    ok = (molecules_dev[:, 5] == 0) & (molecules_dev[:, 4] == 0) & (molecules_dev[:, 3] == 1)
    return int(ok.sum().get())

def ground_state_rate(molecules_dev: cp.ndarray) -> float:
    ok = (molecules_dev[:, 5] == 0) & (molecules_dev[:, 4] == 0) & (molecules_dev[:, 3] == 1)
    gnd = ok & (molecules_dev[:, 0] == 0) & (molecules_dev[:, 1] == 0) & (molecules_dev[:, 2] == 0)
    denom = int(ok.sum().get())
    if denom == 0:
        return 0.0
    return float(gnd.sum().get() / denom)
