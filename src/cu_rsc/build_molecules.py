"""
cu_rsc.build_molecules
----------------------
GPU utilities to build an (N,7) molecule array with a **thermal** distribution
of vibrational quanta along x, y, z.

State layout (int32, device array):
    molecules_dev[:, 0] = n_x
    molecules_dev[:, 1] = n_y
    molecules_dev[:, 2] = n_z
    molecules_dev[:, 3] = state   (mN)  [init: +1]
    molecules_dev[:, 4] = spin    [init: 0]
    molecules_dev[:, 5] = is_lost [init: 0]
    molecules_dev[:, 6] = trap_det (Hz) the trap detuning of the tweezer site

Sampling model per axis i:
    P(n_i = n) ∝ exp(- (n + 1/2) ħ ω_i / (k_B T_i)),  0 ≤ n ≤ n_cap
with n_cap = min(max_n[i]-1, n_limit[i]) so we don't exceed the configured basis
or trap-loss limit.

Returns CuPy device arrays for direct use with GPU pipelines.
"""
from __future__ import annotations

import json
from typing import Iterable, Tuple

import numpy as np
import cupy as cp
import scipy.constants as cts
from importlib.resources import files

# -----------------------------
# Load simulation parameters
# -----------------------------
_pkg = "cu_rsc"

with files(_pkg).joinpath("config.json").open("r") as f:
    _cfg = json.load(f)

amu = 1.66053906660e-27  # kg
mass: float = float(_cfg["mass"]) * amu
trap_freq = np.array(_cfg["trap_freq"], dtype=float) * 2 * np.pi  # ω (rad/s) per axis
k_vec: float = float(2 * np.pi / _cfg["lambda"])  # not used here
trap_depth_J: float = float(_cfg["trap_depth"]) * cts.k
max_n = tuple(map(int, _cfg["max_n"]))

# -----------------------------
# Helpers
# -----------------------------

def compute_n_limit(trap_depth_J: float, omega_rad_s: float) -> int:
    """n_limit such that E(n) = (n + 1/2) ħ ω < trap_depth.
    Using conservative bound n_limit ≈ floor(trap_depth / (h f)) where f = ω/(2π).
    Same as original code: int(trap_depth / (h * f)).
    """
    f_hz = omega_rad_s / (2 * np.pi)
    return int(trap_depth_J / (cts.h * f_hz))


def _axis_boltzmann_probs(T_K: float, omega: float, n_cap: int) -> cp.ndarray:
    """Return normalized Boltzmann probabilities P(n) for n=0..n_cap on GPU."""
    if n_cap < 0:
        # No allowable states; return a single zero-prob vector (handled upstream)
        return cp.asarray([1.0], dtype=cp.float64)
    n = cp.arange(n_cap + 1, dtype=cp.float64)
    E = (n + 0.5) * cts.hbar * float(omega)
    beta = 1.0 / (cts.k * float(T_K))
    p = cp.exp(-beta * E)
    s = p.sum()
    # Guard: if temperature is extremely low, p could underflow; default to ground state
    return cp.where(s > 0, p / s, cp.eye(1, n_cap + 1, 0, dtype=cp.float64)[0])


def _sample_from_probs(prob_row: cp.ndarray, N: int) -> cp.ndarray:
    """Sample N integers from a categorical distribution given by prob_row (1D).
    GPU implementation using CDF + searchsorted.
    """
    cdf = cp.cumsum(prob_row)
    cdf[-1] = 1.0
    u = cp.random.random((N,), dtype=prob_row.dtype)
    return cp.searchsorted(cdf, u, side="left").astype(cp.int32)

# -----------------------------
# Public API
# -----------------------------

def build_thermal_molecules(
    N: int,
    temps_K: Iterable[float],
    *,
    detuning_sigma: float = 0,
    state_init: int = 1,
    spin_init: int = 0,
    seed: int | None = None,
) -> cp.ndarray:
    """Create an (N,6) **device** array of molecules with thermal n along x,y,z.

    Parameters
    ----------
    N : number of molecules
    temps_K : iterable of 3 temperatures [Tx, Ty, Tz] in Kelvin
    detuning_sigma: the trap detuning variation in axial direction among the sites (Hz)
    state_init : initial mN (default +1)
    spin_init : initial spin manifold (default 0)
    seed : optional RNG seed for reproducibility

    Returns
    -------
    molecules_dev : cupy.ndarray (N,6) int32 on device
    """
    if seed is not None:
        cp.random.seed(int(seed))

    temps = list(map(float, temps_K))
    if len(temps) != 3:
        raise ValueError("temps_K must have length 3: [Tx, Ty, Tz] in Kelvin")

    # n_limit per axis from trap depth & frequency
    n_lim = [compute_n_limit(trap_depth_J, w) for w in trap_freq]

    # Cap by configured basis size
    n_cap = [max(0, min(n_lim[i], int(max_n[i]) - 1)) for i in range(3)]

    # Sample n for each axis on GPU
    ns = []
    for i in range(3):
        probs = _axis_boltzmann_probs(temps[i], trap_freq[i], n_cap[i])
        ns_i = _sample_from_probs(probs, N)
        ns.append(ns_i)

    # Assemble (N,7) device array
    mol = cp.zeros((N, 7), dtype=cp.int32)
    mol[:, 0] = ns[0]
    mol[:, 1] = ns[1]
    mol[:, 2] = ns[2]
    mol[:, 3] = int(state_init)  # mN = +1
    mol[:, 4] = int(spin_init)   # spin good
    mol[:, 5] = 0                # not lost

    # Trap detuning (Hz, integer)
    sigma = float(detuning_sigma)
    if sigma > 0.0:
        det_hz = cp.random.normal(
            loc=0.0,
            scale=sigma,
            size=(N,),
            dtype=cp.float32,
        )
        mol[:, 6] = cp.rint(det_hz).astype(cp.int32)
    else:
        mol[:, 6] = 0

    return mol

__all__ = [
    "build_thermal_molecules",
]
