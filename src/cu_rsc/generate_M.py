
from __future__ import annotations

import json
import numpy as np
import cupy as cp
import scipy.constants as cts
from importlib.resources import files, as_file
from pathlib import Path
from tqdm import tqdm

# Do not use hyp1f1, it is unstable for large broadcasted inputs
from cupyx.scipy.special import gammaln

# -----------------------------
# Config & constants (host)
# -----------------------------
_pkg = "cu_rsc"
_rel = "data/M_FACTOR_TABLE.npy"

with files(_pkg).joinpath("config.json").open("r") as f:
    _cfg = json.load(f)

amu = 1.66053906660e-27  # kg
mass = float(_cfg["mass"] * amu)
trap_freq = np.array(_cfg["trap_freq"], dtype=float) * 2 * np.pi  # rad/s
k_vec = float(2 * np.pi / _cfg["lambda"])
trap_depth = float(_cfg["trap_depth"] * cts.k)  # J
max_n = list(map(int, _cfg["max_n"]))
LD_RES = float(_cfg["LD_RES"])

LD_MIN = float(_cfg.get("LD_MIN", 0.0))
LD_MAX = float(_cfg.get("LD_MAX", 1.0))

LD_GRID = cp.arange(LD_MIN, LD_MAX + LD_RES, LD_RES, dtype=cp.float64)
LD_LEN = int(LD_GRID.size)

MAX_N = int(max(max_n))
N_STATES = MAX_N + 1



def _genlaguerre_recurrence(n: cp.ndarray, alpha: cp.ndarray, x: cp.ndarray) -> cp.ndarray:
    # n, alpha, x broadcastable to same shape
    n = cp.asarray(n, dtype=cp.int32)
    alpha = cp.asarray(alpha, dtype=cp.float64)
    x = cp.asarray(x, dtype=cp.float64)

    # L0 and L1
    L0 = cp.ones_like(x, dtype=cp.float64)
    L1 = 1.0 + alpha - x

    # Handle n=0/1 quickly
    out = cp.where(n == 0, L0, cp.where(n == 1, L1, 0.0))

    # Recurrence up to max(n)
    Lkm1 = L0
    Lk = L1

    for k in range(1, MAX_N):
        # compute L_{k+1}
        kf = float(k)
        Lkp1 = ((2.0*kf + 1.0 + alpha - x) * Lk - (kf + alpha) * Lkm1) / (kf + 1.0)

        out = cp.where(n == (k+1), Lkp1, out)

        Lkm1, Lk = Lk, Lkp1

    return out




def _M_factor_matrix(n_i_mat: cp.ndarray, n_f_mat: cp.ndarray, eta: float) -> cp.ndarray:
    if eta == 0.0 or eta < 1e-300:
        return cp.eye(n_i_mat.shape[0], dtype=cp.float64)

    eta = float(eta)
    eta2 = eta * eta

    ge = n_f_mat >= n_i_mat
    delta = cp.abs(n_f_mat - n_i_mat)
    n_small = cp.where(ge, n_i_mat, n_f_mat)

    log_eta = np.log(eta)
    ni_p1 = n_i_mat + 1.0
    nf_p1 = n_f_mat + 1.0

    pref = cp.empty_like(n_i_mat, dtype=cp.float64)
    pref[ge]  = 0.5 * (gammaln(ni_p1[ge])  - gammaln(nf_p1[ge]))  + delta[ge]  * log_eta
    pref[~ge] = 0.5 * (gammaln(nf_p1[~ge]) - gammaln(ni_p1[~ge])) + delta[~ge] * log_eta

    # Key change: broadcast x to the shape of n_small to avoid scalar-related issues
    x_arr = cp.full_like(n_small, eta2, dtype=cp.float64)
    L = _genlaguerre_recurrence(n_small.astype(cp.int32), delta.astype(cp.float64), x_arr)

    return cp.exp(-0.5 * eta2 + pref) * L


# -----------------------------
# Public GPU precompute
# -----------------------------
def precompute_M_factors_gpu(dtype: str = "float32", save_path: Path | None = None) -> np.ndarray:
    out_dtype = cp.float32 if dtype == "float32" else cp.float64
    M_dev = cp.empty((N_STATES, N_STATES, LD_LEN), dtype=out_dtype, order="C")

    ni = cp.arange(N_STATES, dtype=cp.float64)
    nf = cp.arange(N_STATES, dtype=cp.float64)
    n_i_mat, n_f_mat = cp.meshgrid(ni, nf, indexing="ij")

    for ld_idx in tqdm(range(LD_LEN), desc="Precomputing M-factor (GPU)"):
        eta = float(LD_GRID[ld_idx].item())
        M_slice = _M_factor_matrix(n_i_mat, n_f_mat, eta)
        if cp.isnan(M_slice).any():
            raise FloatingPointError(f"NaNs at eta={eta} ld_idx={ld_idx}")

        M_dev[:, :, ld_idx] = M_slice.astype(out_dtype, copy=False)

    M_host = cp.asnumpy(M_dev)

    if save_path is None:
        with as_file(files(_pkg).joinpath(_rel)) as target:
            target = Path(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            np.save(target, M_host)
    else:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_path, M_host)

    return M_host

