
from __future__ import annotations

import json
import numpy as np
import cupy as cp
import scipy.constants as cts
from importlib.resources import files, as_file
from pathlib import Path
from tqdm import tqdm

# Prefer CuPy's special functions; hyp1f1 may be unavailable on older CuPy
try:
    from cupyx.scipy.special import gammaln, hyp1f1  # confluent hypergeometric
    _HAS_HYP1F1 = True
except Exception:
    from cupyx.scipy.special import gammaln
    _HAS_HYP1F1 = False

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

# -----------------------------
# Generalized Laguerre via hyp1f1 or series
# -----------------------------
def _genlaguerre_hyp1f1(n: cp.ndarray, alpha: cp.ndarray, x: cp.ndarray) -> cp.ndarray:
    """Compute L_n^{(alpha)}(x) using confluent hypergeometric: 
       L_n^{(α)}(x) = Γ(n+α+1)/(Γ(α+1) Γ(n+1)) * 1F1(-n, α+1, x).
       Supports broadcasting over n, alpha, x.
    """
    b = alpha + 1.0
    # prefactor = gamma(n+alpha+1)/(gamma(alpha+1)*gamma(n+1))
    pref = cp.exp(gammaln(n + b) - gammaln(b) - gammaln(n + 1.0))
    return pref * hyp1f1(-n, b, x)

def _genlaguerre_series(n: cp.ndarray, alpha: cp.ndarray, x: cp.ndarray, k_max: int | None = None) -> cp.ndarray:
    """Fallback when hyp1f1 is unavailable. Truncated series since a = -n."""
    # Ensure arrays
    n = cp.asarray(n, dtype=cp.float64)
    alpha = cp.asarray(alpha, dtype=cp.float64)
    x = cp.asarray(x, dtype=cp.float64)  # broadcastable (may be scalar)

    b = alpha + 1.0
    if k_max is None:
        k_max = int(cp.max(n).get())

    # Important: shape like n, not x
    out = cp.zeros_like(n, dtype=cp.float64)

    ks = cp.arange(k_max + 1, dtype=cp.float64)
    fact = cp.concatenate([cp.asarray([1.0]), cp.cumprod(ks[1:])])

    gln_n1 = gammaln(n + 1.0)   # Γ(n+1)
    gln_b  = gammaln(b)         # Γ(b)

    for k in range(k_max + 1):
        term_mask = (n >= k)
        if not bool(term_mask.any()):
            continue

        sign = -1.0 if (k % 2 == 1) else 1.0
        num = cp.exp(gln_n1 - gammaln(n - k + 1.0))          # n!/(n-k)!
        poch_neg_n = sign * num
        poch_b = cp.exp(gammaln(b + k) - gln_b)              # (b)_k
        coeff = (poch_neg_n / poch_b) / fact[k]
        xk = x**k                                            # broadcasts if scalar
        term = cp.where(term_mask, coeff * xk, 0.0)
        out = out + term

    pref = cp.exp(gammaln(n + b) - gln_b - gln_n1)           # Γ(n+α+1)/(Γ(α+1)Γ(n+1))
    return pref * out


def _eval_genlaguerre(n, alpha, x):
    if _HAS_HYP1F1:
        return _genlaguerre_hyp1f1(n, alpha, x)
    else:
        return _genlaguerre_series(n, alpha, x)



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
    L = _eval_genlaguerre(n_small.astype(cp.float64), delta.astype(cp.float64), x_arr)

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

