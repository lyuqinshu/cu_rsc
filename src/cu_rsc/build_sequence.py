from __future__ import annotations
import numpy as np
import cupy as cp
from typing import Sequence, Tuple, Optional, Union, Dict, Any
from pathlib import Path
import json

# ---------------------------
# Experiment parameters
# ---------------------------

_CFG_PATH = Path(__file__).with_name("config.json")
# ---------------------------
# Load experiment parameters from config.json
# ---------------------------

_CFG_PATH = Path(__file__).with_name("config.json")

def _as_float_list(x):
    return [float(v) for v in x]

def _validate_lists_equal_length(name: str, a: Sequence[float], b: Sequence[float]):
    if len(a) != len(b):
        raise ValueError(f"{name}: amplitude and duration lists must have the same length; got {len(a)} vs {len(b)}")

def _compute_scalings_if_missing(amp: Dict[str, Sequence[float]],
                                 dur: Dict[str, Sequence[float]]) -> Tuple[float, float, float]:
    """
    Compute per-axis scalings using: pi / 0.3 / (amp[axis][-2] * dur[axis][-2]).
    Requires each axis to have at least 2 entries.
    """
    axes = ("X", "Y", "Z")
    out = []
    for ax in axes:
        a = amp.get(ax)
        d = dur.get(ax)
        if a is None or d is None:
            raise ValueError(f"Missing amp/duration list for axis '{ax}' in config.json")
        if len(a) < 2 or len(d) < 2:
            raise ValueError(f"Axis '{ax}' needs at least 2 entries to compute scaling; got {len(a)} amps, {len(d)} durs")
        s = np.pi / 0.3 / (float(a[-2]) * float(d[-2]))
        out.append(float(s))
    return tuple(out)  # (sx, sy, sz)

def _load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"config.json not found at {path}")
    with open(path, "r") as f:
        cfg = json.load(f)

    # Basic shape: amp_matrix, duration_matrix
    amp_matrix_cfg = cfg.get("amp_matrix")
    dur_matrix_cfg = cfg.get("duration_matrix")
    if amp_matrix_cfg is None or dur_matrix_cfg is None:
        raise ValueError("config.json must include 'amp_matrix' and 'duration_matrix'.")

    # Normalize to lists of floats
    amp_matrix = {
        k: _as_float_list(v)
        for k, v in amp_matrix_cfg.items()
    }
    duration_matrix = {
        k: _as_float_list(v)
        for k, v in dur_matrix_cfg.items()
    }

    # Ensure matching lengths per axis where relevant
    for ax in ("X", "Y", "Z"):
        if ax not in amp_matrix or ax not in duration_matrix:
            raise ValueError(f"Axis '{ax}' missing in amp_matrix or duration_matrix.")
        _validate_lists_equal_length(f"Axis '{ax}'", amp_matrix[ax], duration_matrix[ax])

    # Load or compute scalings
    scalings_cfg = cfg.get("scalings")
    if scalings_cfg is None:
        _SCALINGS = _compute_scalings_if_missing(amp_matrix, duration_matrix)
    else:
        # Accept dict {"X":..,"Y":..,"Z":..} or list/tuple [sx,sy,sz]
        if isinstance(scalings_cfg, dict):
            try:
                _SCALINGS = (float(scalings_cfg["X"]),
                             float(scalings_cfg["Y"]),
                             float(scalings_cfg["Z"]))
            except Exception as e:
                raise ValueError("scalings must have keys 'X','Y','Z' with float values") from e
        else:
            if len(scalings_cfg) != 3:
                raise ValueError("scalings must be length-3 when provided as a list/tuple")
            _SCALINGS = tuple(float(v) for v in scalings_cfg)

    # Non-axis keys like "OP","CO" can exist only in duration_matrix; that's fine.
    return {
        "amp_matrix": amp_matrix,
        "duration_matrix": duration_matrix,
        "SCALINGS": _SCALINGS,
    }

_CFG = _load_config(_CFG_PATH)

# Expose normalized data
amp_matrix: Dict[str, list[float]] = _CFG["amp_matrix"]
duration_matrix: Dict[str, list[float]] = _CFG["duration_matrix"]
_SCALINGS: Tuple[float, float, float] = _CFG["SCALINGS"]

# Map axis->label
_AXIS_KEY = {0: "X", 1: "Y", 2: "Z"}

# ---------------------------
# Core time helpers (unchanged API)
# ---------------------------

def pulse_time(axis: int, delta_n: int) -> float:
    """
    Return the base pulse time for a given axis and Δn (no per-user scale matrix applied).
    axis ∈ {0,1,2}, delta_n < 0 typical for cooling.
    """
    key = _AXIS_KEY[int(axis)]
    idx = -int(delta_n) - 1  # |Δn|-1
    if idx < 0 or idx >= len(amp_matrix[key]):
        raise IndexError(f"delta_n={delta_n} out of range for axis {axis} ({key}).")
    base = _SCALINGS[axis] * amp_matrix[key][idx] * duration_matrix[key][idx]
    return float(base)

def scaled_pulse_time(axis: int, delta_n: int, sm: Optional[np.ndarray]) -> float:
    """
    Apply an optional scale matrix: sm[axis, |Δn|-1] multiplies the base time.
    """
    base = pulse_time(axis, delta_n)
    if sm is None:
        return base
    sm = np.asarray(sm, dtype=float)
    idx = -int(delta_n) - 1
    if sm.shape[0] < 3 or sm.shape[1] <= idx:
        raise ValueError(f"scale matrix shape {sm.shape} incompatible with Δn index {idx}.")
    return float(sm[axis, idx] * base)


# ---------------------------
# GPU sequence builders
# ---------------------------

def get_sequence_unit_gpu(axis: int, delta_n: int, sm: Optional[np.ndarray] = None) -> cp.ndarray:
    """
    GPU-friendly single unit: returns (3,) device array [axis, delta_n, time].
    """
    t = scaled_pulse_time(axis, delta_n, sm)
    # store as float; axis and delta_n will be cast to ints downstream if needed
    return cp.asarray([axis, delta_n, t], dtype=cp.float64)


def _seq_to_device(seq_list: Sequence[Sequence[Union[int, float]]]) -> cp.ndarray:
    """
    Convert a Python list [[axis, delta_n, time], ...] to device (P,3).
    """
    arr = np.asarray(seq_list, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("sequence must be (P,3) [[axis, delta_n, time], ...]")
    # Ensure axis, delta_n are integral in storage
    arr[:, 0] = np.rint(arr[:, 0])
    arr[:, 1] = np.rint(arr[:, 1])
    return cp.asarray(arr, dtype=cp.float64)


def get_original_sequences_gpu(sm: Optional[np.ndarray] = None) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
    """
    Build the five canonical RSC sequences as **device arrays** (P_i,3):
    returns (sequence_XY, sequence_XYZ1, sequence_XYZ2, sequence_XYZ3, sequence_XYZ4)
    each row is [axis, delta_n, time].
    """
    if sm is None:
        sm = np.ones((3, 5), dtype=float)

    # XY
    sequence_XY = [
        [0, -3, sm[0, 2] * pulse_time(0, -3)],
        [1, -3, sm[1, 2] * pulse_time(1, -3)],
        [0, -2, sm[0, 1] * pulse_time(0, -2)],
        [1, -2, sm[1, 1] * pulse_time(1, -2)],
    ]

    # XYZ1
    sequence_XYZ1 = [
        [2, -5, sm[2, 4] * pulse_time(2, -5)],
        [0, -2, sm[0, 1] * pulse_time(0, -2)],
        [2, -4, sm[2, 3] * pulse_time(2, -4)],
        [1, -2, sm[1, 1] * pulse_time(1, -2)],
        [2, -5, sm[2, 4] * pulse_time(2, -5)],
        [0, -1, sm[0, 0] * pulse_time(0, -1)],
        [2, -4, sm[2, 3] * pulse_time(2, -4)],
        [1, -1, sm[1, 0] * pulse_time(1, -1)],
    ]

    # XYZ2
    sequence_XYZ2 = [
        [2, -4, sm[2, 3] * pulse_time(2, -4)],
        [0, -2, sm[0, 1] * pulse_time(0, -2)],
        [2, -3, sm[2, 2] * pulse_time(2, -3)],
        [1, -2, sm[1, 1] * pulse_time(1, -2)],
        [2, -4, sm[2, 3] * pulse_time(2, -4)],
        [0, -1, sm[0, 0] * pulse_time(0, -1)],
        [2, -3, sm[2, 2] * pulse_time(2, -3)],
        [1, -1, sm[1, 0] * pulse_time(1, -1)],
    ]

    # XYZ3
    sequence_XYZ3 = [
        [2, -3, sm[2, 2] * pulse_time(2, -3)],
        [0, -2, sm[0, 1] * pulse_time(0, -2)],
        [2, -2, sm[2, 1] * pulse_time(2, -2)],
        [1, -2, sm[1, 1] * pulse_time(1, -2)],
        [2, -3, sm[2, 2] * pulse_time(2, -3)],
        [0, -1, sm[0, 0] * pulse_time(0, -1)],
        [2, -2, sm[2, 1] * pulse_time(2, -2)],
        [1, -1, sm[1, 0] * pulse_time(1, -1)],
    ]

    # XYZ4
    sequence_XYZ4 = [
        [2, -2, sm[2, 1] * pulse_time(2, -2)],
        [0, -2, sm[0, 1] * pulse_time(0, -2)],
        [2, -1, sm[2, 0] * pulse_time(2, -1)],
        [1, -2, sm[1, 1] * pulse_time(1, -2)],
        [2, -2, sm[2, 1] * pulse_time(2, -2)],
        [0, -1, sm[0, 0] * pulse_time(0, -1)],
        [2, -1, sm[2, 0] * pulse_time(2, -1)],
        [1, -1, sm[1, 0] * pulse_time(1, -1)],
    ]

    return (
        _seq_to_device(sequence_XY),
        _seq_to_device(sequence_XYZ1),
        _seq_to_device(sequence_XYZ2),
        _seq_to_device(sequence_XYZ3),
        _seq_to_device(sequence_XYZ4),
    )


def concat_sequences(*seqs: cp.ndarray) -> cp.ndarray:
    """
    Concatenate multiple (P_i,3) device sequences into a single (P,3) device array.
    """
    if not seqs:
        return cp.zeros((0, 3))
    return cp.concatenate(seqs, axis=0)


# ---------------------------
# Persistence helpers
# ---------------------------

def save_sequence(path: Union[str, Path], sequence_dev: cp.ndarray) -> Path:
    """
    Save a device sequence (P,3) to disk. Format by extension:
      - .npy : NumPy binary (recommended, lossless)
      - .csv : comma-separated 'axis,delta_n,time'
      - .txt : same as csv
    Returns the saved Path.
    """
    p = Path(path)
    arr = cp.asnumpy(sequence_dev).astype(float, copy=False)

    if p.suffix.lower() == ".npy":
        np.save(p, arr)
    elif p.suffix.lower() in (".csv", ".txt"):
        header = "axis,delta_n,time"
        np.savetxt(p, arr, delimiter=",", header=header, comments="", fmt="%.0f,%.0f,%.18e")
    else:
        raise ValueError("Unsupported extension; use .npy, .csv, or .txt")
    return p


def load_sequence_device(path: Union[str, Path], sm: Optional[np.ndarray] = None) -> cp.ndarray:
    """
    Load a sequence file and return a device array (P,3).
    Accepted formats:
      - .npy : expects (P,3) or (P,2). If (P,2): columns [axis, delta_n] → time computed here.
      - .csv/.txt : same; commas or whitespace ok.

    If time is missing (two columns), it will be computed via `scaled_pulse_time` using `sm`.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    if p.suffix.lower() == ".npy":
        arr = np.load(p)
    else:
        # Load flexible csv/txt (delimiter auto)
        try:
            arr = np.loadtxt(p, delimiter=",")
        except Exception:
            arr = np.loadtxt(p)

    arr = np.atleast_2d(arr)
    if arr.shape[1] == 3:
        arr[:, 0] = np.rint(arr[:, 0])
        arr[:, 1] = np.rint(arr[:, 1])
        return cp.asarray(arr, dtype=cp.float64)

    if arr.shape[1] == 2:
        # Need to compute time
        axis = arr[:, 0].astype(int)
        dN = arr[:, 1].astype(int)
        times = np.array([scaled_pulse_time(a, dn, sm) for a, dn in zip(axis, dN)], dtype=float)
        out = np.column_stack([axis, dN, times])
        return cp.asarray(out, dtype=cp.float64)

    raise ValueError("Sequence file must have 2 or 3 columns.")
