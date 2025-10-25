from .RSC_functions import (
    set_seed, to_device_m_table, GPUResources,
    make_device_molecules, make_device_pulses,
    raman_apply, optical_pumping, raman_cool_with_pumping,
    count_survivors, ground_state_rate
)
from .build_molecules import (
    build_thermal_molecules_gpu,
    build_thermal_molecules_host,
)

from.analysis import *

# src/cu_rsc/__init__.py
from importlib.resources import files, as_file
from pathlib import Path
import numpy as np

_pkg = "cu_rsc"
_rel = "data/M_FACTOR_TABLE.npy"

def _ensure_m_table():
    """Check for M_FACTOR_TABLE.npy, build if missing."""
    from .generate_M import precompute_M_factors_gpu

    with as_file(files(_pkg).joinpath(_rel)) as target:
        target = Path(target)
        if not target.exists():
            print("[cu_rsc] M_FACTOR_TABLE.npy not found; generating on GPU...")
            precompute_M_factors_gpu(save_path=target)
        else:
            try:
                np.load(target, allow_pickle=False)
            except Exception:
                print("[cu_rsc] M_FACTOR_TABLE.npy corrupted; regenerating...")
                precompute_M_factors_gpu(save_path=target)

# Optional: call explicitly from user code
def setup_tables(force=False):
    """Force regeneration of M-factor table on GPU."""
    from .generate_M import precompute_M_factors_gpu
    with as_file(files(_pkg).joinpath(_rel)) as target:
        target = Path(target)
        if force or not target.exists():
            precompute_M_factors_gpu(save_path=target)
    return True
