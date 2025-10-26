# cu_rsc

This project simulates the Raman sideband cooling (RSC) of CaF molecules in a tweezer trap following the theoratical proposal from [Caldwell et al.](https://doi.org/10.1103/PhysRevResearch.2.013251) 
and experimantal implementation from [Bao et al.](https://doi.org/10.1103/PhysRevX.14.031002)
It can be easily tailored to other species and experiments by modifying config.json
The package features GPU acceleration

---


## Configuration (`config.json`)

Example:
```python
{
  "mass": 123.0,                 # amu
  "trap_freq": [80e3, 80e3, 20e3], # Hz for x,y,z (converted internally to rad/s)
  "lambda": 531e-9,             # m (e.g., 531 nm)
  "decay_ratio": [0.25, 0.25, 0.5], # P(mN=-1,0,1) after OP
  "branch_ratio": 0.05,          # probability to switch spin manifold during OP
  "trap_depth": 200e-6,          # Kelvin
  "max_n": [200, 200, 200],      # n cutoff per axis for M table and OP, should be larger than the trap depth
  "LD_RES": 0.01,                # η grid resolution
  "LD_MIN": 0.0,
  "LD_MAX": 3.0,
  "angle_pump_sigma": [1.57, 0.0], # [theta, phi] rad for σ-pump, referenced to the axial trap axis
  "angle_pump_pi":    [0.0, 0.0],  # [theta, phi] rad for π-pump, referenced to the axial trap axis
  "LD_raman": [0.5, 0.5, 0.3]    # |Δk|x0 per axis used for Raman pulses
}
```

---
## Getting started

### Initialization
```python
import cu_rsc as cr
import cupy as cp
cr.setup_tables()
M_dev = cr.load_m_table_device()       
res   = cr.resources_from_config(M_dev)
```

### Initialize molecules
```python
temp = [25e-6, 25e-6, 25e-6]
n = int(1e6)
mols_gpu = cr.build_thermal_molecules_gpu(n, temp)
```

### Generate RSC pulses
```python
original_gpu = cr.get_original_sequences_gpu()  # list/tuple of cp.ndarray blocks

# Repeat each block along the first axis, then concatenate in order
blocks = [
    cp.tile(original_gpu[0], (10, 1)),  # repeat 10 times
    cp.tile(original_gpu[1], (5,  1)),  # repeat 5 times
    cp.tile(original_gpu[2], (5,  1)),  # repeat 5 times
    cp.tile(original_gpu[3], (10, 1)),  # repeat 10 times
    cp.tile(original_gpu[4], (10, 1)),  # repeat 10 times
]

seq_gpu = cp.concatenate(blocks, axis=0)
```

### Apply pulses
```python
cr.raman_cool_with_pumping(mols_gpu, seq_gpu, res)
```

---


## Performance benchmark
The run speed has a significant improvment on GPU compare to on CPU.
Here is the bench mark, with CPU: AMD Reyzen 7 5800H, 16 cores 3.2 GHz and GPU: NVIDIA GeForce RTX 3060 Laptop

![alt text](https://github.com/lyuqinshu/cu_rsc/blob/main/images/XY_benchmark.png?raw=true)
![alt text](https://github.com/lyuqinshu/cu_rsc/blob/main/images/full_benchmark.png?raw=true)


## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

---

