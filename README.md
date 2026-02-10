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
  "mass": 123.0,                 # Mass of the partical in the unit of amu
  "trap_freq": [80e3, 80e3, 20e3], # Hz for x,y,z (converted internally to rad/s)
  "lambda": 531e-9,             # Wavelength of the cooling light, in meter (e.g., 531 nm)
  "decay_ratio": [0.25, 0.25, 0.5], # Decaying ratio P(mN=-1,0,1) after OP
  "branch_ratio": 0.05,          # probability to switch spin manifold during OP
  "trap_depth": 200e-6,          # Kelvin
  "max_n": [200, 200, 200],      # n cutoff per axis for M table and OP, should be larger than the trap depth
  "LD_RES": 0.01,                # η grid resolution
  "LD_MIN": 0.0,                 # Minimum η to calculate
  "LD_MAX": 3.0,                 # Maximum η to calculate
  "angle_pump_sigma": [1.57, 0.0], # [theta, phi] rad for σ-pump, referenced to the axial trap axis
  "angle_pump_pi":    [0.0, 0.0],  # [theta, phi] rad for π-pump, referenced to the axial trap axis
  "LD_raman": [0.5, 0.5, 0.3],    # |Δk|x0 per axis used for Raman pulses

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

A molecule sample should be a cupy array with shape (N, 7)
Each row contains (nx, ny, nz, mN, spin, is_lost, teap_freq)
To initialize a thermal sample of molecules:
```python
temp = [25e-6, 25e-6, 25e-6]
n = int(1e6)
mols = cr.build_thermal_molecules(n, temp)
```
You can also add a trap frequency deviation to the sample to simulate an inhomogeneous tweezer array.

### Generate RSC pulses
All raman pulse sequence should be in the form of (N, 4) numpy array
N is the number of the pulse, each pulse in the convension of (axis, Δn, Ω, t)
For example, a sequence of a single pulse cooling in Z axis with Δn=-1, Rabi frequency of 2π*3kHz and a pulse duration of 20μs should be

```python
sequence = np.array([[2, -1, 3e3, 20e-6]])
```

### Apply pulses

Applying the pulse will directly change the molecule array.
```python
cr.raman_cool_with_pumping(
  mols,
  sequence,
  res)
```
Detuning, off-resonance drive and more options are available.

---

### Analyze sample
You can read the state distribution of the molecule sample

```python
dist = cr.get_n_distribution_gpu(mols, max_bins=10, plot=(True, True, True))
result = cr.bootstrap_stats_from_molecules(mol_0)

print("survival rate: ", np.round(result["survival_rate_mean"], 3))
print("N_z bar: ", np.round(result["mot_mean"][2], 3))
print("Ground state rate: ", np.round(result["ground_state_rate_mean"],3))
```

## Performance benchmark
The run speed has a significant improvment on GPU compare to on CPU.
Here is the bench mark, with CPU: AMD Reyzen 7 5800H, 16 cores 3.2 GHz and GPU: NVIDIA GeForce RTX 3060 Laptop

![alt text](https://github.com/lyuqinshu/cu_rsc/blob/main/images/XY_benchmark.png?raw=true)
![alt text](https://github.com/lyuqinshu/cu_rsc/blob/main/images/full_benchmark.png?raw=true)


## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

---

