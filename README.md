# CGem: Coarse Grained Electron Model

This implementation of CGem model uses `Numpy` arrays and `Scipy` optimizers to be fast, efficient,
and flexible. The code is tested (for the case of equal numbers of cores and shells) against the
C-GeM_CHOClNS code (independent implementation using lists and math module).
The code is documented and commented, but this still needs to improve. The API should be stable
at this stage, but it might slightly change.

For a description of the model please refer to
[J. Phys. Chem. Lett. 2019, 10, 6820-6826](https://pubs.acs.org/doi/abs/10.1021/acs.jpclett.9b02771)


Dependencies
------------

- Python >=3.6
- Numpy
- Scipy
- pytest
- [IOData](https://github.com/theochem/iodata)

To install dependencies, create a conda environment (this needs to be done only once):

```bash
conda create -n cgem python=3.6
conda activate cgem
conda install numpy scipy pytest
```

To activate the conda environment (this needs to be done before using the code/scripts each time):

```bash
conda activate cgem
```


Getting Started
---------------

The `setup.py` will be added soon for an easier installation of the library, but at this stage,
the code can be cloned and imported as a Python library by adding its directory to `PYTHONPATH`
environmental variable. To clone, test, and use the code:

```bash
git clone https://github.com/THGLab/CGem.git
cd CGem/
pytest -v cgem/    # make sure all tests pass successfully
export PYTHONPATH=path_to_cloned_cgem:$PYTHONPATH
```

There is a great deal of flexibility in how one can use the code and specify user-defined parameters;
please refer to the code documentation.
The simplest way is by providing the atomic numbers and coordinates of the system as Numpy arrays;
this will use the default global/atomic parameters of the model.

For example, one can easily compute CGem shell positions, energy, and electrostatic potential of
the water molecule by:

```python
import numpy as np
from cgem import CGem

# atomic numbers & coordinates
numbers = np.array([8, 1, 1])
coords = np.array([[2.5369, -0.3375, 0.0], [3.2979, 0.2462, 0.0], [1.776, 0.2462, 0.0]])

# make CGem model
model = CGem.from_molecule(numbers, coords)
print(model.coords_s)   # optimized shell coordinates
print(model.energy)     # energy

# compute electrostatic potential on a set of user-defined points
points = np.array([[2.53693169, -0.33746652,  3.04000502],
                   [2.07608313, -0.00264045,  2.98615819],
                   [0.60768512,  0.92074692, -1.98499664],
                   [2.22311862, -0.81287297,  2.98615819]])
esp = model.compute_electrostatic_potential(points)
print(esp)
```

To compute shell positions for a protein, named `protein.pdb`, one can use:

```python
from iodata import load_one
from cgem import CGem, get_protein_parameters

# load protein
pdb = load_one('protein.pdb')

# get latest CGem protein parameters
data = get_protein_parameters()

# make CGem model
model = CGem.from_molecule(pdb.atnums, pdb.atcoords, coords_s=None, opt_shells=True, min_dist_s=1.0e-5, **data)
print(model.coords_s)   # optimized shell coordinates
print(model.energy)     # energy
```

Soon, there would be command-line scripts to make it even easier to run the model.


C-Gem Scripts
-------------

There are scripts to facilitate using the library and perform various tasks.

For grid search, use `search` task:

```bash
./scripts/script_cgem.py search -h
# for example
./scripts/script_cgem.py search database protein_train_test.log
```

The optimized C-Gem parameters dictionary is dumped into a `parameters_cgem.json` file,
unless the user specifies the optional output filename on the command-line.

Sample usage for drug molecules
-------------
To load and compute cgem electrostatic potential, and get its error relative to the DFT reference, one can use the following code. Here I use the by far most accurate model 'drug_CHNO' which performs atom typing on H, C, N and O. To use different model one can user can chage the atoms to classify and parameters to import. 

```python
from cgem import CGem
from cgem.utils import compute_stats,load_xyz, load_esp
from cgem.parameters import get_parameter 
from cgem.attypes import classify_atom

xyz_file = '../data/neutral_molecules/omegapdb_1d4i/rmsd145-opt.xyz'
atoms, coord = load_xyz(xyz_file)
atoms = classify_atom(atoms,coord,[1,6])
params = get_parameter('drug_CH')
cgem = CGem.from_molecule(atoms, coord, coords_s=None, **params)
points, dft_esp = load_esp('../data/ESP_neutral_molecules/omegapdb_1d4i--rmsd145-opt.esp')
# compute cgem esp on given points
cgem_esp = cgem.compute_electrostatic_potential(points)
# compute error
mae, rmse, min_max,relative_err = compute_stats(cgem_esp, dft_esp)
```
