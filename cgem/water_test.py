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
