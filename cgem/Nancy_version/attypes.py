"""Atom Type Module."""

from scipy.spatial.distance import cdist
import numpy as np

hydrogen_num = [1, 101, 102, 103, 104, 110]
carbon_num = [6, 601, 602, 6001, 610]
nitrogen_num = [7, 701, 702, 703, 704, 705, 706, 707, 7001]
oxygen_num = [8, 801, 802, 803, 804, 805, 806, 820]
elec_negs = [16, 17] + nitrogen_num + oxygen_num


def classify_atom(nums, coords, typed_atom):
    """Do atom typing for given atom types.

		Parameters
		----------
		nums : np.ndarray, shape=(n,)
			Atomic numbers.
		coords: np.ndarray, shape=(n, 3)
			Cartesian coordinates of core positions in Angstrom.
		atomtypes: array of int
			Array of atomic numbers to classify

		Returns
		-------
		nums : np.ndarray, shape=(n,)
			Classified atomic numbers.
		"""
    if typed_atom is not None:
        for atom in typed_atom:
            if atom == 1:
                nums = classify_hydrogen(nums, coords)
            elif atom == 6:
                nums = classify_carbon(nums, coords)
            elif atom == 7:
                nums = classify_nitrogen(nums, coords)
            elif atom == 8:
                nums = classify_oxygen(nums, coords)
            elif atom == 102:
                nums = classify_hydrogen2(nums, coords)
            elif atom == 72:
                nums = classify_nitrogen2(nums, coords)
            elif atom == 73:
                nums = classify_nitrogen3(nums, coords)
            elif atom == 82:
                nums = classify_oxygen2(nums, coords)
            elif atom == 83:
                nums = classify_oxygen3(nums, coords)
            else:
                raise ValueError(
                    f"Atom typing for {atom} is not supported. Supported atoms are: [1,6,7,8]"
                )
    return nums


def classify_carbon(atoms, coords_c):
    """Do atom typing for carbon atoms.

	Classify into two classes according to distance:
	   1. atomic number 601 for carbon adjacent to electronegative atoms (O, N, Cl, S).
	   2. atomic number 602 for carbon not adjacent to electronegative atoms.

	Parameters
	----------
	atoms : np.ndarray, shape=(n,)
		Atomic numbers.
	coords_c: np.ndarray, shape=(n, 3)
		Cartesian coordinates of core positions in Angstrom.

	Returns
	-------
	classified_atoms : np.ndarray, shape=(n,)
		Classified atomic numbers.
	"""
    # atomic numbers for electronegative atoms

    classified_atoms = np.copy(atoms)
    for i, atom in enumerate(atoms):
        if atom == 6:
            classified_atoms[i] = 602
            distance_vec = cdist(
                coords_c[i][np.newaxis, :], coords_c
            )  # shape (1,len(coords_c))
            distance_vec = np.where(
                distance_vec == 0, 10, distance_vec
            )  # replace distance to itself with 10
            # print([atom[i] for i in np.where(distance_vec < 2)[1]])
            if np.any([atoms[i] in elec_negs for i in np.where(distance_vec < 1.8)[1]]):
                classified_atoms[i] = 601
    return classified_atoms

def classify_carbon2(atoms, coords_c):
    """Do atom typing for carbon atoms.

	Classify into two classes according to distance:
	   1. atomic number 601 for carbon adjacent to electronegative atoms (O, N, Cl, S).
	   2. atomic number 602 for carbon not adjacent to electronegative atoms.
	   3. atomic number 603 for carbon adjacent to positively charged atoms
	   3. atomic number 604 for carbon adjacent to negatively charged atoms

	Parameters
	----------
	atoms : np.ndarray, shape=(n,)
		Atomic numbers.
	coords_c: np.ndarray, shape=(n, 3)
		Cartesian coordinates of core positions in Angstrom.

	Returns
	-------
	classified_atoms : np.ndarray, shape=(n,)
		Classified atomic numbers.
	"""
    # atomic numbers for electronegative atoms

    classified_atoms = np.copy(atoms)
    for i, atom in enumerate(atoms):
        if atom == 6:
            classified_atoms[i] = 602
            distance_vec = cdist(
                coords_c[i][np.newaxis, :], coords_c
            )  # shape (1,len(coords_c))
            distance_vec = np.where(
                distance_vec == 0, 10, distance_vec
            )  # replace distance to itself with 10
            if np.any([atoms[i] in [6001,7001] for i in np.where(distance_vec < 1.8)[1]]):
                classified_atoms[i] = 603
            elif np.any([atoms[i] in elec_negs for i in np.where(distance_vec < 1.8)[1]]):
                classified_atoms[i] = 601
    return classified_atoms

def classify_hydrogen(atoms, coords_c):
    """Do atom typing for hydrogen atoms.

	Classify into two classes according to distance:
	   1. atomic number 101 for H adjacent to electronegative atoms (O, N, Cl, S).
	   2. atomic number 102 for H not adjacent to electronegative atoms.

	Parameters
	----------
	atoms : np.ndarray, shape=(n,)
		Atomic numbers.
	coords_c: np.ndarray, shape=(n, 3)
		Cartesian coordinates of core positions in Angstrom.

	Returns
	-------
	classified_atoms : np.ndarray, shape=(n,)
		Classified atomic numbers.
	"""
    # atomic numbers for electronegative atoms
    classified_atoms = np.copy(atoms)
    for i, atom in enumerate(atoms):
        if atom == 1:
            classified_atoms[i] = 102
            distance_vec = cdist(
                coords_c[i][np.newaxis, :], coords_c
            )  # shape (1,len(coords_c))
            distance_vec = np.where(
                distance_vec == 0, 10, distance_vec
            )  # replace distance to itself with 10
            if np.any([atoms[i] in elec_negs for i in np.where(distance_vec < 1.8)[1]]):
                classified_atoms[i] = 101
    return classified_atoms


def classify_hydrogen2(atoms, coords_c):
    """Do atom typing for hydrogen atoms.

	Classify into two classes according to distance:
	   1. atomic number 101 for H adjacent to C
	   2. atomic number 102 for H adjacent to N
	   3. atomic number 103 for H adjacent to O
	   4. atomic number 104 for H adjacent to S


	Parameters
	----------
	atoms : np.ndarray, shape=(n,)
		Atomic numbers.
	coords_c: np.ndarray, shape=(n, 3)
		Cartesian coordinates of core positions in Angstrom.

	Returns
	-------
	classified_atoms : np.ndarray, shape=(n,)
		Classified atomic numbers.
	"""
    # atomic numbers for electronegative atoms
    classified_atoms = np.copy(atoms)
    for i, atom in enumerate(atoms):
        if atom == 1:
            classified_atoms[i] = 102
            distance_vec = cdist(
                coords_c[i][np.newaxis, :], coords_c
            )  # shape (1,len(coords_c))
            distance_vec = np.where(distance_vec == 0, 10, distance_vec)[
                0
            ]  # replace distance to itself with 10
            neighbor = atoms[np.argmin(distance_vec)]
            if neighbor in carbon_num:
                classified_atoms[i] = 101
            elif neighbor in nitrogen_num:
                classified_atoms[i] = 102
            elif neighbor in oxygen_num:
                classified_atoms[i] = 103
            elif neighbor == 16:
                classified_atoms[i] = 104
            else:
                raise ValueError(f"Hydrogen bonded to {neighbor}")
    return classified_atoms

def classify_hydrogen3(atoms, coords_c):
    """Do atom typing for hydrogen atoms.

	Classify into two classes according to distance:
	   1. atomic number 101 for H adjacent to electronegative atoms (O, N, Cl, S).
	   2. atomic number 102 for H not adjacent to electronegative atoms.
	   3. atomic number 103 for H adjacent to positively charged atoms

	Parameters
	----------
	atoms : np.ndarray, shape=(n,)
		Atomic numbers.
	coords_c: np.ndarray, shape=(n, 3)
		Cartesian coordinates of core positions in Angstrom.

	Returns
	-------
	classified_atoms : np.ndarray, shape=(n,)
		Classified atomic numbers.
	"""
    # atomic numbers for electronegative atoms
    classified_atoms = np.copy(atoms)
    for i, atom in enumerate(atoms):
        if atom == 1:
            classified_atoms[i] = 102
            distance_vec = cdist(
                coords_c[i][np.newaxis, :], coords_c
            )  # shape (1,len(coords_c))
            distance_vec = np.where(
                distance_vec == 0, 10, distance_vec
            )  # replace distance to itself with 10
            if np.any([atoms[i] in [6001,7001] for i in np.where(distance_vec < 1.8)[1]]):
                classified_atoms[i] = 103
            elif np.any([atoms[i] in elec_negs for i in np.where(distance_vec < 1.8)[1]]):
                classified_atoms[i] = 101
    return classified_atoms


def classify_oxygen(atoms, coords_c):
    """Do atom typing for oxygen atoms.

	Classify into two classes according to distance:
	   1. atomic number 801 for O with 2 neighbors (sp3).
	   2. atomic number 802 for O with 1 neighbor (sp2, carbonyl).

	Parameters
	----------
	atoms : np.ndarray, shape=(n,)
		Atomic numbers.
	coords_c: np.ndarray, shape=(n, 3)
		Cartesian coordinates of core positions in Angstrom.

	Returns
	-------
	classified_atoms : np.ndarray, shape=(n,)
		Classified atomic numbers.
	"""
    # atomic numbers for electronegative atoms
    classified_atoms = np.copy(atoms)
    for i, atom in enumerate(atoms):
        if atom == 8:
            distance_vec = cdist(
                coords_c[i][np.newaxis, :], coords_c
            )  # shape (1,len(coords_c))
            distance_vec = np.where(
                distance_vec == 0, 10, distance_vec
            )  # replace distance to itself with 10
            n_neighbor = len(np.where(distance_vec < 1.8)[1])
            if n_neighbor == 1:
                classified_atoms[i] = 802
            elif n_neighbor == 2:
                classified_atoms[i] = 801
            else:
                classified_atoms[i] = 801
                print(
                    f"Oxygen with {n_neighbor} neighbors detected. May need to consider charge"
                )
            # raise ValueError(f"Oxygen has {n_neighbor} number of neighbors. Unable to classify")
    return classified_atoms


def classify_oxygen2(atoms, coords_c):
    """Do atom typing for oxygen atoms.

	Classify into two classes according to distance:
	   1. atomic number 801 for O with 2 or 3 neighbors (sp3, might have H-bond) and no electronegative neighbors.
	   2. atomic number 802 for O with 2 or 3 neighbors (sp3) and with indirect electronegative neighbors.
	   3. atomic number 803 for O with 2 or 3 neighbors (sp3) and with direct electronegative neighbors.
	   3. atomic number 804 for O with 1 neighbor (sp2, carbonyl) and no electronegative neighbors.
	   4. atomic number 805 for O with 1 neighbor (sp2, carbonyl) and with indirect electronegative neighbors.
	   5. atomic number 806 for O with 1 neighbor (sp2, carbonyl) and with direct electronegative neighbors.

	Parameters
	----------
	atoms : np.ndarray, shape=(n,)
		Atomic numbers.
	coords_c: np.ndarray, shape=(n, 3)
		Cartesian coordinates of core positions in Angstrom.

	Returns
	-------
	classified_atoms : np.ndarray, shape=(n,)
		Classified atomic numbers.
	"""
    classified_atoms = np.copy(atoms)
    for i, atom in enumerate(atoms):
        if atom == 8:
            distance_vec = cdist(
                coords_c[i][np.newaxis, :], coords_c
            )  # shape (1,len(coords_c))
            distance_vec = np.where(
                distance_vec == 0, 10, distance_vec
            )  # replace distance to itself with 10
            neighbors = np.where(distance_vec < 1.8)[1]
            indirect_neighbors = np.where((distance_vec >= 1.8) & (distance_vec < 2.6))[
                1
            ]
            if len(neighbors) == 1:
                if np.any([atoms[i] in elec_negs for i in neighbors]):
                    classified_atoms[i] = 806
                elif np.any([atoms[i] in elec_negs for i in indirect_neighbors]):
                    classified_atoms[i] = 805
                else:
                    classified_atoms[i] = 804
            elif len(neighbors) == 2 or len(neighbors) == 3:
                if np.any([atoms[i] in elec_negs for i in neighbors]):
                    classified_atoms[i] = 803
                if np.any([atoms[i] in elec_negs for i in indirect_neighbors]):
                    classified_atoms[i] = 802
                else:
                    classified_atoms[i] = 801
            else:
                # print(f"Oxygen with {len(neighbors)} neighbors detected. May need to consider charge")
                raise ValueError(
                    f"Oxygen has {len(neighbors)} number of neighbors. Unable to classify"
                )
    return classified_atoms


def classify_oxygen3(atoms, coords_c):
    """Do atom typing for oxygen atoms.

	Classify into two classes according to distance:
	   1. atomic number 801 for O with direct electronegative neighbors.
	   2. atomic number 802 for O with indirect electronegative neighbors (ideally 2 bonds away).
	   3. atomic number 803 for O with no electronegative neighbors.

	Parameters
	----------
	atoms : np.ndarray, shape=(n,)
		Atomic numbers.
	coords_c: np.ndarray, shape=(n, 3)
		Cartesian coordinates of core positions in Angstrom.

	Returns
	-------
	classified_atoms : np.ndarray, shape=(n,)
		Classified atomic numbers.
	"""
    # atomic numbers for electronegative atoms
    classified_atoms = np.copy(atoms)
    for i, atom in enumerate(atoms):
        if atom == 8:
            distance_vec = cdist(
                coords_c[i][np.newaxis, :], coords_c
            )  # shape (1,len(coords_c))
            distance_vec = np.where(
                distance_vec == 0, 10, distance_vec
            )  # replace distance to itself with 10
            neighbors = np.where(distance_vec < 1.8)[1]
            indirect_neighbors = np.where((distance_vec >= 1.8) & (distance_vec < 2.6))[
                1
            ]
            if np.any([atoms[i] in elec_negs for i in neighbors]):
                classified_atoms[i] = 801
            elif np.any([atoms[i] in elec_negs for i in indirect_neighbors]):
                classified_atoms[i] = 802
            else:
                classified_atoms[i] = 803

    return classified_atoms


def classify_nitrogen(atoms, coords_c):
    """Do atom typing for nitrogen atoms.
	#TODO: more ideally using hybridization state

	Classify into two classes according to distance:
	   1. atomic number 701 for primary N with 2 or 3 H neighbors.
	   2. atomic number 702 for secondary N with 1 H neighbor.
	   3. atomic number 703 for tertiary N with 0 H neighbor.

	Parameters
	----------
	atoms : np.ndarray, shape=(n,)
		Atomic numbers.
	coords_c: np.ndarray, shape=(n, 3)
		Cartesian coordinates of core positions in Angstrom.

	Returns
	-------
	classified_atoms : np.ndarray, shape=(n,)
		Classified atomic numbers.
	"""
    # atomic numbers for electronegative atoms
    classified_atoms = np.copy(atoms)
    for i, atom in enumerate(atoms):
        if atom == 7:
            distance_vec = cdist(
                coords_c[i][np.newaxis, :], coords_c
            )  # shape (1,len(coords_c))
            distance_vec = np.where(
                distance_vec == 0, 10, distance_vec
            )  # replace distance to itself with 10
            neighbors = np.where(distance_vec < 1.8)[1]
            n_H_neighbor = 0
            for j in neighbors:
                if atoms[j] in hydrogen_num:
                    n_H_neighbor += 1
            if n_H_neighbor == 2 or n_H_neighbor == 3:
                classified_atoms[i] = 701
            elif n_H_neighbor == 1:
                classified_atoms[i] = 702
            elif n_H_neighbor == 0:
                classified_atoms[i] = 703
            else:
                raise ValueError(
                    f"Nitrogen has {n_H_neighbor} number of hydrogen neighbors. Unable to classify"
                )
    return classified_atoms


def classify_nitrogen2(atoms, coords_c):
    """Do atom typing for nitrogen atoms.

	Classify into two classes according to distance:
	   1. atomic number 701 for N with 3 neighbors and direct electronegative neighbors.
	   2. atomic number 702 for N with 3 neighbors and indirect electronegative neighbors (ideally 2 bonds away).
	   3. atomic number 703 for N with 3 neighbors and no electronegative neighbors.
	   4. atomic number 704 for N with 2 neighbors and direct electronegative neighbors.
	   5. atomic number 705 for N with 2 neighbors and indirect electronegative neighbors (ideally 2 bonds away).
	   6. atomic number 706 for N with 2 neighbors and no electronegative neighbors.
	   7. atomic number 707 for N with 1 neighbor


	Parameters
	----------
	atoms : np.ndarray, shape=(n,)
		Atomic numbers.
	coords_c: np.ndarray, shape=(n, 3)
		Cartesian coordinates of core positions in Angstrom.

	Returns
	-------
	classified_atoms : np.ndarray, shape=(n,)
		Classified atomic numbers.
	"""
    # atomic numbers for electronegative atoms
    classified_atoms = np.copy(atoms)
    for i, atom in enumerate(atoms):
        if atom == 7:
            distance_vec = cdist(
                coords_c[i][np.newaxis, :], coords_c
            )  # shape (1,len(coords_c))
            distance_vec = np.where(
                distance_vec == 0, 10, distance_vec
            )  # replace distance to itself with 10
            neighbors = np.where(distance_vec < 1.8)[1]
            indirect_neighbors = np.where((distance_vec >= 1.8) & (distance_vec < 2.6))[
                1
            ]
            if len(neighbors) == 3:
                if np.any([atoms[i] in elec_negs for i in neighbors]):
                    classified_atoms[i] = 701
                elif np.any([atoms[i] in elec_negs for i in indirect_neighbors]):
                    classified_atoms[i] = 702
                else:
                    classified_atoms[i] = 703
            elif len(neighbors) == 2:
                if np.any([atoms[i] in elec_negs for i in neighbors]):
                    classified_atoms[i] = 704
                elif np.any([atoms[i] in elec_negs for i in indirect_neighbors]):
                    classified_atoms[i] = 705
                else:
                    classified_atoms[i] = 706
            elif len(neighbors) == 1:
                classified_atoms[i] = 707
            # elif len(neighbors) == 4:
            #     classified_atoms[i] = 708
            else:
                # print(f"Nitrogen with {len(neighbors)} neighbors detected. May need to consider charge")
                raise ValueError(
                    f"Nitrogen has {len(neighbors)} number of neighbors. Unable to classify"
                )

    return classified_atoms


def classify_nitrogen3(atoms, coords_c):
    """Do atom typing for nitrogen atoms.

	Classify into two classes according to distance:
	   1. atomic number 701 for N with direct electronegative neighbors.
	   2. atomic number 702 for N with indirect electronegative neighbors (ideally 2 bonds away).
	   3. atomic number 703 for N with no electronegative neighbors.

	Parameters
	----------
	atoms : np.ndarray, shape=(n,)
		Atomic numbers.
	coords_c: np.ndarray, shape=(n, 3)
		Cartesian coordinates of core positions in Angstrom.

	Returns
	-------
	classified_atoms : np.ndarray, shape=(n,)
		Classified atomic numbers.
	"""
    # atomic numbers for electronegative atoms
    classified_atoms = np.copy(atoms)
    for i, atom in enumerate(atoms):
        if atom == 7:
            distance_vec = cdist(
                coords_c[i][np.newaxis, :], coords_c
            )  # shape (1,len(coords_c))
            distance_vec = np.where(
                distance_vec == 0, 10, distance_vec
            )  # replace distance to itself with 10
            neighbors = np.where(distance_vec < 1.8)[1]
            indirect_neighbors = np.where((distance_vec >= 1.8) & (distance_vec < 2.6))[
                1
            ]
            if np.any([atoms[i] in elec_negs for i in neighbors]):
                classified_atoms[i] = 701
            elif np.any([atoms[i] in elec_negs for i in indirect_neighbors]):
                classified_atoms[i] = 702
            else:
                classified_atoms[i] = 703

    return classified_atoms


def add_shell(coord_s, idx):
    """add a shell at position of atom at specified idx with small random displacement"""
    new_s = coord_s[idx] + (np.random.random(3) - 0.5) * 10 ** (-3)
    return np.concatenate((coord_s, [new_s]), axis=0)


# def classify_charge(atoms, coords_c, charge):
#     """Do atom typing for nitrogen atoms.
# 	#TODO: more ideally using hybridization state
#
# 	Classify for atomic charges:
# 	   1. atomic number 7001 for N with 4 neighbors.
# 	   2. atomic number 6001 for C in ARG guanidino carbon.
#
# 	Parameters
# 	----------
# 	atoms : np.ndarray, shape=(n,)
# 		Atomic numbers.
# 	coords_c: np.ndarray, shape=(n, 3)
# 		Cartesian coordinates of core positions in Angstrom.
# 	charge: int
# 		ideal charge to be assigned
#
# 	Returns
# 	-------
# 	classified_atoms : np.ndarray, shape=(n,)
# 		Classified atomic numbers.
# 	coords_s : np.ndarray, shape=(n,)
# 		Shell positions based on charge.
# 	"""
#     # atomic numbers for electronegative atoms
#     classified_atoms = np.copy(atoms)
#     coords_s = np.copy(coords_c)
#     predicted_charge = 0
#     for i, atom in enumerate(atoms):
#         if atom in nitrogen_num:
#             # 4 bond Nitrogen
#             distance_vec = cdist(
#                 coords_c[i][np.newaxis, :], coords_c
#             )  # shape (1,len(coords_c))
#             distance_vec = np.where(
#                 distance_vec == 0, 10, distance_vec
#             )  # replace distance to itself with 10
#             neighbors = np.where(distance_vec < 1.8)[1]
#             if len(neighbors) == 4:
#                 print(f"Positive nitrogen at : {i}")
#                 classified_atoms[i] = 7001
#                 predicted_charge += 1
#         if atom in carbon_num:
#             distance_vec = cdist(
#                 coords_c[i][np.newaxis, :], coords_c
#             )  # shape (1,len(coords_c))
#             distance_vec = np.where(
#                 distance_vec == 0, 10, distance_vec
#             )  # replace distance to itself with 10
#             neighbors = np.where(distance_vec < 1.8)[1]
#             if len(neighbors) == 3:
#                 # acetate
#                 oxygen_neighbors = np.array(
#                     [i for i in neighbors if atoms[i] in oxygen_num]
#                 )
#                 if len(oxygen_neighbors) == 2:
#                     # first oxygen
#                     distance_vec_1 = cdist(
#                         coords_c[oxygen_neighbors[0]][np.newaxis, :], coords_c
#                     )  # shape (1,len(coords_c))
#                     distance_vec_1 = np.where(
#                         distance_vec_1 == 0, 10, distance_vec_1
#                     )  # replace distance to itself with 10
#                     neighbors_1 = np.where(distance_vec_1 < 1.8)[1]
#                     if len(neighbors_1) == 1:
#                         assert neighbors_1[0] == i
#                         # second oxygen
#                         distance_vec_2 = cdist(
#                             coords_c[oxygen_neighbors[1]][np.newaxis, :], coords_c
#                         )  # shape (1,len(coords_c))
#                         distance_vec_2 = np.where(
#                             distance_vec_2 == 0, 10, distance_vec_2
#                         )  # replace distance to itself with 10
#                         neighbors_2 = np.where(distance_vec_2 < 1.8)[1]
#                         if len(neighbors_2) == 1:
#                             assert neighbors_2[0] == i
#                             coords_s = add_shell(coords_s, oxygen_neighbors[0])
#                             print(f"Negative oxygen at : {oxygen_neighbors[0]}")
#                             predicted_charge -= 1
#                 # ARG carbon center
#                 nitrogen_neighbors = np.array(
#                     [i for i in neighbors if atoms[i] in nitrogen_num]
#                 )
#                 if len(nitrogen_neighbors) == 3:
#                     charged_N = False
#                     for idx in nitrogen_neighbors:
#                         if classified_atoms[idx] == 7001:
#                             charged_N = True
#                             continue
#                         distance_vec_1 = cdist(
#                             coords_c[idx][np.newaxis, :], coords_c
#                         )  # shape (1,len(coords_c))
#                         distance_vec_1 = np.where(
#                             distance_vec_1 == 0, 10, distance_vec_1
#                         )  # replace distance to itself with 10
#                         n_neighbors_1 = len(np.where(distance_vec_1 < 1.8)[1])
#                         if not n_neighbors_1 == 3:
#                             charged_N = True
#                             break
#                     if not charged_N:
#                         print(f"Positive carbon at : {i}")
#                         classified_atoms[i] = 6001
#                         predicted_charge += 1
#
#     assert charge == predicted_charge, print(
#         f"charge:{charge}  predicted charge:{predicted_charge}"
#     )
#     return classified_atoms, coords_s

def classify_charge2(atoms, coords_c, charge):
    """Do atom typing for nitrogen atoms.

	Classify for atomic charges:
	   1. atomic number 7001 for N with 4 neighbors.
	   2. atomic number 6001 for C in ARG guanidino carbon.
	   4. atomic number 110 for H directly bonded to positive N center (including ARG).
	   5. atomic number 610 for C directly bonded to positive N center.

	Parameters
	----------
	atoms : np.ndarray, shape=(n,)
		Atomic numbers.
	coords_c: np.ndarray, shape=(n, 3)
		Cartesian coordinates of core positions in Angstrom.
	charge: int
		ideal charge to be assigned

	Returns
	-------
	classified_atoms : np.ndarray, shape=(n,)
		Classified atomic numbers.
	coords_s : np.ndarray, shape=(n,)
		Shell positions based on charge.
	"""
    # atomic numbers for electronegative atoms
    classified_atoms = np.copy(atoms)
    coords_s = np.copy(coords_c)
    predicted_charge = 0
    for i, atom in enumerate(atoms):
        if atom in nitrogen_num:
            # 4 bond Nitrogen
            distance_vec = cdist(
                coords_c[i][np.newaxis, :], coords_c
            )  # shape (1,len(coords_c))
            distance_vec = np.where(
                distance_vec == 0, 10, distance_vec
            )  # replace distance to itself with 10
            neighbors = np.where(distance_vec < 1.8)[1]
            if len(neighbors) == 4:
                print(f"Positive nitrogen at : {i}")
                classified_atoms[i] = 7001
                predicted_charge += 1
                # change atom types for directly bonded H and C to 110 and 610
                h_neighbors = np.array(
                    [i for i in neighbors if atoms[i] in hydrogen_num]
                )
                for h in h_neighbors:
                    if np.linalg.norm(coords_c[i]-coords_c[h]) < 1.8:
                        classified_atoms[h] = 110
                c_neighbors = np.array(
                    [i for i in neighbors if atoms[i] in carbon_num]
                )
                for c in c_neighbors:
                    if np.linalg.norm(coords_c[i] - coords_c[c]) < 1.8:
                        classified_atoms[c] = 610
        if atom in carbon_num:
            distance_vec = cdist(
                coords_c[i][np.newaxis, :], coords_c
            )  # shape (1,len(coords_c))
            distance_vec = np.where(
                distance_vec == 0, 10, distance_vec
            )  # replace distance to itself with 10
            neighbors = np.where(distance_vec < 1.8)[1]
            if len(neighbors) == 3:
                # acetate
                oxygen_neighbors = np.array(
                    [i for i in neighbors if atoms[i] in oxygen_num]
                )
                if len(oxygen_neighbors) == 2:
                    # first oxygen
                    distance_vec_1 = cdist(
                        coords_c[oxygen_neighbors[0]][np.newaxis, :], coords_c
                    )  # shape (1,len(coords_c))
                    distance_vec_1 = np.where(
                        distance_vec_1 == 0, 10, distance_vec_1
                    )  # replace distance to itself with 10
                    neighbors_1 = np.where(distance_vec_1 < 1.8)[1]
                    if len(neighbors_1) == 1:
                        assert neighbors_1[0] == i
                        # second oxygen
                        distance_vec_2 = cdist(
                            coords_c[oxygen_neighbors[1]][np.newaxis, :], coords_c
                        )  # shape (1,len(coords_c))
                        distance_vec_2 = np.where(
                            distance_vec_2 == 0, 10, distance_vec_2
                        )  # replace distance to itself with 10
                        neighbors_2 = np.where(distance_vec_2 < 1.8)[1]
                        if len(neighbors_2) == 1:
                            assert neighbors_2[0] == i
                            coords_s = add_shell(coords_s, oxygen_neighbors[0])
                            print(f"Negative oxygen at : {oxygen_neighbors[0]}")
                            predicted_charge -= 1

                # ARG carbon center
                nitrogen_neighbors = np.array(
                    [i for i in neighbors if atoms[i] in nitrogen_num]
                )
                if len(nitrogen_neighbors) == 3:
                    charged_N = False
                    neighbors = []
                    for idx in nitrogen_neighbors:
                        if classified_atoms[idx] == 7001:
                            charged_N = True
                            continue
                        distance_vec_1 = cdist(
                            coords_c[idx][np.newaxis, :], coords_c
                        )  # shape (1,len(coords_c))
                        distance_vec_1 = np.where(
                            distance_vec_1 == 0, 10, distance_vec_1
                        )  # replace distance to itself with 10
                        neighbors_1 = np.where(distance_vec_1 < 1.8)[1]
                        if not len(neighbors_1) == 3:
                            charged_N = True
                            break
                        H_neighbor_of_N = np.array(
                            [i for i in neighbors_1 if atoms[i] in hydrogen_num]
                        )
                        if len(H_neighbor_of_N) == 2:
                            neighbors.append(H_neighbor_of_N)
                    if not charged_N:
                        print(f"Positive carbon at : {i}")
                        classified_atoms[i] = 6001
                        predicted_charge += 1
                        assert len(neighbors) == 2
                        for n in neighbors:
                            classified_atoms[n] = 110



    assert charge == predicted_charge, print(
        f"charge:{charge}  predicted charge:{predicted_charge}"
    )
    return classified_atoms, coords_s

def classify_charge(atoms, coords_c, charge):
    """Do atom typing for nitrogen atoms.

	Classify for atomic charges:
	   1. atomic number 7001 for N with 4 neighbors.
	   2. atomic number 6001 for C in ARG guanidino carbon.
	   3. atomic number 820 for acetate negative oxygens.
	   4. atomic number 110 for H directly bonded to positive N center (including ARG).
	   5. atomic number 610 for C directly bonded to positive N center.

	Parameters
	----------
	atoms : np.ndarray, shape=(n,)
		Atomic numbers.
	coords_c: np.ndarray, shape=(n, 3)
		Cartesian coordinates of core positions in Angstrom.
	charge: int
		ideal charge to be assigned

	Returns
	-------
	classified_atoms : np.ndarray, shape=(n,)
		Classified atomic numbers.
	coords_s : np.ndarray, shape=(n,)
		Shell positions based on charge.
	"""
    # atomic numbers for electronegative atoms
    classified_atoms = np.copy(atoms)
    coords_s = np.copy(coords_c)
    predicted_charge = 0
    for i, atom in enumerate(atoms):
        if atom in nitrogen_num:
            # 4 bond Nitrogen
            distance_vec = cdist(
                coords_c[i][np.newaxis, :], coords_c
            )  # shape (1,len(coords_c))
            distance_vec = np.where(
                distance_vec == 0, 10, distance_vec
            )  # replace distance to itself with 10
            neighbors = np.where(distance_vec < 1.8)[1]
            if len(neighbors) == 4:
                print(f"Positive nitrogen at : {i}")
                classified_atoms[i] = 7001
                predicted_charge += 1
                # change atom types for directly bonded H and C to 110 and 610
                h_neighbors = np.array(
                    [i for i in neighbors if atoms[i] in hydrogen_num]
                )
                for h in h_neighbors:
                    if np.linalg.norm(coords_c[i]-coords_c[h]) < 1.8:
                        classified_atoms[h] = 110
                c_neighbors = np.array(
                    [i for i in neighbors if atoms[i] in carbon_num]
                )
                for c in c_neighbors:
                    if np.linalg.norm(coords_c[i] - coords_c[c]) < 1.8:
                        classified_atoms[c] = 610
        if atom in carbon_num:
            distance_vec = cdist(
                coords_c[i][np.newaxis, :], coords_c
            )  # shape (1,len(coords_c))
            distance_vec = np.where(
                distance_vec == 0, 10, distance_vec
            )  # replace distance to itself with 10
            neighbors = np.where(distance_vec < 1.8)[1]
            if len(neighbors) == 3:
                # acetate
                oxygen_neighbors = np.array(
                    [i for i in neighbors if atoms[i] in oxygen_num]
                )
                if len(oxygen_neighbors) == 2:
                    # first oxygen
                    distance_vec_1 = cdist(
                        coords_c[oxygen_neighbors[0]][np.newaxis, :], coords_c
                    )  # shape (1,len(coords_c))
                    distance_vec_1 = np.where(
                        distance_vec_1 == 0, 10, distance_vec_1
                    )  # replace distance to itself with 10
                    neighbors_1 = np.where(distance_vec_1 < 1.8)[1]
                    if len(neighbors_1) == 1:
                        assert neighbors_1[0] == i
                        # second oxygen
                        distance_vec_2 = cdist(
                            coords_c[oxygen_neighbors[1]][np.newaxis, :], coords_c
                        )  # shape (1,len(coords_c))
                        distance_vec_2 = np.where(
                            distance_vec_2 == 0, 10, distance_vec_2
                        )  # replace distance to itself with 10
                        neighbors_2 = np.where(distance_vec_2 < 1.8)[1]
                        if len(neighbors_2) == 1:
                            assert neighbors_2[0] == i
                            coords_s = add_shell(coords_s, oxygen_neighbors[0])
                            print(f"Negative oxygen at : {oxygen_neighbors[0]}")
                            predicted_charge -= 1
                            classified_atoms[oxygen_neighbors] = 820
                # ARG carbon center
                nitrogen_neighbors = np.array(
                    [i for i in neighbors if atoms[i] in nitrogen_num]
                )
                if len(nitrogen_neighbors) == 3:
                    charged_N = False
                    neighbors = []
                    for idx in nitrogen_neighbors:
                        if classified_atoms[idx] == 7001:
                            charged_N = True
                            continue
                        distance_vec_1 = cdist(
                            coords_c[idx][np.newaxis, :], coords_c
                        )  # shape (1,len(coords_c))
                        distance_vec_1 = np.where(
                            distance_vec_1 == 0, 10, distance_vec_1
                        )  # replace distance to itself with 10
                        neighbors_1 = np.where(distance_vec_1 < 1.8)[1]
                        if not len(neighbors_1) == 3:
                            charged_N = True
                            break
                        H_neighbor_of_N = np.array(
                            [i for i in neighbors_1 if atoms[i] in hydrogen_num]
                        )
                        if len(H_neighbor_of_N) == 2:
                            neighbors.append(H_neighbor_of_N)
                    if not charged_N:
                        print(f"Positive carbon at : {i}")
                        classified_atoms[i] = 6001
                        predicted_charge += 1
                        assert len(neighbors) == 2
                        for n in neighbors:
                            classified_atoms[n] = 110



    #assert charge == predicted_charge, print(
    #    f"charge:{charge}  predicted charge:{predicted_charge}"
    #)
    print(f"charge:{predicted_charge}")
    return classified_atoms, coords_s

def classify_charge4(atoms, coords_c, charge):
    """Do atom typing for nitrogen atoms.

	Classify for atomic charges:
	   1. atomic number 7001 for N with 4 neighbors.
	   2. atomic number 6001 for C in ARG guanidino carbon.
	   3. atomic number 820 for acetate negative oxygens.

	Parameters
	----------
	atoms : np.ndarray, shape=(n,)
		Atomic numbers.
	coords_c: np.ndarray, shape=(n, 3)
		Cartesian coordinates of core positions in Angstrom.
	charge: int
		ideal charge to be assigned

	Returns
	-------
	classified_atoms : np.ndarray, shape=(n,)
		Classified atomic numbers.
	coords_s : np.ndarray, shape=(n,)
		Shell positions based on charge.
	"""
    # atomic numbers for electronegative atoms
    classified_atoms = np.copy(atoms)
    coords_s = np.copy(coords_c)
    predicted_charge = 0
    for i, atom in enumerate(atoms):
        if atom in nitrogen_num:
            # 4 bond Nitrogen
            distance_vec = cdist(
                coords_c[i][np.newaxis, :], coords_c
            )  # shape (1,len(coords_c))
            distance_vec = np.where(
                distance_vec == 0, 10, distance_vec
            )  # replace distance to itself with 10
            neighbors = np.where(distance_vec < 1.8)[1]
            if len(neighbors) == 4:
                print(f"Positive nitrogen at : {i}")
                classified_atoms[i] = 7001
                predicted_charge += 1

        if atom in carbon_num:
            distance_vec = cdist(
                coords_c[i][np.newaxis, :], coords_c
            )  # shape (1,len(coords_c))
            distance_vec = np.where(
                distance_vec == 0, 10, distance_vec
            )  # replace distance to itself with 10
            neighbors = np.where(distance_vec < 1.8)[1]
            if len(neighbors) == 3:
                # acetate
                oxygen_neighbors = np.array(
                    [i for i in neighbors if atoms[i] in oxygen_num]
                )
                if len(oxygen_neighbors) == 2:
                    # first oxygen
                    distance_vec_1 = cdist(
                        coords_c[oxygen_neighbors[0]][np.newaxis, :], coords_c
                    )  # shape (1,len(coords_c))
                    distance_vec_1 = np.where(
                        distance_vec_1 == 0, 10, distance_vec_1
                    )  # replace distance to itself with 10
                    neighbors_1 = np.where(distance_vec_1 < 1.8)[1]
                    if len(neighbors_1) == 1:
                        assert neighbors_1[0] == i
                        # second oxygen
                        distance_vec_2 = cdist(
                            coords_c[oxygen_neighbors[1]][np.newaxis, :], coords_c
                        )  # shape (1,len(coords_c))
                        distance_vec_2 = np.where(
                            distance_vec_2 == 0, 10, distance_vec_2
                        )  # replace distance to itself with 10
                        neighbors_2 = np.where(distance_vec_2 < 1.8)[1]
                        if len(neighbors_2) == 1:
                            assert neighbors_2[0] == i
                            coords_s = add_shell(coords_s, oxygen_neighbors[0])
                            print(f"Negative oxygen at : {oxygen_neighbors[0]}")
                            predicted_charge -= 1
                            classified_atoms[oxygen_neighbors] = 820
                # ARG carbon center
                nitrogen_neighbors = np.array(
                    [i for i in neighbors if atoms[i] in nitrogen_num]
                )
                if len(nitrogen_neighbors) == 3:
                    charged_N = False
                    for idx in nitrogen_neighbors:
                        if classified_atoms[idx] == 7001:
                            charged_N = True
                            continue
                        distance_vec_1 = cdist(
                            coords_c[idx][np.newaxis, :], coords_c
                        )  # shape (1,len(coords_c))
                        distance_vec_1 = np.where(
                            distance_vec_1 == 0, 10, distance_vec_1
                        )  # replace distance to itself with 10
                        n_neighbors_1 = len(np.where(distance_vec_1 < 1.8)[1])
                        if not n_neighbors_1 == 3:
                            charged_N = True
                            break
                    if not charged_N:
                        print(f"Positive carbon at : {i}")
                        classified_atoms[i] = 6001
                        predicted_charge += 1

    assert charge == predicted_charge, print(
        f"charge:{charge}  predicted charge:{predicted_charge}"
    )
    return classified_atoms, coords_s
