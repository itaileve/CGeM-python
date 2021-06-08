"""Utility Module."""


# Author: Farnaz Heidar-Zadeh
# Date created: May 20, 2020


import numpy as np
import os
from cgem.periodic import sym2num, num2sym
from scipy.spatial.distance import cdist

#
# __all__ = ["load_lammpstrj_esp", "compute_stats", "load_lammpstrj_xyz","load_xyz", "compute_esp_point_charge",
#            "write_ESPGrid","generate_ESPGrid"]


def load_lammpstrj_esp(fname):
    # print("Loading ", fname)
    with open(fname, "r") as f:
        lines = f.readlines()
    count = int(lines[3])
    # skip header lines
    lines = [line.strip().split() for line in lines[9:]]
    points = np.array([line[0:3] for line in lines], dtype=float)
    esp = np.array([line[3] for line in lines], dtype=float)
    # TODO
    # assert arr.shape[0] == count
    return points, esp


def load_lammpstrj_xyz(fname):
    # print("Loading ", fname)
    with open(fname, "r") as f:
        lines = f.readlines()
    count = int(lines[3])
    # skip header lines
    lines = [line.strip().split() for line in lines[9:]]
    if len(lines) != count or np.any([len(line) != 5 for line in lines]):
        raise ValueError("")
    numbers = np.array([sym2num[line[0].capitalize()] for line in lines], dtype=int)
    coords = np.array([line[1:4] for line in lines], dtype=float)
    charges = np.array([line[4] for line in lines], dtype=float)
    return numbers, coords, charges


def load_xyz(fname):
    """
    load xyz file as atnums and atcoords
    Parameters
    ----------
    fname: name of .xyz file

    Returns
    -------
    numbers: np.ndarray shape (N,)
        atomic numbers for atoms in xyz file
    coords: np.ndarray shape (N,3)
        xyz coordinates for atoms in xyz file
    """
    # print("Loading ", fname)
    with open(fname, "r") as f:
        lines = f.readlines()
    count = int(lines[0])
    # skip header lines
    lines = [line.strip().split() for line in lines[2:]]
    if len(lines) != count or np.any([len(line) != 4 for line in lines]):
        raise ValueError("xyz file format incorrect")
    numbers = np.array([sym2num[line[0].capitalize()] for line in lines], dtype=int)
    coords = np.array([line[1:4] for line in lines], dtype=float)
    return numbers, coords


def load_esp(esp_file):
    """
    load esp file in format
    Parameters
    ----------
    esp_file: name of .esp file

    Returns
    -------
    pts: np.ndarray shape (M,3)
        xyz coordinates for points in esp file
    esp: np.ndarray shape (M,)
        esp value for points in esp file
    """
    with open(esp_file, "r") as f:
        lines = f.readlines()
    i = 0
    while len(lines[i].split()) != 4:
        i += 1
    pts = []
    esp = []
    for line in lines[i:]:
        l = [float(j) for j in line.split()]
        esp.append(l[0])
        pts.append(l[1:])
    pts = np.array(pts) * 0.5291772  # convert bohr to angstrom
    esp = np.array(esp) * 27.2114  # convert hartree to eV
    return pts, esp


def compute_stats(arr_a, arr_b):
    """Compute simple statistics for actual and approximate property values.

    Parameters
    ----------
    arr_a : ndarray
        Estimated property value.
    arr_b : ndarray
        Actual/reference property value

    Returns
    -------
    float, float, (float, float), float
        Mean absolute error (MAE), root mean squared error (RMSE), and (min, max) of |error| and relative error to arr_b.

    """
    # check shape of arrays match
    if arr_a.shape != arr_b.shape:
        raise ValueError("Shape of two arrays do not match!")

    # compute mean absolute error
    error = abs(arr_a - arr_b)
    mae = np.mean(error)
    rmse = np.sqrt(np.mean(error ** 2))
    relative_err = mae / np.mean(abs(arr_b))
    return mae, rmse, (np.min(error), np.max(error)), relative_err


def compute_esp_point_charge(coords, charges, points):
    # check arguments
    if coords.shape != (len(charges), 3):
        raise ValueError()
    if points.shape[1] != 3:
        raise ValueError()
    # compute electrostatic potential
    esp = np.zeros(len(points))
    for coord, charge in zip(coords, charges):
        # TODO: make conversion factor more accurate
        esp += 14.4 * charge / np.linalg.norm(points - coord, axis=1)
    return esp


def compute_dipole_point_charge(coords, charge):
    """Dipole moment vector in eAngstrom."""
    dipole = coords * charge[:, np.newaxis]
    return np.sum(dipole, axis=0)


def generate_ESPGrid(atomic_num, Coords, n_Theta=100, vdW_scale=2.0, unit_flag=1):
    """

    Parameters
    ----------
    atomic_num: np.ndarray shape(N,)
        array of atomic numbers
    Coords: np.ndarray shape (N,3)
        coordinates
    vdW_scale: float
        generate point at vdW_scale * vdW_radius
    unit_flag: int
        1 for Bohr, 0 for Angstrom

    Returns
    -------
    grid: np.ndarray shape (M,3)

    """
    # Unit flag (0=Ang,1=Bohr): unit for generated grid

    vdW_R = [0] * 20
    vdW_R[1] = 1.2 * vdW_scale
    vdW_R[6] = 1.7 * vdW_scale
    vdW_R[7] = 1.55 * vdW_scale
    vdW_R[8] = 1.52 * vdW_scale
    vdW_R[16] = 1.8 * vdW_scale
    vdW_R[17] = 1.75 * vdW_scale

    if unit_flag == 0:
        unit_conver = 1.0
    else:
        unit_conver = 1.88973

    num_atom = len(atomic_num)
    n_phi_param = n_Theta / 10

    # pts per atom type centered at 0
    def pt_atom(i):
        # i is the atomic number
        r = vdW_R[i]
        theta = np.linspace(0, np.pi, n_Theta + 1)
        n_Phi = n_phi_param * 2 * np.pi * np.abs(np.sin(theta) * r)
        n_Phi[n_Phi < 1] = 1
        n_Phi = n_Phi.astype(int)
        linspace = lambda x: np.linspace(0, 2 * np.pi, x + 1)[:-1]
        phi = np.array(list(map(linspace, n_Phi)), dtype=object)
        comb = lambda i: np.array(np.meshgrid(phi[i], theta[i])).T.reshape(-1, 2)
        phi_theta = np.concatenate(list(map(comb, np.arange(len(theta)))), axis=0)
        pts_x = r * np.sin(phi_theta[:, 1]) * np.cos(phi_theta[:, 0])
        pts_y = r * np.sin(phi_theta[:, 1]) * np.sin(phi_theta[:, 0])
        pts_z = r * np.cos(phi_theta[:, 1])
        pts = np.concatenate(
            (pts_x.reshape(-1, 1), pts_y.reshape(-1, 1), pts_z.reshape(-1, 1)), axis=1
        )
        return pts

    atomic_pt = {k: pt_atom(k) for k in np.unique(atomic_num)}
    pt_per_atom = lambda i: atomic_pt[atomic_num[i]] + Coords[i]
    idx = np.arange(len(Coords))
    all_pts = np.concatenate(list(map(pt_per_atom, idx)), axis=0)

    # remove points too close to nuclei
    # approach 1, good for small molecules, but cdist takes too much memory when doing protein
    if num_atom < 100:
        dist = cdist(Coords, all_pts)  # shape (len(Coords),len(all_points))
        vdw = (
            np.array([vdW_R[i] for i in atomic_num]) - 0.00001
        )  # -0.00001 to make sure numerical operation error don't cut out pts
        thresh = np.repeat(vdw.reshape(-1, 1), dist.shape[1], axis=1)
        pts_to_keep = (dist > thresh).T
        all_on_pts_to_keep = lambda i: all(pts_to_keep[i])
        pt_keep = np.array(list(map(all_on_pts_to_keep, np.arange(len(all_pts)))))
        grid = all_pts[pt_keep] * unit_conver
    else:
        # approach 2
        thresh = np.array([vdW_R[i] for i in atomic_num]) - 0.00001
        is_pt_kept = lambda i: np.all(cdist(all_pts[i][np.newaxis, :], Coords) > thresh)
        pt_keep = np.array(list(map(is_pt_kept, np.arange(len(all_pts)))))
        grid = all_pts[pt_keep] * unit_conver

    # print(f"{len(grid)} points for vdW {vdW_scale}")
    return grid


def write_ESPGrid(atom, coords, fname, vdW_scale=2.0, unit_flag=1):
    """
    generate ESPGrid for qchem input and return number of grid point
    Parameters
    ----------
    atom: np.ndarray shape(N,)
        array of atomic numbers
    coords: np.ndarray shape (N,3)
        coordinates
    fname: str
        name of output grid file (ususally path/ESPGrid)
    vdW_scale: float
        generate point at vdW_scale * vdW_radius
    unit_flag: int
        1 for Bohr, 0 for Angstrom

    Returns
    -------
    int
        length of generated grid file
    """
    num_atom = len(atom)
    if num_atom < 100:
        n_Theta = 100
    elif num_atom < 500:
        n_Theta = 50
    elif num_atom < 3000:
        n_Theta = 20
    else:
        n_Theta = 10
    grid = generate_ESPGrid(atom, coords, n_Theta, vdW_scale, unit_flag)
    print(f"grid generated, total of {len(grid)} grid points")
    with open(fname, "w") as f:
        for line in grid:
            f.write(" ".join(["%.8f" % i for i in line]))
            f.write("\n")
    return len(grid)


def write_ESPGrid_multilayer(
    atom,
    coords,
    fname,
    n_Theta=39,
    vdW_scales=np.linspace(1.4, 1.4 + 1.13841995766, 10),
    unit_flag=1,
):
    """
    generate ESPGrid for qchem input and return number of grid point
    Parameters
    ----------
    atom: np.ndarray shape(N,)
        array of atomic numbers
    coords: np.ndarray shape (N,3)
        coordinates
    fname: str
        name of output grid file (ususally path/ESPGrid)
    vdW_scale: float
        generate point at vdW_scale * vdW_radius
    unit_flag: int
        1 for Bohr, 0 for Angstrom

    Returns
    -------
    int
        length of generated grid file
    """
    # vdW_scales=np.linspace(1.4,1.4+1.13841995766,10)
    grids = np.array(
        [
            generate_ESPGrid(atom, coords, n_Theta, vdW_scale, unit_flag)
            for vdW_scale in vdW_scales
        ]
    )
    grid = np.concatenate(grids, axis=0)
    print(f"grid generated, total of {len(grid)} grid points")
    with open(fname, "w") as f:
        for line in grid:
            f.write(" ".join(["%.8f" % i for i in line]))
            f.write("\n")
    return len(grid)


def generate_ESPGrid_multilayer(
    atom,
    coords,
    fname=None,
    max_pt=100000,
    n_Theta=39,
    vdW_scales=np.linspace(1.4, 1.4 + 1.13841995766, 10),
    unit_flag=1,
):
    """
    generate ESPGrid for qchem input and return number of grid point
    Parameters
    ----------
    atom: np.ndarray shape(N,)
        array of atomic numbers
    coords: np.ndarray shape (N,3)
        coordinates
    fname: str
        name of output grid file (ususally path/ESPGrid), None for not writing
    vdW_scale: float
        generate point at vdW_scale * vdW_radius
    unit_flag: int
        1 for Bohr, 0 for Angstrom

    Returns
    -------
    int
        length of generated grid file
    np.ndarray(M,3)
        grid coordinates
    """
    # vdW_scales=np.linspace(1.4,1.4+1.13841995766,10)
    grids = np.array(
        [
            generate_ESPGrid(atom, coords, n_Theta, vdW_scale, unit_flag)
            for vdW_scale in vdW_scales
        ]
    )
    grid = np.concatenate(grids, axis=0)
    if len(grid) > max_pt:
        grid = grid[np.random.choice(len(grid), size=max_pt, replace=False)]
    print(f"grid generated, total of {len(grid)} grid points")
    if fname != None:
        with open(fname, "w") as f:
            for line in grid:
                f.write(" ".join(["%.8f" % i for i in line]))
                f.write("\n")

    return len(grid), grid


def plot_to_eps(plot_file):
    """
    convert plot.esp (atomic unit) generated by QChem to eps (np.ndarray shape (N,4))(unit in eV)
    """
    with open(plot_file, "r") as f:
        lines = f.readlines()

    EPS_scale = 1.0
    count = 0
    eps = []
    points = []
    for line in lines[4:]:
        x, y, z, EPS = line.split()
        EPS = float(EPS) * 27.211 / EPS_scale
        points.append([float(x) * 0.529177, float(y) * 0.529177, float(z) * 0.529177])
        eps.append(float(EPS))

    return np.array(eps), np.array(points)


def eps_to_lammpstrj(eps, points, fname):
    esp = np.concatenate((points, eps.reshape(-1, 1)), axis=1)
    # print(esp)
    with open(fname, "w") as f:
        f.write("ITEM: TIMESTEP\n")
        f.write("0\n")
        f.write("ITEM: NUMBER OF ATOMS\n")
        f.write("%i\n" % (len(eps)))
        f.write("ITEM: BOX BOUNDS pp pp pp\n")
        f.write("-1.0000000000000000e+01 1.0000000000000000e+01\n")
        f.write("-1.0000000000000000e+01 1.0000000000000000e+01\n")
        f.write("-1.0000000000000000e+01 1.0000000000000000e+01\n")
        f.write("ITEM: ATOMS x y z q\n")
        for entry in esp:
            f.write(" ".join([str(i) for i in entry]) + "\n")


def write_xyz(atoms, coords, fname):
    assert len(atoms) == len(coords)
    with open(fname, "w") as f:
        f.write(f"{len(atoms)}\n")
        f.write(f" {fname[:-4]}\n")
        for j, entry in enumerate(coords):
            if atoms[j] == "X":
                f.write("%s " % atoms[j])
            else:
                f.write("%s " % num2sym[atoms[j]])
            f.write(" ".join([str(i) for i in entry]) + "\n")


def generate_vmd(cgem, xyz_file, eps_file, fname, save_dir, suffix=""):
    points, dft_esp = load_lammpstrj_esp(eps_file)
    cgem_eps = cgem.compute_electrostatic_potential(points)
    write_xyz(
        np.full(cgem.coords_s.shape[0], "X"),
        cgem.coords_s,
        f"{save_dir}/{fname}{suffix}_shell.xyz",
    )
    # write_xyz(np.full(cgem.coords_s.shape[0], 'X'), cgem.coords_c, f"{save_dir}/{fname}{suffix}_ini_shell.xyz")
    os.system(f"cp {xyz_file} {save_dir}")
    os.system(f"cp {eps_file} {save_dir}")
    diff_eps = dft_esp - cgem_eps
    if not os.path.isdir(save_dir):
        os.system(f"mkdir {save_dir}")
    eps_to_lammpstrj(cgem_eps, points, f"{save_dir}/{fname}{suffix}_cgem_EPS.lammpstrj")
    eps_to_lammpstrj(diff_eps, points, f"{save_dir}/{fname}{suffix}_diff.lammpstrj")
