import os
import numpy as np
import subprocess
import glob
from cgem.model import CGem
from cgem.utils import load_lammpstrj_xyz, load_lammpstrj_esp, load_xyz
from cgem.utils import compute_stats
from cgem.attypes import *
from numpy.testing import assert_allclose

"""
manage molecules, formulating train and test set
"""


class Molecule:
    def __init__(
        self,
        nums,
        coords,
        points,
        esp,
        coords_s=None,
        charges=None,
        title=None,
        dipole=None,
    ):

        self.nums = nums
        self.coords = coords
        self.points = points
        self.esp = esp
        self.coords_s = coords_s
        if charges is None:
            charges = {}
        else:
            assert isinstance(charges, dict)
        self.charges = charges
        if title is None:
            title = ""
        self.title = title
        self.dipole = dipole


def make_dataset(
    directory,
    folders,
    typed_atom=None,
    print_message=False,
    multi_carbon=False,
    charged=False,
):

    dataset = []
    for i, folder in enumerate(folders):
        if print_message:
            print(f"{i} Loading {folder}")

        # load atomic numbers, coordinates, and EEM charges

        fname_lammpstrj = glob.glob(f"{directory}/{folder}/*_EEM.lammpstrj")

        if len(fname_lammpstrj) == 1:
            nums, coords, charge = load_lammpstrj_xyz(fname_lammpstrj[0])
        else:
            raise ValueError(f"Cannot find *_EEM.lammpstrj in {directory}/{folder}")
        charges = {"eem": charge}
        if os.path.isfile(f"{directory}/{folder}/{folder}_Hir.lammpstrj"):
            charges["Hir"] = load_lammpstrj_xyz(
                f"{directory}/{folder}/{folder}_Hir.lammpstrj"
            )[2]
            charges["HI"] = load_lammpstrj_xyz(
                f"{directory}/{folder}/{folder}_HI.lammpstrj"
            )[2]
            charges["MBIS"] = load_lammpstrj_xyz(
                f"{directory}/{folder}/{folder}_MBIS.lammpstrj"
            )[2]
            if os.path.isfile(f"{directory}/{folder}/{folder}_AVH.lammpstrj"):
                charges["AVH"] = load_lammpstrj_xyz(
                    f"{directory}/{folder}/{folder}_AVH.lammpstrj"
                )[2]
            elif os.path.isfile(f"{directory}/{folder}/{folder}_AVH_PHYS.lammpstrj"):
                charges["AVH-PHYS"] = load_lammpstrj_xyz(
                    f"{directory}/{folder}/{folder}_AVH_PHYS.lammpstrj"
                )[2]
            elif os.path.isfile(f"{directory}/{folder}/{folder}_AVH-PHYS.lammpstrj"):
                charges['AVH-PHYS'] = load_lammpstrj_xyz(f"{directory}/{folder}/{folder}_AVH-PHYS.lammpstrj")[2]

        # xyz = glob.glob(f"{directory}/{folder}/*.xyz")[0]
        # nums2,coords2 = load_xyz(xyz)
        # try:
        #     assert_allclose(nums,nums2)
        #     assert_allclose(coords,coords2,atol=1e-4)
        # except:
        #     print("xyz and eem not match:")
        #     print(folder)
        #     raise

        # load esp points and values
        points, esp = load_lammpstrj_esp(
            glob.glob(f"{directory}/{folder}/*_DFT_EPS.lammpstrj")[0]
        )
        outfile = glob.glob(f"{directory}/{folder}/*.out")
        if len(outfile) == 1:
            dipole = get_qchem_dipole(outfile[0])
        else:
            dipole = None

        coords_s = None

        if charged is not False:
            try:
                # nums1,coords_s1 = get_cores_shells_charged_2(nums,coords,folder)
                # print(nums1,coords_s1)
                charge = folder.split("_")[-1]
                # print(charge)
                if len(charge) == 2 and (charge[0] == "+" or charge[0] == "-"):
                    charge = int(charge)
                else:
                    charge = 0
                if charged == True:
                    nums, coords_s = classify_charge(nums, coords, charge)
                elif charged == 2:
                    nums, coords_s = classify_charge2(nums, coords, charge)
                elif charged == 3:
                    nums, coords_s = classify_charge3(nums, coords, charge)
                elif charged == 4:
                    nums, coords_s = classify_charge4(nums, coords, charge)
            except:
                print(f"Unable to assign charge to {folder}. Skip")
                raise
        # atom typing
        nums = classify_atom(nums, coords, typed_atom)

        if multi_carbon:
            if typed_atom is not None:
                raise ValueError("Cannot do multi carbon while specifying atom type")
            nums = classify_hydrogen(nums, coords)
            nums = classify_carbon(nums, coords)
            nums[np.where(nums == 601)] = 6004
            # nums[np.where(nums == 602)] = 6
            n_4shell_carbon = len(nums[np.where(nums == 6004)])
            # print(n_4shell_carbon)
            # print(coord.shape)
            add_shells = (
                lambda i: np.random.random((3, 3)) * 0.01
                + np.repeat(coords[i], 3).reshape(3, 3).T
            )

            if n_4shell_carbon > 0:
                added_shells = np.concatenate(
                    list(map(add_shells, np.where(nums == 6004)[0])), axis=0
                )
                assert len(added_shells) == 3 * n_4shell_carbon
                coords_s = np.concatenate((coords, added_shells), axis=0)
        # if coords_s is not None:
        #     print(len(coords_s)==len(coords))
        # add instance of Molecule to database
        mol = Molecule(
            nums,
            coords,
            points,
            esp,
            coords_s=coords_s,
            charges=charges,
            title=folder,
            dipole=dipole,
        )
        dataset.append(mol)
    return dataset


def add_shell(coord_s, idx):
    """add a shell at position of atom at specified idx with small random displacement"""
    new_s = coord_s[idx] + (np.random.random(3) - 0.5) * 10 ** (-3)
    return np.concatenate((coord_s, [new_s]), axis=0)


def get_train_test_lists(directory, fname):
    txt = open(f"{directory}/{fname}", "r+")
    content = txt.read()
    txt.close()
    lists_train = (
        content.split("***train***")[1].split("***test***")[0].strip().split("\n")
    )
    try:
        lists_test = content.split("***test***")[1].strip().split("\n")
    except IndexError:
        lists_test = None
    return lists_train, lists_test


def get_train_test_database(
    directory,
    fname,
    typed_atom=None,
    multi_carbon=False,
    print_message=False,
    charged=False,
):

    # list of train and test molecule names (strings)
    lists_train, lists_test = get_train_test_lists(directory, fname)

    # make sure there is no overlap between train and test sets
    for item in lists_train:
        assert item not in lists_test, f"{item} appears in both train and test sets."
    for item in lists_test:
        assert item not in lists_train, f"{item} appears in both test and train sets."

    # list of train and test molecule objects
    print("Loading train database ...")
    database_train = make_dataset(
        directory, lists_train, typed_atom, print_message, multi_carbon, charged=charged
    )
    print("Loading test database ...")
    if len(lists_test) > 0 and len(lists_test[0].strip()) > 0:
        database_test = make_dataset(
            directory,
            lists_test,
            typed_atom,
            print_message,
            multi_carbon,
            charged=charged,
        )
    else:
        database_test = []
    print(f"{len(database_train)} molecules for training!")
    print(f"{len(database_test)} molecules for testing!")

    return database_train, database_test


def get_qchem_dipole(fname):
    # in Debye
    command = f"grep -A 2 'Dipole Moment' {fname}"
    p = subprocess.Popen(
        command,
        universal_newlines=True,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    p = p.stdout.read().strip()
    dipole = p.split("\n")[1].split()
    assert dipole[0] == "X" and dipole[2] == "Y" and dipole[4] == "Z"
    dipole = np.array([dipole[1], dipole[3], dipole[5]], dtype=float)
    return dipole


def compute_dataset_esp_error(dataset, params):
    """Compute MAE & RMSE of ESP over a dataset of molecules given C-Gem parameters."""
    error_esp = np.zeros((len(dataset), 2))
    for i, mol in enumerate(dataset):
        # make instance of C-Gem and predict ESP
        cgem = CGem.from_molecule(
            mol.nums, mol.coords, coords_s=mol.coords_s, opt_shells=True, **params
        )
        esp = cgem.compute_electrostatic_potential(mol.points)
        # compute MAE & RMSE of actual and predicted ESP
        mae, rmse, _, _ = compute_stats(esp, mol.esp)
        error_esp[i] = [mae, rmse]
    return np.mean(error_esp, axis=0)
