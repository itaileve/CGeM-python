"""Test of Utility Module."""


# Author: Farnaz Heidar-Zadeh
# Date created: May 20, 2020


import numpy as np

from cgem.utils import load_lammpstrj_esp, load_lammpstrj_xyz
from cgem.utils import compute_esp_point_charge, compute_stats


def test_mae_hcl():
    points1, esp1 = load_lammpstrj_esp("cgem/test/data/HCl_DFT_EPS.lammpstrj")
    points2, esp2 = load_lammpstrj_esp("cgem/test/data/HCl_EEM_EPS.lammpstrj")
    points3, esp3 = load_lammpstrj_esp("cgem/test/data/HCl_HIRSH_EPS.lammpstrj")
    # check points coordinates
    assert np.all(abs(points1 - points2) < 1.0e-7)
    assert np.all(abs(points1 - points3) < 1.0e-7)
    # check esp
    assert abs(compute_stats(esp1, esp2)[0] - 0.12167284025528437) < 1.0e-8
    assert abs(compute_stats(esp1, esp3)[0] - 0.06179102221175952) < 1.0e-8
    assert abs(compute_stats(esp2, esp3)[0] - 0.07753136122410496) < 1.0e-8


def test_mae_water():
    points1, esp1 = load_lammpstrj_esp("cgem/test/data/Water_DFT_EPS.lammpstrj")
    points2, esp2 = load_lammpstrj_esp("cgem/test/data/Water_EEM_EPS.lammpstrj")
    points3, esp3 = load_lammpstrj_esp("cgem/test/data/Water_HIRSH_EPS.lammpstrj")
    # check points coordinates
    assert np.all(abs(points1 - points2) < 1.0e-7)
    assert np.all(abs(points1 - points3) < 1.0e-7)
    # check esp
    assert abs(compute_stats(esp1, esp2)[0] - 0.062229278272808754) < 1.0e-8
    assert abs(compute_stats(esp1, esp3)[0] - 0.10355650456094717) < 1.0e-8
    assert abs(compute_stats(esp2, esp3)[0] - 0.13691504777198968) < 1.0e-8


def test_load_lammpstrj_xyz_hcl():
    # expected results
    nums = np.array([17, 1])
    coords = np.array([[3.1378, 0.0, 0.0], [1.8622, 0.0, 0.0]])
    qs_eem = np.array([-0.34999, 0.34999])
    qs_hir = np.array([-0.25, 0.25])

    ns, cs, qs = load_lammpstrj_xyz("cgem/test/data/HCl_EEM.lammpstrj")
    assert np.all(ns == nums)
    assert np.all(abs(cs - coords) < 1.0e-4)
    assert np.all(abs(qs - qs_eem) < 1.0e-4)

    ns, cs, qs = load_lammpstrj_xyz("cgem/test/data/HCl_HIRSH.lammpstrj")
    assert np.all(ns == nums)
    assert np.all(abs(cs - coords) < 1.0e-4)
    assert np.all(abs(qs - qs_hir) < 1.0e-4)


def test_load_lammpstrj_xyz_water():
    # expected results
    nums = np.array([8, 1, 1])
    coords = np.array([[2.5369, -0.3375, 0.0], [3.2979, 0.2462, 0.0], [1.776, 0.2462, 0.0]])
    qs_eem = np.array([-0.617072, 0.308525, 0.308547])
    qs_hir = np.array([-0.911, 0.456, 0.456])

    ns, cs, qs = load_lammpstrj_xyz("cgem/test/data/Water_EEM.lammpstrj")
    assert np.all(ns == nums)
    assert np.all(abs(cs - coords) < 1.0e-4)
    assert np.all(abs(qs - qs_eem) < 1.0e-4)

    ns, cs, qs = load_lammpstrj_xyz("cgem/test/data/Water_HIRSH.lammpstrj")
    assert np.all(ns == nums)
    assert np.all(abs(cs - coords) < 1.0e-4)
    assert np.all(abs(qs - qs_hir) < 1.0e-4)


def test_compute_esp_point_charge_hcl():
    # expected results
    coords = np.array([[3.1378, 0.0, 0.0], [1.8622, 0.0, 0.0]])
    qs_eem = np.array([-0.34999, 0.34999])
    qs_hir = np.array([-0.25, 0.25])
    points = np.array(
        [
            [-0.04742739, -0.35094465, -1.41068693],
            [+3.13777095, +0.00000000, -3.50000578],
            [+3.35753810, +0.00000000, -3.49309933],
            [+2.49918676, +0.14943446, -3.43801106],
            [+1.83536734, +1.07089449, -3.06707846],
        ]
    )
    esp_emm = np.array([0.66052488, -0.08704797, -0.11357732, 0.00012195, 0.11136009])
    esp_hir = np.array([0.47181697, -0.06217890, -0.08112898, 0.00008711, 0.07954520])
    assert np.all(abs(compute_esp_point_charge(coords, qs_eem, points) - esp_emm) < 1.0e-8)
    assert np.all(abs(compute_esp_point_charge(coords, qs_hir, points) - esp_hir) < 1.0e-8)


def test_compute_esp_point_charge_water():
    # expected results
    coords = np.array([[2.5369, -0.3375, 0.0], [3.2979, 0.2462, 0.0], [1.776, 0.2462, 0.0]])
    qs_eem = np.array([-0.617072, 0.308525, 0.308547])
    qs_hir = np.array([-0.911, 0.456, 0.456])
    points = np.array(
        [
            [2.53693169, -0.33746652, +3.04000502],
            [2.22311862, +0.13793990, +2.98615819],
            [3.07689453, +0.43125792, +2.89121659],
            [1.83124823, +1.21874379, +2.51432911],
            [1.98841570, +2.00114320, -1.86324053],
        ]
    )
    esp_emm = np.array([-0.13541696, -0.05336433, 0.00514713, 0.17242792, 0.35221125])
    esp_hir = np.array([-0.19540232, -0.07413660, 0.01235086, 0.25956569, 0.52527680])
    assert np.all(abs(compute_esp_point_charge(coords, qs_eem, points) - esp_emm) < 1.0e-8)
    assert np.all(abs(compute_esp_point_charge(coords, qs_hir, points) - esp_hir) < 1.0e-8)
