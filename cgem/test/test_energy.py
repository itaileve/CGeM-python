"""Test for Energy Terms of CGem Model."""


# Author: Farnaz Heidar-Zadeh
# Date created: May 20, 2020

import numpy as np
from numpy.testing import assert_allclose

from cgem.test.common import get_data_hcl, get_data_h2o, get_data_ch3oh
from cgem.test.common import get_data_energy_hcl, get_data_energy_h2o, get_data_energy_ch3oh

from cgem.model import compute_coulomb_cc, compute_coulomb_ss, compute_coulomb_cs
from cgem.model import compute_gaussian_ss, compute_gaussian_cs
from cgem.parameters import get_water_parameter
from cgem import CGem


def test_coulomb_interaction_example1_hcl():
    # test against C-GeM_CHOClNS code; 14.4 pre-factor was used in this code to convert to eV
    coords_c, coords_s,q_c, q_s, a_c, a_s, _, _, _, _ = get_data_hcl()
    expected_cc, expected_ss, expected_cs, _, _ = get_data_energy_hcl(shells="core", terms=True)

    # check core-core, shell-shell, & core-shell coulomb energy (shells at core coordinates)
    cc = compute_coulomb_cc(coords_c, q_c, a_c)
    assert_allclose(expected_cc, cc * 14.4, rtol=1.0e-7, atol=0.0)

    ss = compute_coulomb_ss(coords_c, q_s, a_s)
    assert_allclose(expected_ss[0], ss * 14.4, rtol=1.0e-7, atol=0.0)
    ss, d_ss = compute_coulomb_ss(coords_c, q_s, a_s, deriv=True)
    assert_allclose(expected_ss[0], ss * 14.4, rtol=1.0e-7, atol=0.0)
    assert_allclose(expected_ss[1], d_ss * 14.4, rtol=1.0e-7, atol=0.0)

    cs = compute_coulomb_cs(coords_c, q_c, a_c, coords_c, q_s, a_s)
    assert_allclose(expected_cs[0], cs * 14.4, rtol=1.0e-7, atol=0.0)
    cs, d_cs = compute_coulomb_cs(coords_c, q_c, a_c, coords_c, q_s, a_s, deriv=True)
    assert_allclose(expected_cs[0], cs * 14.4, rtol=1.0e-7, atol=0.0)
    assert_allclose(expected_cs[1], d_cs * 14.4, rtol=1.0e-7, atol=0.0)


def test_coulomb_interaction_example2_hcl():
    # test against C-GeM_CHOClNS code; 14.4 pre-factor was used in this code to convert to eV
    coords_c, coords_s, q_c, q_s, a_c, a_s, _, _, _, _ = get_data_hcl()
    expected_cc, expected_ss, expected_cs, _, _ = get_data_energy_hcl(shells="opt", terms=True)

    # check shell-shell & core-shell coulomb energy (shells at optimized position)
    cc = compute_coulomb_cc(coords_c, q_c, a_c)
    assert_allclose(expected_cc, cc * 14.4, rtol=1.0e-7, atol=0.0)

    ss = compute_coulomb_ss(coords_s, q_s, a_s)
    assert_allclose(expected_ss[0], ss * 14.4, rtol=1.0e-7, atol=0.0)
    ss, d_ss = compute_coulomb_ss(coords_s, q_s, a_s, deriv=True)
    assert_allclose(expected_ss[0], ss * 14.4, rtol=1.0e-7, atol=0.0)
    assert_allclose(expected_ss[1], d_ss * 14.4, rtol=1.0e-7, atol=0.0)

    cs = compute_coulomb_cs(coords_c, q_c, a_c, coords_s, q_s, a_s)
    assert_allclose(expected_cs[0], cs * 14.4, rtol=1.0e-7, atol=0.0)
    cs, d_cs = compute_coulomb_cs(coords_c, q_c, a_c, coords_s, q_s, a_s, deriv=True)
    assert_allclose(expected_cs[0], cs * 14.4, rtol=1.0e-7, atol=0.0)
    assert_allclose(expected_cs[1], d_cs * 14.4, rtol=1.0e-7, atol=0.0)


def test_gaussian_interaction_example1_hcl():
    # test against C-GeM_CHOClNS code; 14.4 pre-factor was used in this code to convert to eV
    coords_c, coords_s, q_c, q_s, a_c, a_s, g_c, g_s, b_c, b_s = get_data_hcl()
    _, _, _, expected_ss, expected_cs = get_data_energy_hcl(shells="core", terms=True)

    # check shell-shell & core-shell gaussian interaction (shells at core coordinates)
    ss = compute_gaussian_ss(coords_c, g_s, b_s)
    assert_allclose(expected_ss[0], ss, rtol=1.0e-5, atol=0.0)
    ss, d_ss = compute_gaussian_ss(coords_c, g_s, b_s, deriv=True)
    assert_allclose(expected_ss[0], ss, rtol=1.0e-5, atol=0.0)
    assert_allclose(expected_ss[1], d_ss, rtol=1.0e-6, atol=0.0)

    cs = compute_gaussian_cs(coords_c, g_c, b_c, coords_c)
    assert_allclose(expected_cs[0], cs, rtol=1.0e-7, atol=0.0)
    cs, d_cs = compute_gaussian_cs(coords_c, g_c, b_c, coords_c, deriv=True)
    assert_allclose(expected_cs[0], cs, rtol=1.0e-7, atol=0.0)
    assert_allclose(expected_cs[1], d_cs, rtol=1.0e-7, atol=0.0)


def test_gaussian_interaction_example2_hcl():
    # test against C-GeM_CHOClNS code; 14.4 pre-factor was used in this code to convert to eV
    coords_c, coords_s, q_c, q_s, a_c, a_s, g_c, g_s, b_c, b_s = get_data_hcl()
    _, _, _, expected_ss, expected_cs = get_data_energy_hcl(shells="opt", terms=True)

    # check shell-shell & core-shell gaussian interaction (shells at optimized positions)
    ss = compute_gaussian_ss(coords_s, g_s, b_s)
    assert_allclose(expected_ss[0], ss, rtol=1.0e-5, atol=0.0)
    ss, d_ss = compute_gaussian_ss(coords_s, g_s, b_s, deriv=True)
    assert_allclose(expected_ss[0], ss, rtol=1.0e-5, atol=0.0)
    assert_allclose(expected_ss[1], d_ss, rtol=1.0e-5, atol=0.0)

    cs = compute_gaussian_cs(coords_c, g_c, b_c, coords_s)
    assert_allclose(expected_cs[0], cs, rtol=1.0e-7, atol=0.0)
    cs, d_cs = compute_gaussian_cs(coords_c, g_c, b_c, coords_s, deriv=True)
    assert_allclose(expected_cs[0], cs, rtol=1.0e-7, atol=0.0)
    assert_allclose(expected_cs[1], d_cs, rtol=1.0e-7, atol=0.0)


def test_coulomb_interaction_example1_h2o():
    # test against C-GeM_CHOClNS code; 14.4 pre-factor was used in this code to convert to eV
    coords_c, coords_s, q_c, q_s, a_c, a_s, _, _, _, _ = get_data_h2o()
    expected_cc, expected_ss, expected_cs, _, _ = get_data_energy_h2o(shells="core", terms=True)

    # check core-core, shell-shell, & core-shell coulomb energy (shells at core coordinates)
    cc = compute_coulomb_cc(coords_c, q_c, a_c)
    assert_allclose(expected_cc, cc * 14.4, rtol=1.0e-7, atol=0.0)

    ss = compute_coulomb_ss(coords_c, q_s, a_s)
    assert_allclose(expected_ss[0], ss * 14.4, rtol=1.0e-7, atol=0.0)
    ss, d_ss = compute_coulomb_ss(coords_c, q_s, a_s, deriv=True)
    assert_allclose(expected_ss[0], ss * 14.4, rtol=1.0e-7, atol=0.0)
    assert_allclose(expected_ss[1], d_ss * 14.4, rtol=1.0e-7, atol=0.0)

    cs = compute_coulomb_cs(coords_c, q_c, a_c, coords_c, q_s, a_s)
    assert_allclose(expected_cs[0], cs * 14.4, rtol=1.0e-7, atol=0.0)
    cs, d_cs = compute_coulomb_cs(coords_c, q_c, a_c, coords_c, q_s, a_s, deriv=True)
    assert_allclose(expected_cs[0], cs * 14.4, rtol=1.0e-7, atol=0.0)
    assert_allclose(expected_cs[1], d_cs * 14.4, rtol=1.0e-7, atol=0.0)


def test_coulomb_interaction_example2_h2o():
    # test against C-GeM_CHOClNS code; 14.4 pre-factor was used in this code to convert to eV
    coords_c, coords_s, q_c, q_s, a_c, a_s, _, _, _, _ = get_data_h2o()
    expected_cc, expected_ss, expected_cs, _, _ = get_data_energy_h2o(shells="opt", terms=True)

    # check shell-shell & core-shell coulomb energy (shells at optimized position)
    cc = compute_coulomb_cc(coords_c, q_c, a_c)
    assert_allclose(expected_cc, cc * 14.4, rtol=1.0e-7, atol=0.0)

    ss = compute_coulomb_ss(coords_s, q_s, a_s)
    assert_allclose(expected_ss[0], ss * 14.4, rtol=1.0e-7, atol=0.0)
    ss, d_ss = compute_coulomb_ss(coords_s, q_s, a_s, deriv=True)
    assert_allclose(expected_ss[0], ss * 14.4, rtol=1.0e-7, atol=0.0)
    assert_allclose(expected_ss[1], d_ss * 14.4, rtol=1.0e-7, atol=0.0)

    cs = compute_coulomb_cs(coords_c, q_c, a_c, coords_s, q_s, a_s)
    assert_allclose(expected_cs[0], cs * 14.4, rtol=1.0e-7, atol=0.0)
    cs, d_cs = compute_coulomb_cs(coords_c, q_c, a_c, coords_s, q_s, a_s, deriv=True)
    assert_allclose(expected_cs[0], cs * 14.4, rtol=1.0e-7, atol=0.0)
    assert_allclose(expected_cs[1], d_cs * 14.4, rtol=1.0e-7, atol=0.0)


def test_gaussian_interaction_example1_h2o():
    # test against C-GeM_CHOClNS code; 14.4 pre-factor was used in this code to convert to eV
    coords_c, coords_s, q_c, q_s, a_c, a_s, g_c, g_s, b_c, b_s = get_data_h2o()
    _, _, _, expected_ss, expected_cs = get_data_energy_h2o(shells="core", terms=True)

    # check shell-shell & core-shell gaussian interaction (shells at core coordinates)
    ss = compute_gaussian_ss(coords_c, g_s, b_s)
    assert_allclose(expected_ss[0], ss, rtol=1.0e-5, atol=0.0)
    ss, d_ss = compute_gaussian_ss(coords_c, g_s, b_s, deriv=True)
    assert_allclose(expected_ss[0], ss, rtol=1.0e-5, atol=0.0)
    assert_allclose(expected_ss[1], d_ss, rtol=1.0e-5, atol=0.0)

    cs = compute_gaussian_cs(coords_c, g_c, b_c, coords_c)
    assert_allclose(expected_cs[0], cs, rtol=1.0e-7, atol=0.0)
    cs, d_cs = compute_gaussian_cs(coords_c, g_c, b_c, coords_c, deriv=True)
    assert_allclose(expected_cs[0], cs, rtol=1.0e-7, atol=0.0)
    assert_allclose(expected_cs[1], d_cs, rtol=1.0e-6, atol=0.0)


def test_gaussian_interaction_example2_h2o():
    # test against C-GeM_CHOClNS code; 14.4 pre-factor was used in this code to convert to eV
    coords_c, coords_s, q_c, q_s, a_c, a_s, g_c, g_s, b_c, b_s = get_data_h2o()
    _, _, _, expected_ss, expected_cs = get_data_energy_h2o(shells="opt", terms=True)

    # check shell-shell & core-shell gaussian interaction (shells at optimized positions)
    ss = compute_gaussian_ss(coords_s, g_s, b_s)
    assert_allclose(expected_ss[0], ss, rtol=1.0e-5, atol=0.0)
    ss, d_ss = compute_gaussian_ss(coords_s, g_s, b_s, deriv=True)
    assert_allclose(expected_ss[0], ss, rtol=1.0e-5, atol=0.0)
    assert_allclose(expected_ss[1], d_ss, rtol=1.0e-5, atol=0.0)

    cs = compute_gaussian_cs(coords_c, g_c, b_c, coords_s)
    assert_allclose(expected_cs[0], cs, rtol=1.0e-6, atol=0.0)
    cs, d_cs = compute_gaussian_cs(coords_c, g_c, b_c, coords_s, deriv=True)
    assert_allclose(expected_cs[0], cs, rtol=1.0e-6, atol=0.0)
    assert_allclose(expected_cs[1], d_cs, rtol=1.0e-6, atol=0.0)


def test_coulomb_interaction_example1_ch3oh():
    # test against C-GeM_CHOClNS code; 14.4 pre-factor was used in this code to convert to eV
    coords_c, coords_s, q_c, q_s, a_c, a_s, _, _, _, _ = get_data_ch3oh()
    expected_cc, expected_ss, expected_cs, _, _ = get_data_energy_ch3oh(shells="core", terms=True)

    cc = compute_coulomb_cc(coords_c, q_c, a_c)
    assert_allclose(expected_cc, cc * 14.4, rtol=1.0e-7, atol=0.0)

    ss = compute_coulomb_ss(coords_c, q_s, a_s)
    assert_allclose(expected_ss[0], ss * 14.4, rtol=1.0e-7, atol=0.0)
    ss, d_ss = compute_coulomb_ss(coords_c, q_s, a_s, deriv=True)
    assert_allclose(expected_ss[0], ss * 14.4, rtol=1.0e-7, atol=0.0)
    assert_allclose(expected_ss[1], d_ss * 14.4, rtol=1.0e-7, atol=0.0)

    cs = compute_coulomb_cs(coords_c, q_c, a_c, coords_c, q_s, a_s)
    assert_allclose(expected_cs[0], cs * 14.4, rtol=1.0e-7, atol=0.0)
    cs, d_cs = compute_coulomb_cs(coords_c, q_c, a_c, coords_c, q_s, a_s, deriv=True)
    assert_allclose(expected_cs[0], cs * 14.4, rtol=1.0e-7, atol=0.0)
    assert_allclose(expected_cs[1], d_cs * 14.4, rtol=1.0e-7, atol=0.0)


def test_coulomb_interaction_example2_ch3oh():
    # test against C-GeM_CHOClNS code; 14.4 pre-factor was used in this code to convert to eV
    coords_c, coords_s, q_c, q_s, a_c, a_s, _, _, _, _ = get_data_ch3oh()
    expected_cc, expected_ss, expected_cs, _, _ = get_data_energy_ch3oh(shells="opt", terms=True)

    cc = compute_coulomb_cc(coords_c, q_c, a_c)
    assert_allclose(expected_cc, cc * 14.4, rtol=1.0e-7, atol=0.0)

    ss = compute_coulomb_ss(coords_s, q_s, a_s)
    assert_allclose(expected_ss[0], ss * 14.4, rtol=1.0e-7, atol=0.0)
    ss, d_ss = compute_coulomb_ss(coords_s, q_s, a_s, deriv=True)
    assert_allclose(expected_ss[0], ss * 14.4, rtol=1.0e-7, atol=0.0)
    assert_allclose(expected_ss[1], d_ss * 14.4, rtol=1.0e-7, atol=1.0e-7)

    cs = compute_coulomb_cs(coords_c, q_c, a_c, coords_s, q_s, a_s)
    assert_allclose(expected_cs[0], cs * 14.4, rtol=1.0e-7, atol=0.0)
    cs, d_cs = compute_coulomb_cs(coords_c, q_c, a_c, coords_s, q_s, a_s, deriv=True)
    assert_allclose(expected_cs[0], cs * 14.4, rtol=1.0e-7, atol=0.0)
    assert_allclose(expected_cs[1], d_cs * 14.4, rtol=1.0e-7, atol=1.0e-7)


def test_gaussian_interaction_example1_ch3o():
    # test against C-GeM_CHOClNS code; 14.4 pre-factor was used in this code to convert to eV
    coords_c, coords_s, q_c, q_s, a_c, a_s, g_c, g_s, b_c, b_s = get_data_ch3oh()
    _, _, _, expected_ss, expected_cs = get_data_energy_ch3oh(shells="core", terms=True)

    # check shell-shell & core-shell gaussian interaction (shells at core coordinates)

    ss = compute_gaussian_ss(coords_c, g_s, b_s, penalty_param=(0,200))
    assert_allclose(expected_ss[0], ss, rtol=1.0e-5, atol=0.0)
    ss, d_ss = compute_gaussian_ss(coords_c, g_s, b_s, deriv=True, penalty_param=(0,200))
    assert_allclose(expected_ss[0], ss, rtol=1.0e-5, atol=0.0)
    assert_allclose(expected_ss[1], d_ss, rtol=1.0e-7, atol=1.0e-7)

    cs = compute_gaussian_cs(coords_c, g_c, b_c, coords_c)
    assert_allclose(expected_cs[0], cs, rtol=1.0e-7, atol=0.0)
    cs, d_cs = compute_gaussian_cs(coords_c, g_c, b_c, coords_c, deriv=True)
    assert_allclose(expected_cs[0], cs, rtol=1.0e-7, atol=0.0)
    assert_allclose(expected_cs[1], d_cs, rtol=1.0e-7, atol=0.0)


def test_gaussian_interaction_example2_ch3o():
    # test against C-GeM_CHOClNS code; 14.4 pre-factor was used in this code to convert to eV
    coords_c, coords_s, q_c, q_s, a_c, a_s, g_c, g_s, b_c, b_s = get_data_ch3oh()
    _, _, _, expected_ss, expected_cs = get_data_energy_ch3oh(shells="opt", terms=True)

    # check shell-shell & core-shell gaussian interaction (shells away from core)
    ss = compute_gaussian_ss(coords_s, g_s, b_s, penalty_param=(0,200))
    assert_allclose(expected_ss[0], ss, rtol=1.0e-5, atol=0.0)
    ss, d_ss = compute_gaussian_ss(coords_s, g_s, b_s, deriv=True, penalty_param=(0,200))
    assert_allclose(expected_ss[0], ss, rtol=1.0e-5, atol=0.0)
    assert_allclose(expected_ss[1], d_ss, rtol=1.0e-5, atol=0.0)

    cs = compute_gaussian_cs(coords_c, g_c, b_c, coords_s)
    assert_allclose(expected_cs[0], cs, rtol=1.0e-7, atol=0.0)
    cs, d_cs = compute_gaussian_cs(coords_c, g_c, b_c, coords_s, deriv=True)
    assert_allclose(expected_cs[0], cs, rtol=1.0e-7, atol=0.0)
    assert_allclose(expected_cs[1], d_cs, rtol=1.0e-7, atol=1.0e-7)


def test_cgem_forces_water():
    atnums = np.array([8,1,1,8,1,1])
    coords_c = np.array([[-0.328759, 0.480981, 0.429003], [0.347488, 0.593649, 1.10142], [-1.1352, 0.123382, 0.87853],
                         [2.07155, 0.672402, 1.60723], [1.89096, 0.228659, 2.44428], [2.60603, 1.50711, 1.76977]])
    coords_s = np.array([[-0.328759, 0.490981, 0.429003], [0.347488, 0.603649, 1.10142], [-1.1352, 0.133382, 0.87853],
                         [2.07155, 0.682402, 1.60723], [1.89096, 0.238659, 2.44428], [2.60603, 1.51711, 1.76977]])
    force_c = np.array([[-1.12323, 6.68206, 11.2368], [63.6927, 22.6583, 55.4689], [-91.0912, -32.5791, 34.3574],
                        [-19.4674, 11.8366, 2.15887], [-21.3478, -53.566, 88.6342], [52.7556, 88.1254, 4.71133]])
    force_s = np.array([[1.41519, -10.5105, 16.6487], [-46.0523, -18.4105, -66.7392], [89.7946, 33.6536, -48.2088],
                        [3.67051, -1.97835, 12.734], [21.8278, 45.5031, -94.7706], [-54.0745, -91.4146, -16.2316]])
    params = get_water_parameter()
    cgem = CGem.from_molecule(atnums,coords_c,coords_s=coords_s,opt_shells=False,r_cut=10,**params)
    _,cgem_force_s = cgem.compute_energy(coords_s,True)
    cgem_force_c = cgem.force
    #convert kcal/mol/A to eV/A
    print(cgem_force_c*	23.0609)
    print(cgem_force_s* 23.0609)
    # assert_allclose(force_s,cgem_force_s,rtol=1.0e-7, atol=1.0e-7)


def test_cgem_forces_fdc_water():
    #finite difference check
    atnums = np.array([8,1,1,8,1,1])
    coords_c = np.array([[-0.328759, 0.480981, 0.429003], [0.347488, 0.593649, 1.10142], [-1.1352, 0.123382, 0.87853],
                         [2.07155, 0.672402, 1.60723], [1.89096, 0.228659, 2.44428], [2.60603, 1.50711, 1.76977]])
    coords_s = np.array([[-0.328759, 0.490981, 0.429003], [0.347488, 0.603649, 1.10142], [-1.1352, 0.133382, 0.87853],
                         [2.07155, 0.682402, 1.60723], [1.89096, 0.238659, 2.44428], [2.60603, 1.51711, 1.76977]])

    # force_c = np.array([[-1.12323, 6.68206, 11.2368], [63.6927, 22.6583, 55.4689], [-91.0912, -32.5791, 34.3574],
    #                     [-19.4674, 11.8366, 2.15887], [-21.3478, -53.566, 88.6342], [52.7556, 88.1254, 4.71133]])
    # force_s = np.array([[1.41519, -10.5105, 16.6487], [-46.0523, -18.4105, -66.7392], [89.7946, 33.6536, -48.2088],
    #                     [3.67051, -1.97835, 12.734], [21.8278, 45.5031, -94.7706], [-54.0745, -91.4146, -16.2316]])
    diff = 0.000001
    params = get_water_parameter()
    rcut = None
    cgem = CGem.from_molecule(atnums,coords_c,coords_s=coords_s,opt_shells=False, penalty_param=(0,200),r_cut=rcut,**params)
    cgem_energy, cgem_force_s = cgem.compute_energy(coords_s, True)
    cgem_force_c = cgem.force
    fd_force_core = np.zeros((len(coords_c),3))
    fd_force_shell = np.zeros((len(coords_s),3))
    for i in range(len(coords_c)):
        for j in range(len(coords_c[0])):
            coords_s_2 = np.copy(coords_s)
            coords_s_2[i][j]+=diff
            coords_c_2 = np.copy(coords_c)
            coords_c_2[i][j] += diff
            cgem2 = CGem.from_molecule(atnums, coords_c, coords_s=coords_s_2, opt_shells=False, penalty_param=(0,200), r_cut=rcut, **params)
            cgem3 = CGem.from_molecule(atnums,coords_c_2,coords_s=coords_s,opt_shells=False, penalty_param=(0,200),r_cut=rcut,**params)
            cgem_energy_2, cgem_force_s_2 = cgem2.compute_energy(coords_s_2, True)
            cgem_energy_3, cgem_force_s_3 = cgem3.compute_energy(coords_s, True)
            fd_force_shell[i][j] = -(cgem_energy_2 - cgem_energy) / diff
            fd_force_core[i][j] = -(cgem_energy_3-cgem_energy)/diff
    #convert kcal/mol/A to eV/A
    # print(cgem_force_s*	23.0609)
    # print(fd_force_shell * 23.0609)
    # print(cgem_force_c* 23.0609)
    # print(fd_force_core * 23.0609)
    assert_allclose(fd_force_shell,cgem_force_s,rtol=1.0e-3, atol=1.0e-3)
    assert_allclose(fd_force_core, cgem_force_c, rtol=1.0e-3, atol=1.0e-3)

if __name__ =="__main__":
    # test_cgem_forces_water()
    test_cgem_forces_fdc_water()
