"""Test of CGem Class."""


# Author: Farnaz Heidar-Zadeh
# Date created: May 20, 2020


import pytest
import numpy as np

from numpy.testing import assert_allclose

from cgem.test.common import get_data_hcl, get_data_h2o, get_data_ch3oh, get_data_cgem_parameters
from cgem.test.common import get_data_energy_hcl, get_data_energy_h2o, get_data_energy_ch3oh

from cgem.model import CGem


def test_raise_min_shell_distance():
    # test against C-GeM_CHOClNS code; 14.4 pre-factor was used in this code to convert to eV
    coords_c, coords_s, q_c, q_s, a_c, a_s, g_c, g_s, b_c, b_s = get_data_ch3oh()

    with pytest.raises(ValueError) as error:
        CGem(coords_c, coords_s, q_c, q_s, a_c, a_s, g_c, g_s, b_c, b_s, opt_shells=False,
             min_dist_s=1.0e-5)
    assert str(error.value) == "Minimum shell distance 4.04e-07 is less than 1e-05."

    with pytest.raises(ValueError) as error:
        CGem(coords_c, coords_c, q_c, q_s, a_c, a_s, g_c, g_s, b_c, b_s, opt_shells=True,
             min_dist_s=1.5e-6, penalty_param=(0,200),method="BFGS")
    assert str(error.value).startswith("Minimum shell distance")
    assert str(error.value).endswith("is less than 1.5e-06.")


def test_cgem_example1_hcl():
    # test against C-GeM_CHOClNS code; 14.4 pre-factor was used in this code to convert to eV
    coords_c, coords_s, q_c, q_s, a_c, a_s, g_c, g_s, b_c, b_s = get_data_hcl()
    coulomb, gaussian, total = get_data_energy_hcl(shells="core", terms=False)

    # check Coulomb/Gaussian/Total energy terms & derivatives for shells at the position of core
    cgem = CGem(coords_c, coords_c, q_c, q_s, a_c, a_s, g_c, g_s, b_c, b_s, opt_shells=False)

    result = cgem.compute_coulomb_terms(coords_c, deriv=False)
    assert_allclose(coulomb[0], result, rtol=1.0e-7, atol=0)
    result = cgem.compute_coulomb_terms(coords_c, deriv=True)
    assert_allclose(coulomb[0], result[0], rtol=1.0e-7, atol=0)
    assert_allclose(coulomb[1], result[1], rtol=1.0e-7, atol=0)

    result = cgem.compute_gaussian_terms(coords_c, deriv=False)
    assert_allclose(gaussian[0], result, rtol=1.0e-7, atol=0)
    result = cgem.compute_gaussian_terms(coords_c, deriv=True)
    assert_allclose(gaussian[0], result[0], rtol=1.0e-7, atol=0)
    assert_allclose(gaussian[1], result[1], rtol=1.0e-7, atol=0)

    result = cgem.compute_energy(coords_c, deriv=False)
    assert_allclose(total[0], result, rtol=1.e-7, atol=0.0)
    result = cgem.compute_energy(coords_c, deriv=True)
    assert_allclose(total[0], result[0], rtol=1.e-7, atol=0.0)
    assert_allclose(total[1], result[1], rtol=1.e-7, atol=0.0)

    # check dipole moment
    assert_allclose(cgem.dipole, np.array([0.0, 0.0, 0.0]), rtol=1.e-7, atol=0)


def test_cgem_example2_hcl():
    # test against C-GeM_CHOClNS code; 14.4 pre-factor was used in this code to convert to eV
    coords_c, coords_s, q_c, q_s, a_c, a_s, g_c, g_s, b_c, b_s = get_data_hcl()
    coulomb, gaussian, total = get_data_energy_hcl(shells="opt", terms=False)

    # check Coulomb/Gaussian/Total energy terms & derivatives for shells at optimized position
    cgem = CGem(coords_c, coords_s, q_c, q_s, a_c, a_s, g_c, g_s, b_c, b_s, opt_shells=False)

    result = cgem.compute_coulomb_terms(coords_s, deriv=False)
    assert_allclose(coulomb[0], result, rtol=1.0e-7, atol=0)
    result = cgem.compute_coulomb_terms(coords_s, deriv=True)
    assert_allclose(coulomb[0], result[0], rtol=1.0e-7, atol=0)
    assert_allclose(coulomb[1], result[1], rtol=1.0e-7, atol=0)

    result = cgem.compute_gaussian_terms(coords_s, deriv=False)
    assert_allclose(gaussian[0], result, rtol=1.0e-7, atol=0)
    result = cgem.compute_gaussian_terms(coords_s, deriv=True)
    assert_allclose(gaussian[0], result[0], rtol=1.0e-7, atol=0)
    assert_allclose(gaussian[1], result[1], rtol=1.0e-7, atol=0)

    result = cgem.compute_energy(coords_s, deriv=False)
    assert_allclose(total[0], result, rtol=1.e-7, atol=0.0)
    result = cgem.compute_energy(coords_s, deriv=True)
    assert_allclose(total[0], result[0], rtol=1.e-7, atol=0.0)
    assert_allclose(total[1], result[1], rtol=1.e-7, atol=1.e-7)

    # check dipole moment
    assert_allclose(cgem.dipole, np.array([-0.228831958877, 0.0, 0.0]), rtol=1.e-7, atol=0)


def test_cgem_example1_h2o():
    # test against C-GeM_CHOClNS code; 14.4 pre-factor was used in this code to convert to eV
    coords_c, coords_s, q_c, q_s, a_c, a_s, g_c, g_s, b_c, b_s = get_data_h2o()
    coulomb, gaussian, total = get_data_energy_h2o(shells="core", terms=False)

    # check Coulomb/Gaussian/Total energy terms & derivatives for shells at the position of core
    cgem = CGem(coords_c, coords_c, q_c, q_s, a_c, a_s, g_c, g_s, b_c, b_s, opt_shells=False)

    result = cgem.compute_coulomb_terms(coords_c, deriv=False)
    assert_allclose(coulomb[0], result, rtol=1.0e-7, atol=0)
    result = cgem.compute_coulomb_terms(coords_c, deriv=True)
    assert_allclose(coulomb[0], result[0], rtol=1.0e-7, atol=0)
    assert_allclose(coulomb[1], result[1], rtol=1.0e-7, atol=0)

    result = cgem.compute_gaussian_terms(coords_c, deriv=False)
    assert_allclose(gaussian[0], result, rtol=1.0e-7, atol=0)
    result = cgem.compute_gaussian_terms(coords_c, deriv=True)
    assert_allclose(gaussian[0], result[0], rtol=1.0e-7, atol=0)
    assert_allclose(gaussian[1], result[1], rtol=1.0e-7, atol=1.0e-7)

    result = cgem.compute_energy(coords_c)
    assert_allclose(total[0], result, rtol=1.e-7, atol=0.0)
    result = cgem.compute_energy(coords_c, deriv=True)
    assert_allclose(total[0], result[0], rtol=1.0e-7, atol=0)
    assert_allclose(total[1], result[1], rtol=1.0e-7, atol=0)

    # check dipole moment
    assert_allclose(cgem.dipole, np.array([0.0, 0.0, 0.0]), rtol=1.e-7, atol=0)


def test_cgem_example2_h2o():
    # test against C-GeM_CHOClNS code; 14.4 pre-factor was used in this code to convert to eV
    coords_c, coords_s, q_c, q_s, a_c, a_s, g_c, g_s, b_c, b_s = get_data_h2o()
    coulomb, gaussian, total = get_data_energy_h2o(shells="opt", terms=False)

    # check Coulomb/Gaussian/Total energy terms & derivatives for shells at optimized position
    cgem = CGem(coords_c, coords_s, q_c, q_s, a_c, a_s, g_c, g_s, b_c, b_s, opt_shells=False)

    result = cgem.compute_coulomb_terms(coords_s, deriv=False)
    assert_allclose(coulomb[0], result, rtol=1.0e-7, atol=0)
    result = cgem.compute_coulomb_terms(coords_s, deriv=True)
    assert_allclose(coulomb[0], result[0], rtol=1.0e-7, atol=0)
    assert_allclose(coulomb[1], result[1], rtol=1.0e-7, atol=0)

    result = cgem.compute_gaussian_terms(coords_s, deriv=False)
    assert_allclose(gaussian[0], result, rtol=1.0e-6, atol=0)
    result = cgem.compute_gaussian_terms(coords_s, deriv=True)
    assert_allclose(gaussian[0], result[0], rtol=1.0e-6, atol=0)
    assert_allclose(gaussian[1], result[1], rtol=1.0e-6, atol=0)

    result = cgem.compute_energy(coords_s, deriv=False)
    assert_allclose(total[0], result, rtol=1.e-7, atol=0.0)
    result = cgem.compute_energy(coords_s, deriv=True)
    assert_allclose(total[0], result[0], rtol=1.e-7, atol=0.0)
    assert_allclose(total[1], result[1], rtol=1.e-7, atol=1.e-7)

    # check dipole moment
    assert_allclose(cgem.dipole, [4.4766464e-05, 0.404272398273, 0.0], rtol=1.0e-7, atol=0)


def test_cgem_example1_ch3oh():
    # test against C-GeM_CHOClNS code; 14.4 pre-factor was used in this code to convert to eV
    coords_c, coords_s, q_c, q_s, a_c, a_s, g_c, g_s, b_c, b_s = get_data_ch3oh()
    coulomb, gaussian, total = get_data_energy_ch3oh(shells="core", terms=False)

    # check Coulomb/Gaussian/Total energy terms & derivatives for shells at the position of core
    cgem = CGem(coords_c, coords_c, q_c, q_s, a_c, a_s, g_c, g_s, b_c, b_s, opt_shells=False,penalty_param=(0,200))

    result = cgem.compute_coulomb_terms(coords_c, deriv=False)
    assert_allclose(coulomb[0], result, rtol=1.0e-7, atol=0)
    result = cgem.compute_coulomb_terms(coords_c, deriv=True)
    assert_allclose(coulomb[0], result[0], rtol=1.0e-7, atol=0)
    assert_allclose(coulomb[1], result[1], rtol=1.0e-7, atol=0)

    result = cgem.compute_gaussian_terms(coords_c, deriv=False)
    assert_allclose(gaussian[0], result, rtol=1.0e-7, atol=0)
    result = cgem.compute_gaussian_terms(coords_c, deriv=True)
    assert_allclose(gaussian[0], result[0], rtol=1.0e-7, atol=0)
    assert_allclose(gaussian[1], result[1], rtol=1.0e-6, atol=0)

    result = cgem.compute_energy(coords_c, deriv=False)
    assert_allclose(total[0], result, rtol=1.e-7, atol=0.0)
    result = cgem.compute_energy(coords_c, deriv=True)
    assert_allclose(total[0], result[0], rtol=1.e-7, atol=0.0)
    assert_allclose(total[1], result[1], rtol=1.e-7, atol=1.e-7)

    # check dipole moment
    assert_allclose(cgem.dipole, np.array([0.0, 0.0, 0.0]), rtol=1.e-7, atol=0)


def test_cgem_example2_ch3oh():
    # test against C-GeM_CHOClNS code; 14.4 pre-factor was used in this code to convert to eV
    coords_c, coords_s, q_c, q_s, a_c, a_s, g_c, g_s, b_c, b_s = get_data_ch3oh()
    coulomb, gaussian, total = get_data_energy_ch3oh(shells="opt", terms=False)

    # check Coulomb/Gaussian/Total energy terms & derivatives for shells at optimized position
    cgem = CGem(coords_c, coords_s, q_c, q_s, a_c, a_s, g_c, g_s, b_c, b_s, opt_shells=False, penalty_param=(0,200))

    result = cgem.compute_coulomb_terms(coords_s, deriv=False)
    assert_allclose(coulomb[0], result, rtol=1.0e-7, atol=0)
    result = cgem.compute_coulomb_terms(coords_s, deriv=True)
    assert_allclose(coulomb[0], result[0], rtol=1.0e-7, atol=0)
    assert_allclose(coulomb[1], result[1], rtol=1.0e-7, atol=1.0e-7)

    result = cgem.compute_gaussian_terms(coords_s, deriv=False)
    assert_allclose(gaussian[0], result, rtol=1.0e-7, atol=0)
    result = cgem.compute_gaussian_terms(coords_s, deriv=True)
    assert_allclose(gaussian[0], result[0], rtol=1.0e-7, atol=0)
    assert_allclose(gaussian[1], result[1], rtol=1.0e-7, atol=1.0e-7)

    result = cgem.compute_energy(coords_s, deriv=False)
    assert_allclose(total[0], result, rtol=1.e-7, atol=0.0)
    result = cgem.compute_energy(coords_s, deriv=True)
    assert_allclose(total[0], result[0], rtol=1.e-7, atol=0.0)
    assert_allclose(total[1], result[1], rtol=1.e-7, atol=1.e-7)

    # check dipole moment
    dipole_moment = [-0.051154742611, -0.310569240314, 1.491350763919e-15]
    assert_allclose(cgem.dipole, dipole_moment, rtol=1.0e-7, atol=1.0e-7)


def test_cgem_optimize_shell_hcl():
    # test against C-GeM_CHOClNS code; 14.4 pre-factor was used in this code to convert to eV
    coords_c, coords_s, q_c, q_s, a_c, a_s, g_c, g_s, b_c, b_s = get_data_hcl()

    # use core coordinates as initial guess
    cgem = CGem(coords_c, coords_c, q_c, q_s, a_c, a_s, g_c, g_s, b_c, b_s, opt_shells=True)
    assert_allclose(cgem.coords_s, coords_s, rtol=1.e-5, atol=1.e-6)
    assert_allclose(cgem.energy, -35.437909135966, rtol=1.e-7, atol=0.)
    assert_allclose(cgem.dipole, np.array([-0.228831958877, 0.0, 0.0]), rtol=1.e-5, atol=1.e-5)

    # use shell coordinates as initial guess
    cgem = CGem(coords_c, coords_s, q_c, q_s, a_c, a_s, g_c, g_s, b_c, b_s, opt_shells=True)
    assert_allclose(cgem.coords_s, coords_s, rtol=1.e-5, atol=1.e-6)
    assert_allclose(cgem.energy, -35.437909135966, rtol=1.e-7, atol=0.)
    assert_allclose(cgem.dipole, np.array([-0.228831958877, 0.0, 0.0]), rtol=1.e-5, atol=1.e-5)


def test_cgem_optimize_shell_h2o():
    # test against C-GeM_CHOClNS code; 14.4 pre-factor was used in this code to convert to eV
    coords_c, coords_s, q_c, q_s, a_c, a_s, g_c, g_s, b_c, b_s = get_data_h2o()

    # use core coordinates as initial guess
    cgem = CGem(coords_c, coords_c, q_c, q_s, a_c, a_s, g_c, g_s, b_c, b_s)
    assert_allclose(cgem.coords_s, coords_s, rtol=1.0e-4, atol=1.e-5)
    assert_allclose(cgem.energy, -50.784432648039, rtol=1.e-7, atol=0)
    assert_allclose(cgem.dipole, [4.4766464e-05, 0.404272398273, 0.0], rtol=1.0e-7, atol=1.0e-4)

    # use shell coordinates as initial guess
    cgem = CGem(coords_c, coords_s, q_c, q_s, a_c, a_s, g_c, g_s, b_c, b_s, opt_shells=True)
    assert_allclose(cgem.coords_s, coords_s, rtol=1.e-4, atol=1.e-5)
    assert_allclose(cgem.energy, -50.784432648039, rtol=1.e-7, atol=0)
    assert_allclose(cgem.dipole, [4.4766464e-05, 0.404272398273, 0.0], rtol=1.0e-7, atol=1.0e-4)


def test_cgem_optimize_shell_ch3oh():
    # test against C-GeM_CHOClNS code; 14.4 pre-factor was used in this code to convert to eV
    coords_c, coords_s, q_c, q_s, a_c, a_s, g_c, g_s, b_c, b_s = get_data_ch3oh()

    # use core coordinates as initial guess
    cgem = CGem(coords_c, coords_c, q_c, q_s, a_c, a_s, g_c, g_s, b_c, b_s, penalty_param=(0,200), method="BFGS")
    assert_allclose(cgem.coords_s, coords_s, rtol=1.e-4, atol=1.e-5)
    assert_allclose(cgem.energy, -95.598952188766, rtol=1.e-7, atol=0)

    # use shell coordinates as initial guess
    cgem = CGem(coords_c, coords_s, q_c, q_s, a_c, a_s, g_c, g_s, b_c, b_s, opt_shells=True,
                penalty_param=(0,200), method="BFGS")
    assert_allclose(cgem.coords_s, coords_s, rtol=1.e-5, atol=1.e-5)
    assert_allclose(cgem.energy, -95.598952188766, rtol=1.e-7, atol=0)


def test_cgem_from_molecule_hcl():
    # test against C-GeM_CHOClNS code; 14.4 pre-factor was used in this code to convert to eV
    coords_c, coords_s, q_c, q_s, a_c, a_s, g_c, g_s, b_c, b_s = get_data_hcl()
    params = get_data_cgem_parameters()

    # check parameters without optimizing shell positions
    cgem = CGem.from_molecule(np.array([17, 1]), coords_c, coords_c, opt_shells=False, **params)

    assert_allclose(cgem.charge_c, q_c, rtol=1.e-7, atol=0)
    assert_allclose(cgem.charge_s, q_s, rtol=1.e-7, atol=0)
    assert_allclose(cgem.alpha_c, a_c, rtol=1.e-7, atol=0)
    assert_allclose(cgem.alpha_s, a_s, rtol=1.e-7, atol=0)
    assert_allclose(cgem.gamma_c, g_c, rtol=1.e-7, atol=0)
    assert_allclose(cgem.gamma_s, g_s, rtol=1.e-7, atol=0)
    assert_allclose(cgem.beta_c, b_c, rtol=1.e-7, atol=0)
    assert_allclose(cgem.beta_s, b_s, rtol=1.e-6, atol=0)
    assert_allclose(cgem.coords_c, coords_c, rtol=1.e-7, atol=0)
    assert_allclose(cgem.coords_s, coords_c, rtol=1.e-7, atol=0)
    assert_allclose(cgem.energy, -35.176214097689, rtol=1.e-7, atol=0.)
    assert_allclose(cgem.dipole, np.array([0.0, 0.0, 0.0]), rtol=1.e-7, atol=0)

    # check parameters with optimizing shell positions
    cgem = CGem.from_molecule(np.array([17, 1]), coords_c, None, opt_shells=True, **params)

    assert_allclose(cgem.charge_c, q_c, rtol=1.e-7, atol=0)
    assert_allclose(cgem.charge_s, q_s, rtol=1.e-7, atol=0)
    assert_allclose(cgem.alpha_c, a_c, rtol=1.e-7, atol=0)
    assert_allclose(cgem.alpha_s, a_s, rtol=1.e-7, atol=0)
    assert_allclose(cgem.gamma_c, g_c, rtol=1.e-7, atol=0)
    assert_allclose(cgem.gamma_s, g_s, rtol=1.e-7, atol=0)
    assert_allclose(cgem.beta_c, b_c, rtol=1.e-7, atol=0)
    assert_allclose(cgem.beta_s, b_s, rtol=1.e-6, atol=0)
    assert_allclose(cgem.coords_c, coords_c, rtol=1.e-7, atol=0)
    assert_allclose(cgem.coords_s, coords_s, rtol=1.e-5, atol=1.e-6)
    assert_allclose(cgem.energy, -35.437909135966, rtol=1.e-7, atol=0.)
    assert_allclose(cgem.dipole, np.array([-0.228831958877, 0.0, 0.0]), rtol=1.e-5, atol=1.e-5)


def test_cgem_from_molecule_h2o():
    # test against C-GeM_CHOClNS code; 14.4 pre-factor was used in this code to convert to eV
    coords_c, coords_s, q_c, q_s, a_c, a_s, g_c, g_s, b_c, b_s = get_data_h2o()
    params = get_data_cgem_parameters()

    # check parameters without optimizing shell positions
    cgem = CGem.from_molecule(np.array([8, 1, 1]), coords_c, coords_c, opt_shells=False, **params)

    assert_allclose(cgem.charge_c, q_c, rtol=1.e-7, atol=0)
    assert_allclose(cgem.charge_s, q_s, rtol=1.e-7, atol=0)
    assert_allclose(cgem.alpha_c, a_c, rtol=1.e-7, atol=0)
    assert_allclose(cgem.alpha_s, a_s, rtol=1.e-7, atol=0)
    assert_allclose(cgem.gamma_c, g_c, rtol=1.e-7, atol=0)
    assert_allclose(cgem.gamma_s, g_s, rtol=1.e-7, atol=0)
    assert_allclose(cgem.beta_c, b_c, rtol=1.e-7, atol=0)
    assert_allclose(cgem.beta_s, b_s, rtol=1.e-6, atol=0)
    assert_allclose(cgem.coords_c, coords_c, rtol=1.e-7, atol=0)
    assert_allclose(cgem.coords_s, coords_c, rtol=1.e-7, atol=0)
    assert_allclose(cgem.energy, -48.779482927731, rtol=1.e-7, atol=0.)
    assert_allclose(cgem.dipole, np.array([0.0, 0.0, 0.0]), rtol=1.e-7, atol=0)

    # check parameters with optimizing shell positions
    cgem = CGem.from_molecule(np.array([8, 1, 1]), coords_c, opt_shells=True, **params)

    assert_allclose(cgem.charge_c, q_c, rtol=1.e-7, atol=0)
    assert_allclose(cgem.charge_s, q_s, rtol=1.e-7, atol=0)
    assert_allclose(cgem.alpha_c, a_c, rtol=1.e-7, atol=0)
    assert_allclose(cgem.alpha_s, a_s, rtol=1.e-7, atol=0)
    assert_allclose(cgem.gamma_c, g_c, rtol=1.e-7, atol=0)
    assert_allclose(cgem.gamma_s, g_s, rtol=1.e-7, atol=0)
    assert_allclose(cgem.beta_c, b_c, rtol=1.e-7, atol=0)
    assert_allclose(cgem.beta_s, b_s, rtol=1.e-6, atol=0)
    assert_allclose(cgem.coords_c, coords_c, rtol=1.e-7, atol=0)
    assert_allclose(cgem.coords_s, coords_s, rtol=1.0e-4, atol=1.e-5)
    assert_allclose(cgem.energy, -50.784432648039, rtol=1.e-7, atol=0.)
    assert_allclose(cgem.dipole, [4.4766464e-05, 0.404272398273, 0.0], rtol=1.0e-7, atol=1.0e-4)


def test_cgem_from_molecule_ch3oh():
    # test against C-GeM_CHOClNS code; 14.4 pre-factor was used in this code to convert to eV
    coords_c, coords_s, q_c, q_s, a_c, a_s, g_c, g_s, b_c, b_s = get_data_ch3oh()
    params = get_data_cgem_parameters()
    numbers = np.array([8, 6, 1, 1, 1, 1])

    # check parameters without optimizing shell positions
    cgem = CGem.from_molecule(numbers, coords_c, coords_c, opt_shells=False, penalty_param=(0,200), **params)

    assert_allclose(cgem.charge_c, q_c, rtol=1.e-7, atol=0)
    assert_allclose(cgem.charge_s, q_s, rtol=1.e-7, atol=0)
    assert_allclose(cgem.alpha_c, a_c, rtol=1.e-7, atol=0)
    assert_allclose(cgem.alpha_s, a_s, rtol=1.e-7, atol=0)
    assert_allclose(cgem.gamma_c, g_c, rtol=1.e-7, atol=0)
    assert_allclose(cgem.gamma_s, g_s, rtol=1.e-7, atol=0)
    assert_allclose(cgem.beta_c, b_c, rtol=1.e-7, atol=0)
    assert_allclose(cgem.beta_s, b_s, rtol=1.e-6, atol=0)
    assert_allclose(cgem.coords_c, coords_c, rtol=1.e-7, atol=0)
    assert_allclose(cgem.coords_s, coords_c, rtol=1.e-7, atol=0)
    assert_allclose(cgem.energy, -94.100999691983, rtol=1.e-7, atol=0.)
    assert_allclose(cgem.dipole, np.array([0.0, 0.0, 0.0]), rtol=1.e-7, atol=0)

    # check parameters with optimizing shell positions
    cgem = CGem.from_molecule(numbers, coords_c, opt_shells=True, penalty_param=(0,200), method="BFGS", **params)

    assert_allclose(cgem.charge_c, q_c, rtol=1.e-7, atol=0)
    assert_allclose(cgem.charge_s, q_s, rtol=1.e-7, atol=0)
    assert_allclose(cgem.alpha_c, a_c, rtol=1.e-7, atol=0)
    assert_allclose(cgem.alpha_s, a_s, rtol=1.e-7, atol=0)
    assert_allclose(cgem.gamma_c, g_c, rtol=1.e-7, atol=0)
    assert_allclose(cgem.gamma_s, g_s, rtol=1.e-7, atol=0)
    assert_allclose(cgem.beta_c, b_c, rtol=1.e-7, atol=0)
    assert_allclose(cgem.beta_s, b_s, rtol=1.e-6, atol=0)
    assert_allclose(cgem.coords_c, coords_c, rtol=1.e-7, atol=0)
    assert_allclose(cgem.coords_s, coords_s, rtol=1.0e-4, atol=1.e-5)
    assert_allclose(cgem.energy, -95.598952188766, rtol=1.e-7, atol=0.)
    # check dipole moment
    dipole_moment = [-0.051154742611, -0.310569240314, 1.491350763919e-15]
    assert_allclose(cgem.dipole, dipole_moment, rtol=1.0e-7, atol=1.0e-5)


def test_cgem_electrostatic_potential_hcl():
    # test against C-GeM_CHOClNS code; 14.4 pre-factor was used in this code to convert to eV
    coords_c, coords_s, q_c, q_s, a_c, a_s, g_c, g_s, b_c, b_s = get_data_hcl()

    # check
    points = np.array([[ 3.13777095, 0.00000000, 3.50000578], [ 3.24770879, 0.00000000, 3.49827876],
                       [-0.08212635,-0.39957689,-1.34900235], [-0.44301919, 0.29928846,-0.59685673],
                       [ 3.84068956,-0.29806323, 3.41571429]])

    expected = np.array([-0.08281072, -0.08745577, 0.45339949, 0.52547680, -0.10764282])
    cgem = CGem(coords_c, coords_s, q_c, q_s, a_c, a_s, g_c, g_s, b_c, b_s, opt_shells=False)
    result = cgem.compute_electrostatic_potential_gaussian_charge(points)
    assert_allclose(expected, result, rtol=1.e-7, atol=0)

    cgem = CGem(coords_c, coords_c, q_c, q_s, a_c, a_s, g_c, g_s, b_c, b_s, opt_shells=True)
    result = cgem.compute_electrostatic_potential_gaussian_charge(points)
    assert_allclose(expected, result, rtol=1.e-5, atol=1.e-5)


def test_cgem_electrostatic_potential_h2o():
    # test against C-GeM_CHOClNS code; 14.4 pre-factor was used in this code to convert to eV
    coords_c, coords_s, q_c, q_s, a_c, a_s, g_c, g_s, b_c, b_s = get_data_h2o()

    # check
    points = np.array([[2.53693169, -0.33746652,  3.04000502],
                       [2.07608313, -0.00264045,  2.98615819],
                       [0.60768512,  0.92074692, -1.98499664],
                       [2.22311862, -0.81287297,  2.98615819 ]])
    expected = np.array([-0.35178665, -0.28787408, 0.27473619, -0.40445939])

    cgem = CGem(coords_c, coords_s, q_c, q_s, a_c, a_s, g_c, g_s, b_c, b_s, opt_shells=False)
    result = cgem.compute_electrostatic_potential_gaussian_charge(points)
    assert_allclose(expected, result, rtol=1.e-7, atol=0)

    cgem = CGem(coords_c, coords_c, q_c, q_s, a_c, a_s, g_c, g_s, b_c, b_s, opt_shells=True)
    result = cgem.compute_electrostatic_potential_gaussian_charge(points)
    assert_allclose(expected, result, rtol=1.e-4, atol=0)


def test_cgem_electrostatic_potential_ch3oh():
    # test against C-GeM_CHOClNS code; 14.4 pre-factor was used in this code to convert to eV
    coords_c, coords_s, q_c, q_s, a_c, a_s, g_c, g_s, b_c, b_s = get_data_ch3oh()

    # check
    points = np.array([[1.24548321,  0.27449969, 3.03400627],
                       [1.75799389, -1.99294705, -2.02639035],
                       [1.21012593, -0.66753869, 2.91929763]])
    expected = np.array([-0.25168226, 0.40602477, -0.08071066])

    cgem = CGem(coords_c, coords_s, q_c, q_s, a_c, a_s, g_c, g_s, b_c, b_s, opt_shells=False, penalty_param=(0,200))
    result = cgem.compute_electrostatic_potential_gaussian_charge(points)
    result_pt_charge = cgem.compute_electrostatic_potential(points)
    assert_allclose(expected, result, rtol=1.e-7, atol=0)
    assert_allclose(expected, result_pt_charge, rtol=1.e-4, atol=0)
    # check dipole moment
    dipole_moment = [-0.051154742611, -0.310569240314, 1.491350763919e-15]
    assert_allclose(cgem.dipole, dipole_moment, rtol=1.0e-7, atol=1.0e-7)

    cgem = CGem(coords_c, coords_c, q_c, q_s, a_c, a_s, g_c, g_s, b_c, b_s, opt_shells=True, penalty_param=(0,200),
                method="BFGS")
    result = cgem.compute_electrostatic_potential_gaussian_charge(points)
    result_pt_charge = cgem.compute_electrostatic_potential(points)
    assert_allclose(expected, result, rtol=1.e-4, atol=0)
    assert_allclose(expected,result_pt_charge,rtol=1.e-4,atol=0)
    # check dipole moment
    dipole_moment = [-0.051154742611, -0.310569240314, 1.491350763919e-15]
    assert_allclose(cgem.dipole, dipole_moment, rtol=1.0e-7, atol=1.0e-5)
