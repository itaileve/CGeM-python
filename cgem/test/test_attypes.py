"""Atom Type Module."""


import numpy as np

from numpy.testing import assert_allclose

from cgem.attypes import *


def test_carbon_classification():
    #methanol data
    atomic_nums = np.array([6, 1, 1, 1, 8, 1])
    coord_c = np.array([[-4.45028, 4.013166, 0.005401], [-4.8541, 5.049979, 0.022709],
                        [-4.79625, 3.470473, 0.913222], [-4.837763, 3.490458, -0.896556],
                        [-3.039411, 4.042686, -0.033892], [-2.737196, 4.502238, 0.762117]])
    classified = classify_carbon(atomic_nums, coord_c)
    assert_allclose(classified, [601, 1, 1, 1, 8, 1], rtol=1.0e-7, atol=1.0e-5)

    #pyrrole data
    atomic_nums = np.array([7, 1, 6, 1, 6, 1, 6, 1, 6, 1])
    coord_c = np.array([[-0.204112, 3.185019, -0.075741], [-0.221492, 4.179387, 0.130716],
                        [-1.276336, 2.387919, -0.226908], [-2.293503, 2.745807, -0.13927],
                        [-0.845624, 1.133207, -0.492811], [-1.470564, 0.265525, -0.664519],
                        [0.508669, 1.159459, -0.505233], [1.163508, 0.316585, -0.68868],
                        [0.895242, 2.430013, -0.246826], [1.899212, 2.827078, -0.177728]])
    classified = classify_carbon(atomic_nums, coord_c)
    assert_allclose(classified, [7, 1, 601, 1, 602, 1, 602, 1, 601, 1], rtol=1.0e-7, atol=1.0e-5)

def test_oxygen_classification():
    #methanol data
    atomic_nums = np.array([6, 1, 1, 1, 8, 1])
    coord_c = np.array([[-4.45028, 4.013166, 0.005401], [-4.8541, 5.049979, 0.022709],
                        [-4.79625, 3.470473, 0.913222], [-4.837763, 3.490458, -0.896556],
                        [-3.039411, 4.042686, -0.033892], [-2.737196, 4.502238, 0.762117]])
    classified = classify_oxygen(atomic_nums, coord_c)
    assert_allclose(classified, [6, 1, 1, 1, 801, 1], rtol=1.0e-7, atol=1.0e-5)

def test_nitrogen_classification():
    #pyrrole data
    atomic_nums = np.array([7, 1, 6, 1, 6, 1, 6, 1, 6, 1])
    coord_c = np.array([[-0.204112, 3.185019, -0.075741], [-0.221492, 4.179387, 0.130716],
                        [-1.276336, 2.387919, -0.226908], [-2.293503, 2.745807, -0.13927],
                        [-0.845624, 1.133207, -0.492811], [-1.470564, 0.265525, -0.664519],
                        [0.508669, 1.159459, -0.505233], [1.163508, 0.316585, -0.68868],
                        [0.895242, 2.430013, -0.246826], [1.899212, 2.827078, -0.177728]])
    classified = classify_nitrogen(atomic_nums, coord_c)
    assert_allclose(classified, [702, 1, 6, 1, 6, 1, 6, 1, 6, 1], rtol=1.0e-7, atol=1.0e-5)

def test_classify_charge():
    atoms = np.array([7, 6, 1, 1, 1, 1, 1, 1] )
    coords_c = np.array([[ 7.57041e-01, -6.91100e-05,  1.06000e-06],
       [-7.49473e-01,  1.34000e-05,  2.68700e-05],
       [ 1.12943e+00, -3.13881e-01, -9.00518e-01],
       [ 1.12957e+00, -6.22865e-01,  7.22135e-01],
       [ 1.12937e+00,  9.36760e-01,  1.78267e-01],
       [-1.09123e+00, -1.01534e+00, -1.93229e-01],
       [-1.09117e+00,  3.40283e-01,  9.76023e-01],
       [-1.09113e+00,  6.75097e-01, -7.82706e-01]])
    classified,_ = classify_charge(atoms,coords_c,1)
    assert_allclose(classified,[7001, 6, 1, 1, 1, 1, 1, 1], rtol=1.0e-7, atol=1.0e-5)
