"""Data Used for CGem Testing."""


# Author: Farnaz Heidar-Zadeh
# Date created: May 20, 2020


import numpy as np


def get_data_cgem_parameters():
    data = {
        "omega_c": 0.48683651,
        "gamma_s": 5.28037196,
        "lambda_c": 1.93948730,
        "lambda_s": 1.9394873,
        "shell_r": 0.78598114,
        "shell_ip": 14.40828174,
        "atomic_r": ((1, 0.556), (6, 0.717), (8, 0.501), (17, 0.994), (601, 0.717), (602, 0.717)),
        "atomic_ip": ((1, -14.95), (6, -15.76), (8, -19.35), (17, -18.12), (601, -15.76), (602, -15.76)),
    }
    return data


def get_data_hcl():
    # coordinates and core/shell charges of Cl & H
    coords_c = np.array([[3.1378, 0.0, 0.0], [1.8622, 0.0, 0.0]])
    coords_s = np.array([[3.116037762079, 0.0, 0.0], [2.112794196797, 0.0, 0.0]])
    q_c = np.array([1.0, 1.0])
    q_s = np.array([-1.0, -1.0])
    # alpha exponent for coulomb interaction
    a_c = np.array([0.98148615, 3.13694830])
    a_s = np.array([1.56975796, 1.56975796])
    # gamma exponents & beta pre-factors for gaussian interaction
    g_c = np.array([0.48977516, 0.87560523])
    g_s = np.array([5.28037196, 5.28037196])
    b_c = np.array([-5.49300586, 1.6699239])
    b_s = np.array([0.01304204, 0.01304204])
    return coords_c, coords_s, q_c, q_s, a_c, a_s, g_c, g_s, b_c, b_s


def get_data_h2o():
    # coordinates and core/shell charges of O, H, & H
    coords_c = np.array([[2.5369, -0.3375, 0.0], [3.2979, 0.2462, 0.0], [1.776, 0.2462, 0.0]])
    coords_s = np.array([[2.536821662462639, -0.12048395247146021, 0.0],
                         [2.808497967243062, -0.06442267867730732, 0.0],
                         [2.265435603830418, -0.06446576712443078, 0.0]])
    q_c = np.array([1.0, 1.0, 1.0])
    q_s = np.array([-1.0, -1.0, -1.0])
    # alpha exponent for coulomb interaction
    a_c = np.array([3.86350512, 3.13694830, 3.13694830])
    a_s = np.array([1.56975796, 1.56975796, 1.56975796])
    # gamma exponents & beta pre-factors for gaussian interaction
    g_c = np.array([0.97172956, 0.87560523, 0.87560523])
    g_s = np.array([5.28037196, 5.28037196, 5.28037196])
    b_c = np.array([-2.18300512, 1.6699239, 1.6699239])
    b_s = np.array([0.01304204, 0.01304204, 0.01304204])
    return coords_c, coords_s, q_c, q_s, a_c, a_s, g_c, g_s, b_c, b_s


def get_data_ch3oh():
    # coordinates and core/shell charges of O, C, & 4 H
    coords_c = np.array([[1.0849, 0.1713, 0.0], [-0.3366, 0.1504, 0.0],
                         [-0.7076, 1.1754, 0.0], [-0.7006, -0.3636, 0.89],
                         [-0.7006, -0.3636, -0.89], [1.3606, -0.7699, 0.0]])
    coords_s = np.array([[1.1706328105796915, -0.12349646566348321, 1.8449427505150131e-16],
                         [-0.02707104553986363, 0.17091532075574475, 3.180962322701279e-16],
                         [-0.7569813249778815, 1.0464085201828268, 2.4061489015513796e-16],
                         [-0.7529793615600723, -0.329881005665833, 0.7884561381557949],
                         [-0.7529793615600708, -0.32988100566583384, -0.7884561381557973],
                         [1.1706330256692143, -0.12349612362913182, 1.739837730181232e-16]])
    q_c = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    q_s = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
    # alpha exponent for coulomb interaction
    a_c = np.array([3.86350512, 1.88633417, 3.13694830, 3.13694830, 3.13694830, 3.13694830])
    a_s = np.array([1.56975796, 1.56975796, 1.56975796, 1.56975796, 1.56975796, 1.56975796])
    # gamma exponents & beta pre-factors for gaussian interaction
    g_c = np.array([0.97172956, 0.67899095, 0.87560523, 0.87560523, 0.87560523, 0.87560523])
    g_s = np.array([5.28037196, 5.28037196, 5.28037196, 5.28037196, 5.28037196, 5.28037196])
    b_c = np.array([-2.18300512, -0.71990570, 1.6699239, 1.6699239, 1.6699239, 1.6699239])
    b_s = np.array([0.01304204, 0.01304204, 0.01304204, 0.01304204, 0.01304204, 0.01304204])
    return coords_c, coords_s, q_c, q_s, a_c, a_s, g_c, g_s, b_c, b_s


def get_data_energy_hcl(shells, terms=False):
    # values computed by C-GeM_CHOClNS code
    if shells == "core":
        # energy terms & their derivatives for shells at the position of core
        coulomb_cc, coulomb_ss, coulomb_cs = 9.947521327, 10.047051436, -49.273710199
        coulomb_d_ss = np.array([[4.72961308, 0.0, 0.0], [-4.72961308, 0.0, 0.0]])
        coulomb_d_cs = np.array([[-5.8998886, 0.0, 0.0], [ 3.72006058, 0.0, 0.0]])

        gaussian_ss, gaussian_cs = 2.4204367153060364e-06, -5.897079082566
        gaussian_d_ss = np.array([[3.26063927e-05, 0.0, 0.0], [-3.26063927e-05, 0.0, 0.0]])
        gaussian_d_cs = np.array([[0.8974088, 0.0, 0.0], [3.09346039, 0.0, 0.0]])

    elif shells == "opt":
        # energy terms & their derivatives for optimized shells
        coulomb_cc, coulomb_ss, coulomb_cs = 9.947521327, 11.356921360, -49.9693683439604
        coulomb_d_ss = np.array([[ 4.80798356, 0.0, 0.0], [-4.80798356, 0.0, 0.0]])
        coulomb_d_cs = np.array([[-5.85141094, 0.0, 0.0], [ 0.81825645, 0.0, 0.0]])

        gaussian_ss, gaussian_cs = 6.41517612997215e-05, -6.773047631025148
        gaussian_d_ss = np.array([[0.00067969, 0.0, 0.0], [-0.00067969, 0.0, 0.0]])
        gaussian_d_cs = np.array([[1.04272438, 0.0, 0.0], [ 3.99037454, 0.0, 0.0]])
    else:
        raise ValueError(f"Argument shells={shells} not recognized! Choose either 'core' or 'opt'")

    coulomb_terms = (coulomb_cc, coulomb_ss, coulomb_cs, coulomb_d_ss, coulomb_d_cs)
    gaussian_terms = (gaussian_ss, gaussian_d_ss, gaussian_cs, gaussian_d_cs)
    return prepare_data_energy(coulomb_terms, gaussian_terms, terms)


def get_data_energy_h2o(shells, terms=False):
    # values computed by C-GeM_CHOClNS code
    if shells == "core":
        # energy terms & their derivatives for shells at the position of core
        coulomb_cc, coulomb_ss, coulomb_cs = 37.193285208, 32.064203799, -119.339992433
        coulomb_d_ss = np.array([[-2.49102595e-04, -5.80689399e+00, 0.0],
                                 [ 8.11507062e+00,  2.90335176e+00, 0.0],
                                 [-8.11482152e+00,  2.90354223e+00, 0.0]])
        coulomb_d_cs = np.array([[ 2.38990401e-04,  7.84771542e+00, 0.0],
                                 [-1.05254346e+01, -4.17906752e+00, 0.0],
                                 [ 1.05252067e+01, -4.17944187e+00, 0.0]])

        gaussian_ss, gaussian_cs = 0.000202911366, 1.302817586207858
        gaussian_d_ss = np.array([[ 5.47888966e-07, -1.25041544e-03, 0.0],
                                  [ 8.15811256e-04,  6.24956506e-04, 0.0],
                                  [-8.16359145e-04,  6.25458935e-04, 0.0]])
        gaussian_d_cs = np.array([[ 1.83367756e-06, -1.52581079e+00, 0.0],
                                  [-7.35125166e-01, -1.01306557e+00, 0.0],
                                  [ 7.35146923e-01, -1.01321540e+00, 0.0]])

    elif shells == "opt":
        # energy terms & their derivatives for optimized shells
        coulomb_cc, coulomb_ss, coulomb_cs = 37.193285208, 41.57939720821391, -129.19532187558647
        coulomb_d_ss = np.array([[-1.95889104e-03, -8.14308910e-01, 0.0],
                                 [ 5.54047692e+00,  4.07578511e-01, 0.0],
                                 [-5.53851803e+00,  4.06730399e-01, 0.0]])
        coulomb_d_cs = np.array([[ 1.62249451e-03, 2.85234708e+00, 0.0],
                                 [-4.60818440e+00, 1.59541664e+00, 0.0],
                                 [ 4.60662686e+00, 1.59631251e+00, 0.0]])

        gaussian_ss, gaussian_cs = 0.020129921090800706, -0.3819231103543623
        gaussian_d_ss = np.array([[-5.26153518e-06, -1.02869316e-02, 0.0],
                                  [ 4.06851573e-02,  5.14448768e-03, 0.0],
                                  [-4.06798957e-02,  5.14244397e-03, 0.0]])
        gaussian_d_cs = np.array([[ 3.23124304e-04, -2.02776328e+00, 0.0],
                                  [-9.72971327e-01, -2.00812829e+00, 0.0],
                                  [ 9.72586090e-01, -2.00817995e+00, 0.0]])
    else:
        raise ValueError(f"Argument shells={shells} not recognized! Choose either 'core' or 'opt'")

    coulomb_terms = (coulomb_cc, coulomb_ss, coulomb_cs, coulomb_d_ss, coulomb_d_cs)
    gaussian_terms = (gaussian_ss, gaussian_d_ss, gaussian_cs, gaussian_d_cs)
    return prepare_data_energy(coulomb_terms, gaussian_terms, terms)


def get_data_energy_ch3oh(shells, terms=False):
    # values computed by C-GeM_CHOClNS code
    if shells == "core":
        # energy terms & their derivatives for shells at the position of core
        coulomb_cc, coulomb_ss, coulomb_cs = 129.793093854, 123.036102842, -351.609517642
        coulomb_d_ss = np.array([[11.25116772,  4.74285794, 0.0],
                                 [-2.63379575,  1.56810602, 0.0],
                                 [-5.69375894,  13.80182669, 0.0],
                                 [-6.67058614, -5.86625818, 11.95422809],
                                 [-6.67058614, -5.86625818, -11.95422809],
                                 [10.41755926, -8.38027430, 0.0]])
        coulomb_d_cs = np.array([[-11.5352266, -6.33244042, 0.0],
                                 [2.46503391, -1.68192683, 0.0],
                                 [6.03162765, -15.02246111, 0.0],
                                 [7.06236238, 6.46043321, -13.03640108],
                                 [7.06236238, 6.46043321, 13.03640108],
                                 [-11.2384331, 10.43840334, 0.0]])
        gaussian_ss, gaussian_cs = 0.000155072264, 4.679166182405
        gaussian_d_ss = np.array([[-2.31882364e-04,  8.07174475e-04, 0.0],
                                  [ 2.80079952e-04, -2.77866123e-08, 0.0],
                                  [-9.62586874e-05,  2.65967340e-04, 0.0],
                                  [-9.41803674e-05, -1.33002960e-04, 2.30296441e-04],
                                  [-9.41803674e-05, -1.33002960e-04, -2.30296441e-04],
                                  [ 2.36421834e-04, -8.07108107e-04, 0.0]])
        gaussian_d_cs = np.array([[-0.32034011,  1.18221532, 0.0],
                                  [ 1.79142778,  0.11797789, 0.0],
                                  [ 0.27987858,  0.05268511, 0.0],
                                  [ 0.21680023, -0.01016812, 0.06762934],
                                  [ 0.21680023, -0.01016812, -0.06762934],
                                  [-0.45984370,  1.60995676, 0.0]])
    elif shells == "opt":
        # energy terms & their derivatives for optimized shells
        coulomb_cc, coulomb_ss, coulomb_cs = 129.793093854, 132.6028467637502, -360.92361235381856
        coulomb_d_ss = np.array([[ 1.25521237e+01, -1.93858401e+00,  3.63178988e-15],
                                 [-2.72768431e-01,  2.64098863e+00,  9.88346504e-15],
                                 [-7.73148945e+00,  1.37928983e+01,  2.74434375e-15],
                                 [-8.54999620e+00, -6.27836338e+00,  1.18056658e+01],
                                 [-8.54999620e+00, -6.27836338e+00, -1.18056658e+01],
                                 [ 1.25521265e+01, -1.93857614e+00,  4.21884749e-15]])
        coulomb_d_cs = np.array([[-1.18470058e+01, -5.19015332e-01, -5.17376050e-15],
                                 [-2.09525669e+00, -2.86586174e+00, -6.18686959e-15],
                                 [ 7.56049592e+00, -1.36330043e+01, -4.15621572e-15],
                                 [ 8.45724183e+00,  6.24249934e+00, -1.16958930e+01],
                                 [ 8.45724183e+00,  6.24249934e+00,  1.16958930e+01],
                                 [-1.18470098e+01, -5.19023205e-01, -4.41527652e-15]])
        gaussian_ss, gaussian_cs = 0.013080371891484782, 2.9156391758123696
        gaussian_d_ss = np.array([[ 5.35477115e-05, -1.32171246e-05,  1.44167370e-18],
                                  [ 1.21772003e-04, -1.48466015e-05, -8.60915118e-19],
                                  [-1.05393321e-04,  1.27056772e-04, -1.74171150e-20],
                                  [-6.17666382e-05, -4.29350890e-05,  6.77052880e-05],
                                  [-6.17666382e-05, -4.29350890e-05, -6.77052880e-05],
                                  [ 5.36068825e-05, -1.31228682e-05, -1.45409659e-18]])
        gaussian_d_cs = np.array([[-7.05171233e-01,  2.45761402e+00, -3.45187378e-16],
                                  [ 2.36788328e+00,  2.24894864e-01, -8.08792640e-16],
                                  [ 1.71097783e-01, -1.60016763e-01,  3.06083414e-16],
                                  [ 9.28215541e-02,  3.59022620e-02, -1.09839544e-01],
                                  [ 9.28215541e-02,  3.59022620e-02,  1.09839544e-01],
                                  [-7.05171694e-01,  2.45761309e+00, -4.35232328e-16]])
    else:
        raise ValueError(f"Argument shells={shells} not recognized! Choose either 'core' or 'opt'")

    coulomb_terms = (coulomb_cc, coulomb_ss, coulomb_cs, coulomb_d_ss, coulomb_d_cs)
    gaussian_terms = (gaussian_ss, gaussian_d_ss, gaussian_cs, gaussian_d_cs)
    return prepare_data_energy(coulomb_terms, gaussian_terms, terms)


def prepare_data_energy(coulomb_terms, gaussian_terms, terms):
    coulomb_cc, coulomb_ss, coulomb_cs, coulomb_d_ss, coulomb_d_cs = coulomb_terms
    gaussian_ss, gaussian_d_ss, gaussian_cs, gaussian_d_cs = gaussian_terms

    if terms:
        # put the energy terms & their derivatives together
        coulomb_ss = (coulomb_ss, coulomb_d_ss)
        coulomb_cs = (coulomb_cs, coulomb_d_cs)
        gaussian_ss = (gaussian_ss, gaussian_d_ss)
        gaussian_cs = (gaussian_cs, gaussian_d_cs)
        return coulomb_cc, coulomb_ss, coulomb_cs, gaussian_ss, gaussian_cs
    else:
        # compute coulomb & gaussian terms and their derivatives
        coulomb = coulomb_cc + coulomb_ss + coulomb_cs
        coulomb_d = coulomb_d_ss + coulomb_d_cs
        gaussian = gaussian_ss + gaussian_cs
        gaussian_d = gaussian_d_ss + gaussian_d_cs
        total = coulomb + gaussian
        total_d = coulomb_d + gaussian_d
        return (coulomb, coulomb_d), (gaussian, gaussian_d), (total, total_d)
