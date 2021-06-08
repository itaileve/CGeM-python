"""CGem Module."""


# Author: Farnaz Heidar-Zadeh
# Date created: May 20, 2020


import numpy as np

from scipy.special import erf
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.optimize import minimize


class CGem(object):
    def __init__(
        self,
        coords_c,
        coords_s,
        charges_c,
        charges_s,
        alphas_c,
        alphas_s,
        gamma_c,
        gamma_s,
        beta_c,
        beta_s,
        opt_shells=True,
        min_dist_s=0.0,
        penalty_param=(10, 200),
        method="L-BFGS-B",
        r_cut=None,
        options=None,
    ):
        """Initialize C-Gem class with n cores and m shells.

        Parameters
        ----------
        coords_c : np.ndarray, shape=(n, 3)
            Cartesian coordinates of core positions in Angstrom.
        coords_s : np.ndarray, shape=(m, 3)
            Cartesian coordinates of shell positions in Angstrom.
        charges_c : np.ndarray, shape=(n,)
            Charge value of cores.
        charges_s : np.ndarray, shape=(m,)
            Charge value of shells.
        alphas_c : np.ndarray, shape=(n,)
            Alpha parameter (i.e. width) of core Gaussian distribution functions.
        alphas_s : np.ndarray, shape=(m,)
            Alpha parameter (i.e. width) of shell Gaussian distribution functions.
        gamma_c : np.ndarray, shape=(n,)
            Gamma parameter of cores in Gaussian interaction term.
        gamma_s : np.ndarray, shape=(m,)
            Gamma parameter of shells in Gaussian interaction term.
        beta_c : np.ndarray, shape=(n,)
            Beta parameter (e.g. A) of cores in Gaussian interaction term.
        beta_s : np.ndarray, shape=(m,)
            Beta parameter (e.g. A) of shells in Gaussian interaction term.
        opt_shells : bool, optional
            Whether to optimize shell positions. If True, the `coords_s` argument is used as
            an initial guess.
        min_dist_s : float, optional
            The minimum distance allowed between shell positions, otherwise an error is raised.
        penalty_param: [a,b], optional
            Penalty term that get added to gaussian_ss term in form a*np.exp(-b*dist).
            Set a to 0 to turn off penalty.
        method: str
            optimization method that goes into the scipy.optimize.minimize function
        r_cut: float/int, optional
            cutoff radius in Angstrom for tapering function. Defualt is None (don't use tapering).
            In Itai's water model cuttoff is set to 10 angstrom.
        options: dict, optional
            A dictionary of solver options; check options argument of scipy.optimize.minimize.

        """
        if len(charges_c) != len(coords_c) or charges_c.ndim != 1:
            raise ValueError("")

        if len(alphas_c) != len(coords_c) or alphas_c.ndim != 1:
            raise ValueError("")

        if len(gamma_c) != len(coords_c) or gamma_c.ndim != 1:
            raise ValueError("")

        if len(beta_c) != len(coords_c) or beta_c.ndim != 1:
            raise ValueError("")

        # print(f"CGem Model: {len(coords_c)} cores & {len(coords_s)} shells!")

        self._coords_c = coords_c
        self._charegs_c = charges_c
        self._charges_s = charges_s
        self._alphas_c = alphas_c
        self._alphas_s = alphas_s
        self._gamma_c = gamma_c
        self._gamma_s = gamma_s
        self._beta_c = beta_c
        self._beta_s = beta_s
        self._penalty_param = penalty_param
        self._r_cut = r_cut

        # pre-compute core-core Coulomb term as it is fixed
        # self._coulomb_cc = compute_coulomb_cc(
        #     self.coords_c, self.charge_c, self.alpha_c, r_cut=self.r_cut
        # )
        # pre-compute factors used in core-core coulomb interaction
        # to take advantage of the more efficient _compute_coulomb_ss implementation (Coulomb
        # interaction for shell-shell interactions which only computes upper-triangular elements),
        # the factors are computed for core-core, but are assigned to the shell-shell attributes.
        self._f1ss_c = np.sqrt(np.outer(alphas_c, alphas_c) / np.add.outer(alphas_c, alphas_c))
        self._f1ss_c = self._f1ss_c[np.triu_indices_from(self._f1ss_c, k=1)]
        self._f2ss_c = np.outer(charges_c, charges_c)
        self._f2ss_c = self._f2ss_c[np.triu_indices_from(self._f2ss_c, k=1)]
        self._coulomb_cc = self._compute_coulomb_ss(
            coords_c, charges_c, alphas_c, deriv=False, r_cut=self.r_cut
        )

        self._prefactor = 2.0 / np.sqrt(np.pi)

        # pre-compute factors used in shell-shell coulomb interaction
        self._f1ss_c = np.sqrt(np.outer(alphas_s, alphas_s) / np.add.outer(alphas_s, alphas_s))
        self._f1ss_c = self._f1ss_c[np.triu_indices_from(self._f1ss_c, k=1)]
        self._f2ss_c = np.outer(charges_s, charges_s)
        self._f2ss_c = self._f2ss_c[np.triu_indices_from(self._f2ss_c, k=1)]

        # pre-compute factors used in core-shell coulomb interaction
        self._f1cs_c = np.sqrt(np.outer(alphas_c, alphas_s) / np.add.outer(alphas_c, alphas_s))
        self._f2cs_c = np.outer(charges_c, charges_s)

        self._g1ss_g = np.ones((len(gamma_s), len(gamma_s))) * gamma_s[:, np.newaxis]
        self._g1ss_g = self._g1ss_g[np.triu_indices_from(self._g1ss_g, k=1)]
        self._b1ss_g = np.ones((len(beta_s), len(beta_s))) * beta_s[:, np.newaxis]
        self._b1ss_g = self._b1ss_g[np.triu_indices_from(self._b1ss_g, k=1)]

        # optimize shell coordinates
        if opt_shells:
            default = {'ftol': 1.0e-8}
            if options is not None:
                default.update(options)
            coords_s = self._optimize_shell_coordinates(coords_s, method=method, options=default)

        # check minimum distance of shell coordinates
        if len(coords_s) > 1:
            min_dist = np.min(pdist(coords_s))
            if min_dist < min_dist_s:
                raise ValueError(
                    f"Minimum shell distance {min_dist:.2e} is less than {min_dist_s}."
                )

        self._coords_s = coords_s

    @classmethod
    def from_molecule(
        cls,
        atomic_numbers,
        atomic_coords,
        coords_s=None,
        opt_shells=True,
        min_dist_s=0.0,
        penalty_param=(10, 200),
        method="L-BFGS-B",
        r_cut=None,
        options=None,
        **kwargs,
    ):
        """Initialize C-Gem class from molecule.

        Parameters
        ----------
        atomic_numbers : np.ndarray, shape=(n,)
            Atomic numbers.
        atomic_coords : np.ndarray, shape=(n, 3)
            Cartesian coordinates of atomic positions in Angstrom.
        coords_s : np.ndarray, shape=(m, 3), optional
            Cartesian coordinates of shell positions in Angstrom. If not provided, `atomic_coords`
            are used for shell coordinates as well.
        opt_shells : bool, optional
            Whether to optimize shell positions. If True, the `coords_s` argument is used as
            an initial guess.
        min_dist_s : float, optional
            The minimum distance allowed between shell positions, otherwise an error is raised.
        penalty_param: [a,b], optional
            Penalty term that get added to gaussian_ss term in form a*np.exp(-b*dist).
            Set a to 0 to turn off penalty.
        method: str
            optimization method that goes into the scipy.optimize.minimize function
        r_cut: float/int, optional
            cutoff radius in Angstrom for tapering function. Defualt is None (don't use tapering).
            In Itai's water model cuttoff is set to 10 angstrom.
        options: dict, optional
            A dictionary of solver options; check options argument of scipy.optimize.minimize.
        kwargs : optional
            Keyword arguments to pass to `get_cgem_parameters` to set up model parameters.
            These include global parameters & atomic parameters used by `get_cgem_parameters`.
        """
        if len(atomic_numbers) != len(atomic_coords):
            raise ValueError(
                f"The atomic numbers & coordinates don't have same number of atoms!"
            )
        #         if not np.any([int(number) != number for number in atomic_numbers]):
        #             raise ValueError(f"Expected atomic_numbers={atomic_numbers} to be all integers!")

        # assign shell coordinates
        if coords_s is None:
            coords_s = np.copy(atomic_coords)

        # get CGem parameters using default global/atomic parameters
        nums, nshell = atomic_numbers, len(coords_s)
        q_c, q_s, a_c, a_s, g_c, g_s, b_c, b_s = get_cgem_parameters(
            nums, nshell, **kwargs
        )

        return cls(
            atomic_coords,
            coords_s,
            q_c,
            q_s,
            a_c,
            a_s,
            g_c,
            g_s,
            b_c,
            b_s,
            opt_shells,
            min_dist_s,
            penalty_param,
            method,
            r_cut,
            options,
        )

    @property
    def coords_c(self):
        """Cartesian coordinates of core positions in Angstrom."""
        return self._coords_c

    @property
    def coords_s(self):
        """Cartesian coordinates of shell positions in Angstrom."""
        return self._coords_s

    @property
    def charge_c(self):
        """Charge value of cores."""
        return self._charegs_c

    @property
    def charge_s(self):
        """Charges value of shells."""
        return self._charges_s

    @property
    def alpha_c(self):
        """Alpha parameter (i.e. width) of core Gaussian distribution functions."""
        return self._alphas_c

    @property
    def alpha_s(self):
        """Alpha parameter (i.e. width) of shell Gaussian distribution functions."""
        return self._alphas_s

    @property
    def gamma_c(self):
        """Gamma parameter of cores in Gaussian interaction term."""
        return self._gamma_c

    @property
    def gamma_s(self):
        """Gamma parameter of shells in Gaussian interaction term."""
        return self._gamma_s

    @property
    def beta_c(self):
        """Beta parameter (e.g. A) of cores in Gaussian interaction term."""
        return self._beta_c

    @property
    def beta_s(self):
        """Beta parameter (e.g. A) of shells in Gaussian interaction term."""
        return self._beta_s

    @property
    def r_cut(self):
        """ cutoff radius in Angstrom for tapering function. Defualt is None (don't use tapering)"""
        return self._r_cut

    @property
    def energy(self):
        """Total interaction energy."""
        return self.compute_energy(self.coords_s)
        #return 0.0

    @property
    def force(self):
        """Force with respect to change of core positions"""
        # TODO
        return self.compute_force(self.coords_s)
        #return 0.0
    
    @property
    def dipole(self):
        """Dipole moment vector in eAngstrom."""
        dipole_c = self.coords_c * self.charge_c[:, np.newaxis]
        dipole_s = self.coords_s * self.charge_s[:, np.newaxis]
        # if shell less than core, pad zeros to shell. If core less than shell, pad zeros to core value.
        diff = len(self.coords_s) - len(self.coords_c)
        if diff < 0:
            dipole_s = np.pad(dipole_s, ((0, -diff), (0, 0)), "constant")

        elif diff > 0:
            dipole_c = np.pad(dipole_c, ((0, diff), (0, 0)), "constant")
        dipole = dipole_c + dipole_s
        return np.sum(dipole, axis=0)

    def _compute_coulomb_ss(self, coords, charges, alphas, deriv=False, r_cut=None):

        # compute a condensed distance matrix (for i & j where i < j < m & m is # of shells)
        # i.e., upper-triangular elements of the distance matrix
        dist = pdist(coords)
        
        # compute Coulomb interaction energy (Eq. 3a)
        value = erf(self._f1ss_c * dist)
        #print(self._f1ss_c)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            value = self._f2ss_c * value / dist
        
        # compute values for shell-shell interactions that are closer than 1.0e-8
        has_close_shells = np.any(dist < 1.0e-8)
        if has_close_shells:
            mask = np.where(dist < 1.0e-8)
            c1_m, c2_m = charges[mask[0]], charges[mask[1]]
            a1_m, a2_m = alphas[mask[0]], alphas[mask[1]]
            value[mask] = 2 * c1_m * c2_m * np.sqrt((a1_m * a2_m) / (a1_m + a2_m)) / np.pi ** 0.5
        
        if deriv:
            d_value = self._prefactor * self._f1ss_c * np.exp(-((self._f1ss_c * dist) ** 2))
            with np.errstate(divide='ignore', invalid='ignore'):
                d_value = (self._f2ss_c * d_value - value) / dist**2
            if has_close_shells:
                d_value[mask] = 0.0
            # TODO: See whether making a square matrix can be avoinded
            d_value = squareform(d_value)
            # tapering
            # if r_cut is not None:
            #     r_cut = np.full(dist.shape, r_cut)
            #     d_value = d_tap(dist, r_cut) * value + tap(dist, r_cut) * d_value
            #     value = value * tap(dist, np.full(dist.shape, r_cut))

            # convert d_value wrt r to x,y,z component, shape is (len(coord1), len(coord2), 3)
            pairwise = coords[:, np.newaxis] - coords
            # pairwise derivatives with shape of (len(coord1), len(coord2), 3)
            d_value = pairwise * d_value.reshape(len(coords), len(coords), 1)
            return np.sum(value), np.sum(d_value, axis=0)
        #print(value)
        #print(np.sum(value))
        if r_cut is not None:
            value = value * tap(dist, np.full(dist.shape, r_cut))
        #print(23.060549*np.sum(value)*14.4)
        return np.sum(value)

    def _compute_coulomb_cs(
            self,
            coord1,
            charge1,
            alpha1,
            coord2,
            charge2,
            alpha2,
            deriv=False,
            threshold=1.0e-8,
            r_cut=None,
    ):
        # compute pair-wise distance matrix
        dist = cdist(coord1, coord2)
        # compute Coulomb interaction energy (Eq. 3a)
        value = erf(self._f1cs_c * dist)
        with np.errstate(divide='ignore', invalid='ignore'):
            value = self._f2cs_c * value / dist
        # compute values for interactions that are closer than 1.0e-8
        has_close_shells = np.any(dist < 1.0e-8)
        if has_close_shells:
            mask = np.where(dist < threshold)
            c1_m, c2_m = charge1[mask[0]], charge2[mask[1]]
            a1_m, a2_m = alpha1[mask[0]], alpha2[mask[1]]
            value[mask] = compute_pairwise_coulomb_limit(c1_m, a1_m, c2_m, a2_m)

        if deriv:
            d_value = self._prefactor * self._f1cs_c * np.exp(-((self._f1cs_c * dist) ** 2))
            with np.errstate(divide='ignore', invalid='ignore'):
                d_value = (self._f2cs_c * d_value - value) / dist**2
            if has_close_shells:
                d_value[mask] = 0.0
            # # tapering
            # if r_cut is not None:
            #     r_cut = np.full(dist.shape, r_cut)
            #     d_value = d_tap(dist, r_cut) * value + tap(dist, r_cut) * d_value
            #     value = value * tap(dist, np.full(dist.shape, r_cut))
            # convert d_value wrt r to x,y,z component with shape (len(coord1), len(coord2), 3)
            pairwise = coord1[:, np.newaxis] - coord2
            # pairwise derivative shape (len(coord1), len(coord2), 3)
            d_value = pairwise * d_value.reshape(len(coord1), len(coord2), 1)
            return np.sum(value), np.sum(d_value, axis=0)

        if r_cut is not None:
            value = value * tap(dist, np.full(dist.shape, r_cut))
        #print(23.060549*np.sum(value)*14.4)
        return np.sum(value)

    def _compute_gaussian_ss(self, coords, deriv=False, penalty_param=(10, 200), r_cut=None):
        dist = pdist(coords)
        value = self._b1ss_g * np.exp(-(dist ** 2) * self._g1ss_g)
        # add penalty for very short shell-shell distance
        # penalty = penalty_param[0] * np.exp(-penalty_param[1] * dist)
        # penalty =0
        if deriv:
            # d_value = -2 * gamma[:, np.newaxis] * value
            d_value = -2 * self._g1ss_g * value
            # d_value += (
            #         -penalty_param[0] * penalty_param[1] * np.exp(-penalty_param[1] * dist)
            # )
            d_value = squareform(d_value)
            # deriv for penalty
            # tapering
            # if r_cut is not None:
            #     r_cut = np.full(dist.shape, r_cut)
            #     d_value = d_tap(dist, r_cut) * value + tap(dist, r_cut) * d_value
            #     value = value * tap(dist, np.full(dist.shape, r_cut))
            pairwise = coords[:, np.newaxis] - coords
            d_value = pairwise * d_value.reshape(len(coords), len(coords), 1)
            # value = value[np.triu_indices(len(dist), k=1)]
            return np.sum(value), np.sum(d_value, axis=0)
        # value = value + penalty
        if r_cut is not None:
            value = value * tap(dist, np.full(dist.shape, r_cut))
        # value = value[np.triu_indices(len(dist), k=1)]
        #print(23.060549*value*14.4)
        return np.sum(value)

    def compute_coulomb_terms(self, coords_s, deriv=False):
        """Compute Coulomb interaction energy and its derivatives.

        Parameters
        ----------
        coords_s : np.ndarray, shape=(m, 3)
            Cartesian coordinates of shell positions in Angstrom.
        deriv : bool, optional
            Whether to compute derivative of Coulomb interaction energy w.r.t. shell coordinates.

        Returns
        -------
        float
            The Coulomb interaction energy in eV.
        np.ndarray, shape=(m, 3), optional
            If deriv=True, derivative of Coulomb interaction terms w.r.t. shell coordinates is
            returned.

        """
        if len(coords_s) == 0:  # no shell
            if deriv:
                return 14.4 * self._coulomb_cc, 0
            else:
                return 14.4 * self._coulomb_cc
        # compute shell-shell & core-shell Coulomb interactions
        ss = self._compute_coulomb_ss(
            coords_s, self.charge_s, self.alpha_s, deriv=deriv, r_cut=self.r_cut
        )
        cs = self._compute_coulomb_cs(
            self.coords_c,
            self.charge_c,
            self.alpha_c,
            coords_s,
            self.charge_s,
            self.alpha_s,
            deriv=deriv,
            r_cut=self.r_cut,
        )
        if deriv:
            # computed values are tuples of value & its derivative
            value = self._coulomb_cc + ss[0] + cs[0]
            return 14.4 * value, 14.4 * (ss[1] + cs[1])
        # add core-core pre-computed Coulomb term
        value = self._coulomb_cc + ss + cs
        #print(14.4*value*23.060549)
        return 14.4 * value

    def compute_coulomb_forces(self, coords_s):
        if len(coords_s) == 0:  # no shell
            return 0
        # compute shell-shell & core-shell Coulomb interactions
        # shape(n,3) n is number of core, m is number of shell
        cc = compute_coulomb_cc_force(
            self.coords_c, self.charge_c, self.alpha_c, r_cut=self.r_cut
        )
        # shape (n,3)
        cs = compute_coulomb_cs_force(
            self.coords_c,
            self.charge_c,
            self.alpha_c,
            coords_s,
            self.charge_s,
            self.alpha_s,
            r_cut=self.r_cut,
        )

        return 14.4 * (cc + cs)

    def compute_gaussian_terms(self, coords_s, deriv=False):
        """Compute Gaussian interaction energy and its derivatives.

        Parameters
        ----------
        coords_s : np.ndarray, shape=(m, 3)
            Cartesian coordinates of shell positions in Angstrom.
        deriv : bool, optional
            Whether to compute derivative of Coulomb interaction energy w.r.t. shell coordinates.

        Returns
        -------

        """
        if len(coords_s) == 0 or len(self.coords_c) == 0:
            if deriv:
                return 0.0, 0.0
            else:
                return 0.0
        # compute shell-shell & core-shell Gaussian interactions
        ss = self._compute_gaussian_ss(
            coords_s,
            deriv=deriv,
            penalty_param=self._penalty_param,
            r_cut=self.r_cut,
        )
        cs = compute_gaussian_cs(
            self.coords_c,
            self.gamma_c,
            self.beta_c,
            coords_s,
            deriv=deriv,
            r_cut=self.r_cut,
        )
        if deriv:
            # computed values are tuples of value & its derivative
            return cs[0] + ss[0], cs[1] + ss[1]
        return cs + ss

    def compute_gaussian_forces(self, coords_s):
        """Compute Gaussian interaction derivatives.

        Parameters
        ----------
        coords_s : np.ndarray, shape=(m, 3)
            Cartesian coordinates of shell positions in Angstrom.

        Returns
        -------
        np.ndarray, shape=(n, 3)

        """
        if len(coords_s) == 0 or len(self.coords_c) == 0:
            return 0.0

        d_cs = compute_gaussian_cs_force(
            self.coords_c, self.gamma_c, self.beta_c, coords_s, r_cut=self.r_cut
        )
        return d_cs

    def compute_energy(self, coords_s, deriv=False):
        """Compute total interaction energy and its derivatives.

        Parameters
        ----------
        coords_s : np.ndarray, shape=(m, 3)
            Cartesian coordinates of shell positions in Angstrom.
        deriv : bool, optional
            Whether to compute derivative of total interaction energy w.r.t. shell coordinates.

        Returns
        -------
        float
            Total C-Gem energy in eV.
        np.ndarray, optional
            If deriv=True, derivative of total C-Gem energy w.r.t. shell coordinates is returned.

        """
        if len(coords_s) != len(self.charge_s):
            raise ValueError(f"Expected {len(self.charge_s)} shell coordinates.")
        # compute Coulomb & Gaussian terms of energy
        term_c = self.compute_coulomb_terms(coords_s, deriv=deriv)
        term_g = self.compute_gaussian_terms(coords_s, deriv=deriv)
        if deriv:
            # computed values are tuples of value & its derivative
            return term_c[0] + term_g[0], term_c[1] + term_g[1]
        return term_c + term_g

    def compute_force(self, coords_s):
        """Compute total force with respect to core positions.

        Parameters
        ----------
        coords_s : np.ndarray, shape=(m, 3)
            Cartesian coordinates of shell positions in Angstrom.

        Returns
        -------
        np.ndarray
            minus derivative of total C-Gem energy w.r.t. core coordinates is returned. In unit  of eV/Angstrom

        """
        if len(coords_s) != len(self.charge_s):
            raise ValueError(f"Expected {len(self.charge_s)} shell coordinates.")
        # compute Coulomb & Gaussian terms of energy
        term_c = self.compute_coulomb_forces(coords_s)
        term_g = self.compute_gaussian_forces(coords_s)

        return -(term_c + term_g)

    def _optimize_shell_coordinates(self, guess, method, options):
        """Optimize shell coordinates.

        Parameters
        ----------
        guess : np.ndarray, shape=(m, 3)
            Initial guess for Cartesian coordinates of m shells in Angstrom.

        Returns
        -------
        np.ndarray
            Optimized Cartesian coordinates of shells in Angstrom.

        """

        def objective(x, shape):
            x = x.reshape(shape)
            e, d_e = self.compute_energy(x, deriv=True)
            return e, -d_e.flatten()

        result = minimize(
            objective,
            guess.flatten(),
            jac=True,
            args=(guess.shape,),
            method=method,
            options=options,
        )
        # print("Shell Optimization:")
        # print("Message : ", result.message)
        # print("Success : ", result.success)
        # print("# iter  : ", result.nit)
        # print("Energy  : ", result.fun)

        def evaluate(result, guess, displacement_decimal=3):
            if displacement_decimal < 1:
                displacement_decimal = 1
            sucess = True
            if not result.success:
                # raise ValueError("Optimization of shell coordinates failed!")
                sucess = False
                print(
                    "Optimization of shell coordinates failed! Displacing coords and try again..."
                )
                coords_s = guess + (np.random.random((len(guess), 3)) - 0.5) * 10 ** (
                    -displacement_decimal
                )
            else:
                coords_s = result.x.reshape(guess.shape)

            return coords_s, sucess

        count = 0
        guess, sucess = evaluate(result, guess)
        while not sucess and count < 8:
            result = minimize(
                objective,
                guess.flatten(),
                jac=True,
                args=(guess.shape,),
                method=method,
                # options={'gtol': 1.0e-5},
            )
            guess, sucess = evaluate(result, guess, displacement_decimal=(6 - count))
            count += 1
        if not sucess:
            print(result)
            raise ValueError("Optimization of shell coordinates failed!")

        return result.x.reshape(guess.shape)

    def compute_electrostatic_potential_gaussian_charge(self, points,r_cut=None):
        """Compute electrostatic potential of the cores & shells at the given points using gaussian charges.

        Parameters
        ----------
        points : np.ndarray, shape=(k, 3)
            Cartesian coordinates of points in Angstrom.

        Returns
        -------
        np.ndarray, shape=(k,)
            Electrostatic potential at the points in eV.

        """
        if points.ndim != self.coords_c.ndim:
            raise ValueError(
                f"Argument points should have {self.coords_c.ndim} dimentions!"
            )

        charge = np.ones(len(points))
        alphas = np.repeat(1569.7579613660428, len(points))
        # electrostatic potential of cores
        value = compute_pairwise_coulomb(
            self.coords_c, self.charge_c, self.alpha_c, points, charge, alphas,r_cut=r_cut
            #self.coords_c, self.charge_c, self.alpha_c, points, charge, self.alpha_c,r_cut=r_cut
        )
        # electrostatic potential of shells
        shell_value = compute_pairwise_coulomb(
            self.coords_s, self.charge_s, self.alpha_s, points, charge, alphas,r_cut=r_cut
        )
        #print(shell_value)
        #print(value)
        # if shell less than core, pad zeros to shell. If core less than shell, pad zeros to core value.
        diff = len(self.coords_s) - len(self.coords_c)
        if diff < 0:
            shell_value = np.pad(shell_value, ((0, -diff), (0, 0)), "constant")

        elif diff > 0:
            value = np.pad(value, ((0, diff), (0, 0)), "constant")

        value += shell_value

        return 14.4 * np.sum(value, axis=0)

    def compute_electrostatic_potential(self, points):
        """Compute electrostatic potential of the cores & shells at the given points using point charges.

        Parameters
        ----------
        points : np.ndarray, shape=(k, 3)
            Cartesian coordinates of points in Angstrom.

        Returns
        -------
        np.ndarray, shape=(k,)
            Electrostatic potential at the points in eV.

        """
        if points.ndim != self.coords_c.ndim:
            raise ValueError(
                f"Argument points should have {self.coords_c.ndim} dimentions!"
            )

        # from cores
        dist_c = cdist(self.coords_c, points)

        #dist_c = np.where(dist_c < 0.0, 1000, dist_c)
        
                
        charge = np.ones(len(points))
        value = np.outer(self.charge_c, charge) / dist_c

        # from shells
        dist_s = cdist(self.coords_s, points)
        #dist_s = np.where(dist_s < 0.0, 1000, dist_s)
                
        value_s = np.outer(self.charge_s, charge) / dist_s
        
                
        diff = len(self.coords_s) - len(self.coords_c)
        if diff < 0:
            shell_value = np.pad(value_s, ((0, -diff), (0, 0)), "constant")

        elif diff > 0:
            value = np.pad(value, ((0, diff), (0, 0)), "constant")

        value += value_s

        return 14.4 * np.sum(value, axis=0)


def compute_pairwise_coulomb_limit(charge1, alpha1, charge2, alpha2):
    # Equation (3b) of [J. Phys. Chem. Lett. 2019, 10, 6820-6826]
    # Note that there should be pi**0.5 in the denominator not pi (as written in the paper)
    if charge1.ndim != 1 or alpha1.ndim != 1 or charge1.shape != alpha1.shape:
        raise ValueError(
            "Arguments charge1 & alpha1 should be 1D array of same length!"
        )
    if charge2.ndim != 1 or alpha2.ndim != 1 or charge2.shape != alpha2.shape:
        raise ValueError(
            "Arguments charge2 & alpha2 should be 1D array of same length!"
        )
    if charge1.shape != charge2.shape:
        # pad zeros to keep shape the same. Padded region contribute 0 to energy
        diff = len(charge1) - len(charge2)
        if diff > 0:
            charge2 = np.pad(charge2, (0, diff), "constant")
            alpha2 = np.pad(alpha2, (0, diff), "constant")
        elif diff < 0:
            charge1 = np.pad(charge1, (0, -diff), "constant")
            alpha1 = np.pad(alpha1, (0, -diff), "constant")
        # raise ValueError("Expect charge1 & charge2 arguments to have the same shape!")
    return (
        2
        * charge1
        * charge2
        * np.sqrt((alpha1 * alpha2) / (alpha1 + alpha2))
        / np.pi ** 0.5
    )


def compute_pairwise_coulomb(
    coord1,
    charge1,
    alpha1,
    coord2,
    charge2,
    alpha2,
    deriv=False,
    threshold=1.0e-8,
    r_cut=None,
):
    """
    Equation (3a) of [J. Phys. Chem. Lett. 2019, 10, 6820-6826]

    Parameters
    ----------
    coord1: np.ndarray, shape=(n, 3)
            Cartesian coordinates of positions in Angstrom.
    charge1: np.ndarray, shape=(n,)
            Charge value of core/shells.
    alpha1: np.ndarray, shape=(n,)
            alpha values of core/shells
    coord2: np.ndarray, shape=(m, 3)
    charge2: np.ndarray, shape=(m,)
    alpha2: np.ndarray, shape=(m,)
    deriv: boolean
            True if return pairwise coulomb energies and forces. False only returns energies.
    threshold: float
            threshold for distance. Use compute_pairwise_coulomb_limit when smaller than threshold.
    r_cut: float/int
            cutoff radius in Angstrom for tapering function. Defualt is None (don't use tapering).
            In Itai's water model cuttoff is set to 10 angstrom.

    Returns
    -------
    value: np.ndarray, shape=(n,m)
            pairwise coulomb energy in eV

    """
    #r_cut = 10.0
    #print(r_cut)
    dist = cdist(coord1, coord2)
    mask = np.where(dist < threshold)
    # dist = np.ma.masked_where(dist < threshold, dist)
    value = erf(np.sqrt(np.outer(alpha1, alpha2) / np.add.outer(alpha1, alpha2)) * dist)
    #print(np.sqrt(np.outer(alpha1, alpha2) / np.add.outer(alpha1, alpha2)))
    #print("erf=%.4f r_ij=$.4f alpha=%.4f" % (value,dist,np.sqrt(np.outer(alpha1, alpha2) / np.add.outer(alpha1, alpha2)) ) )
    with np.errstate(divide='ignore', invalid='ignore'):
        value = np.outer(charge1, charge2) * value / dist
    c1_m, c2_m = charge1[mask[0]], charge2[mask[1]]
    a1_m, a2_m = alpha1[mask[0]], alpha2[mask[1]]
    value_m = compute_pairwise_coulomb_limit(c1_m, a1_m, c2_m, a2_m)
    value[mask] = value_m
    if deriv:
        factor = np.sqrt(np.outer(alpha1, alpha2) / np.add.outer(alpha1, alpha2))
        d_value = (2.0 / np.sqrt(np.pi)) * factor * np.exp(-((factor * dist) ** 2))
        with np.errstate(divide='ignore', invalid='ignore'):
            d_value = np.outer(charge1, charge2) * d_value / dist
            d_value -= value / dist
            d_value /= dist
        # d_value = d_value.filled(0.0)
        d_value[mask] = 0.0

        # tapering
        if r_cut is not None:
            r_cut = np.full(dist.shape, r_cut)
            d_value = d_tap(dist, r_cut) * value + tap(dist, r_cut) * d_value
            value = value * tap(dist, np.full(dist.shape, r_cut))
        # convert d_value wrt r to x,y,z component
        pairwise = coord1[:, np.newaxis] - coord2  # shape(len(coord1),len(coord2),3)
        # pairwise forces shape(len(coord1),len(coord2),3)
        d_value = pairwise * d_value.reshape(len(coord1), len(coord2), 1)

        return value, np.sum(d_value, axis=0)

    if r_cut is not None:
        value = value * tap(dist, np.full(dist.shape, r_cut))
    #print(23.060549*value*14.4)
    return value


def compute_pairwise_coulomb_force(
    coord1, charge1, alpha1, coord2, charge2, alpha2, threshold=1.0e-8, r_cut=None
):
    # force for Equation (3a) of [J. Phys. Chem. Lett. 2019, 10, 6820-6826]
    dist = cdist(coord1, coord2)
    mask = np.where(dist < threshold)
    dist = np.ma.masked_where(dist < threshold, dist)
    value = erf(np.sqrt(np.outer(alpha1, alpha2) / np.add.outer(alpha1, alpha2)) * dist)
    value = np.outer(charge1, charge2) * value / dist
    c1_m, c2_m = charge1[mask[0]], charge2[mask[1]]
    a1_m, a2_m = alpha1[mask[0]], alpha2[mask[1]]
    value_m = compute_pairwise_coulomb_limit(c1_m, a1_m, c2_m, a2_m)

    factor = np.sqrt(np.outer(alpha1, alpha2) / np.add.outer(alpha1, alpha2))
    d_value = (2.0 / np.sqrt(np.pi)) * factor * np.exp(-((factor * dist) ** 2))
    d_value = np.outer(charge1, charge2) * d_value / dist
    d_value -= value / dist
    d_value /= dist
    d_value = d_value.filled(0.0)
    # tapering
    if r_cut is not None:
        r_cut = np.full(dist.shape, r_cut)
        d_value = d_tap(dist, r_cut) * value + tap(dist, r_cut) * d_value
        value = value * tap(dist, np.full(dist.shape, r_cut))
    pairwise = coord1[:, np.newaxis] - coord2
    d_value = pairwise * d_value.reshape(len(coord1), len(coord2), 1)
    value[mask] = value_m
    return np.sum(d_value, axis=1)


def compute_gaussian_cs(coords_c, gamma_c, beta_c, coords_s, deriv=False, r_cut=None):
    dist = cdist(coords_c, coords_s)
    value = np.exp(-(dist ** 2) * gamma_c[:, np.newaxis])
    value = value * beta_c[:, np.newaxis]
    if deriv:
        d_value = -2 * gamma_c[:, np.newaxis] * value

        # tapering
        if r_cut is not None:
            r_cut = np.full(dist.shape, r_cut)
            d_value = d_tap(dist, r_cut) * value + tap(dist, r_cut) * d_value
            value = value * tap(dist, np.full(dist.shape, r_cut))
        # convert d_value wrt r to x,y,z component
        pairwise = coords_c[:, np.newaxis] - coords_s
        d_value = pairwise * d_value.reshape(len(coords_c), len(coords_s), 1)
        return np.sum(value), np.sum(d_value, axis=0)

    if r_cut is not None:
        value = value * tap(dist, np.full(dist.shape, r_cut))
        
    return np.sum(value)


def compute_gaussian_cs_force(coords_c, gamma_c, beta_c, coords_s, r_cut=None):
    dist = cdist(coords_c, coords_s)
    value = np.exp(-(dist ** 2) * gamma_c[:, np.newaxis])
    value = value * beta_c[:, np.newaxis]

    d_value = -2 * gamma_c[:, np.newaxis] * value

    # tapering
    if r_cut is not None:
        r_cut = np.full(dist.shape, r_cut)
        d_value = d_tap(dist, r_cut) * value + tap(dist, r_cut) * d_value
        # value = value * tap(dist, np.full(dist.shape, r_cut))
    # convert d_value wrt r to x,y,z component
    pairwise = coords_c[:, np.newaxis] - coords_s
    d_value = pairwise * d_value.reshape(len(coords_c), len(coords_s), 1)

    return np.sum(d_value, axis=1)


def compute_gaussian_ss(
    coords, gamma, beta, deriv=False, penalty_param=(10, 200), r_cut=None
):
    dist = cdist(coords, coords)
    value = beta * np.exp(-(dist ** 2) * gamma[:, np.newaxis])
    # add penalty for very short shell-shell distance
    penalty = penalty_param[0] * np.exp(-penalty_param[1] * (dist))
    # penalty =0
    if deriv:
        d_value = -2 * gamma[:, np.newaxis] * value
        d_value += (
            -penalty_param[0] * penalty_param[1] * np.exp(-penalty_param[1] * (dist))
        )  # deriv for penalty
        # tapering
        if r_cut is not None:
            r_cut = np.full(dist.shape, r_cut)
            d_value = d_tap(dist, r_cut) * value + tap(dist, r_cut) * d_value
            value = value * tap(dist, np.full(dist.shape, r_cut))
        pairwise = coords[:, np.newaxis] - coords
        d_value = pairwise * d_value.reshape(len(coords), len(coords), 1)
        value = value[np.triu_indices(len(dist), k=1)]
        return np.sum(value), np.sum(d_value, axis=0)
    value = value + penalty
    if r_cut is not None:
        value = value * tap(dist, np.full(dist.shape, r_cut))
    value = value[np.triu_indices(len(dist), k=1)]
    return np.sum(value)


def compute_coulomb_cc(coords, charges, alphas, r_cut=None):
    if len(coords) <= 1:
        return 0.0
    cc = compute_pairwise_coulomb(
        coords, charges, alphas, coords, charges, alphas, deriv=False, r_cut=r_cut,
    )
    cc = np.sum(cc[np.triu_indices(len(cc), k=1)])
    return cc


def compute_coulomb_cc_force(coords, charges, alphas, r_cut=None):
    if len(coords) <= 1:
        return 0.0
    d_cc = compute_pairwise_coulomb_force(
        coords, charges, alphas, coords, charges, alphas, r_cut=r_cut
    )
    # cc = np.sum(cc[np.triu_indices(len(cc), k=1)])
    return d_cc


def compute_coulomb_ss(coords, charges, alphas, deriv=False, r_cut=None):
    if len(coords) <= 1:
        if deriv:
            return 0, 0
        else:
            return 0
    if deriv:
        ss, d_ss = compute_pairwise_coulomb(
            coords, charges, alphas, coords, charges, alphas, deriv=deriv, r_cut=r_cut
        )
    else:
        ss = compute_pairwise_coulomb(
            coords, charges, alphas, coords, charges, alphas, deriv=False, r_cut=r_cut
        )
    ss = np.sum(ss[np.triu_indices(len(ss), k=1)])
    if deriv:
        return ss, d_ss
    return ss


def compute_coulomb_cs(
    coords_c,
    charges_c,
    alphas_c,
    coords_s,
    charges_s,
    alphas_s,
    deriv=False,
    r_cut=None,
):
    if len(coords_c) < 1 or len(coords_s) < 1:
        cs, d_cs = 0, 0
    if deriv:
        cs, d_cs = compute_pairwise_coulomb(
            coords_c,
            charges_c,
            alphas_c,
            coords_s,
            charges_s,
            alphas_s,
            deriv=True,
            r_cut=r_cut,
        )
    else:
        cs = compute_pairwise_coulomb(
            coords_c,
            charges_c,
            alphas_c,
            coords_s,
            charges_s,
            alphas_s,
            deriv=False,
            r_cut=r_cut,
        )
    cs = np.sum(cs)
    if deriv:
        return cs, d_cs
    return cs


def compute_coulomb_cs_force(
    coords_c, charges_c, alphas_c, coords_s, charges_s, alphas_s, r_cut=None
):
    if len(coords_c) < 1 or len(coords_s) < 1:
        d_cs = 0, 0
    else:
        d_cs = compute_pairwise_coulomb_force(
            coords_c, charges_c, alphas_c, coords_s, charges_s, alphas_s, r_cut=r_cut
        )

    return d_cs


def get_cgem_parameters(
    atomic_numbers,
    nshell,
    omega_c=0.44000000,
    gamma_s=2.50000000,
    lambda_c=1.93498308,
    lambda_s=1.93498308,
    shell_r=0.78582406,
    shell_ip=14.70285714,
    atomic_r={
        1: 0.556,
        6: 0.717,
        7: 0.57142857,
        8: 0.501,
        16: 1.1,
        17: 0.994,
        601: 0.717,
        602: 0.717,
    },
    atomic_ip={
        1: -14.95,
        6: -15.76,
        7: -17.5,
        8: -19.35,
        16: -17.42857143,
        17: -18.12,
        601: -15.76,
        602: -15.76,
    },
    **kwargs,
):
    """Compute C-Gem parameters given global & atomic parameters for atomic numbers and shells.

    Parameters
    ----------
    atomic_numbers : np.ndarray, shape=(n,)
        Atomic numbers.
    nshell : int
        Number of shells.
    omega_c : float, optional
        Global parameter representing core omega value.
    gamma_s : float, optional
        Global parameter representing shell gamma value (used instead of having shell omega).
    lambda_c : float, optional
        Global parameter representing core lambda value.
    lambda_s : float, optional
        Global parameter representing shell lambda value.
    shell_r : float, optional
        Global parameter representing shell radius in angstrom.
    shell_ip : float, optional
        Global parameter representing shell ionization potential in eV.
    atomic_r : dict, optional
        Dictionary of atomic number (key) and atomic radius (value).
    atomic_ip : dict, optional
        Dictionary of atomic number (key) and atomic ionization potential (value).

    Returns
    -------
    charges_c : np.ndarray
        Core charges.
    charges_s : np.ndarray
        Shell charges.
    alpha_c : np.ndarray
        Core alpha values.
    alpha_s : np.ndarray
        Shell alpha values.
    gamma_c : np.ndarray
        Core gamma values.
    gamma_s : np.ndarray
        Shell gamma values.
    beta_c : np.ndarray
        Core beta values.
    beta_s : np.ndarray
        Shell beta values.

    """
    # enforce same lambda
    lambda_s = np.copy(lambda_c)
    if shell_ip < 0:
        raise ValueError(f"Expected given shell_ip={shell_ip} to be positive values!")

    atomic_r = dict(atomic_r)
    atomic_ip = dict(atomic_ip)

    if np.any([number not in atomic_r.keys() for number in atomic_numbers]):
        raise ValueError(
            f"Not all atomic_numbers={atomic_numbers} exist in data_r={atomic_r}!"
        )

    if np.any([number not in atomic_ip.keys() for number in atomic_numbers]):
        raise ValueError(
            f"Not all atomic_numbers={atomic_numbers} exist in data_ip={atomic_ip}!"
        )

    if np.any([ip > 0 for ip in atomic_ip.values()]):
        raise ValueError(
            f"Expected all IP values in data_ip={atomic_ip} to be negative!"
        )

    # set charge of core and shell
    charges_c = np.ones(len(atomic_numbers))
    charges_s = -1 * np.ones(nshell)

    # deal with positive atoms
    if 7001 in atomic_numbers:
        charges_c[np.where(atomic_numbers == 7001)] = 2
    if 6001 in atomic_numbers:
        charges_c[np.where(atomic_numbers == 6001)] = 2
    if 6004 in atomic_numbers:
        charges_c[np.where(atomic_numbers == 6004)] = 4
    if 12 in atomic_numbers:
        charges_c[np.where(atomic_numbers == 12)] = 3
    if 20 in atomic_numbers:
        charges_c[np.where(atomic_numbers == 20)] = 3
    if 25 in atomic_numbers:
        charges_c[np.where(atomic_numbers == 25)] = 3
    if 26 in atomic_numbers:
        charges_c[np.where(atomic_numbers == 26)] = 3
    if 27 in atomic_numbers:
        charges_c[np.where(atomic_numbers == 27)] = 3
    if 28 in atomic_numbers:
        charges_c[np.where(atomic_numbers == 28)] = 3
    if 29 in atomic_numbers:
        charges_c[np.where(atomic_numbers == 29)] = 3
    if 30 in atomic_numbers:
        charges_c[np.where(atomic_numbers == 30)] = 3

    # H partial charge
    if "charge_H101" in kwargs:
        charges_c[np.where(atomic_numbers == 101)] = kwargs["charge_H101"]
        charges_s[np.where(atomic_numbers == 101)] = -kwargs["charge_H101"]

    # set radius parameter for core and shell
    radius_c = np.array([atomic_r[num] for num in atomic_numbers])
    radius_s = np.repeat(shell_r, nshell)

    # set ip parameter for core and shell
    ip_c = np.array([atomic_ip[num] for num in atomic_numbers])
    ip_s = np.repeat(shell_ip, nshell)

    # compute alpha value for core & shell
    # Eq. (2a & 2b) with the difference that core & shell have different lambda values
    alpha_c = lambda_c / (2.0 * radius_c ** 2)
    alpha_s = lambda_s / (2.0 * radius_s ** 2)

    # compute gamma parameter for core and shell
    # Eq. (5) however, for core that 2 in the denominator is included in omega_c & for shell
    # technically gamma_s is optimized or omega_s includes (2 * radius_s) in the denominator
    gamma_c = omega_c / radius_c
    gamma_s = np.repeat(gamma_s, nshell)

    # compute beta parameter for core and shell
    # remove padded shells to keep shape constant
    diff = nshell - len(atomic_numbers)
    if diff > 0:
        e_elec = compute_pairwise_coulomb_limit(charges_c, alpha_c, charges_s, alpha_s)
        for i in range(diff):
            e_elec = np.delete(e_elec, -1)
        beta_c = ip_c - 14.4 * e_elec
    else:
        beta_c = ip_c - 14.4 * compute_pairwise_coulomb_limit(
            charges_c, alpha_c, charges_s, alpha_s
        )
    beta_s = ip_s - 14.4 * compute_pairwise_coulomb_limit(
        charges_s, alpha_s, charges_s, alpha_s
    )
    #print(beta_c)
    #print(beta_s)
    return charges_c, charges_s, alpha_c, alpha_s, gamma_c, gamma_s, beta_c, beta_s


def tap(r, r_cut):
    """
    Tappering function used to modulate the lengthsclae over which the interation is operative.
    This tapering function comes from Itai's paper(#TODO ref link),
    which adopts a tapering fucntion of the form used in ReaxFF.
    r,r_cut are np.ndarray of same shape
    """
    mask = np.where(r > r_cut)
    ratio = r / r_cut
    result = 20 * ratio ** 7 - 70 * ratio ** 6 + 84 * ratio ** 5 - 35 * ratio ** 4 + 1
    result[mask] = 0
    return result


def d_tap(r, r_cut):
    """derivative of the tapering function"""
    mask = np.where(r > r_cut)
    result = (
        140 * (r ** 6) / (r_cut ** 7)
        - 420 * (r ** 5) / (r_cut ** 6)
        + 420 * (r ** 4) / (r_cut ** 5)
        - 140 * (r ** 3) / (r_cut ** 4)
    )
    result[mask] = 0
    return result
