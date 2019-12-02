#!/usr/bin/env python
# pylint: disable=anomalous-backslash-in-string
"""Some useful functions for the Water Cloud Model (WCM)
used to retrieve parameters from Sentinel 1 data. The model
is first presented in Attema & Ulaby (1978)

The WCM predicts backscatter in a polarisation `pp` as a function
of some parameters:
$$
\sigma_{pp}^{0} =
    A\cdot V_{1}\left[1 - \exp\left(-\frac{-2B\cdot V_{2}}
                     {\cos\theta}\right)\right]
        + \exp\left(-\frac{-2B\cdot V_{2}}
                     {\cos\theta}\right)\cdot\left(C + D\cdot M_{v}\right).
$$

`A*V_1` is basically the backscattering coefficient, whereas
`B*V_2` is the extinction coefficient. `C` relates `VSM` (volumetric
soil moisture in [%]) to backscatter. In general, all the "constants"
(`A`, `B`, `C`, `D`) are polarisation dependent. `V1` and `V2` have to do with
the scatterers within the turbid medium, and are usually related to LAI.
"""
# pylint: enable=anomalous-backslash-in-string

import numpy as np

# from numba import jit  # unused import


def wcm(x, theta=30.):
    """The Water Cloud Model for one polarisation. This function phrases
    the WCM for a time series: We assume that the A, B, and C terms are
    constant over the time series, and that the V1, V2 and VSM parameters
    are time-varying (e.g. they change per pixel). The ordering matters:
    the input vector `x` order is:
    1. A (single)
    2. B (single)
    3. C (single)
    4. V1 (array of `n_obs` scalars)
    5. V2 (array of `n_obs` scalars)
    6. VSM (array of `n_obs` scalars)

    Arguments:
        x {array} -- An array of input parameters. Order is A, B, C,
        V1[1:n_obs], V2[1:n_obs] and VSM[1:n_obs]

    Keyword Arguments:
        theta {float} -- Angle of incidence (default: 30)

    Returns:
        [array] -- Backscatter
    """
    my_m = np.cos(np.deg2rad(theta))
    n_obs = int((x.shape[0] - 3) / 3)
    my_a, my_b, my_c = x[:3]
    my_v1 = x[3:(3 + n_obs)]
    my_v2 = x[(3 + n_obs):(3 + 2 * n_obs)]
    s = x[(3 + 2 * n_obs):]

    tau = np.exp(-2 * my_b * my_v2 / my_m)
    sigma_soil = tau * (my_c + s)
    sigma_veg = my_a * my_v1 * (1 - tau)
    return sigma_soil + sigma_veg


def wcm_jac(x, theta=30.):
    """The Jacobian of the WCM model for one polarisation. See above
    Arguments:
    x {array} -- An array of input parameters. Order is A, B, C,
                 V1[1:n_obs], V2[1:n_obs] and VSM[1:n_obs]

    Keyword Arguments:
        theta {float} -- Angle of incidence (default: 30)

    Returns:
        [array] -- Backscatter"""

    my_m = np.cos(np.deg2rad(theta))
    n_obs = int((x.shape[0] - 3) / 3)
    my_a, my_b, my_c = x[:3]
    my_v1 = x[3:(3 + n_obs)]
    my_v2 = x[(3 + n_obs):(3 + 2 * n_obs)]
    my_s = x[(3 + 2 * n_obs):]
    tau = np.exp(-2 * my_b * my_v2 / my_m)

    der_da = my_v1 - my_v1 * tau
    # der_dV1 = a - a * tau   unused variable
    der_db = (-2 * my_v2 / my_m) * tau * (-my_a * my_v1 + my_s)
    # der_dV2 = (-2 * b / m) * tau * (-a * V1 + s)   unused variable
    der_dc = tau
    der_dsigmasoil = tau

    # return [der_dA, der_dB, der_dC, der_dV1, der_dV2, der_dsigmasoil]
    return [der_da, der_db, der_dc, der_dsigmasoil]


def wcm_hess(x, theta=30.):
    """The Hessian of the WCM model for one polarisation. See above
    Arguments:
    x {array} -- An array of input parameters. Order is A, B, C,
                 V1[1:n_obs], V2[1:n_obs] and VSM[1:n_obs]

    Keyword Arguments:
        theta {float} -- Angle of incidence (default: 30)

    Returns:
        [array] -- Backscatter Hessian"""
    my_m = np.cos(np.deg2rad(theta))
    my_a, my_b, my_c = x[:3]
    n_obs = int((x.shape[0] - 3) / 3)

    my_v1 = x[3:(3 + n_obs)]
    my_v2 = x[(3 + n_obs):(3 + 2 * n_obs)]
    my_s = x[(3 + 2 * n_obs):]

    tau = np.exp(-2 * my_b * my_v2 / my_m)
    d_tau = -2 * my_v2 * tau / my_m * np.ones_like(my_s)
    v1_d_tau = -my_v1 * d_tau
    zero_vector = np.zeros_like(my_s)

    d_aa = zero_vector
    d_ab = v1_d_tau
    d_ac = zero_vector
    d_as = zero_vector

    d_ba = v1_d_tau
    d_bb = (4 * tau * my_v2 * my_v2 * (my_c + my_s - my_a * my_v1)) / my_m**2
    d_bc = d_tau
    d_bs = d_tau

    d_ca = zero_vector
    d_cb = d_tau
    d_cc = zero_vector
    d_cs = zero_vector

    d_sa = zero_vector
    d_sb = d_tau
    d_sc = zero_vector
    d_ss = zero_vector

# hess =
#  np.hstack
#   ([np.vstack
#       ([np.diag(d_aa), np.diag(d_ab), np.diag(d_ac), np.diag(d_as)]),
#     np.vstack([np.diag(d_ba), np.diag(d_bb), np.diag(d_bc), np.diag(d_bs)]),
#     np.vstack([np.diag(d_ca), np.diag(d_cb), np.diag(d_cc), np.diag(d_cs)]),
#     np.vstack([np.diag(d_sa), np.diag(d_sb), np.diag(d_sc), np.diag(d_ss)]),
#     ])

    return (d_aa, d_ab, d_ac, d_as,
            d_ba, d_bb, d_bc, d_bs,
            d_ca, d_cb, d_cc, d_cs,
            d_sa, d_sb, d_sc, d_ss)


def hessian_time_residual(hess_xx, diff_xx):
    """Hessian times residual term.
    This next function calculates the $H'' C_{obs}^{-1}(y-H(x))$ term.
    This term is the second order contribution of the non-linear observation
    operator to the cost function. The main issue here is that we need to
    combine the A, B and C terms per observation and add them up as they are
    constant within the observation window. The `VSM` terms are per time step
    and as such, are not added

    Arguments:
        hess_xx [list] -- A 16-element list of arrays with the relevant
                          4x4 elements of the WCM hessian matrix
        diff_xx [array] -- The `n_obs` residual between the model and
                            observations.
    Returns:
        A tuple of arrays in this order:
        (   d_aa, d_ab, d_ac, d_as,
            d_ba, d_bb, d_bc, d_bs,
            d_ca, d_cb, d_cc, d_cs,
            d_sa, d_sb, d_sc, d_ss
        )
    """
    # the next bit is the big hairy H''C_{obs}residual term
    # Since we define A, B and C to be constant over the inversion window,
    # we need to add them up together
    # S is related to individual observations
    # hess_xx is returned as a tuple of arrays:
    # (    d_aa, d_ab, d_ac, d_as,
    #            d_ba, d_bb, d_bc, d_bs,
    #            d_ca, d_cb, d_cc, d_cs,
    #            d_sa, d_sb, d_sc, d_ss )
    # First row
    my_aa = np.sum(hess_xx[0] * diff_xx)  # daa
    my_ab = np.sum(hess_xx[1] * diff_xx)
    my_ac = np.sum(hess_xx[2] * diff_xx)
    # AS = hess_xx[3] * diff_xx   unused variable
#     ABCS = np.concatenate([np.array([AA, AB, AC]), AS])
    # Second row
    my_ba = np.sum(hess_xx[4] * diff_xx)
    my_bb = np.sum(hess_xx[5] * diff_xx)
    my_bc = np.sum(hess_xx[6] * diff_xx)
    # BS = hess_xx[7] * diff_xx   unused variable
#     BACS = np.concatenate([np.array([BA, BB, BC]), BS])
    # Third row
    my_ca = np.sum(hess_xx[8] * diff_xx)
    my_cb = np.sum(hess_xx[9] * diff_xx)
    my_cc = np.sum(hess_xx[10] * diff_xx)
    # CS = hess_xx[11] * diff_xx   unused variable
#     CABS = np.concatenate([np.array([CA, CB, CC]), CS])
    # All other rows
    my_sa = hess_xx[12] * diff_xx
    my_sb = hess_xx[13] * diff_xx
    my_sc = hess_xx[14] * diff_xx
    my_ss = hess_xx[15] * diff_xx

    # SA = np.array([np.sum(hess_xx[0]*i) for i in hess_xx[15]])*diff_xx
    # SB = np.array([np.sum(hess_xx[5]*i) for i in hess_xx[15]])*diff_xx
    # SC = np.array([np.sum(hess_xx[10]*i) for i in hess_xx[15]])*diff_xx

    # So the top left corner of the matrix is 3x3
    # and contains the correlatins between A, B and C:

    abc_abc = np.array([[my_aa, my_ab, my_ac],
                        [my_ba, my_bb, my_bc],
                        [my_ca, my_cb, my_cc],
                        ])
    # These are the AS, AB and AC contributions
    abcs = np.array([my_sa, my_sb, my_sc])
    return abc_abc, abcs, my_ss


def cost(x, svh, svv, theta, sigma=0.5):
    """The SAR WCM cost function.
    Function takes a parameter vector for both VV and VH polarisations.
    The vector is given has polarisation dependent parameters (`A`, `B`
    and `C`), and polarisation common parameters (`V1`, `V2` and `sigmas`).
    We assume that `A`, `B` and `C` are fixed within the inversion window,
    and that `V1`, `V2` and `sigmas` are time-varying. In some cases, we
    may want to solve for `V1` and `V2`, in some others, we may not.
    Arguments:
        x [array] -- Parameter array. Given as [A_vv, B_VV, C_vv,
                                                A_vh, B_vh, C_vh,
                                                V1[1...n_obs],
                                                V2[1...n_obs],
                                                sigma_s[1...n_obs]
                                                ]
        svh [array] -- Array of backscatter measurements VH polarisation
        svv [array] -- Array of backscatter measurements VV polarisation
        theta [array] -- Angle of incidence (per observation)

    Keyword Arguments:
        sigma {float|array} -- Backscatter uncertainty (default: 0.5)

    Returns:
        Cost -- Cost function assuming all observations are independent
    """
    a_vv, b_vv, c_vv = x[:3]
    a_vh, b_vh, c_vh = x[3:6]
    n_obs = int((x.shape[0] - 6) / 3)

    my_v1 = x[6:(6 + n_obs)]
    my_v2 = x[(6 + n_obs):(6 + 2 * n_obs)]
    my_s = x[(6 + 2 * n_obs):]
    x_vv = np.r_[a_vv, b_vv, c_vv, my_v1, my_v2, my_s]
    x_vh = np.r_[a_vh, b_vh, c_vh, my_v1, my_v2, my_s]
    # x_vv = np.array([a_vv, b_vv, c_vv, V1, V2, s])
    # x_vh = np.array([a_vh, b_vh, c_vh, V1, V2, s])

    sigma_vv = wcm(x_vv, theta=theta)
    sigma_vh = wcm(x_vh, theta=theta)
    diff_vv = (svv - sigma_vv)
    diff_vh = (svh - sigma_vh)
    my_cost = 0.5 * (diff_vv**2 + diff_vh**2) / (sigma**2)
    cost0 = np.sum(my_cost)
    return cost0


def cost_jac(x, svh, svv, theta, sigma=0.5):
    """The **Jacobian** of the SAR WCM cost function.
    Function takes a parameter vector for both VV and VH polarisations.
    The vector is given has polarisation dependent parameters (`A`, `B`
    and `C`), and polarisation common parameters (`V1`, `V2` and `sigmas`).
    We assume that `A`, `B` and `C` are fixed within the inversion window,
    and that `V1`, `V2` and `sigmas` are time-varying. In some cases, we
    may want to solve for `V1` and `V2`, in some others, we may not.
    Arguments:
        x [array] -- Parameter array. Given as [A_vv, B_VV, C_vv,
                                                A_vh, B_vh, C_vh,
                                                V1[1...n_obs],
                                                V2[1...n_obs],
                                                sigma_s[1...n_obs]
                                                ]
        svh [array] -- Array of backscatter measurements VH polarisation
        svv [array] -- Array of backscatter measurements VV polarisation
        theta [array] -- Angle of incidence (per observation)

    Keyword Arguments:
        sigma {float|array} -- Backscatter uncertainty (default: 0.5)

    Returns:
        Cost -- Jacobian of cost function assuming all observations
        are independent
    """

    a_vv, b_vv, c_vv = x[:3]
    a_vh, b_vh, c_vh = x[3:6]
    n_obs = int((x.shape[0] - 6) / 3)

    my_v1 = x[6:(6 + n_obs)]
    my_v2 = x[(6 + n_obs):(6 + 2 * n_obs)]
    my_s = x[(6 + 2 * n_obs):]
    x_vv = np.r_[a_vv, b_vv, c_vv, my_v1, my_v2, my_s]
    x_vh = np.r_[a_vh, b_vh, c_vh, my_v1, my_v2, my_s]
    # x_vv = np.array([a_vv, b_vv, c_vv, V1, V2, my_s])
    # x_vh = np.array([a_vh, b_vh, c_vh, V1, V2, my_s])
    sigma_vv = wcm(x_vv, theta=theta)
    sigma_vh = wcm(x_vh, theta=theta)
    diff_vv = (svv - sigma_vv)
    diff_vh = (svh - sigma_vh)
    dvv = wcm_jac(x_vv, theta=theta)
    dvh = wcm_jac(x_vh, theta=theta)

    jac = \
        np.concatenate([np.array([np.sum(dvv[0] * diff_vv),
                                  np.sum(dvv[1] * diff_vv),
                                  np.sum(dvv[2] * diff_vv),
                                  # np.sum(dvv[3]*diff_vv), # Removed D param
                                  np.sum(dvh[0] * diff_vh),
                                  np.sum(dvh[1] * diff_vh),
                                  np.sum(dvh[2] * diff_vh)]),
                        # np.sum(dvh[3]*diff_vh)]),# Removed D parameter
                        dvv[-1] * diff_vv + dvh[-1] * diff_vh])
    return -jac / sigma**2


def cost_hess(x, svh, svv, theta, sigma=0.5):
    """The **Hessian** of the SAR WCM cost function.
    Function takes a parameter vector for both VV and VH polarisations.
    The vector is given has polarisation dependent parameters (`A`, `B`
    and `C`), and polarisation common parameters (`V1`, `V2` and `sigmas`).
    We assume that `A`, `B` and `C` are fixed within the inversion window,
    and that `V1`, `V2` and `sigmas` are time-varying. In some cases, we
    may want to solve for `V1` and `V2`, in some others, we may not.
    Arguments:
        x [array] -- Parameter array. Given as [A_vv, B_VV, C_vv,
                                                A_vh, B_vh, C_vh,
                                                V1[1...n_obs],
                                                V2[1...n_obs],
                                                sigma_s[1...n_obs]
                                                ]
        svh [array] -- Array of backscatter measurements VH polarisation
        svv [array] -- Array of backscatter measurements VV polarisation
        theta [array] -- Angle of incidence (per observation)

    Keyword Arguments:
        sigma {float|array} -- Backscatter uncertainty (default: 0.5)

    Returns:
        Cost -- Hessian of cost function assuming all observations
        are independent
    """
    a_vv, b_vv, c_vv = x[:3]
    a_vh, b_vh, c_vh = x[3:6]
    n_obs = int((x.shape[0] - 6) / 3)

    my_v1 = x[6:(6 + n_obs)]
    my_v2 = x[(6 + n_obs):(6 + 2 * n_obs)]
    my_s = x[(6 + 2 * n_obs):]
    x_vv = np.r_[a_vv, b_vv, c_vv, my_v1, my_v2, my_s]
    x_vh = np.r_[a_vh, b_vh, c_vh, my_v1, my_v2, my_s]
    # x_vv = np.array([a_vv, b_vv, c_vv, V1, V2, s])
    # x_vh = np.array([a_vh, b_vh, c_vh, V1, V2, s])

    sigma_vv = wcm(x_vv, theta=theta)
    sigma_vh = wcm(x_vh, theta=theta)

    diff_vv = (svv - sigma_vv)
    diff_vh = (svh - sigma_vh)

    dvv = wcm_jac(x_vv, theta=theta)
    dvh = wcm_jac(x_vh, theta=theta)
    hess_vv = wcm_hess(x_vv, theta=theta)
    hess_vh = wcm_hess(x_vh, theta=theta)
    # The hessian contribution to the cost function is given by
    # H'C_{obs}H'^{T} - H''C_{obs}(H(x)-y)
    # since C_obs is diagonal, the first term is diagonal
    linear_hess_term = \
        np.diag(np.concatenate([np.array([np.sum(dvv[0]),
                                          np.sum(dvv[1]),
                                          np.sum(dvv[2]),
                                          # np.sum(dvv[3]*diff_vv),
                                          # Removed D parameter
                                          np.sum(dvh[0]),
                                          np.sum(dvh[1]),
                                          np.sum(dvh[2])
                                         ]),
                                # np.sum(dvh[3]*diff_vh)]),
                                # Removed D parameter
                                dvv[-1] + dvh[-1]])**2) / (sigma**2)

    abc_abc_vv, abcs_vv, ss_vv = hessian_time_residual(hess_vv, diff_vv)
    # top_rows = np.vstack([ABC_ABC, ABCS.T])
    # bot_rows = np.vstack([ABCS, np.diag(SS)])
    # hessian_res_vv  = np.hstack([top_rows, bot_rows])
    abc_abc_vh, abcs_vh, ss_vh = hessian_time_residual(hess_vh, diff_vh)

    # top_rows = np.vstack([ABC_ABC, ABCS.T])
    # bot_rows = np.vstack([ABCS, np.diag(SS)])
    # hessian_res_vh  = np.hstack([top_rows, bot_rows])
    top_rows_vv = np.vstack([abc_abc_vv, np.zeros_like(abc_abc_vv), abcs_vv.T])
    top_rows_vh = np.vstack([np.zeros_like(abc_abc_vh),
                             abc_abc_vh, abcs_vh.T])
    bot_rows = np.vstack([abcs_vv, abcs_vh, np.diag(ss_vv + ss_vh)])
    hessian_residual = np.hstack([top_rows_vv, top_rows_vh, bot_rows])
    cost_f_hessian = linear_hess_term - hessian_residual / sigma**2
    return cost_f_hessian  # , linear_hess_term
