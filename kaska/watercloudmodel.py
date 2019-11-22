#!/usr/bin/env python
"""Some useful functions for the Water Cloud Model (WCM)
used to retrieve parameters from Sentinel 1 data. The model
is first presented in Attema & Ulaby (1978)

The WCM predicts backscatter in a polarisation `pp` as a function
of some parameters:
$$
\sigma_{pp}^{0} = A\cdot V_{1}\left[1 - \exp\left(-\frac{-2B\cdot V_{2}}{\cos\theta}\right)\right] + \exp\left(-\frac{-2B\cdot V_{2}}{\cos\theta}\right)\cdot\left(C + D\cdot M_{v}\right).
$$

`A*V_1` is basically the backscattering coefficient, whereas 
`B*V_2` is the extinction coefficient. `C` relates `VSM` (volumetric
soil moisture in [%]) to backscatter. In general, all the "constants" 
(`A`, `B`, `C`, `D`) are polarisation dependent. `V1` and `V2` have to do with
the scatterers within the turbid medium, and are usually related to LAI.
"""

import numpy as np
import scipy.stats as SS_vh


def wcm_jac_(A, V1, B, V2, R, alpha, C, theta=23):
    """WCM model and jacobian calculations. The main
    assumption here is that we only consider first
    order effects. The vegetation backscatter contribution
    is given by `A*V1`, which is often related to scatterer
    (e.g. leaves, stems, ...) properties. The attenuation
    due to the canopy is controlled by `B*V2`, which is
    often related to canopy moisture content (this is polarisation
    and frequency dependent). The soil backscatter is modelled as
    an additive model (in dB units, multiplicative in linear), with
    a roughness term and a moisture-controlled term. The soil moisture
    term can be interpreted in weird and wonderful manners once retrieved
    (eg inverting the dielectric constant)
    This function returns the gradient for all parameters (A, B,
    V1, V2 and C)."""
    mu = np.cos(np.deg2rad(theta))
    tau = np.exp(-2 * B * V2 / mu)
    veg = A * V1 * mu * (1 - tau)
    sigma_soil = R+alpha
    soil = tau * sigma_soil + C

    der_dA = V1 * mu - V1 * mu * tau
    der_dV1 = A * mu - A * mu * tau
    der_dB = (-2 * V2 / mu) * tau * (-A * V1 * mu + sigma_soil)
    der_dV2 = (-2 * B / mu) * tau * (-A * V1 * mu + sigma_soil)
    der_dC = 1
    der_dR = tau
    der_dalpha = tau

    # Also returns der_dV1 and der_dV2
    return (
        veg + soil,
        [der_dA, der_dB, der_dC, der_dR, der_dalpha, der_dV1, der_dV2]
    )

def fwd_model_(x, svh, svv, theta):
    """Running the model forward to predict backscatter"""
    n_obs = len(svv)
    A_vv, B_vv, C_vv, A_vh, B_vh, C_vh = x[:6]
    alpha = x[6 : (6 + n_obs)]
    R = x[(6 + n_obs):(6 + 2*n_obs)]
    lai = x[(6 + 2*n_obs) :]
    sigma_vv, dvv = wcm_jac_(A_vv, lai, B_vv, lai, C_vv, R, alpha, theta=theta)
    sigma_vh, dvh = wcm_jac_(A_vh, lai, B_vh, lai, C_vh, R, alpha, theta=theta)
    return sigma_vv, sigma_vh

def cost_obs_(x, svh, svv, theta, unc=0.5):
    """Cost function. Order of parameters is
    A_vv, B_vv, C_vv, A_vh, B_vh, C_vh,
    vsm_0, ..., vsm_N,
    LAI_0, ..., LAI_N
    We assume that len(svh) == N
    Uncertainty is the uncertainty in backscatter, and
    assume that there are two polarisations (VV and VH),
    although these are just labels!
    """
    n_obs = svh.shape[0]
    A_vv, B_vv, C_vv, A_vh, B_vh, C_vh = x[:6]
    alpha = x[6 : (6 + n_obs)]
    R = x[(6 + n_obs):(6 + 2*n_obs)]
    lai = x[(6 + 2*n_obs) :]
    sigma_vv, dvv = wcm_jac_(A_vv, lai, B_vv, lai, C_vv, R, alpha, theta=theta)
    sigma_vh, dvh = wcm_jac_(A_vh, lai, B_vh, lai, C_vh, R, alpha, theta=theta)
    diff_vv = svv - sigma_vv
    diff_vh = svh - sigma_vh
    #NOTE!!!!! Only fits the VV channel!!!!
    # Soil misture in VH is complicated
    diff_vh = 0.
    cost = 0.5 * (diff_vv ** 2 + diff_vh ** 2) / (unc ** 2)
    jac = np.concatenate(
        [##[der_dA, der_dB, der_dC, der_dR, der_dalpha, der_dV1, der_dV2]
            np.array(
                [
                    np.sum(dvv[0] * diff_vv),  # A_vv
                    np.sum(dvv[1] * diff_vv),  # B_vv
                    np.sum(dvv[2] * diff_vv),  # C_vv
                    np.sum(dvh[0] * diff_vh),  # A_vh
                    np.sum(dvh[1] * diff_vh),  # B_vh
                    np.sum(dvh[2] * diff_vh),
                ]
            ),  # C_vh
            dvv[3] * diff_vv + dvh[3] * diff_vh,  # R
            dvv[4] * diff_vv + dvh[4] * diff_vh,  # alpha
            (dvv[5] + dvv[6]) * diff_vv + (dvh[5] + dvh[6]) * diff_vh,  # LAI
        ]
    )
    return np.nansum(cost), -jac / (unc ** 2)


def cost_prior_(x, svh, svv, theta, prior_mean, prior_unc):
    """A Gaussian cost function prior. We assume no correlations
    between parameters, only mean and standard deviation.
    Cost function. Order of parameters is
    A_vv, B_vv, C_vv, A_vh, B_vh, C_vh,
    alpha_0, ..., alpha_N,
    ruff_0, ..., ruff_N,
    LAI_0, ..., LAI_N
    We assume that len(svh) == N
    """
    n_obs = len(svh)
    prior_cost = 0.5 * (prior_mean - x) ** 2 / prior_unc ** 2
    dprior_cost = -(prior_mean - x) / prior_unc ** 2
    dprior_cost[:6] = 0.0
    # Ruff->No prior!
    dprior_cost[(6 + n_obs):(6 + 2*n_obs)] = 0.
    cost0 = prior_cost[6:(6+n_obs)].sum() # alpha cost
    cost1 = prior_cost[(6+2*n_obs):].sum() # LAI cost
    return cost0 + cost1, dprior_cost


def cost_smooth_(x, gamma):
    """A smoother for one parameter (e.g. LAI or whatever).
    `gamma` controls the magnitude of the smoothing (higher
    `gamma`, more smoothing)
    """
    # Calculate differences
    p_diff1 = x[1:-1] - x[2:]
    p_diff2 = x[1:-1] - x[:-2]
    # Cost function
    xcost_model = 0.5 * gamma * np.sum(p_diff1 ** 2 + p_diff2 ** 2)
    # Jacobian
    xdcost_model = 1 * gamma * (p_diff1 + p_diff2)
    # Note that we miss the first and last elements of the Jacobian
    # They're zero!
    return xcost_model, xdcost_model


def cost_function(x, svh, svv, theta, gamma, prior_mean, prior_unc, unc=0.8):
    """A combined cost function that calls the prior, fit to the observations
    """
    # Fit to the observations
    cost1, dcost1 = cost_obs_(x, svh, svv, theta, unc=unc)
    # Fit to the prior
    cost2, dcost2 = cost_prior_(x, svh, svv, theta, prior_mean, prior_unc)
    # Smooth evolution of LAI
    n_obs = len(svv)
    lai = x[(6 + 2*n_obs) :]
    cost3, dcost3 = cost_smooth_(lai, gamma[1])
    tmp = np.zeros_like(dcost1)
    tmp[(7 + 2*n_obs) : -1] = dcost3
    # Smooth evolution of ruffness
    R = x[(6 + n_obs):(6 + 2*n_obs)]
    cost4, dcost4 = cost_smooth_(R, gamma[0])
    tmp[(7 + n_obs) : (5 + 2*n_obs)] = dcost4
    return cost1 + cost2 + cost3 + cost4, dcost1 + dcost2 + tmp
