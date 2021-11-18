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
import pdb

from sense.canopy import OneLayer
from sense.soil import Soil
from sense import model
from sense.dielectric import Dobson85
from sense.core import Reflectivity
from sense.util import f2lam

# def wcm_jac_(A, V1, B, V2, R, alpha, C, theta=23):
#     """WCM model and jacobian calculations. The main
#     assumption here is that we only consider first
#     order effects. The vegetation backscatter contribution
#     is given by `A*V1`, which is often related to scatterer
#     (e.g. leaves, stems, ...) properties. The attenuation
#     due to the canopy is controlled by `B*V2`, which is
#     often related to canopy moisture content (this is polarisation
#     and frequency dependent). The soil backscatter is modelled as
#     an additive model (in dB units, multiplicative in linear), with
#     a roughness term and a moisture-controlled term. The soil moisture
#     term can be interpreted in weird and wonderful manners once retrieved
#     (eg inverting the dielectric constant)
#     This function returns the gradient for all parameters (A, B,
#     V1, V2 and C)."""
#     mu = np.cos(np.deg2rad(theta))
#     tau = np.exp(-2 * B * V2 / mu)
#     veg = A * V1 * mu * (1 - tau)
#     sigma_soil = R+alpha
#     soil = tau * sigma_soil + C

#     der_dA = V1 * mu - V1 * mu * tau
#     der_dV1 = A * mu - A * mu * tau
#     der_dB = (-2 * V2 / mu) * tau * (-A * V1 * mu + sigma_soil)
#     der_dV2 = (-2 * B / mu) * tau * (-A * V1 * mu + sigma_soil)
#     der_dC = 1
#     der_dR = tau
#     der_dalpha = tau

#     # Also returns der_dV1 and der_dV2
#     return (
#         veg + soil,
#         [der_dA, der_dB, der_dC, der_dR, der_dalpha, der_dV1, der_dV2]
#     )


# def fresnel(e):
#     return np.abs( (1.-np.sqrt(e))/(1.+np.sqrt(e))   )**2.

# def refelctivity(eps, theta):
#     """
#     table 2.5 Ulaby (2014)
#     assumes specular surface
#     Parameters
#     ----------
#     eps : complex
#         relative dielectric permitivity
#     theta : float, ndarray
#         incidence angle [rad]
#         can be specified
#     """
#     co = np.cos(theta)
#     si2 = np.sin(theta)**2.
#     rho_v = (eps*co-np.sqrt(eps-si2))/(eps*co+np.sqrt(eps-si2))
#     rho_h = (co-np.sqrt(eps-si2))/(co+np.sqrt(eps-si2))

#     v = np.abs(rho_v)**2.
#     h = np.abs(rho_h)**2.

#     return v, h

# def ssrt_jac_oh92_(eps, coef, LAI, H, theta):
#     theta = np.deg2rad(theta)
#     mu = np.cos(theta)
#     omega = 0.027
#     freq = 5.405
#     k = 2.*np.pi / f2lam(freq)
#     s = 0.0115

#     ks = k * s

#     c = 1./(3.*fresnel(eps))
#     p = (1. - (2.*theta/np.pi)**c * np.exp(-ks))**2.

#     v, h = refelctivity(eps,theta)
#     a = 0.7*(1.-np.exp(-0.65*ks**1.8)) * np.cos(theta)**3.
#     b = (v+h) / np.sqrt(p)

#     sigma_soil = a*b

#     tau = np.exp(-coef * np.sqrt(LAI) * H / mu)**2

#     soil = tau * sigma_soil
#     veg = omega * mu / 2 * (1 - tau)

#     co = np.cos(theta)
#     si = np.sin(theta)
#     si2 = np.sin(theta)**2.
#     hoch = (np.sqrt(eps)+1)**2./(3*(1-np.sqrt(eps))**2.)
#     d = np.exp(-ks)
#     f = 2.*theta/np.pi

#     k = sigma_soil
#     m = np.sqrt(LAI) * H / mu
#     l = omega * mu / 2


#     part_one = (2*a*(co-1/2*np.sqrt(eps-si2))*(eps*co-np.sqrt(eps-si2))) / (p * (eps*co+np.sqrt(eps-si2))**2)
#     part_two = (2*a*(co+1/2*np.sqrt(eps-si2))*(eps*co-np.sqrt(eps-si2))**2) / (p * (eps*co+np.sqrt(eps-si2))**3)
#     part_three = (2*a*d*( (np.sqrt(eps)+1)**2)/(3*(1-np.sqrt(eps))**3*np.sqrt(eps)) + (np.sqrt(eps)+1)/(3*(1-np.sqrt(eps))**2*np.sqrt(eps)) * f**c * np.log(f) * (eps*co-np.sqrt(eps-si2))**2) / ((1-d*f**c)**3*(np.sqrt(eps-si2)+eps*co)**2)
#     p1 = part_one-part_two+part_three

#     part_four = (-a*co-np.sqrt(eps-si2)) / (2*(1-d*f**c)**2 * np.sqrt(eps-si2) * (np.sqrt(eps-si2)+co)**2)
#     part_five = (2*d*((np.sqrt(eps)+1)**2 / (3*(1-np.sqrt(eps))**3*np.sqrt(eps)) + (np.sqrt(eps)+1)/(3*(1-np.sqrt(eps))**2*np.sqrt(eps))) *f**c * np.log(f) * (a*co-np.sqrt(eps-si2))) / ((1-d*f**c)**3 * (np.sqrt(eps-si2)+co))
#     part_six = 1/ (2*(1-d*f**c)**2 * np.sqrt(eps-si2) * (np.sqrt(eps*si2)+co))
#     p2 = part_four + part_five - part_six

#     der_mv = p1 + p2
#     # pdb.set_trace()
#     # part_one = (a*( ((co-1/2*np.sqrt(eps-si2))/(eps*co+np.sqrt(eps-si2))) - (1/(2*np.sqrt(eps-si2)*(co+np.sqrt(eps-si2)))) - ((1/2*np.sqrt(eps-si2)+co)*(eps*co-np.sqrt(eps-si2))/(eps*co+np.sqrt(eps-si2))**2) )) / p
#     # part_two = (2*a*d*((((np.sqrt(eps)+1)**2)/(3*(1-np.sqrt(eps))**3 * np.sqrt(eps))) + (np.sqrt(eps)+1)/(3*(1-np.sqrt(eps))**2 * np.sqrt(eps))) * f**c * np.log(f) * (v+h)) / (1-d*f**c)**3

#     # der_mv = part_one + part_two
#     der_coef = -2*l*(k-m)*np.exp(-2*l-coef)

#     return (
#         veg + soil,
#         [der_mv, der_coef]
#     )


def ssrt_jac_vwc(mv, vwc, s, omega, b, theta):
    """"""

    # omega = 0.027
    freq = 5.405
    k = 2.*np.pi / f2lam(freq)
    # s = 0.0115

    clay = 0.0738
    sand = 0.2408
    bulk = 1.45

    ks = k * s

    mu = np.cos(np.deg2rad(theta))
    sin = np.sin(1.5*np.deg2rad(theta))
    a = 0.11 * mv**0.7 * mu**2.2
    bb = 1 - np.exp(-0.32 * ks**1.8)
    q = 0.095 * (0.13 + sin)**1.4 * (1-np.exp(-1.3 * ks**0.9))
    sigma_soil = a * bb / q

    tau = np.exp(-b * vwc / mu)**2

    # soil2 = tau * sigma_soil
    # veg = omega * mu / 2 * (1 - tau)
    # pdb.set_trace()
    # Sense
    models = {'surface': 'Oh04', 'canopy': 'turbid_isotropic'}
    can = 'turbid_isotropic'
    ke = b * vwc
    if np.nanmean(theta) > 5.:
        theta = np.deg2rad(theta)

    # soil
    soil = Soil(mv=mv, s=s, f=freq, clay=clay, sand=sand, bulk=bulk)

    # canopy
    can = OneLayer(canopy=can, ke_h=ke, ke_v=ke, d=1., ks_h = omega * ke, ks_v = omega*ke)

    S = model.RTModel(surface=soil, canopy=can, models=models, theta=theta, freq=freq)
    S.sigma0()
    S.__dict__['stot']['vv'[::-1]], S.__dict__['stot']['vh'[::-1]]

    s0g = S.__dict__['s0g']['vv']
    s0c = S.__dict__['s0c']['vv']
    s0cgt = S.__dict__['s0cgt']['vv']
    s0gcg = S.__dict__['s0gcg']['vv']
    stot = S.__dict__['stot']['vv']

    eps =  Dobson85(clay=clay, sand=sand, bulk=bulk, mv=mv, freq=freq).eps
    v = Reflectivity(eps,theta).v

    sec = 1/np.cos(theta)

    der_omega1 = (1/2 * np.cos(theta) * (1-tau))
    der_omega2 = (4 * b * vwc * tau * v)
    der_omega3 = 1/2 * np.cos(theta) * v * (np.sqrt(tau)-tau)
    der_b1 = -2 * s0g * vwc * sec * tau
    der_vwc1 = -2 * s0g * b * sec * tau
    der_b2 = omega * vwc * tau
    der_vwc2 = omega * b * tau
    der_b3 = -4 * omega * vwc * v * tau * (2 * vwc * b * sec - 1)
    der_vwc3 = -4 * omega * b * v * tau * (2 * vwc * b * sec - 1)
    der_b4 = - omega * vwc * v * tau**2 * (tau -2 )
    der_vwc4 = - omega * b * v * tau**2 * (tau -2 )

    der_omega = der_omega1+der_omega2+der_omega3
    der_b = der_b1+der_b2+der_b3+der_b4
    der_vwc = der_vwc1+der_vwc2+der_vwc3+der_vwc4

    der_s = S.G.rt_s.der_s_vv
    der_mv = S.G.rt_s.der_mv_vv

    return (
        stot ,
        [der_omega, der_s, der_mv, der_vwc, der_b]
    )

    # return (
    #     stot ,
    #     [der_omega, der_s, der_mv, der_vwc]
    # )

def ssrt_vwc(mv, vwc, s, omega, b, theta):
    """"""

    # omega = 0.027
    freq = 5.405
    k = 2.*np.pi / f2lam(freq)
    # s = 0.0115

    clay = 0.0738
    sand = 0.2408
    bulk = 1.45

    ks = k * s

    mu = np.cos(np.deg2rad(theta))
    sin = np.sin(1.5*np.deg2rad(theta))
    a = 0.11 * mv**0.7 * mu**2.2
    bb = 1 - np.exp(-0.32 * ks**1.8)
    q = 0.095 * (0.13 + sin)**1.4 * (1-np.exp(-1.3 * ks**0.9))
    sigma_soil = a * bb / q

    tau = np.exp(-b * vwc / mu)**2

    # soil2 = tau * sigma_soil
    # veg = omega * mu / 2 * (1 - tau)
    # pdb.set_trace()
    # Sense
    models = {'surface': 'Oh04', 'canopy': 'turbid_isotropic'}
    can = 'turbid_isotropic'
    ke = b * vwc
    if np.nanmean(theta) > 5.:
        theta = np.deg2rad(theta)

    # soil
    soil = Soil(mv=mv, s=s, f=freq, clay=clay, sand=sand, bulk=bulk)

    # canopy
    can = OneLayer(canopy=can, ke_h=ke, ke_v=ke, d=1., ks_h = omega * ke, ks_v = omega*ke)

    S = model.RTModel(surface=soil, canopy=can, models=models, theta=theta, freq=freq)
    S.sigma0()
    S.__dict__['stot']['vv'[::-1]], S.__dict__['stot']['vh'[::-1]]

    s0g = S.__dict__['s0g']['vv']
    s0c = S.__dict__['s0c']['vv']
    s0cgt = S.__dict__['s0cgt']['vv']
    s0gcg = S.__dict__['s0gcg']['vv']
    stot = S.__dict__['stot']['vv']

    eps =  Dobson85(clay=clay, sand=sand, bulk=bulk, mv=mv, freq=freq).eps
    v = Reflectivity(eps,theta).v

    sec = 1/np.cos(theta)

    der_omega1 = (1/2 * np.cos(theta) * (1-tau))
    der_omega2 = (4 * b * vwc * tau * v)
    der_omega3 = 1/2 * np.cos(theta) * v * (np.sqrt(tau)-tau)
    der_b1 = -2 * s0g * vwc * sec * tau
    der_vwc1 = -2 * s0g * b * sec * tau
    der_b2 = omega * vwc * tau
    der_vwc2 = omega * b * tau
    der_b3 = -4 * omega * vwc * v * tau * (2 * vwc * b * sec - 1)
    der_vwc3 = -4 * omega * b * v * tau * (2 * vwc * b * sec - 1)
    der_b4 = - omega * vwc * v * tau**2 * (tau -2 )
    der_vwc4 = - omega * b * v * tau**2 * (tau -2 )

    der_omega = der_omega1+der_omega2+der_omega3
    der_b = der_b1+der_b2+der_b3+der_b4
    der_vwc = der_vwc1+der_vwc2+der_vwc3+der_vwc4

    der_s = S.G.rt_s.der_s_vv
    der_mv = S.G.rt_s.der_mv_vv

    # return (
    #     stot ,
    #     [der_s, der_omega, der_mv, der_vwc, der_b]
    # )

    return (
        stot ,s0g, s0c
    )



# def fwd_model_(x, svh, svv, theta):
#     """Running the model forward to predict backscatter"""
#     n_obs = len(svv)
#     A_vv, B_vv, C_vv, A_vh, B_vh, C_vh = x[:6]
#     alpha = x[6 : (6 + n_obs)]
#     R = x[(6 + n_obs):(6 + 2*n_obs)]
#     lai = x[(6 + 2*n_obs) :]
#     sigma_vv, dvv = wcm_jac_(A_vv, lai, B_vv, lai, C_vv, R, alpha, theta=theta)
#     sigma_vh, dvh = wcm_jac_(A_vh, lai, B_vh, lai, C_vh, R, alpha, theta=theta)
#     return sigma_vv, sigma_vh

# def cost_obs_(x, svh, svv, theta, unc=0.5):
#     """Cost function. Order of parameters is
#     A_vv, B_vv, C_vv, A_vh, B_vh, C_vh,
#     vsm_0, ..., vsm_N,
#     LAI_0, ..., LAI_N
#     We assume that len(svh) == N
#     Uncertainty is the uncertainty in backscatter, and
#     assume that there are two polarisations (VV and VH),
#     although these are just labels!
#     """
#     n_obs = svh.shape[0]
#     A_vv, B_vv, C_vv, A_vh, B_vh, C_vh = x[:6]
#     alpha = x[6 : (6 + n_obs)]
#     R = x[(6 + n_obs):(6 + 2*n_obs)]
#     lai = x[(6 + 2*n_obs) :]
#     sigma_vv, dvv = wcm_jac_(A_vv, lai, B_vv, lai, C_vv, R, alpha, theta=theta)
#     sigma_vh, dvh = wcm_jac_(A_vh, lai, B_vh, lai, C_vh, R, alpha, theta=theta)
#     diff_vv = svv - sigma_vv
#     diff_vh = svh - sigma_vh
#     #NOTE!!!!! Only fits the VV channel!!!!
#     # Soil misture in VH is complicated
#     diff_vh = 0.
#     cost = 0.5 * (diff_vv ** 2 + diff_vh ** 2) / (unc ** 2)
#     jac = np.concatenate(
#         [##[der_dA, der_dB, der_dC, der_dR, der_dalpha, der_dV1, der_dV2]
#             np.array(
#                 [
#                     np.sum(dvv[0] * diff_vv),  # A_vv
#                     np.sum(dvv[1] * diff_vv),  # B_vv
#                     np.sum(dvv[2] * diff_vv),  # C_vv
#                     np.sum(dvh[0] * diff_vh),  # A_vh
#                     np.sum(dvh[1] * diff_vh),  # B_vh
#                     np.sum(dvh[2] * diff_vh),
#                 ]
#             ),  # C_vh
#             dvv[3] * diff_vv + dvh[3] * diff_vh,  # R
#             dvv[4] * diff_vv + dvh[4] * diff_vh,  # alpha
#             (dvv[5] + dvv[6]) * diff_vv + (dvh[5] + dvh[6]) * diff_vh,  # LAI
#         ]
#     )

#     return np.nansum(cost), -jac / (unc ** 2)


def cost_obs_vwc(x, svh, svv, theta, unc=0.5, data=0):
    """Cost function. Order of parameters is
    s, omega,
    b_0, ..., b_N,
    vwc_0, ..., vwc_N
    mv_0, ..., mv_N
    We assume that len(svh) == N
    Uncertainty is the uncertainty in backscatter, and
    assume that there are two polarisations (VV and VH),
    although these are just labels!
    """
    n_obs = svh.shape[0]
    omega = x[:1]
    s = x[1 : (1 + n_obs)]
    mv = x[(1 + n_obs) : (1 + 2*n_obs)]
    vwc = x[(1 + 2*n_obs) : (1 + 3*n_obs)]
    b = x[(1 + 3*n_obs) : (1 + 4*n_obs)]

    sigma_vv, dvv = ssrt_jac_vwc(mv, vwc, s, omega, b, theta=theta)
    # sigma_vh, dvh = ssrt_jac_vwc(A_vh, lai, B_vh, lai, C_vh, R, alpha, theta=theta)
    diff_vv = svv - sigma_vv
    ### in dB ???
    diff_vv = 10*np.log10(svv) - 10*np.log10(sigma_vv)
    # diff_vh = svh - sigma_vh
    #NOTE!!!!! Only fits the VV channel!!!!
    # Soil misture in VH is complicated
    diff_vh = 0.
    cost = 0.5 * (diff_vv ** 2 + diff_vh ** 2) / (unc ** 2)

    jac = np.concatenate(
        [##[der_omega, der_s, der_mv, der_vwc, der_b]
            np.array(
                [
                    np.sum(dvv[0] * diff_vv),  # omega
                ]),
                dvv[1] * diff_vv,  # s
                dvv[2] * diff_vv,  # mv
                dvv[3] * diff_vv,  # vwc
                dvv[4] * diff_vv,  # b

        ]
    )
    # pdb.set_trace()
    return np.nansum(cost), -jac / (unc ** 2)



# def cost_obs_ssrt(x, svh, svv, theta, data, unc=0.3):
#     """
#     """
#     n_obs = svh.shape[0]
#     mv = x[:n_obs]
#     coef = x[n_obs:2*n_obs]
#     lai = data[1:(2*n_obs)]
#     h = data[0]

#     sigma_vv, dvv = ssrt_jac_(mv, coef, lai, h, theta=theta)

#     diff_vv = 10*np.log10(svv) - 10*np.log10(sigma_vv)
#     # diff_vv = svv - sigma_vv
#     # diff_vv = 10 ** (svv/10) - sigma_vv
#     # pdb.set_trace()
#     cost = 0.5 * (diff_vv ** 2) / (unc ** 2)

#     jac = np.concatenate(
#         ##[der_dA, der_dB, der_dC, der_dR, der_dalpha, der_dV1, der_dV2]
#             np.array(
#                 [
#                     (dvv[0] * diff_vv),  # mv
#                     (dvv[1] * diff_vv),  # coef
#                     # (dvv[2] * diff_vv),  # lai
#                     # (dvv[3] * diff_vv),  # height
#                 ]
#             )

#     )
#     # pdb.set_trace()
#     return np.nansum(cost), -jac / (unc ** 2)

# def cost_obs_ssrt_oh92_(x, svh, svv, theta, data, unc=0.3):
#     """
#     """
#     n_obs = svh.shape[0]
#     mv = x[:n_obs]
#     coef = x[n_obs:2*n_obs]
#     lai = data[n_obs:(2*n_obs)]
#     h = data[:n_obs]

#     sigma_vv, dvv = ssrt_jac_oh92_(mv, coef, lai, h, theta=theta)

#     diff_vv = svv - 10*np.log10(sigma_vv)

#     cost = 0.5 * (diff_vv ** 2) / (unc ** 2)

#     jac = np.concatenate(
#         ##[der_dA, der_dB, der_dC, der_dR, der_dalpha, der_dV1, der_dV2]
#             np.array(
#                 [
#                     (dvv[0] * diff_vv),  # mv
#                     (dvv[1] * diff_vv),  # coef
#                     # (dvv[2] * diff_vv),  # lai
#                     # (dvv[3] * diff_vv),  # height
#                 ]
#             )

#     )
#     # pdb.set_trace()
#     return np.nansum(cost), -jac / (unc ** 2)



# def cost_prior_(x, svh, svv, theta, prior_mean, prior_unc):
#     """A Gaussian cost function prior. We assume no correlations
#     between parameters, only mean and standard deviation.
#     Cost function. Order of parameters is
#     A_vv, B_vv, C_vv, A_vh, B_vh, C_vh,
#     alpha_0, ..., alpha_N,
#     ruff_0, ..., ruff_N,
#     LAI_0, ..., LAI_N
#     We assume that len(svh) == N
#     """
#     n_obs = len(svh)
#     prior_cost = 0.5 * (prior_mean - x) ** 2 / prior_unc ** 2
#     dprior_cost = -(prior_mean - x) / prior_unc ** 2
#     dprior_cost[:6] = 0.0
#     # Ruff->No prior!
#     dprior_cost[(6 + n_obs):(6 + 2*n_obs)] = 0.
#     cost0 = prior_cost[6:(6+n_obs)].sum() # alpha cost
#     cost1 = prior_cost[(6+2*n_obs):].sum() # LAI cost
#     return cost0 + cost1, dprior_cost


def cost_prior_vwc(x, svh, svv, theta, prior_mean, prior_unc):
    """A Gaussian cost function prior. We assume no correlations
    between parameters, only mean and standard deviation.
    Cost function. Order of parameters is
    s, omega,
    mv_0, ..., b_N,
    vwc_0, ..., vwc_N
    b_0, ..., mv_N
    We assume that len(svh) == N
    """
    n_obs = len(svh)

    # mean_prior = np.nanmean(prior_mean[(2) : (2 + n_obs)])
    # mean_x = np.nanmean(x[(2) : (2 + n_obs)])

    # ppp = prior_mean *1.
    # pppp = prior_mean *1.

    # ppp[(2) : (2 + n_obs)] = prior_mean[(2) : (2 + n_obs)] - mean_prior
    # pppp[(2) : (2 + n_obs)]  = x[(2) : (2 + n_obs)] - mean_x

    prior_cost = 0.5 * (prior_mean - x) ** 2 / prior_unc ** 2
    # prior_cost = 0.5 * (ppp - pppp) ** 2 / prior_unc ** 2



    dprior_cost = -(prior_mean - x) / prior_unc ** 2

    # dprior_cost = -(prior_mean/mean_prior*mean_x - x) / prior_unc ** 2


    dprior_cost[:1] = 0.0
    # Ruff->No prior!
    dprior_cost[(1) : (1 + n_obs)] = 0. # rms
    # dprior_cost[(1 + n_obs) : (1 + 2*n_obs)] = 0. # mv
    dprior_cost[(1 + 2*n_obs) : (1 + 3*n_obs)] = 0. # vwc
    dprior_cost[(1 + 3*n_obs) : (1 + 4*n_obs)] = 0. # b
    cost0 = prior_cost[(1 + n_obs) : (1 + 2*n_obs)].sum() # mv cost
    # pdb.set_trace()
    # print(cost0)
    return cost0 , dprior_cost

# def cost_prior_ssrt(x, svh, svv, theta, prior_mean, prior_unc):
#     """A Gaussian cost function prior. We assume no correlations
#     between parameters, only mean and standard deviation.
#     Cost function. Order of parameters is
#     A_vv, B_vv, C_vv, A_vh, B_vh, C_vh,
#     alpha_0, ..., alpha_N,
#     ruff_0, ..., ruff_N,
#     LAI_0, ..., LAI_N
#     We assume that len(svh) == N
#     """
#     # pdb.set_trace()
#     n_obs = len(svh)
#     prior_cost = 0.5 * (prior_mean - x) ** 2 / prior_unc ** 2
#     dprior_cost = -(prior_mean - x) / prior_unc ** 2
#     # coef->No prior!
#     # dprior_cost[(n_obs):(2*n_obs)] = 0.
#     # dprior_cost[:n_obs] = 0.
#     cost0 = prior_cost[:(n_obs)].sum() # mv cost
#     cost1 = prior_cost[n_obs:2*n_obs].sum() # coef cost
#     # cost0=0

#     return cost0 + cost1, dprior_cost


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


# def cost_function(x, svh, svv, theta, gamma, prior_mean, prior_unc, unc=0.8):
#     """A combined cost function that calls the prior, fit to the observations
#     """
#     # Fit to the observations
#     cost1, dcost1 = cost_obs_(x, svh, svv, theta, unc=unc)
#     # Fit to the prior
#     cost2, dcost2 = cost_prior_(x, svh, svv, theta, prior_mean, prior_unc)
#     # Smooth evolution of LAI
#     n_obs = len(svv)
#     lai = x[(6 + 2*n_obs) :]
#     cost3, dcost3 = cost_smooth_(lai, gamma[1])
#     tmp = np.zeros_like(dcost1)
#     tmp[(7 + 2*n_obs) : -1] = dcost3
#     # Smooth evolution of ruffness
#     R = x[(6 + n_obs):(6 + 2*n_obs)]
#     cost4, dcost4 = cost_smooth_(R, gamma[0])
#     tmp[(7 + n_obs) : (5 + 2*n_obs)] = dcost4
#     # pdb.set_trace()
#     return cost1 + cost2 + cost3 + cost4, dcost1 + dcost2 + tmp

def cost_function_vwc(x, svh, svv, theta, gamma, prior_mean, prior_unc, unc=0.8, data=0):
    """A combined cost function that calls the prior, fit to the observations
    """
    # Fit to the observations
    cost1, dcost1 = cost_obs_vwc(x, svh, svv, theta, unc=unc, data=data)
    # Fit to the prior
    cost2, dcost2 = cost_prior_vwc(x, svh, svv, theta, prior_mean, prior_unc)
    # print(cost1)
    # print(cost2)
    # print(x)
    # cost2 = 0
    # dcost2=0
    cost3 = 0
    cost4 = 0
    tmp = 0
    # Smooth evolution of sm
    # n_obs = len(svv)
    # lai = x[2 : (2 + n_obs)]
    # cost3, dcost3 = cost_smooth_(lai, gamma[1])
    # # pdb.set_trace()
    # tmp = np.zeros_like(dcost1)
    # tmp[3 : (2 + n_obs)-1] = dcost3
    # # Smooth evolution of ruffness
    # R = x[(6 + n_obs):(6 + 2*n_obs)]
    # cost4, dcost4 = cost_smooth_(R, gamma[0])
    # tmp[(7 + n_obs) : (5 + 2*n_obs)] = dcost4
    # pdb.set_trace()
    return cost1 + cost2 + cost3 + cost4, dcost1 + dcost2 + tmp




# def cost_function2(x, svh, svv, theta, gamma, prior_mean, prior_unc, data, unc=0.3):
#     """A combined cost function that calls the prior, fit to the observations
#     """
#     # Fit to the observations
#     cost1, dcost1 = cost_obs_ssrt(x, svh, svv, theta, data, unc=unc)
#     # cost1, dcost1 = cost_obs_ssrt_oh92_(x, svh, svv, theta, data, unc=unc)
#     # pdb.set_trace()
#     # Fit to the prior
#     cost2, dcost2 = cost_prior_ssrt(x, svh, svv, theta, prior_mean, prior_unc)
#     # pdb.set_trace()
#     # Smooth evolution of LAI
#     # n_obs = len(svv)
#     # lai = x[2*n_obs:3*n_obs]
#     # cost3, dcost3 = cost_smooth_(lai, gamma[1])
#     # tmp = np.zeros_like(dcost1)
#     # tmp[2*n_obs+1:-1] = dcost3
#     tmp=0
#     cost3=0

#     # # Smooth evolution of ruffness
#     # R = x[(6 + n_obs):(6 + 2*n_obs)]
#     # cost4, dcost4 = cost_smooth_(R, gamma[0])
#     # tmp[(7 + n_obs) : (5 + 2*n_obs)] = dcost4
#     # return cost1 + cost2 + cost3 + cost4, dcost1 + dcost2 + tmp
#     # pdb.set_trace()
#     return cost1 + cost2 + cost3, dcost1 + dcost2 + tmp
