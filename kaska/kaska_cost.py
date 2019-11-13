#!/usr/bin/env python
from collections import namedtuple

import numpy as np

common_computations = namedtuple('common_computations',
                                 'diff y_unc p_diff tstep dH_fwd')


class CostWrapper(object):
    def __init__(self, time_grid, current_data,
                 gamma, emu,
                 mu_prior, c_prior_inv):

        self.gamma = gamma
        self.time_grid = time_grid
        self.current_data = current_data
        self.mu_prior = mu_prior
        self.c_prior_inv = c_prior_inv
        self.doy_obs = self.current_data.doy

        self.n_tsteps = self.time_grid.shape[0]
        self.n_params = self.mu_prior.shape[0]//self.n_tsteps

        self.emu = emu

        self.common_computations = None
        self.x = None

    def do_common_calculations(self, x):
        idx = np.argmin(np.abs(np.array(self.doy_obs)[:, None] -
                        np.array(self.time_grid)),
                        axis=1)
        if x == self.x:  # Check np.equal or whatever
            return self.common_computations
            # self.COMMON_compt (with a better name!) is a structure that stores the common
            # calculations (see the namedtuple definition for what can go there)
        else:
            diff = []
            y_unc = []
            step = []
            dH_fwd = []
            for tstep in np.unique(idx):
                x_f = x[tstep*self.n_params:((tstep+1)*self.n_params)]
                obs_list = np.where(idx == tstep)
                for j in obs_list[0]:

                    refl = self.current_data.rho_surf[j]
                    rho_unc = self.current_data.rho_unc[j]
                    sza = self.current_data.sza[j]
                    vza = self.current_data.vza[j]
                    raa = self.current_data.raa[j]
                    x_tstep = np.r_[x_f, sza, vza, raa]
                    emu_fwd = self.emu.predict(x_tstep, cal_jac=True)
                    for band in range(len(refl)):
                        y = refl[band]
                        y_unc.append(rho_unc[band])
                        y_fwd = emu_fwd[band][0].squeeze()
                        dH_fwd.append(emu_fwd[band][1][:-3])
                        diff.append(y_fwd - y)
                        step.append(tstep)

            p_diff = {}
            for param in range(self.n_params):
                xp = x[param::self.n_params]
                # p_diff = 1*np.gradient(xp[::-1], edge_order=2)
                p_diff[param] = xp[1:-1] - xp[2:] + xp[1:-1] - xp[:-2]
            self.x = x
            self.common_computations = common_computations(diff, y_unc, p_diff, step, dH_fwd)
            return self.common_computations

    def calc_cost(self, x):
        computations = self.do_common_calculations(x)
        obs_cost = 0.
        for diff, y_unc in zip(computations.diff, computations.y_unc):
            obs_cost += 0.5*(diff**2)/y_unc**2

        d = (x - self.mu_prior)
        cost_prior = 0.5*(d@self.c_prior_inv@d)

        cost_model = 0
        for param in range(self.n_params):
            if isinstance(self.gamma, list):
                xcost_model = 0.5*self.gamma[param]*np.sum(computations.p_diff[param]**2)
            else:
                xcost_model = 0.5*self.gamma*np.sum(computations.p_diff[param]**2)
            cost_model += xcost_model

        cost = obs_cost + cost_prior + cost_model
        return cost  # scalar

    def calc_dcost(self, x):
        computations = self.do_common_calculations(x)

        obs_dcost = np.zeros_like(x)
        for diff, y_unc, tstep, dH_fwd in zip(computations.diff, computations.y_unc,
                                              computations.tstep, computations.dH_fwd):
            obs_dcost[tstep*self.n_params:((tstep+1)*self.n_params)] += \
                      dH_fwd*diff/y_unc**2

        d = (x - self.mu_prior)
        dcost_prior = self.c_prior_inv@d

        dcost_model = np.zeros_like(x)
        for param in range(self.n_params):
            if isinstance(self.gamma, list):
                xdcost_model = 1*self.gamma[param]*computations.p_diff[param]
            else:
                xdcost_model = 1*self.gamma*computations.p_diff[param]
            dcost_model[param::self.n_params][1:-1] += xdcost_model

        dcost = obs_dcost + dcost_prior + dcost_model
        return dcost  # vector shape as x```

    def calc_cost_dcost(self, x, *args):

        idx = np.argmin(np.abs(np.array(self.doy_obs)[:, None] -
                        np.array(self.time_grid)),
                        axis=1)
        obs_cost = 0.
        obs_dcost = np.zeros_like(x)
        for tstep in np.unique(idx):
            x_f = x[tstep*self.n_params:((tstep+1)*self.n_params)]
            obs_list = np.where(idx == tstep)
            for j in obs_list[0]:

                refl = self.current_data.rho_surf[j]
                rho_unc = self.current_data.rho_unc[j]
                sza = self.current_data.sza[j]
                vza = self.current_data.vza[j]
                raa = self.current_data.raa[j]
                x_tstep = np.r_[x_f, sza, vza, raa]
                emu_fwd = self.emu.predict(x_tstep, cal_jac=True)
                for band in range(len(refl)):
                    y = refl[band]
                    y_unc = rho_unc[band]
                    y_fwd = emu_fwd[band][0].squeeze()
                    dH_fwd = emu_fwd[band][1][:-3]
                    diff = y_fwd - y
                    obs_cost += 0.5*(diff**2)/y_unc**2
                    obs_dcost[tstep*self.n_params:((tstep+1)*self.n_params)] += \
                        dH_fwd*diff/y_unc**2
        d = (x - self.mu_prior)
        cost_prior = 0.5*(d@self.c_prior_inv@d)
        dcost_prior = self.c_prior_inv@d
        cost_model = 0
        dcost_model = np.zeros_like(x)
        for param in range(self.n_params):
            xp = x[param::self.n_params]
            # p_diff = 1*np.gradient(xp[::-1], edge_order=2)
            p_diff = xp[1:-1] - xp[2:] + xp[1:-1] - xp[:-2]
            if isinstance(self.gamma, list):
                xcost_model = 0.5*self.gamma[param]*np.sum(p_diff**2)
                xdcost_model = 1*self.gamma[param]*p_diff
            else:
                xcost_model = 0.5*self.gamma*np.sum(p_diff**2)
                xdcost_model = 1*self.gamma*p_diff
            dcost_model[param::self.n_params][1:-1] += xdcost_model
            cost_model += xcost_model
        return (obs_cost + cost_prior + cost_model,
                obs_dcost + dcost_prior + dcost_model)
