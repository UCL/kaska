import numpy as np
import datetime as dt
from pathlib import Path 
import matplotlib.pyplot as plt
import scipy.sparse as sp
import gp_emulator

from smoothn import smoothn
from Two_NN import Two_NN

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from collections import namedtuple

import numpy as np

from scipy.optimize import minimize

common_computations = namedtuple('common_computations', 
                    'y y_unc y_fwd dH_fwd diff prior_cost dprior_cost')
s2_obs = namedtuple("s2_obs",
                    "doy mask rho_surf sza vza raa rho_unc")

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
        

        self.n_tsteps =  self.time_grid.shape[0]
        self.n_params = self.mu_prior.shape[0]//self.n_tsteps
    
        self.emu = emu

        self.common_computations= None
        self.x = None

    def calc_cost(self, x, *args):

        idx = np.argmin(np.abs(np.array(self.doy_obs)[:, None] - np.array(self.time_grid) ),
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
        d = (x- self.mu_prior )
        cost_prior = 0.5*(d@self.c_prior_inv@d)
        dcost_prior = self.c_prior_inv@d
        cost_model = 0
        dcost_model = np.zeros_like(x)
        for param in range(self.n_params):
            xp = x[param::self.n_params]
            #p_diff = 1*np.gradient(xp[::-1], edge_order=2)
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
                


def do_real_data(fi, field, shps):  
    from glob import glob           
    import datetime                 
    import gdal
    from reproject import reproject_data

    #s2_dir = '/data/netapp_3/ucfajlg/python/KaFKA_Validation/LMU/'
#
#    shps  = glob(s2_dir + '/carto/MNI_2017.shp')

    bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10','B11', 'B12']
    b_ind = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    ns = 1
    ress  = [60, 10, 10, 10, 20, 20, 20, 10, 20, 60, 60, 20, 20]

    dats = []                       

    angs = []    

    i = 0

    
    fi = fi.split("B02_sur.tif")[0]
    for _, ii in enumerate(b_ind):              

        band = bands[ii]

        res  = ress[ii]

        fname = fi + band + '_sur.tif'

        if _ ==0:

            g = gdal.Warp('', fname, xRes=10, yRes=10, resampleAlg=0, cutlineWhere =  field,\

                             cutlineDSName=shps, cropToCutline=True, format='MEM')

            data = g.ReadAsArray()

        else:

            data  = reproject_data(fname, g, xRes=10, yRes=10, resample=0).data

        dats.append(data)

    sas   = reproject_data(fi.split('IMG_DATA')[0] + 'ANG_DATA/SAA_SZA.tif',     g, xRes=10, yRes=10, resample=0).data

    vas   = reproject_data(fi.split('IMG_DATA')[0] + 'ANG_DATA/VAA_VZA_B05.tif', g, xRes=10, yRes=10, resample=0).data

    rgb   = reproject_data(fi.split('IMG_DATA')[0] + 'IMG_DATA/BOA_RGB.tif',     g, xRes=10, yRes=10, resample=0).g.ReadAsArray()

    cloud = reproject_data(fi.split('IMG_DATA')[0] + 'cloud.tif',                g, xRes=10, yRes=10, resample=0).data  

    temp = np.ones_like(data) * 100 

    temp[np.where(cloud >=0)] = cloud[cloud >=0]

    cloud = temp > 30  

    dats = np.array(dats) / 10000.  

    mask = np.all(dats>0, axis=0) & (~cloud)

    #dats = dats[:,mask]

 

    sza = np.cos(np.deg2rad(sas[1].mean() / 100.))   

    vza = np.cos(np.deg2rad(vas[1].mean() / 100.))

    saa = sas[0].mean() / 100.

    vaa = vas[0].mean() / 100.     

    raa = np.cos(np.deg2rad(vaa - saa))

    
    '''
    AAT  = np.dot(comps.T, comps)   
    ATy  = np.dot(comps.T, dats[:, mask])    

    x    = np.linalg.solve(AAT, ATy)

    simu = np.dot(comps, x)         

    omega = np.dot(comps[:,ns:], x[ns:])

    X = np.hstack([x[ns:].T, np.array([np.ones_like(x[0])*sza]).T])

    '''

    data = data * np.nan

    

    return dats, None, mask, sza, vza, raa

def read_emulator(emulator_file="/home/ucfafyi/DATA/Prosail/prosail_2NN.npz"):
    f = np.load(str(emulator_file))
    emulator = Two_NN(Hidden_Layers=f.f.Hidden_Layers,
                      Output_Layers=f.f.Output_Layers)
    return emulator


def get_n_obs(fnames, field_name="lai"):
    akk = []
    for i, fname in enumerate(fnames):
        f = np.load(fname)
        x = f[field_name]
        x[x<0] = np.nan
        akk.append(x)
    return np.nansum(np.isfinite(akk), axis=0)
        
    



def get_field(fname, field_name="lai"):
    f = np.load(fname)
    datex = fname.name.split(".")[0]
    date = dt.datetime.strptime(datex, 
                        "%Y%m%d")
    x = f[field_name]
    x[x<0] = np.nan
    return date, np.nanpercentile(x, 75)


def get_pxl_obs(fname, xloc, yloc, field_name="lai"):
    datex = fname.name.split(".")[0]
    date = dt.datetime.strptime(datex, 
                        "%Y%m%d")
    f = np.load(fname)
    x = f[field_name]
    x[x<0] = np.nan
    return date, x[xloc, yloc]

f =  np.load('/home/ucfafyi/DATA/From_TOA_TO_LAI/S2_TOA_TO_LAI/data/Third_comps.npz')
comps = f.f.comps

inverse_param_model  = tf.keras.models.load_model('/home/ucfafyi/DATA/Prosail/Prosail_5_paras.h5')

#uk_fields = ["Appletree","Barn Field","Big Lawn", "Kells","Retters","Rushbottom",
#             "Shrubbery", "West Farm"]
#uk_fields = ["Barn Field", "West Farm", "Big Lawn"]
lmu_fields = ["508", "301","542","515","319"]
tif_names = np.loadtxt("LMU_files.txt", dtype=str)
for field in lmu_fields:
    M = []
    
    #pix_x_loc, pix_y_loc = slice(9,14),slice(25,30) #11, 28
    pix_x_loc, pix_y_loc = slice(2,20),slice(20,40) #11, 28
    s2_refl = []
    s2_mask = []
    sza = [] ; vza=[] ; raa = []
    lai = [] ; cab = [] ; cbrown = []
    obs_dates=[]

    for fi in tif_names:
        this_date = dt.datetime.strptime(fi.split("/")[-1].split("_")[1].split("T")[0], "%Y%m%d")                           
        if (this_date >= dt.datetime(2017, 1, 1, 0, 0)) and (this_date <= dt.datetime(2017, 12,31, 0, 0)):

            fieldx = f"NUMMER={field:s}"
            data, X, mask, xsza, xvza, xraa = do_real_data(fi, fieldx, "/home/ucfajlg/Data/python/KaFKA_Validation/LMU/carto/MNI_2017.shp")
            ##this_date = dt.datetime.strptime(fi.split("/")[-1].split("_")[1].split("T")[0], "%Y%m%d")                           
            ##if this_date in [dt.datetime(2018, 3, 8, 0, 0),
                            ##dt.datetime(2018, 3, 23, 0, 0),
                            ##dt.datetime(2018, 4, 4, 0, 0),
                            ##dt.datetime(2018, 5, 4, 0, 0)]:
                ##continue
            if data[0, pix_x_loc, pix_y_loc].mean() > 0.18:
                continue
            if mask.sum() > 0:
                X = np.vstack([data, np.array([np.ones_like(data[0])*xsza, np.ones_like(data[0])*xvza, np.ones_like(data[0])*xraa])])
                s2_refl.append(data[:, pix_x_loc, pix_y_loc])
                s2_mask.append(mask[pix_x_loc, pix_y_loc])
                sza.append(xsza)
                vza.append(xvza)
                raa.append(xraa)
                temp = np.ones_like(mask).astype(np.float32)
                
                xn, xcab, xcbrown, xlai, xala = inverse_param_model.predict(X[:, mask].T).T
                temp[mask] = xlai.ravel()
                lai.append(-2*np.log(temp)[pix_x_loc, pix_y_loc])
                obs_dates.append(dt.datetime.strptime(fi.split("/")[-1].split("_")[1].split("T")[0], "%Y%m%d"))
                
                temp[mask] = xcab.ravel()
                cab.append(-100*np.log(temp)[pix_x_loc, pix_y_loc])
                
                temp[mask] = xcbrown.ravel()
                cbrown.append(temp[pix_x_loc, pix_y_loc])
            

        



    # fix sza for high sza values
    sza = np.cos(np.deg2rad(np.minimum(np.rad2deg(np.arccos(sza)), 60)))
    time_grid=np.arange(0, 366, 5)
    obs_doy = [int(t.strftime("%j")) for t in obs_dates]
    L = np.array(lai)
    C = np.array(cab)
    B = np.array(cbrown)
    L[L<=0] = np.nan
    C[C<=0] = np.nan
    B[B<=0] = np.nan
    lai = np.nanmean(L, axis=(1,2))
    cab = np.nanmean(C, axis=(1,2))
    cbrown = np.nanmean(B, axis=(1,2))
    idx = np.argmin(np.abs(time_grid-np.array(obs_doy)[:, None]),axis=1) 
    L = np.zeros_like(time_grid, dtype=np.float32)
    C = np.zeros_like(time_grid, dtype=np.float32)
    B = np.zeros_like(time_grid, dtype=np.float32)
    Weight = np.zeros_like(time_grid, np.float32)
    for ii, tstep in enumerate(np.unique(idx)):
        obs_list = np.where(idx == tstep)[0]
        L[tstep] = lai[obs_list].mean()
        C[tstep] = cab[obs_list].mean()
        B[tstep] = cbrown[obs_list].mean()
        Weight[tstep] = len(obs_list)
    


    slai, _, _, slai_sigma = smoothn(L, W=Weight, isrobust=False, s=2, TolZ=1e-6)  
    slai_sigma[slai_sigma<=0.1]=0.1

    scab, _, _, scab_sigma = smoothn(C, W=Weight, isrobust=False, s=2, TolZ=1e-6)  
    scab_sigma[scab_sigma<=0.1]=0.1
    scbrown, _, _, scbrown_sigma = smoothn(B, W=Weight, isrobust=False, s=0.5, TolZ=1e-6)  
    scbrown_sigma[scbrown_sigma<=0.01]=0.01


    mean_pixel = np.array([2.1, np.exp(-30. / 100.),
                        np.exp(-7.0 / 100.), 0.1,
                        np.exp(-50 * 0.018), np.exp(-50. * 0.005),
                        np.exp(-4. / 2.), 70. / 90., 0.1, 0])

    sigma_pixel = np.array([0.01, .2, 0.05, 0.2,
                                0.005, 0.01,
                                .8, 0.05, 0.5, 0.01])

    mu_prior = np.tile(mean_pixel, len(time_grid))
    diag_sigma_prior = np.tile(sigma_pixel, len(time_grid))
    mu_prior[1::10] = np.exp(-scab/100.) 
    mu_prior[6::10] = np.exp(-slai/2.)
    #mu_prior[3::10] = scbrown
    #mu_prior[1::10] = np.exp(-slai/2.) 
    diag_sigma_prior[6::10] = 0.002/slai_sigma
    diag_sigma_prior[1::10] = 0.05/scab_sigma
    #diag_sigma_prior[3::10] = 0.0001
    n_params = 10
    lower_bounds = [ 0.8  ,  0.35,  0.819,  0.   ,  0.023,  np.exp(-50*0.007),  0.018,  0.011, -0.6  , -0.6  ]
    upper_bounds = [2.5  , 0.951, 0.951, 1.0  , 0.807, np.exp(-50*0.004), 0.999   , .95   ,
                                    0.95  , 0.95  ]
    bounds = list(zip(lower_bounds*len(time_grid), upper_bounds*len(time_grid)))

    A=sp.dia_matrix((time_grid.shape[0]*n_params, time_grid.shape[0]*n_params)) 
    A.setdiag(1/diag_sigma_prior**2)        

    emu=read_emulator()

    S2_obs = s2_obs(np.array(obs_doy),
                    np.ones_like(obs_doy, dtype=np.bool)*True,
                    np.array(s2_refl).mean(axis=(2,3)),
                    np.array(sza), np.array(vza), np.array(raa),
                    np.array(s2_refl).mean(axis=(2,3))*0.005)

    cost_wrapper = CostWrapper(time_grid, S2_obs, [0, 500000, 0, 100000, 0, 0, 1000, 0, 100000,100000],
                            emu, mu_prior, A)

    retval = minimize(cost_wrapper.calc_cost, mu_prior,  jac=True, method="L-BFGS-B", 
                    bounds=bounds,
                    options={"iprint":1, "maxcor":400, "maxiter":500,})  


    cost_wrapper2 = CostWrapper(time_grid, S2_obs, [0, 50000, 0, 10000, 0, 0, 500, 0, 1000,1000],   emu, mu_prior, A)

    H = hessian(retval.x, cost_wrapper2.calc_cost, epsilon=0.01)

    cov_post = np.linalg.inv(H)

    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(12,8))
    colors = ["#FC8D62", "#66C2A5", "#8DA0CB", "#E78AC3", "#A6D854"]
    axs = axs.flatten()
    veg_time = time_grid[-2*np.log(retval.x[6::10])>0.8]
    for i in range(4):
        axs[i].axvspan(veg_time.min(), veg_time.max(), color="0.8")



    axs[0].vlines(time_grid, -2*np.log(retval.x[6::10] -3*np.sqrt(cov_post.diagonal())[6::10]),  -2*np.log(retval.x[6::10] +3*np.sqrt(cov_post.diagonal())[6::10]))
    axs[0].plot(time_grid, -2*np.log(retval.x[6::10]), 'o-', color=colors[0], lw=3, mfc="none")
    axs[0].set_ylabel("LAI $[m^{2}m^{-2}]$")

    axs[1].vlines(time_grid, -100*np.log(retval.x[1::10] -3*np.sqrt(cov_post.diagonal())[1::10]),  -100*np.log(retval.x[1::10] +3*np.sqrt(cov_post.diagonal())[1::10]))
    axs[1].plot(time_grid, -100*np.log(retval.x[1::10]), 'o-', color=colors[1], lw=3, mfc="none")
    axs[1].set_ylabel("$C_{ab}\; [\mu\cdot g cm^{-2}]$")


    axs[2].vlines(time_grid, retval.x[3::10] -3*np.sqrt(cov_post.diagonal())[3::10],  retval.x[3::10] +3*np.sqrt(cov_post.diagonal())[3::10])
    axs[2].plot(time_grid, retval.x[3::10], 'o-', color=colors[2], lw=3, mfc="none")
    axs[2].set_ylabel("$C_{brown}$")
    axs[3].vlines(time_grid, retval.x[8::10] -3*np.sqrt(cov_post.diagonal())[8::10],  retval.x[8::10] +3*np.sqrt(cov_post.diagonal())[8::10])
    axs[3].plot(time_grid, retval.x[8::10], 'o-', color=colors[3], lw=3, mfc="none")
    axs[3].set_ylabel(r"$\rho_{s,b}$")
    axs[2].set_xlabel("Day of Year [d]")
    axs[3].set_xlabel("Day of Year [d]")

    fig.tight_layout()
    np.savetxt(f"LMUField_{field:s}.txt", retval.x)
#class CostWrapper(object):
#    def __init__(self, time_grid, current_data,
#                 gamma, emu,
#                 mu_prior, c_prior_inv):




#####files = sorted([f for f in (Path.cwd()/"suffolk_v1/").glob("2018????.npz")])

#####pix_x_loc = 24
#####pix_y_loc = 18
#####lai_meas = dict([get_pxl_obs(f, pix_x_loc, pix_y_loc) for f in files])
#####cab_meas = dict([get_pxl_obs(f, pix_x_loc, pix_y_loc, field_name="chol") for f in files])
#####cbrown_meas = dict([get_pxl_obs(f, pix_x_loc, pix_y_loc, field_name="cbrown") for f in files])

#####t = np.array([t for t in lai_meas.keys()])
#####lai = np.array([lai_meas[x] for x in t])
#####cab = np.array([cab_meas[x] for x in t])
#####cbrown = np.array([cbrown_meas[x] for x in t])
#####slai = smoothn(lai, isrobust=False, s=2, TolZ=1e-6)[0]
#####scab = smoothn(cab, isrobust=False, s=2, TolZ=1e-6)[0]
#####scbrown = smoothn(cbrown, isrobust=False,
#####s=2,TolZ=1e-6)[0]
#####plt.figure()
#####plt.plot_date(t, slai, '-')
#####plt.figure()
#####plt.plot_date(t, scab, '-')
#####plt.figure()
#####plt.plot_date(t, scbrown, '-')

#####doys = [int(x.strftime("%j")) for x in t]
#####temporal_grid = np.arange(1, 365, 5)
#####emulator = read_emulator()


