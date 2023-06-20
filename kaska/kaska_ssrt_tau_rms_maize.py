
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import MonthLocator
# import matplotlib.ticker
import numpy as np
# from sense.canopy import OneLayer
# from sense.soil import Soil
# from sense import model
import scipy.stats
from scipy.optimize import minimize
import pdb
from z_helper import *
# from z_optimization import *
import datetime
from matplotlib import gridspec
import datetime
from matplotlib.lines import Line2D
import copy
import os
import osr
import gdal
import datetime
import numpy as np
from netCDF4 import Dataset
from scipy.ndimage import label
from utils import reproject_data
from skimage.filters import sobel
from collections import namedtuple
from scipy.optimize import minimize
from scipy.interpolate import interp1d
# from watercloudmodel import cost_function
# from watercloudmodel import cost_function2
from scipy.ndimage.filters import gaussian_filter1d
import pdb
from z_helper import *
# from watercloudmodel import ssrt_jac_
from watercloudmodel_vwc_rms import cost_function_vwc, ssrt_jac_vwc, ssrt_vwc



def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def do_one_pixel_field(data_field, vv, vh, vwc, theta, time, sm, sm_std, b, b_std, omega, rms, rms_std, orbits, unc):

    ps   = []
    vwcs    = []
    bs = []
    sms    = []
    srms = []
    times = []

    uorbits = np.unique(orbits)
    # uorbits = np.array([95])
    for orbit in uorbits:
    # for jj in range(len(vv)):
        # pdb.set_trace()
        orbit_mask = orbits == orbit
        # orbit_mask = (orbits == 44) | (orbits == 168)
        # orbit_mask = (orbits == 95) | (orbits == 117)
        # orbit_mask = (orbits == 44) | (orbits == 95) | (orbits == 117) | (orbits == 168)
        # orbit_mask = (orbits == 168)
        # orbit_mask = (orbits == 44) | (orbits == 95) | (orbits == 117)
        ovv, ovh, ovwc, otheta, otime = vv[orbit_mask], vh[orbit_mask], vwc[orbit_mask], theta[orbit_mask], time[orbit_mask]
        osm, osm_std, osb, osb_std = sm[orbit_mask], sm_std[orbit_mask], b[orbit_mask], b_std[orbit_mask]


        ovwc_std = np.ones_like(osb)*0.05

        # alpha     = _calc_eps(osm)
        # alpha = osm
        # alpha_std = np.ones_like(alpha)*10
        # alpha_std = osm_std
        # pdb.set_trace()

        # prior_mean = np.concatenate([[0,   ]*2, osm,     ovwc,     osb])
        # prior_unc  = np.concatenate([[10., ]*2, osm_std, ovwc_std, osb_std])

        # xvv = np.array([rms, omega])


        # x0 = np.concatenate([xvv, osm, ovwc, osb])

        # bounds = (
        #     [[0.013, 0.013]] # s
        #   + [[0.0107, 0.0107]] # omega
        #   + [[0.01,   0.7]] * osb.shape[0] # mv
        #   + [[0,     7.5]] * osb.shape[0] # vwc
        #   + [[0.01,       0.6]] * osb.shape[0] # b
        #   )

        # pdb.set_trace()
        prior_mean = np.concatenate([[0,   ], [rms], osm,     ovwc,     osb])
        prior_unc  = np.concatenate([[10., ], [rms_std], osm_std, ovwc_std, osb_std])

        xvv = np.array([omega])


        x0 = np.concatenate([xvv, np.array([rms]), osm, ovwc, osb])

        bounds = (
            [[0.027, 0.027]] # omega
          + [[0.005, 0.03]]  # s=rms
          + [[0.01,   0.7]] * osb.shape[0] # mv
          + [[0,     7.5]] * osb.shape[0] # vwc
          + [[0.01,       0.6]] * osb.shape[0] # b
          )



        data = osb

        gamma = [10, 10]
        # pdb.set_trace()
        retval = minimize(cost_function_vwc,
                            x0,
                            args=(ovh, ovv, otheta, gamma, prior_mean, prior_unc, unc, data),
                            jac=True,
                            bounds = bounds,
                            options={"disp": True})

        posterious_rms   = retval.x[1]
        posterious_sm   = retval.x[2 : 2+len(osb)]
        posterious_vwc    = retval.x[2+len(osb)   : 2+2*len(osb)]
        posterious_b = retval.x[2+2*len(osb)   : 2+3*len(osb)]

        srms.append(posterious_rms)
        sms.append(posterious_sm)
        vwcs.append(posterious_vwc)
        bs.append(posterious_b)
        times.append(otime)
        ps.append(retval.x[:1])

    order = np.argsort(np.hstack(times))
    times  = np.hstack(times )[order]
    vwcs   = np.hstack(vwcs  )[order]
    bs    = np.hstack(bs   )[order]
    sms    = np.hstack(sms   )[order].real
    # pdb.set_trace()
    # srms = np.hstack(srms)[order]
    return times, vwcs, bs, sms, np.array(srms), np.array(ps), orbit_mask


def _simple_ew():
    """
    eq. 4.69
    simplistic approach with T=23°C, bulk density = 1.7 g/cm3
    """
    f0 = 18.64   # relaxation frequency [GHz]
    f = 5.405
    hlp = f/f0
    e1 = 4.9 + (74.1)/(1.+hlp**2.)
    # e2 =(74.1*hlp)/(1.+hlp**2.) + 6.46 * self.sigma/self.f
    # return e1 + 1.j * e2
    return e1

def _calc_eps(mv):
    """
    calculate dielectric permittivity
    Eq. 4.66 (Ulaby et al., 2014)
    """
    clay = 0.0738
    sand = 0.2408
    bulk = 1.45
    alpha = 0.65
    beta1 = 1.27-0.519*sand - 0.152*clay
    beta2 = 2.06 - 0.928*sand -0.255*clay
    sigma = -1.645 + 1.939*bulk - 2.256*sand + 1.594*clay


    e1 = (1.+0.66*bulk+mv**beta1*_simple_ew()**alpha - mv)**(1./alpha)
    # e2 = np.imag(self.ew)*self.mv**self.beta2
    # return e1 + 1.j*e2
    return e1

# def quad_approx_solver(alphas):
#     x = np.arange(0.01, 0.5, 0.01)
#     p = np.polyfit(x, _calc_eps(x), 2)
#     # 2nd order polynomial
#     #solve
#     solutions = [np.roots([p[0], p[1], p[2]-aa]) for aa in alphas]
#     return solutions

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx




### Data preparation df_agro!!!! ###
#-----------------------------------------------------------------
# storage information
path = '/media/tweiss/Work/z_final_mni_data_2017'
file_name = 'in_situ_s1_buffer_100' # theta needs to be changed to for norm multi
extension = '.csv'

path_agro = '/media/nas_data/2017_MNI_campaign/field_data/meteodata/agrarmeteorological_station'
path_agro = '/media/tweiss/Work/Paper/in_progress/RT_model_comparison/images'
file_name_agro = 'Daily_Freising'
extension_agro = '.csv'

field = '508_high'
pol = 'vv'

df, df_agro, field_data, field_data_orbit, theta_field, sm_field, height_field, lai_field, vwc_field, pol_field, vv_field, vh_field, relativeorbit, vwcpro_field = read_data(path, file_name, extension, field, path_agro, file_name_agro, extension_agro, pol)

aggregation = ['','_buffer_30','_buffer_50','_buffer_100','_Field_buffer_30']
pre_processing = ['multi', 'norm_multi']
aggregation = ['_buffer_50','_Field_buffer_30']
pre_processing = ['multi']
aggregation = ['_buffer_100']
# aggregation = ['_Field_buffer_30']
surface_list = ['Oh92', 'Oh04', 'Dubois95', 'WaterCloud', 'I2EM']
canopy_list = ['turbid_isotropic', 'water_cloud']

surface_list = ['Oh92', 'I2EM']
canopy_list = ['turbid_isotropic']

surface_list = ['Oh04']
# surface_list = ['Oh92']
# canopy_list = ['water_cloud']
field = ['508_high']
field = ['508_low']
field = ['508_med']
field = ['301_high']
field = ['301_low']
field = ['301_med']
field = ['542_high']
field = ['542_low']
field = ['542_med']

field = ['508_high','508_low','508_med','301_high','301_low','301_med','542_high','542_low','542_med']

### option for time invariant or variant calibration of parameter
#-------------------------------
opt_mod = ['time_variant']
#---------------------------

years = ['_2017','_2018']
years = ['_2017']
numbers = [1,3,5,7,9]
numbers = [1]

for zzz in numbers:

    for p in pre_processing:

        for pp in aggregation:

            for year in years:
                if year == '_2017':
                    field_list = ['508_high','508_low','508_med','301_high','301_low','301_med','542_high','542_low','542_med']
                    field_list = ['319_high','319_low','319_med','515_high','515_low','515_med']
                elif year == '_2018':
                    field_list = ['525_high','525_low','525_med','317_high','317_low','317_med']
                else:
                    pass

                # versions = ['everything','','44_117','95_168','44_168','117_95','44_95','117_168','44_117_95','44_117_168','44_95_168','117_95_168']
                # ver = ['','','44','95','44','117','44','117','44','44','44','117']
                # ver2 = ['','','117','168','168','95','95','168','117','117','95','95']
                # ver3 = ['','','','','','','','','95','168','168','168']

                # versions = ['','everything']
                # ver = ['','']
                # ver2 = ['','']
                # ver3 = ['','']


                versions = ['everything']
                ver = ['']
                ver2 = ['']
                ver3 = ['']

                # versions = ['44_168']
                # ver = ['44']
                # ver2 = ['168']
                # ver3 = ['']

                for i, ii in enumerate(versions):

                    if ii == 'everything':
                        orbit_list = [None]
                        orbit1=None
                        orbit2=None
                        orbit3=None
                        orbit4=None
                        plot_output_path = '/media/tweiss/Work/paper3/'+year[1:]+'/z_dense_s1_time_series_n'+str(zzz)+p+pp+'_all'+'/'
                        csv_output_path = plot_output_path+'csv/None_'
                    elif ii == '':
                        orbit_list = [44,117,95,168]
                        orbit2=None
                        orbit3=None
                        orbit4=None
                        plot_output_path = '/media/tweiss/Work/paper3/'+year[1:]+'/z_dense_s1_time_series_n'+str(zzz)+p+pp+'/'
                        csv_output_path = plot_output_path+'csv/'
                    else:
                        plot_output_path = '/media/tweiss/Work/paper3/'+year[1:]+'/z_dense_s1_time_series_n'+str(zzz)+p+pp+'_'+ii+'/'
                        csv_output_path = plot_output_path+'csv/'+ver[i]+'_'
                        orbit_list = [int(ver[i])]
                        orbit2 = int(ver2[i])
                        if ver3[i] == '':
                            orbit3 = None
                        else:
                            orbit3 = int(ver3[i])


                    data = pd.read_csv(csv_output_path+'all'+pp+'.csv',header=[0,1,2,3,4,5],index_col=0)






                    # fig, ax = plt.subplots(figsize=(17, 13))
                    # gs = gridspec.GridSpec(5, 1, height_ratios=[14, 3, 3, 3, 3])
                    # ax = plt.subplot(gs[0])

                    # plt.ylabel('Backscatter [dB]', fontsize=18)
                    # plt.xlabel('Date', fontsize=18)
                    # plt.tick_params(labelsize=17)

                    # ax.set_ylim([-21.5,-8.5])


                    # colormaps = ['Greens', 'Purples', 'Blues', 'Oranges', 'Reds', 'Greys', 'pink', 'bone', 'Blues', 'Blues', 'Blues']
                    # r = 0

                    # colormap = plt.get_cmap(colormaps[r])
                    # colors = [colormap(rr) for rr in np.linspace(0.35, 1., 3)]

                    for kkk in opt_mod:
                        for kkkk in field_list:
                            for k in surface_list:
                                for kk in canopy_list:

                                    if k == 'Oh92':
                                        hm = 'Oh92'
                                        colors = 'b'
                                    elif k == 'Oh04':
                                        hm = 'Oh04'
                                        colors = 'r'
                                    elif k == 'Dubois95':
                                        hm='Dubois95'
                                        colors = 'y'
                                    elif k == 'WaterCloud':
                                        hm = 'WCM'
                                        colors = 'm'
                                    elif k == 'I2EM':
                                        hm = 'IEM_B'
                                        colors = 'g'

                                    data_field = data.filter(like=k).filter(like=kk).filter(like=kkk).filter(like=kkkk)
                                    data_field.index = pd.to_datetime(data_field.index)
                                    date = data_field.index

                                    ### b mean

                                    data_b = data.filter(like=k).filter(like=kk).filter(like=kkk).filter(like='coef')
                                    mean_b = data_b.mean(axis=1)





                                    vv = data_field.filter(like='S1_vv').values.flatten()
                                    vv = 10*np.log10(vv)
                                    vh = data_field.filter(like='S1_vh').values.flatten()
                                    vh = 10*np.log10(vh)

                                    time = date
                                    time2 = np.array(time)
                                    for jj in range(len(time)):
                                        time2[jj] = time[jj].replace(microsecond=0).replace(second=0).replace(minute=0)
                                    time2 = pd.to_datetime(time2)
                                    time3 = time2.normalize()
                                    theta = data_field.filter(like='theta').values.flatten()
                                    theta = np.rad2deg(theta)




                                    # lai = data_field.filter(like='LAI_insitu').values.flatten()
                                    # lai = lai




                                    # s2_data = pd.read_csv('/media/tweiss/Daten/data_AGU/S2_'+kkkk+pp+'.csv',header=[0],index_col=0)
                                    # s2_data.index = pd.to_datetime(s2_data.index).floor('Min').floor('H')
                                    # s2_lai = s2_data.loc[time2]['lai'].values.flatten()
                                    # s2_cab = s2_data.loc[time2]['cab'].values.flatten()
                                    # s2_cbrown = s2_data.loc[time2]['cbrown'].values.flatten()

                                    sm_insitu = data_field.filter(like='SM_insitu').values.flatten()





                                    api_data = pd.read_csv('/media/tweiss/Daten/data_AGU/api'+year+'_radolan.csv',header=[0],index_col=0)
                                    api_data.index = pd.to_datetime(api_data.index)
                                    print(kkkk+year)
                                    api_field = api_data.filter(like=kkkk)
                                    api_sm = api_field.loc[time2].values.flatten()

                                    vwc_data = pd.read_csv('/media/tweiss/Work/z_final_mni_data_2017/vwc_sentinel_2'+pp+year+'_paper3_gao.csv', header=[0,1],index_col=0)


                                    vwc_data.index = pd.to_datetime(vwc_data.index)
                                    vwc_data = vwc_data.resample('D').mean().interpolate()
                                    vwc_data = vwc_data.loc[time2.normalize()]

                                    vwc_field = vwc_data.filter(like=kkkk)
                                    vwc_sentinel_2 = vwc_field.filter(like='m_pos_ag_vwc')

                                    sm_insitu = data_field.filter(like='SM_insitu').values.flatten()
                                    # pdb.set_trace()


                                    # sm = smooth(sm,2)
                                    # sm[:] = 0.25
                                    # sm = data_field.filter(like='SM_insitu').values.flatten()
                                    sm = api_sm
                                    # sm[:] = 0.2
                                    sm_std = data_field.filter(like='SM_insitu').values.flatten()
                                    # ooo = np.abs(sm[1:]-sm[:-1])*20
                                    # sm_std[0] = ooo[-1]
                                    # sm_std[1:] = ooo
                                    sm_std[:] = 0.3
                                    # sm_std[:] = 0.5

                                    b = data_field.filter(like='coef').values.flatten()
                                    b_old = data_field.filter(like='coef').values.flatten()
                                    b_std = data_field.filter(like='SM_insitu').values.flatten()
                                    # b = data_field.filter(like='coef').rolling(4).mean().values.flatten()
                                    # b[0] = b_old[0]
                                    # b[1] = b_old[1]
                                    # b[2] = b_old[2]
                                    # b[3] = b_old[3]
                                    # b = mean_b.values.flatten()


                                    # # b=b-0.1
                                    # b_std[:] = 0.4
                                    # b[:] = 0.4
                                    # # height = data_field.filter(like='height').values.flatten()
                                    orbits = data_field.filter(like='relativeorbit').values.flatten()
                                    orbits95 = orbits==95
                                    orbits168 = orbits==168
                                    orbits44 = orbits==44
                                    orbits117 = orbits==117
                                    orbits44_168 = (orbits == 44) | (orbits == 168)
                                    # b[:] = 0.4
                                    b[orbits95] = 0.4
                                    b[orbits117] = 0.4
                                    b[orbits44] = 0.6
                                    b[orbits168] = 0.6



                                    # pdb.set_trace()

                                    omega = 0.027
                                    unc = 0.7

                                    vwc_insitu = data_field.filter(like='VWC').values.flatten()
                                    vwc = vwc_sentinel_2.values.flatten()
                                    vwc[vwc < 0.01] = 0.02
                                    # vwc = vwc_insitu
                                    # pdb.set_trace()

                                    orbits95[0:np.argmax(vwc)] = False
                                    orbits117[0:np.argmax(vwc)] = False
                                    orbits44[0:np.argmax(vwc)] = False
                                    orbits168[0:np.argmax(vwc)] = False

                                    b[orbits95] = 0.1
                                    b[orbits117] = 0.1
                                    b[orbits44] = 0.2
                                    b[orbits168] = 0.2


                                    rms = 0.0115
                                    rms = 0.02

                                    # rms = data_field.filter(like='SM_insitu').values.flatten()
                                    # rms[:] = 0.027
                                    # rms_std = data_field.filter(like='SM_insitu').values.flatten()
                                    rms_std = 0.01

                                    # unc_array = np.arange(0,2,0.1)
                                    # coef_array = np.arange(0,2,0.1)
                                    # sm_array = np.arange(0,2,0.1)

                                    # hm = {}
                                    # for r in unc_array:
                                    #     for rr in coef_array:
                                    #         for rrr in sm_array:
                                    #             unc = r
                                    #             coef_std[:] = rr
                                    #             sm_std[:] = rrr
                                    #             times, lais, coefs, sms, orbit_mask = do_one_pixel_field(data_field, vv, vh, lai, theta, time, sm, sm_std, coef, coef_std, height, orbits,unc=unc)
                                    #             rmse_vv = rmse_prediction(sm_insitu,sms)
                                    #             bias_vv = bias_prediction(sm_insitu,sms)
                                    #             ubrmse_vv = ubrmse_prediction(rmse_vv,bias_vv)
                                    #             hm[(r,rr,rrr)] = ubrmse_vv

                                    # pdb.set_trace()
                                    # min(hm, key=hm.get)
                                    # hm[min(hm, key=hm.get)]
                                    uncs = np.arange(0.1,2,0.3)
                                    uncs = np.array([0.9,1.9,2.5])
                                    b_stds = np.arange(0.1,1,0.4)
                                    b_stds = np.array([0.5])
                                    sm_stds = np.arange(0.1,0.5,0.1)
                                    sm_stds = np.array([0.1,0.2,0.3,0.7])
                                    # uncs = np.array([1.9])
                                    vv = 10 ** (vv/10)
                                    # pdb.set_trace()
                                    for unc in uncs:
                                        for t in b_stds:
                                            for tt in sm_stds:

                                                b_std[:] = t
                                                sm_std[:] = tt

                                                # pdb.set_trace()

                                                pdb.set_trace()
                                                times, vwcs, bs, sms, srms, ps, orbit_mask = do_one_pixel_field(data_field, vv, vh, vwc, theta, time, sm, sm_std, b, b_std, omega, rms, rms_std, orbits,unc=unc)


                                                uorbits = np.unique(orbits)
                                                rms_2 = np.ones_like(orbits)*rms
                                                srms_2 = np.ones_like(orbits)
                                                for hh, hhh in enumerate(uorbits):
                                                    if len(srms) == 1:
                                                        srms_2[:] = srms[0]
                                                    else:
                                                        srms_2[orbits == hhh] = srms[hh]

                                                # pdb.set_trace()
                                                fig, ax = plt.subplots(figsize=(17, 13))
                                                gs = gridspec.GridSpec(5, 1, height_ratios=[5, 5, 5, 5, 5])
                                                ax = plt.subplot(gs[0])


                                                # sm_insitu = sm_insitu[orbit_mask]
                                                # api_sm = api_sm[orbit_mask]
                                                # vwc = vwc[orbit_mask]
                                                # b = b[orbit_mask]
                                                # b_old = b_old[orbit_mask]
                                                # vv = vv[orbit_mask]
                                                # theta = theta[orbit_mask]
                                                # sm = sm[orbit_mask]




                                                ax.plot(times,sm_insitu, label='insitu')




                                                rmse_vv = rmse_prediction(sm_insitu,api_sm)
                                                bias_vv = bias_prediction(sm_insitu,api_sm)
                                                ubrmse_vv = ubrmse_prediction(rmse_vv,bias_vv)
                                                ax.plot(times,api_sm, label='prior RMSE:'+str(rmse_vv)[0:6]+' ubRMSE:'+str(ubrmse_vv)[0:6])

                                                rmse_vv = rmse_prediction(sm_insitu,sms)
                                                bias_vv = bias_prediction(sm_insitu,sms)
                                                ubrmse_vv = ubrmse_prediction(rmse_vv,bias_vv)
                                                ax.plot(times,sms, label='model RMSE:'+str(rmse_vv)[0:6]+' ubRMSE:'+str(ubrmse_vv)[0:6])
                                                plt.ylabel('Soil moisture', fontsize=18)
                                                plt.ylim(0.05,0.45)
                                                plt.grid(linestyle='dotted')
                                                plt.legend()
                                                plt.subplots_adjust(hspace=.0)
                                                plt.setp(ax.get_xticklabels(), visible=False)

                                                ax1 = plt.subplot(gs[1])

                                                ax1.plot(times,vwc_insitu,label='insitu')
                                                ax1.plot(times,vwc,label='input vwc')
                                                ax1.plot(times,vwcs,label='model vwc')
                                                plt.ylabel('VWC', fontsize=18)
                                                plt.ylim(0,6)
                                                plt.grid(linestyle='dotted')
                                                plt.legend()
                                                plt.subplots_adjust(hspace=.0)
                                                plt.setp(ax1.get_xticklabels(), visible=False)

                                                ax2 = plt.subplot(gs[2])

                                                ax2.plot(times,b,label='input b')
                                                ax2.plot(times,bs,label='model b')
                                                ax2.plot(times,b_old,label='b calibrated')

                                                plt.ylabel('b', fontsize=18)
                                                plt.ylim(0,1)
                                                plt.grid(linestyle='dotted')
                                                plt.legend()
                                                plt.subplots_adjust(hspace=.0)
                                                plt.setp(ax1.get_xticklabels(), visible=False)

                                                ax3 = plt.subplot(gs[4])

                                                sigma_vv, vv_g, vv_c = ssrt_vwc(sms, vwc, rms, omega, bs, theta)

                                                ax3.plot(times,10*np.log10(vv),label='S1')
                                                ax3.plot(times,10*np.log10(sigma_vv),label='model')
                                                ax3.plot(times,10*np.log10(vv_g),label='ground')
                                                ax3.plot(times,10*np.log10(vv_c),label='canopy')
                                                plt.ylabel('VV [dB]', fontsize=18)
                                                plt.ylim(-30,-5)
                                                plt.grid(linestyle='dotted')
                                                plt.legend()
                                                # plt.setp(ax1.get_xticklabels(), visible=False)

                                                ax4 = plt.subplot(gs[3])

                                                ax4.plot(times,rms_2,label='input rms')
                                                ax4.plot(times,srms_2,label='model rms')
                                                # ax4.plot(times,b_old,label='b calibrated')

                                                plt.ylabel('rms'+str(rms), fontsize=18)
                                                plt.ylim(0.005,0.03)
                                                plt.grid(linestyle='dotted')
                                                plt.legend()
                                                plt.subplots_adjust(hspace=.0)
                                                plt.setp(ax4.get_xticklabels(), visible=False)

                                                ax3.set_xlabel('Date', fontsize=18)
                                                # plt.tick_params(labelsize=17)
                                                plt.subplots_adjust(hspace=.0)
                                                rmse_vv = rmse_prediction(sm_insitu,sm)
                                                bias_vv = bias_prediction(sm_insitu,sm)
                                                ubrmse_vv = ubrmse_prediction(rmse_vv,bias_vv)



                                                ax.set_title('omega: 0.027, omega model:'+str(ps[0,0]))


                                                # plt.show()
                                                plt.savefig('/media/tweiss/Work/paper3/plot/maize/'+year[1:]+'/'+kkkk+ii+'unc:'+str(unc)+'_sm_std'+str(tt)[:3]+'.png', bbox_inches = 'tight')
                                                # pdb.set_trace()


                                                # noprior/bmean_s/oh04_unc10_apism025_'+kkkk, bbox_inches = 'tight')
                                                plt.close()
pdb.set_trace()


