
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
from watercloudmodel import cost_function2
from scipy.ndimage.filters import gaussian_filter1d
import pdb
from z_helper import *
from watercloudmodel import ssrt_jac_




def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def do_one_pixel_field(data_field, vv, vh, lai, theta, time, sm, sm_std, coef, coef_std, height, orbits, unc):

    lais   = []
    coefs    = []
    sms    = []
    times  = []

    uorbits = np.unique(orbits)
    uorbits = np.array([95])
    for orbit in uorbits:
    # for jj in range(len(vv)):
        # pdb.set_trace()
        # orbit_mask = orbits == orbit
        # orbit_mask = (orbits == 44) | (orbits == 168)
        orbit_mask = (orbits == 44) | (orbits == 95) | (orbits == 117) | (orbits == 168)
        # orbit_mask = (orbits == 168)
        # orbit_mask = (orbits == 44) | (orbits == 95) | (orbits == 117)
        ovv, ovh, olai, otheta, otime = vv[orbit_mask], vh[orbit_mask], lai[orbit_mask], theta[orbit_mask], time[orbit_mask]
        osm, osm_std, oscoef, oscoef_std  = sm[orbit_mask], sm_std[orbit_mask], coef[orbit_mask], coef_std[orbit_mask]

        oheight = height[orbit_mask]

        # ovv, ovh, olai, otheta, otime = np.array([vv[jj]]), np.array([vh[jj]]), np.array([lai[jj]]), np.array([theta[jj]]), np.array([time[jj]])
        # osm, osm_std, oscoef, oscoef_std  = np.array([sm[jj]]), np.array([sm_std[jj]]), np.array([coef[jj]]), np.array([coef_std[jj]])

        # oheight = np.array([height[jj]])



        # pdb.set_trace()
        olai_std = np.ones_like(olai)*0.05

        alpha     = _calc_eps(osm)
        alpha = osm
        alpha_std = np.ones_like(alpha)*10
        alpha_std = osm_std
        # pdb.set_trace()
        prior_mean = np.concatenate([alpha,oscoef])
        prior_unc  = np.concatenate([alpha_std,oscoef_std])

        x0 = np.concatenate([alpha,oscoef])
        data =  np.concatenate([oheight,olai])
        bounds = (
          # [[2.5,   30]] * olai.shape[0]
          [[0.01,   0.5]] * olai.shape[0]
          + [[0.0000001,     3]] * olai.shape[0]
          )

        gamma = [500, 500]

        retval = minimize(cost_function2,
                            x0,
                            args=(ovh, ovv, otheta, gamma, prior_mean, prior_unc, data, unc),
                            jac=True,
                            bounds = bounds,
                            options={"disp": True},)

        # posterious_lai   = retval.x[2*len(olai) : 3*len(olai)]
        posterious_coef    = retval.x[len(olai)   : +2*len(olai)]
        posterious_mv = retval.x[             : +len(olai)]
        # lais.append(posterious_lai)
        coefs.append(posterious_coef)
        # x = np.arange(0.01, 0.5, 0.001)
        # xx = _calc_eps(x)
        # sols=[]
        # for i in posterious_mv:
        #     p, pp = find_nearest(xx,i)
        #     sols.append(x[pp])
        # sols = np.array(sols)

        sms.append(posterious_mv)
        # sms.append(sols)
        times.append(otime)

    order = np.argsort(np.hstack(times))
    times  = np.hstack(times )[order]
    # lais   = np.hstack(lais  )[order]
    lais=0
    coefs    = np.hstack(coefs   )[order]
    # coefs=0
    sms    = np.hstack(sms   )[order].real
    # orbit_mask = (orbits == 44) | (orbits == 95) | (orbits == 117) | (orbits == 168)
    return times, lais, coefs, sms, orbit_mask



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
file_name = 'in_situ_s1_buffer_50' # theta needs to be changed to for norm multi
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
# field = ['508_low']
# field = ['508_med']
# field = ['301_high']
field = ['301_low']
# field = ['301_med']
# field = ['542_high']
# field = ['542_low']
# field = ['542_med']

### option for time invariant or variant calibration of parameter
#-------------------------------
opt_mod = ['time_variant']
#---------------------------


for p in pre_processing:

    for pp in aggregation:

        # versions = ['everything','','44_117','95_168','44_168','117_95','44_95','117_168','44_117_95','44_117_168','44_95_168','117_95_168']
        # ver = ['','','44','95','44','117','44','117','44','44','44','117']
        # ver2 = ['','','117','168','168','95','95','168','117','117','95','95']
        # ver3 = ['','','','','','','','','95','168','168','168']

        versions = ['','everything']
        ver = ['','']
        ver2 = ['','']
        ver3 = ['','']


        # versions = ['everything']
        # ver = ['']
        # ver2 = ['']
        # ver3 = ['']

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
                plot_output_path = '/media/tweiss/Work/paper2/z_dense_s1_time_series_n7'+p+pp+'_all'+'/'
                csv_output_path = plot_output_path+'csv/None_'
            elif ii == '':
                orbit_list = [44,117,95,168]
                orbit2=None
                orbit3=None
                orbit4=None
                plot_output_path = '/media/tweiss/Work/paper2/z_dense_s1_time_series_n7'+p+pp+'/'
                csv_output_path = plot_output_path+'csv/'
            else:
                plot_output_path = '/media/tweiss/Work/paper2/z_dense_s1_time_series_n7'+p+pp+'_'+ii+'/'
                csv_output_path = plot_output_path+'csv/'+ver[i]+'_'+ver[i]+'_'
                orbit_list = [int(ver[i])]
                orbit2 = int(ver2[i])
                if ver3[i] == '':
                    orbit3 = None
                else:
                    orbit3 = int(ver3[i])


            data = pd.read_csv(csv_output_path+'all_50.csv',header=[0,1,2,3,4,5],index_col=0)






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
                for kkkk in field:
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



                            vv = data_field.filter(like='S1_vv').values.flatten()
                            vv = 10*np.log10(vv)
                            vh = data_field.filter(like='S1_vh').values.flatten()
                            vh = 10*np.log10(vh)
                            lai = data_field.filter(like='LAI_insitu').values.flatten()
                            lai = lai
                            theta = data_field.filter(like='theta').values.flatten()
                            theta = np.rad2deg(theta)
                            time = date
                            time2 = np.array(time)
                            for jj in range(len(time)):
                                time2[jj] = time[jj].replace(microsecond=0).replace(second=0).replace(minute=0)
                            time2 = pd.to_datetime(time2)


                            s2_data = pd.read_csv('/media/tweiss/Daten/data_AGU/S2_'+kkkk+pp+'.csv',header=[0],index_col=0)
                            s2_data.index = pd.to_datetime(s2_data.index).floor('Min').floor('H')
                            s2_lai = s2_data.loc[time2]['lai'].values.flatten()
                            s2_cab = s2_data.loc[time2]['cab'].values.flatten()
                            s2_cbrown = s2_data.loc[time2]['cbrown'].values.flatten()
                            lai = s2_lai
                            sm_insitu = data_field.filter(like='SM_insitu').values.flatten()
                            api_data = pd.read_csv('/media/tweiss/Daten/data_AGU/api_sm.csv',header=[0],index_col=0)
                            api_data.index = pd.to_datetime(api_data.index)
                            api_sm = api_data.loc[time2].values.flatten()
                            sm = data_field.filter(like='SM_insitu').values.flatten()
                            # sm = smooth(sm,2)
                            sm[:] = 0.25
                            # sm = data_field.filter(like='SM_insitu').values.flatten()
                            sm = api_sm
                            sm_std = data_field.filter(like='SM_insitu').values.flatten()
                            ooo = np.abs(sm[1:]-sm[:-1])*20
                            sm_std[0] = ooo[-1]
                            sm_std[1:] = ooo
                            sm_std[:] = 0.21
                            coef = data_field.filter(like='coef').values.flatten()
                            coef_std = data_field.filter(like='SM_insitu').values.flatten()
                            coef_std[:] = 0.01
                            height = data_field.filter(like='height').values.flatten()
                            orbits = data_field.filter(like='relativeorbit').values.flatten()
                            unc = 2.1

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

                            vv = 10 ** (vv/10)

                            # pdb.set_trace()
                            times, lais, coefs, sms, orbit_mask = do_one_pixel_field(data_field, vv, vh, lai, theta, time, sm, sm_std, coef, coef_std, height, orbits,unc=unc)
                            # pdb.set_trace()
                            plt.rcParams["figure.figsize"] = (10,7)
                            # plt.plot(time,sm_insitu, label='insitu')
                            plt.plot(times,sm_insitu[orbit_mask], label='insitu')
                            rmse_vv = rmse_prediction(sm_insitu,sm)
                            bias_vv = bias_prediction(sm_insitu,sm)
                            ubrmse_vv = ubrmse_prediction(rmse_vv,bias_vv)
                            # plt.plot(time,sm, label='prior RMSE:'+str(rmse_vv)[0:6]+' ubRMSE:'+str(ubrmse_vv)[0:6])
                            plt.plot(times,sm[orbit_mask], label='prior RMSE:'+str(rmse_vv)[0:6]+' ubRMSE:'+str(ubrmse_vv)[0:6])
                            # rmse_vv = rmse_prediction(sm_insitu,sms)
                            # bias_vv = bias_prediction(sm_insitu,sms)
                            rmse_vv = rmse_prediction(sm_insitu[orbit_mask],sms)
                            bias_vv = bias_prediction(sm_insitu[orbit_mask],sms)
                            ubrmse_vv = ubrmse_prediction(rmse_vv,bias_vv)
                            plt.plot(times,sms, label='model RMSE:'+str(rmse_vv)[0:6]+' ubRMSE:'+str(ubrmse_vv)[0:6])
                            # plt.plot(times,coefs, label='coef')
                            # pdb.set_trace()
                            # #orbit_mask
                            # plt.plot(time[orbit_mask],sm_insitu[orbit_mask])
                            # rmse_vv = rmse_prediction(sm_insitu[orbit_mask],sm[orbit_mask])
                            # bias_vv = bias_prediction(sm_insitu[orbit_mask],sm[orbit_mask])
                            # ubrmse_vv = ubrmse_prediction(rmse_vv,bias_vv)
                            # plt.plot(time[orbit_mask],sm[orbit_mask], label='prior RMSE:'+str(rmse_vv)[0:6]+' ubRMSE:'+str(ubrmse_vv)[0:6])
                            # rmse_vv = rmse_prediction(sm_insitu[orbit_mask],sms)
                            # bias_vv = bias_prediction(sm_insitu[orbit_mask],sms)
                            # ubrmse_vv = ubrmse_prediction(rmse_vv,bias_vv)
                            # plt.plot(times,sms, label='model RMSE:'+str(rmse_vv)[0:6]+' ubRMSE:'+str(ubrmse_vv)[0:6])


                            plt.legend()
                            plt.grid()
                            plt.ylabel('Soil Moisture')
                            plt.xlabel('Time')
                            # plt.savefig('/media/tweiss/Daten/data_AGU/test_kaska/oh04_unc10_lai_flat', bbox_inches = 'tight')
                            plt.show()
                            pdb.set_trace()
                            plt.plot(time[orbit_mask],coef)
                            plt.plot(times,coefs)
                            # plt.show()
                            pdb.set_trace()
                            pdb.set_trace()




































                            if kkk == 'time invariant':
                                if kk == 'turbid_isotropic':
                                    ax.plot(date, 10*np.log10(data_field.filter(like='vv_model')), color=colors, marker='s', linestyle='dashed', label = hm+ ' + ' +  'SSRT')
                                else:
                                    ax.plot(date, 10*np.log10(data_field.filter(like='vv_model')), color=colors, marker='s', label = hm+ ' + ' +  'WCM')
                            else:
                                if kk == 'turbid_isotropic':
                                    ax.plot(date, 10*np.log10(data_field.filter(like='vv_model')), color=colors, marker='s', linestyle='dashed', label = hm+ ' + ' +  'SSRT')
                                else:
                                    ax.plot(date, 10*np.log10(data_field.filter(like='vv_model')), color=colors, marker='s', label = hm+ ' + ' +  'WCM')
                a = 0
                b = 0
                c = 0
                d = 0

                relativeorbit = data_field.filter(like='relativeorbit')
                for j in range(len(relativeorbit)):
                    relativeorbit.index[j]
                    x = relativeorbit.index[j] - datetime.timedelta(days=0.4)
                    xx = relativeorbit.index[j] + datetime.timedelta(days=0.4)
                    if relativeorbit.values.flatten()[j] == 95:
                        if a == 0:
                            ax.axvspan(x,xx, color='red', alpha=0.2, label = 'Incidence angle 43°, Descending track')
                            a += 1
                        else:
                            ax.axvspan(x,xx, color='red', alpha=0.2)
                    elif relativeorbit.values.flatten()[j] == 117:
                        if b == 0:
                            ax.axvspan(x,xx, color='blue', alpha=0.2, label = 'Incidence angle 45°, Ascending track')
                            b += 1
                        else:
                            ax.axvspan(x,xx, color='blue', alpha=0.2)
                    elif relativeorbit.values.flatten()[j] == 168:
                        if c == 0:
                            ax.axvspan(x,xx, color='orange', alpha=0.2, label = 'Incidence angle 35°, Descending track')
                            c += 1
                        else:
                            ax.axvspan(x,xx, color='orange', alpha=0.2)
                    elif relativeorbit.values.flatten()[j] == 44:
                        if d == 0:
                            ax.axvspan(x,xx, color='green', alpha=0.2, label = 'Incidence angle 36°, Ascending track')
                            d += 1
                        else:
                            ax.axvspan(x,xx, color='green', alpha=0.2)
                    else:
                        pass

                ax.plot(date,10*np.log10(data_field.filter(like='S1_vv')), '-', color='black', label='Sentinel-1', linewidth=3, marker='s')

                ax.set_xlim([datetime.date(2017, 3, 22), datetime.date(2017, 7, 18)])
                plt.legend(prop={'size': 14}, loc=3)

                plt.grid(linestyle='dotted')

                plt.setp(ax.get_xticklabels(), visible=False)

                ax0 = plt.subplot(gs[1])
                plt.tick_params(labelsize=17)
                for kkkk in field:
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


                            ground = data.filter(like=k).filter(like=kk).filter(like=kkk).filter(like=kkkk).filter(like='part_g')
                            ground = ground[ground.columns[0]]

                            lai = data.filter(like=k).filter(like=kk).filter(like=kkk).filter(like=kkkk).filter(like='LAI_insitu').values
                            theta = data.filter(like=k).filter(like=kk).filter(like=kkk).filter(like=kkkk).filter(like='theta').values

                            if kk == 'turbid_isotropic':
                                coef = data.filter(like=k).filter(like=kk).filter(like=kkk).filter(like=kkkk).filter(like='coef').values
                                d = data.filter(like=k).filter(like=kk).filter(like=kkk).filter(like=kkkk).filter(like='height').values
                                T = np.exp(-coef*np.sqrt(lai)*d/np.cos(theta))
                                T=T**2
                                ax0.plot(date,T.flatten(), color=colors, marker='s', linestyle='dashed')
                            else:
                                B_vv = data.filter(like=k).filter(like=kk).filter(like=kkk).filter(like=kkkk).filter(like='B_vv').values
                                T = np.exp(-2*B_vv*lai/np.cos(theta))
                                ax0.plot(date,T.flatten(), color=colors, marker='s')

                a = 0
                b = 0
                c = 0
                d = 0

                relativeorbit = data_field.filter(like='relativeorbit')
                for j in range(len(relativeorbit)):
                    relativeorbit.index[j]
                    x = relativeorbit.index[j] - datetime.timedelta(days=0.4)
                    xx = relativeorbit.index[j] + datetime.timedelta(days=0.4)
                    if relativeorbit.values.flatten()[j] == 95:
                        if a == 0:
                            ax0.axvspan(x,xx, color='red', alpha=0.2, label = 'descending 43°')
                            a += 1
                        else:
                            ax0.axvspan(x,xx, color='red', alpha=0.2)
                    elif relativeorbit.values.flatten()[j] == 117:
                        if b == 0:
                            ax0.axvspan(x,xx, color='blue', alpha=0.2, label = 'ascending 43°')
                            b += 1
                        else:
                            ax0.axvspan(x,xx, color='blue', alpha=0.2)
                    elif relativeorbit.values.flatten()[j] == 168:
                        if c == 0:
                            ax0.axvspan(x,xx, color='orange', alpha=0.2, label = 'descending 35°')
                            c += 1
                        else:
                            ax0.axvspan(x,xx, color='orange', alpha=0.2)
                    elif relativeorbit.values.flatten()[j] == 44:
                        if d == 0:
                            ax0.axvspan(x,xx, color='green', alpha=0.2, label = 'ascending 36°')
                            d += 1
                        else:
                            ax0.axvspan(x,xx, color='green', alpha=0.2)
                    else:
                        pass
                plt.ylabel('Transmissivity\nT', fontsize=18)
                ax0.set_xlim([datetime.date(2017, 3, 22), datetime.date(2017, 7, 18)])
                ax0.set_ylim(-0.2,1.1)
                plt.grid(linestyle='dotted')
                plt.setp(ax0.get_xticklabels(), visible=False)


                ax1 = plt.subplot(gs[2], sharex = ax)
                plt.tick_params(labelsize=17)
                # remove vertical gap between subplots
                plt.subplots_adjust(hspace=.0)
                plt.grid(linestyle='dotted')
                plt.setp(ax1.get_xticklabels(), visible=False)

                lai_field = data_field.filter(like='LAI_insitu')
                height_field = data_field.filter(like='height')


                ax1.plot(date,lai_field,color='green',linewidth=2,label='LAI')
                ax2 = ax1.twinx()
                plt.tick_params(labelsize=17)
                ax2.plot(date,height_field,color='black', linewidth=2, label='Height')
                ax1.set_ylabel('LAI', fontsize=16)
                ax2.set_ylabel('Height\n[m]', fontsize=16)


                # add std for LAI and height for field 508 (data from field measurements)
                lai_old = copy.deepcopy(lai_field)
                height_old = copy.deepcopy(height_field)
                if field == '508_high':
                    lai_field[lai_field.index>'2017-03-28'] = 0.2218
                    lai_field[lai_field.index>'2017-04-05'] = 0.1367
                    lai_field[lai_field.index>'2017-04-10'] = 0.4054
                    lai_field[lai_field.index>'2017-04-21'] = 0.3247
                    lai_field[lai_field.index>'2017-05-02'] = 0.5546
                    lai_field[lai_field.index>'2017-05-10'] = 0.5852
                    lai_field[lai_field.index>'2017-05-16'] = 0.3058
                    lai_field[lai_field.index>'2017-05-26'] = 0.5373
                    lai_field[lai_field.index>'2017-05-29'] = 0.332
                    lai_field[lai_field.index>'2017-06-02'] = 0.2856
                    lai_field[lai_field.index>'2017-06-13'] = 0.4717
                    lai_field[lai_field.index>'2017-06-26'] = 0.2982
                    lai_field[lai_field.index>'2017-07-06'] = 0.253

                    height_field[height_field.index>'2017-03-28'] =  0.005774
                    height_field[height_field.index>'2017-04-05'] = 0.015275
                    height_field[height_field.index>'2017-04-10'] = 0.026458
                    height_field[height_field.index>'2017-04-21'] = 0.049329
                    height_field[height_field.index>'2017-05-02'] = 0.01
                    height_field[height_field.index>'2017-05-10'] = 0.01
                    height_field[height_field.index>'2017-05-26'] = 0.028868
                    height_field[height_field.index>'2017-05-29'] = 0.028868
                    height_field[height_field.index>'2017-06-02'] = 0.028868
                    height_field[height_field.index>'2017-06-13'] = 0.020817
                    height_field[height_field.index>'2017-06-26'] = 0.025166
                    height_field[height_field.index>'2017-07-06'] = 0.015275

                    ax1.fill_between(lai_field.index,lai_old.values.flatten()-lai_field.values.flatten(), lai_old.values.flatten()+lai_field.values.flatten(), color='green', alpha=0.2, label='Standard Deviation')
                    ax2.fill_between(height_field.index,height_old.values.flatten()-height_field.values.flatten(), height_old.values.flatten()+height_field.values.flatten(), color='black', alpha=0.2, label='Standard Deviation')

                ax2.legend(bbox_to_anchor=(.965, 0.45), prop={'size': 14})
                ax1.legend(loc=2, prop={'size': 14})

                # ax1.set_xticks([])
                ax1.set_ylim(0,6.7)
                ax2.set_ylim(0,1)
                start, end = ax1.get_ylim()
                ax1.yaxis.set_ticks(np.arange(start, end, 2))

                # soil moisture and rainfall
                ax3 = plt.subplot(gs[3], sharex = ax)
                plt.tick_params(labelsize=17)
                # remove vertical gap between subplots
                plt.subplots_adjust(hspace=.0)
                plt.grid(linestyle='dotted')
                ax3.plot(date,data_field.filter(like='SM_insitu'),color='blue', linewidth=2, label='Soil Moisture')
                ax3.set_ylabel('Soil Moisture\n$[cm^3/cm^3]$', fontsize=16)
                ax5 = ax3.twinx()
                date_agro = pd.to_datetime(df_agro['date'], format='%d.%m.%Y')
                agro_sum = df_agro['SUM_NN050'][87:192]
                ax5.bar(agro_sum.index, agro_sum, width=0.8, label='Precipitation')
                ax3.legend(loc=2, prop={'size': 14})
                ax5.legend(loc=1, prop={'size': 14})
                ax5.set_ylabel('Precipita-\ntion [mm]', fontsize=16)
                ax5.set_ylim(0,39)
                ax3.set_ylim(0.17,0.38)
                plt.setp(ax3.get_xticklabels(), visible=False)
                plt.tick_params(labelsize=17)

                ax4 = plt.subplot(gs[4], sharex = ax)
                plt.tick_params(labelsize=17)
                # remove vertical gap between subplots
                plt.subplots_adjust(hspace=.0)
                plt.grid(linestyle='dotted')
                bbch = pd.read_csv('/media/tweiss/Work/z_final_mni_data_2017/bbch_2017.csv',header=[0,1])
                bbch = bbch.set_index(pd.to_datetime(bbch['None']['None'], format='%Y-%m-%d'))
                bbch.index = pd.to_datetime(bbch.index)

                lai_field['bbch'] = 0

                bbch_new = bbch.filter(like=kkkk[0:3])
                for t, tt in enumerate(bbch.index):
                    if t == 0:
                        start_date = '2017-03-29'
                    else:
                        start_date = bbch.index[t]
                    try:
                        end_date = bbch.index[t+1]
                    except IndexError:
                        start_date = bbch.index[t]
                        end_date = '2017-07-30'
                    mask = (lai_field.index > start_date) & (lai_field.index <= end_date)

                    bbbb = lai_field['bbch'].where(~mask, other=2)
                    if bbch.index[t] < datetime.datetime.strptime('2017-03-29', '%Y-%m-%d'):
                        pass
                    else:
                        if bbch_new.values[t] < 30 and bbch_new.values[t] >= 20:
                            # lai_field['bbch'] = lai_field['bbch'].where(~mask, other=2)
                            n2 = max(lai_field['bbch'][mask].index)
                        elif bbch_new.values[t] < 40 and bbch_new.values[t] >= 30:
                            # lai_field['bbch'] = lai_field['bbch'].where(~mask, other=3)
                            n3 = max(lai_field['bbch'][mask].index)
                        elif bbch_new.values[t] < 50 and bbch_new.values[t] >= 40:
                            # lai_field['bbch'] = lai_field['bbch'].where(~mask, other=4)
                            n4 = max(lai_field['bbch'][mask].index)
                        elif bbch_new.values[t] < 60 and bbch_new.values[t] >= 50:
                            # lai_field['bbch'] = lai_field['bbch'].where(~mask, other=5)
                            n5 = max(lai_field['bbch'][mask].index)
                        elif bbch_new.values[t] < 70 and bbch_new.values[t] >= 60:
                            # lai_field['bbch'] = lai_field['bbch'].where(~mask, other=6)
                            n6 = max(lai_field['bbch'][mask].index)
                        elif bbch_new.values[t] < 80 and bbch_new.values[t] >= 70:
                            # lai_field['bbch'] = lai_field['bbch'].where(~mask, other=7)
                            n7 = max(lai_field['bbch'][mask].index)
                        elif bbch_new.values[t] < 90 and bbch_new.values[t] >= 80:
                            # lai_field['bbch'] = lai_field['bbch'].where(~mask, other=8)
                            n8 = max(lai_field['bbch'][mask].index)
                        elif bbch_new.values[t] < 100 and bbch_new.values[t] >= 90:
                            # lai_field['bbch'] = lai_field['bbch'].where(~mask, other=9)
                            n9 = max(lai_field['bbch'][mask].index)
                # bbch_ = lai_field['bbch'].value_counts().sort_index().values
                bbch_ = [n2-datetime.datetime.strptime('2017-03-22', '%Y-%m-%d'),n3-n2,n4-n3,n5-n4,n6-n5,n7-n6,n8-n7,n9-n8]

                #Plot BBCH
                hm = lai_field.filter(like='bbch')
                label = ['','BBCH','']
                width = 0.3
                legend_items = ['Tillering','Stem elongation','Booting','Heading','Flowering','Fruit development','Ripening', 'Senescence']

                a_508 = 0

                aa_508 = mdates.date2num(lai_field['bbch'].index[0])

                for xxxx, kkkkk in enumerate(bbch_):
                    a_508 = a_508 + bbch_[xxxx].total_seconds() /60/60/24
                    ax4.barh(label,[0,a_508,0],width, label=legend_items[xxxx], left=[0,aa_508,0])

                    aa_508 = mdates.date2num(lai_field['bbch'].index[0]) + a_508


                xmin, xmax = ax4.get_xlim()

                ax4.barh(label,[0,200,0],width, left=[0,xmax-1,0], color='white')
                ax4.set_ylim(0,1.7)
                plt.legend(bbox_to_anchor=(.935, 0.4),ncol=8)

                plt.text(0.98, 0.05, "(a)", transform=ax.transAxes, fontsize=20, horizontalalignment='center', verticalalignment='center')
                plt.text(0.98, 0.2, "(b)", transform=ax0.transAxes, fontsize=20, horizontalalignment='center', verticalalignment='center')
                plt.text(0.98, 0.2, "(c)", transform=ax2.transAxes, fontsize=20, horizontalalignment='center', verticalalignment='center')
                plt.text(0.98, 0.2, "(d)", transform=ax3.transAxes, fontsize=20, horizontalalignment='center', verticalalignment='center')
                plt.text(0.98, 0.2, "(e)", transform=ax4.transAxes, fontsize=20, horizontalalignment='center', verticalalignment='center')


                plt.savefig(plot_output_path+pol+'_all_'+kkk+kkkk, bbox_inches = 'tight')

                plt.close()

