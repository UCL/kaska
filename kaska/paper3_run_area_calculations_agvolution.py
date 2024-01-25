#!/usr/bin/env python

import os
# import osr
from osgeo import gdal
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
from watercloudmodel_vwc_rms import cost_function_vwc, ssrt_jac_vwc, ssrt_vwc
from scipy.ndimage.filters import gaussian_filter1d
import pdb
from z_helper import *
import matplotlib.pyplot as plt
from netCDF4 import date2num
from netCDF4 import num2date
import glob
from paper3_plotting import *
from paper3_plot_scatter import *
from paper3_plot_esu import *
from pandas.plotting import register_matplotlib_converters
from agv_plot_input_output import *
from sm_helper_data_preparation import get_sm_input, inference_preprocessing
from sm_run_SenSARP import run_SenSARP

def get_api_folder(api_folder):

    filelist = glob.glob(api_folder+'**/**/*.nc', recursive = True)
    filelist.sort()

    for file in filelist:
        data = Dataset(file)
        api_name = file.replace('.nc', '.tif')

        if not os.path.exists(api_name):
            gg = gdal.Open('NETCDF:"%s":ssm'%file)
            geo = gg.GetGeoTransform()
            save_to_tif(api_name, data['ssm'], geo)
def read_api_ssm(api_folder, state_mask):
    api_ssm = namedtuple('api_data', 'time api')
    filelist = glob.glob(api_folder+'**/*.tif', recursive = True)
    filelist.sort()

    time = []
    api = []

    for file in filelist:
        g = gdal.Open(file)
        ssm_array = reproject_data(file, output_format="MEM", target_img=state_mask)
        ssm_array = ssm_array.ReadAsArray()
        porosity = 0.45
        ssm_array_absolute = ssm_array*porosity
        # ssm_array_absolute = ssm_array/100 * porosity

        time.append(datetime.datetime.strptime(file.split('/')[-1][13:21], '%Y%m%d'))
        api.append(ssm_array_absolute)

    return api_ssm(time, api)


def do_one_pixel_field(vv, vh, vwc, vwc_std, theta, time, sm, sm_std, b, b_std, omega, rms, rms_std, orbits, unc):

    ps   = []
    vwcs    = []
    bs = []
    sms    = []
    srms = []
    times = []

    uorbits = np.unique(orbits)
    uorbits = np.array([95])
    for orbit in uorbits:
    # for jj in range(len(vv)):
        # orbit_mask = orbits == orbit
        # orbit_mask = (orbits == 44) | (orbits == 168)
        # orbit_mask = (orbits == 95) | (orbits == 117)
        orbit_mask = (orbits == 44) | (orbits == 95) | (orbits == 117) | (orbits == 168)
        orbit_mask = (orbits == 146) | (orbits == 168) | (orbits == 44) | (orbits == 95)
        # orbit_mask = (orbits == 168)
        # orbit_mask = (orbits == 44) | (orbits == 95) | (orbits == 117)
        ovv, ovh, ovwc, ovwc_std, otheta, otime = vv[orbit_mask], vh[orbit_mask], vwc[orbit_mask], vwc_std[orbit_mask], theta[orbit_mask], time[orbit_mask]
        osm, osm_std, osb, osb_std = sm[orbit_mask], sm_std[orbit_mask], b[orbit_mask], b_std[orbit_mask]


        prior_mean = np.concatenate([[0,   ], [rms], osm,     ovwc,     osb])
        prior_unc  = np.concatenate([[10., ], [rms_std], osm_std, ovwc_std, osb_std])


        x0 = np.concatenate([np.array([omega]), np.array([rms]), osm, ovwc, osb])

        # bounds for b related to expected curve
        xxx = []
        for jjj, jj in enumerate(osb):
            if jj <= 0.2:
                xxx.append([0.01,osb[jjj]+0.2])
            else:
                xxx.append([osb[jjj]-0.2,osb[jjj]+0.2])

        bounds = (
            [[0.027, 0.027]] # omega
          + [[0.005, 0.03]]  # s=rms
          + [[0.01,   0.7]] * osb.shape[0] # mv
          + [[0,     7.5]] * osb.shape[0] # vwc
          + xxx #[[0.01,       0.6]] * osb.shape[0] # b
          )


        data = osb

        gamma = [10, 10]

        retval = minimize(cost_function_vwc,
                            x0,
                            args=(ovh, ovv, otheta, gamma, prior_mean, prior_unc, unc, data),
                            jac=True,
                            bounds = bounds,
                            options={"disp": False})

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

    return times, vwcs, bs, sms, np.array(srms), np.array(ps), orbit_mask




def do_inversion(sar_inference_data, state_mask, year=None, version=None, passes=None):

    orbits = sar_inference_data.relorbit[sar_inference_data.time_mask]
    uorbits = np.unique(orbits)

    out_shape   = sar_inference_data.vwc[sar_inference_data.time_mask].shape
    vwc_outputs = np.zeros(out_shape )
    sm_outputs  = np.zeros(out_shape )
    b_outputs  = np.zeros(out_shape )
    rms_outputs  = np.zeros(out_shape )

    g = gdal.Open(state_mask)
    state_mask = g.ReadAsArray()
    # state_mask = state_mask > 0
    state_mask = state_mask >= 1

    vv_all = sar_inference_data.vv.ReadAsArray()[sar_inference_data.time_mask]
    vh_all    = sar_inference_data.vh.ReadAsArray()[sar_inference_data.time_mask]
    theta_all = sar_inference_data.ang.ReadAsArray()[sar_inference_data.time_mask]
    time_all = np.array(sar_inference_data.time)[sar_inference_data.time_mask]

    vwc_all = sar_inference_data.vwc[sar_inference_data.time_mask]
    ndwi_all = sar_inference_data.ndwi[sar_inference_data.time_mask]

    ### vwc needs to be changed!!!! NDWI1!!!
    vwc_std = vwc_all[:,0,0]
    vwc_std[:] = 0.1
    sm_all = sar_inference_data.api[sar_inference_data.time_mask]
    sm_all = sm_all / 100.
    sm_std = np.copy(sm_all[:,0,0])
    sm_std[:] = 0.2

    b = np.copy(sm_all[:,0,0])
    b[:] = 0
    b_std = np.copy(sm_all[:,0,0])
    b_std[:] = 0.5 # not used anyway
    rms = sm_all[:,0,0]
    rms = 0.2
    rms_std = 0.1 # not used anyway

    unc = 1.9
    omega = 0.027

    sm_retrieved = sm_all * np.nan

    if not os.path.exists('/media/AUF/userdata/agvolution/inversion/'+passes):
        os.makedirs('/media/AUF/userdata/agvolution/inversion/'+passes)


    np.save('/media/AUF/userdata/agvolution/inversion/'+passes+'/'+year+version+'_input_vv.npy', vv_all)
    np.save('/media/AUF/userdata/agvolution/inversion/'+passes+'/'+year+version+'_input_vwc.npy', vwc_all)
    np.save('/media/AUF/userdata/agvolution/inversion/'+passes+'/'+year+version+'_input_sm_api.npy', sm_all)
    np.save('/media/AUF/userdata/agvolution/inversion/'+passes+'/'+year+version+'_input_ndwi.npy', ndwi_all)
    np.save('/media/AUF/userdata/agvolution/inversion/'+passes+'/'+year+version+'_input_theta.npy', theta_all)

    for z in range(len(state_mask)):
        print(z)
        for zz in range(len(state_mask[0])):
            if state_mask[z,zz] == False:
                pass
            else:
                vv = vv_all[:,z,zz]
                vh = vh_all[:,z,zz]
                theta = theta_all[:,z,zz]
                vwc = vwc_all[:,z,zz]
                vwc[vwc < 0.01] = 0.02

                orbits95 = orbits==95
                orbits168 = orbits==168
                orbits44 = orbits==44
                orbits117 = orbits==117
                # orbits44_168 = (orbits == 44) | (orbits == 168)
                # b[:] = 0.4
                b[orbits95] = 0.4
                b[orbits117] = 0.4
                b[orbits44] = 0.6
                b[orbits168] = 0.6

                if passes == 'b_0515':
                    orbits95[0:np.where(time_all == min(time_all, key=lambda x: abs(x-datetime.datetime(2017,5,15))))[0][0]] = False
                    orbits117[0:np.where(time_all == min(time_all, key=lambda x: abs(x-datetime.datetime(2017,5,15))))[0][0]] = False
                    orbits44[0:np.where(time_all == min(time_all, key=lambda x: abs(x-datetime.datetime(2017,5,15))))[0][0]] = False
                    orbits168[0:np.where(time_all == min(time_all, key=lambda x: abs(x-datetime.datetime(2017,5,15))))[0][0]] = False
                else:
                    orbits95[0:np.argmax(vwc)] = False
                    orbits117[0:np.argmax(vwc)] = False
                    orbits44[0:np.argmax(vwc)] = False
                    orbits168[0:np.argmax(vwc)] = False

                if passes == 'b_veg':
                    norm = (vwc- np.nanmin(vwc)) / (np.nanmax(vwc) - np.nanmin(vwc))
                    norm_ref = np.abs(norm-1)
                    b = b * norm_ref
                elif passes == 'normal':
                    b[orbits95] = 0.1
                    b[orbits117] = 0.1
                    b[orbits44] = 0.2
                    b[orbits168] = 0.2
                else:
                    norm = (vwc- np.nanmin(vwc)) / (np.nanmax(vwc) - np.nanmin(vwc))
                    norm_ref = np.abs(norm-1)
                    b = b * norm_ref
                pdb.set_trace()

                if passes == 'unc_15':
                    unc = 1.5
                elif passes == 'unc_13':
                    unc = 1.3
                elif passes == 'unc_10':
                    unc = 1.0
                elif passes == 'unc_05':
                    unc = 0.5
                elif passes == 'unc_21':
                    unc = 2.1
                elif passes == 'unc_19':
                    unc = 1.9
                elif passes == 'unc_25':
                    unc = 2.5
                else:
                    unc = 1.0


                if passes == 'sm_std_001':
                    sm_std[:] = 0.01
                elif passes == 'sm_std_003':
                    sm_std[:] = 0.03
                elif passes == 'sm_std_005':
                    sm_std[:] = 0.05
                elif passes == 'sm_std_007':
                    sm_std[:] = 0.07
                elif passes == 'sm_std_010':
                    sm_std[:] = 0.1
                elif passes == 'sm_std_013':
                    sm_std[:] = 0.13
                elif passes == 'sm_std_015':
                    sm_std[:] = 0.15
                elif passes == 'sm_std_017':
                    sm_std[:] = 0.17
                elif passes == 'sm_std_020':
                    sm_std[:] = 0.20
                else:
                    sm_std[:] = 0.2

                if passes == 'sm_std_001_1':
                    sm_std[:] = 0.01
                    unc = 0.4
                elif passes == 'sm_std_003_1':
                    sm_std[:] = 0.03
                    unc = 0.4
                elif passes == 'sm_std_005_1':
                    sm_std[:] = 0.05
                    unc = 0.4
                elif passes == 'sm_std_007_1':
                    sm_std[:] = 0.07
                    unc = 0.4
                elif passes == 'sm_std_010_1':
                    sm_std[:] = 0.1
                    unc = 0.4
                elif passes == 'sm_std_013_1':
                    sm_std[:] = 0.13
                    unc = 0.4
                elif passes == 'sm_std_015_1':
                    sm_std[:] = 0.15
                    unc = 0.4
                elif passes == 'sm_std_017_1':
                    sm_std[:] = 0.17
                    unc = 0.4
                elif passes == 'sm_std_020_1':
                    sm_std[:] = 0.20
                    unc = 0.4
                else:
                   pass

                sm = sm_all[:,z,zz]
                print(unc)
                print(sm_std[0])

                times, svwc, sb, sms, srms, ps, orbit_mask = do_one_pixel_field(vv, vh, vwc, vwc_std, theta, time_all, sm, sm_std, b, b_std, omega, rms, rms_std, orbits,unc=unc)

                vwc_outputs[:,z,zz] = svwc
                sm_outputs[:,z,zz]  = sms
                b_outputs[:,z,zz]  = sb
                rms_outputs[:,z,zz]  = srms

    np.save('/media/AUF/userdata/agvolution/inversion/'+passes+'/'+year+version+'_sm'+'.npy', sm_outputs)
    np.save('/media/AUF/userdata/agvolution/inversion/'+passes+'/'+year+version+'_vwc'+'.npy', vwc_outputs)
    np.save('/media/AUF/userdata/agvolution/inversion/'+passes+'/'+year+version+'_b'+'.npy', b_outputs)
    np.save('/media/AUF/userdata/agvolution/inversion/'+passes+'/'+year+version+'_rms'+'.npy', rms_outputs)
    np.save('/media/AUF/userdata/agvolution/inversion/'+passes+'/'+year+version+'_times.npy',times)

    return 'done'

def save_output(fname, Array, GeoT, projction, time):
    if os.path.exists(fname):
        os.remove(fname)
    ds = gdal.GetDriverByName('GTiff').Create(fname, Array.shape[2], Array.shape[1], Array.shape[0], gdal.GDT_Float32)
    ds.SetGeoTransform(GeoT)
    ds.SetProjection(projction)
    for i, image in enumerate(Array):
        ds.GetRasterBand(i+1).SetMetadata({'date': time[i]})
        ds.GetRasterBand(i+1).WriteArray( image )
    ds.FlushCache()
    return fname

def save_ps_output(fname, Array, GeoT, projction, orbit):
    if os.path.exists(fname):
        os.remove(fname)
    ds = gdal.GetDriverByName('GTiff').Create(fname, Array.shape[2], Array.shape[1], Array.shape[0], gdal.GDT_Float32)
    ds.SetGeoTransform(GeoT)
    ds.SetProjection(projction)
    for i, image in enumerate(Array):
        ds.GetRasterBand(i+1).SetMetadata({'orbit': str(int(orbit[i]))})
        ds.GetRasterBand(i+1).WriteArray( image )
    ds.FlushCache()
    return fname



class KaSKASAR(object):
    """A class to process Sentinel 1 SAR data using S2 data as
    an input"""

    def __init__(self, s1_ncfile, state_mask, s2_wvc, rad_api, year, vv_version, passes, orbit1=None,orbit2=None):
        self.s1_ncfile = s1_ncfile
        self.state_mask = state_mask
        self.s2_wvc    = s2_vwc
        self.rad_api = rad_api
        self.year = year
        self.version = version
        self.passes = passes

        self.orbit1     = None
        self.orbit2    = None
        if orbit1 != None:
            self.orbit1     = orbit1
            if orbit2 != None:
                self.orbit2 = orbit2

    def sentinel1_inversion(self):

        path_s1data = '/media/AUF/GG/Geodatenverzeichnis-Uni/Fernerkundung/Satellitenbilder/Sentinel1/'
        output_folder = '/media/AUF/userdata/agvolution_new'
        sample_config_file = os.path.expanduser('~/sar-pre-processing/docs/notebooks/sample_config_file') #todo: change this location for automation
        name_tag = 'agvolution_new'
        year = '2023'
        lr_lat = 53.49579
        lr_lon = 13.11798
        ul_lat = 53.62974
        ul_lon = 12.87880
        multi_speck = '5'

        run_SenSARP(path_s1data,output_folder,sample_config_file,name_tag,year=year,lr_lat=lr_lat,lr_lon=lr_lon,ul_lat=ul_lat,ul_lon=ul_lon,multi_speck=multi_speck)

        s1_data, vwc_data, api_data = get_sm_input(s1_ncfile,version,s2_vwc,rad_api,self.state_mask,year)


        # api = get_api_folder(rad_api)
        # api_data = read_api_ssm(rad_api, self.state_mask)
        output_folder = '/media/tweiss/data/test_data/output'
        pdb.set_trace()
        sar_inference_data = inference_preprocessing(s1_data, vwc_data, api_data, output_folder)
        pdb.set_trace()

        xxx = do_inversion(sar_inference_data, self.state_mask, year, version, passes)

        # gg = gdal.Open('NETCDF:"%s":sigma0_vv_multi'%self.s1_ncfile)
        # geo = gg.GetGeoTransform()
        #
        # projction = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.01745329251994328,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]]'
        #
        # time = [i.strftime('%Y-%m-%d') for i in np.array(sar_inference_data.time)[sar_inference_data.time_mask]]
        #
        # sm_name  = self.s1_ncfile.replace('.nc', '_sar_sm.tif')
        # sr_name  = self.s1_ncfile.replace('.nc', '_sar_sr.tif')
        # lai_name = self.s1_ncfile.replace('.nc', '_sar_lai.tif')
        #
        # save_output(sm_name,  sm_outputs,  geo, projction, time)
        # save_output(sr_name,  sr_outputs,  geo, projction, time)
        # save_output(lai_name, lai_outputs, geo, projction, time)




if __name__ == '__main__':


    # years = ['2017','2018']
    year = '2023'
    versions = ['_multi', '_single']
    versions = ['_multi']

    esus = ['high', 'med', 'low']

    esu_size_tiff = '_ESU_buffer_100.tif' # buffer around ESU 100, 50, 30 etc

    time_contrainst = ['no'] # if yes time period march to july will be investigated

    pas = ['b_veg']
    # pas = ['analysis']
    # pas = ['b_0515']
    pas = ['unc_15','unc_13','unc_10']
    pas = ['analysis','b_veg','unc_15','unc_13','unc_10','unc_25', 'unc_05', 'unc_19']
    pas = ['sm_std_001','sm_std_003','sm_std_007','sm_std_010','sm_std_013','sm_std_015','sm_std_017','sm_std_020',]
    pas = ['sm_std_010']
    pas = ['b_veg']

    pas = ['sm_std_007', 'sm_std_007_1','sm_std_010', 'sm_std_010_1','sm_std_013','sm_std_013_1','sm_std_001','sm_std_003','sm_std_015','sm_std_017','sm_std_020']

    pas = ['sm_std_013']

    start = datetime.datetime.now()
    for passes in pas:

        for version in versions:
            s1_ncfile = '/media/tweiss/data/test_data/agvolution.nc'
            # rad_api = '/media/AUF/GG/Geodatenverzeichnis-Uni/Fernerkundung/Satellitenbilder/SSM/'

            state_mask = '/media/tweiss/data/test_data/fields_1.tif'

            rad_api = '/media/tweiss/data/test_data/API_RADOLAN_19768.01021216689_-0.05_6.996020755659563_5.0cm_2022-202306_2023-11-08T07:12:30.nc'

            s2_vwc = '/media/tweiss/data/test_data/tif/'

            sarsar = KaSKASAR(s1_ncfile, state_mask, s2_vwc, rad_api, year, version, passes)
            sarsar.sentinel1_inversion()
            pass
        # todo: path right now hard coded in different classes, this need to be changed!!!!
        path = '/media/AUF/userdata/agvolution/inversion/'
        # year = '2017'
        # path = '/media/tweiss/data/Arbeit_einordnen/mni/'

        plot_input_output(path, passes,year)

        # plot1 = datetime.datetime.now()
        # plot_scatter(years, esus, passes, esu_size_tiff)
        # plot2 = datetime.datetime.now()
        # plot_paper_3(years, esus, passes,time_contrainst)
        # plot3 = datetime.datetime.now()
        # plot_esu(years, esus, passes, esu_size_tiff)

    end = datetime.datetime.now()
    print('start:'+str(start))
    # print('start plot 1:'+str(plot1))
    # print('start plot 2:'+str(plot2))
    # print('start plot 3:'+str(plot3))
    print('end:'+str(end))

pdb.set_trace()
