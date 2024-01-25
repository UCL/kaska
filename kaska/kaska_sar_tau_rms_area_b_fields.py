#!/usr/bin/env python

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
from watercloudmodel_vwc_rms import cost_function_vwc, ssrt_jac_vwc, ssrt_vwc
from scipy.ndimage.filters import gaussian_filter1d
import pdb
from z_helper import *
import matplotlib.pyplot as plt
from netCDF4 import date2num
import glob

def ndwi1_mag(ndwi1):
    vwc = 13.2*ndwi1**2+1.62*ndwi1
    return vwc

def ndwi1_cos_maize(ndwi1):
    vwc = 9.39*ndwi1+1.26
    return vwc

def save_to_tif(fname, Array, GeoT):
    if os.path.exists(fname):
        os.remove(fname)
    ds = gdal.GetDriverByName('GTiff').Create(fname, Array.shape[2], Array.shape[1], Array.shape[0], gdal.GDT_Float32)
    ds.SetGeoTransform(GeoT)
    wkt = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.01745329251994328,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]]'
    ds.SetProjection(wkt)
    for i, image in enumerate(Array):
        # ds.GetRasterBand(i+1).SetMetadata({'date': prior_time[i]})
        ds.GetRasterBand(i+1).WriteArray( image )
    ds.FlushCache()
    return fname

def get_sar(s1_nc_file, version):
    s1_data = namedtuple('s1_data', 'time lat lon satellite  relorbit orbitdirection ang_name vv_name, vh_name')
    data = Dataset(s1_nc_file)
    relorbit            = data['relorbit'][:]
    localIncidenceAngle = data['theta'][:]
    satellite           = data['satellite'][:]
    orbitdirection      = data['orbitdirection'][:]
    time                = data['time'][:]
    lat = data['lat'][:]
    lon = data['lon'][:]

    vv_name = s1_nc_file.replace('.nc', '_vv'+version+'.tif')
    vh_name = s1_nc_file.replace('.nc', '_vh'+version+'.tif')
    ang_name = s1_nc_file.replace('.nc', '_ang'+version+'.tif')

    if not os.path.exists(vv_name):
        gg = gdal.Open('NETCDF:"%s":sigma0_vv"%s"'%(s1_nc_file,version))
        geo = gg.GetGeoTransform()
        sigma0_vv = data['sigma0_vv'+version][:]
        save_to_tif(vv_name, sigma0_vv, geo)

    if not os.path.exists(vh_name):
        gg = gdal.Open('NETCDF:"%s":sigma0_vh"%s"'%(s1_nc_file,version))
        geo = gg.GetGeoTransform()
        sigma0_vh = data['sigma0_vh'+version][:]
        save_to_tif(vh_name, sigma0_vh, geo)

    if not os.path.exists(ang_name):
        gg = gdal.Open('NETCDF:"%s":theta'%s1_nc_file)
        geo = gg.GetGeoTransform()
        localIncidenceAngle = data['theta'][:]
        save_to_tif(ang_name, localIncidenceAngle, geo)

    return s1_data(time, lat, lon, satellite, relorbit, orbitdirection, ang_name, vv_name, vh_name)

def get_api(api_nc_file,year):
    api_data = namedtuple('api_data', 'time lat lon api')
    data = Dataset(api_nc_file)

    xxx = date2num(datetime.datetime.strptime(year+'0201', '%Y%m%d'), units ='hours since 2000-01-01 00:00:00', calendar='gregorian')
    yyy = date2num(datetime.datetime.strptime(year+'1001', '%Y%m%d'), units ='hours since 2000-01-01 00:00:00', calendar='gregorian')

    time = data['time'][np.where(data['time'][:]==xxx)[0][0]:np.where(data['time'][:]==yyy)[0][0]]
    lat = data['lat'][:]
    lon = data['lon'][:]

    api_name = api_nc_file.replace('.nc', '_api'+year+'.tif')

    if not os.path.exists(api_name):
        gg = gdal.Open('NETCDF:"%s":api'%api_nc_file)
        geo = gg.GetGeoTransform()
        save_to_tif(api_name, data['api'][np.where(data['time'][:]==xxx)[0][0]:np.where(data['time'][:]==yyy)[0][0],:,:], geo)

    return api_data(time, lat, lon, api_name)

def read_sar(sar_data, state_mask):
    s1_data = namedtuple('s1_data', 'time lat lon satellite  relorbit orbitdirection ang vv vh')
    ang = reproject_data(sar_data.ang_name, output_format="MEM", target_img=state_mask)
    vv  = reproject_data(sar_data.vv_name, output_format="MEM", target_img=state_mask)
    vh  = reproject_data(sar_data.vh_name, output_format="MEM", target_img=state_mask)

    time = [datetime.datetime(1970,1,1) + datetime.timedelta(days=float(i)) for i in  sar_data.time]

    return s1_data(time, sar_data.lat, sar_data.lon, sar_data.satellite, sar_data.relorbit, sar_data.orbitdirection, ang, vv, vh)

def read_vwc(vwc_data, state_mask):
    s2_data = namedtuple('s2_vwc', 'time vwc ndwi')
    filelist = glob.glob(vwc_data+'*.tif')
    filelist.sort()
    time = []
    vwc = []
    ndwi = []
    for file in filelist:
        g = gdal.Open(file)
        ndwi_array = reproject_data(file, output_format="MEM", target_img=state_mask)
        ndwi_array = ndwi_array.ReadAsArray()
        vwc_array = ndwi1_mag(ndwi_array)
        time.append(datetime.datetime.strptime(file.split('/')[-1][14:22], '%Y%m%d'))
        vwc.append(vwc_array)
        ndwi.append(ndwi_array)

    return s2_data(time, vwc, ndwi)

def read_api(api_data, state_mask):
    s1_data = namedtuple('api_data', 'time lat lon api')

    api = reproject_data(api_data.api, output_format="MEM", target_img=state_mask)
    time = [datetime.datetime(2000,1,1) + datetime.timedelta(hours=float(i)) for i in  api_data.time]

    return s1_data(time, api_data.lat, api_data.lon, api)


def inference_preprocessing(s1_data, vwc_data, api_data, state_mask, orbit1=None, orbit2=None):
    """Resample S2 smoothed output to match S1 observations
    times"""
    # Move everything to DoY to simplify interpolation

    sar_inference_data = namedtuple('sar_inference_data', 'time lat lon satellite  relorbit orbitdirection ang vv vh vwc api time_mask ndwi')


    vwc_doys = np.array([ int(i.strftime('%j')) for i in vwc_data.time])
    s1_doys = np.array([ int(i.strftime('%j')) for i in s1_data.time])


    time = np.array(s1_data.time)
    for jj in range(len(s1_data.time)):
        time[jj] = s1_data.time[jj].replace(microsecond=0).replace(second=0).replace(minute=0)

    index=[]
    xxx = np.array(api_data.time)
    for jj in range(len(time)):
        oje = np.where(xxx==time[jj])
        try:
            ojet = oje[0][0]
            index.append(ojet)
        except IndexError:
            pass
    api_doys = np.array([ int(i.strftime('%j')) for i in np.array(api_data.time)[index]])

    f = interp1d(vwc_doys, np.array(vwc_data.vwc), axis=0, bounds_error=False)
    vwc_s1 = f(s1_doys)

    f = interp1d(vwc_doys, np.array(vwc_data.ndwi), axis=0, bounds_error=False)
    ndwi_s1 = f(s1_doys)

    api_s1 = api_data.api.ReadAsArray()[index]
    f = interp1d(api_doys, api_s1, axis=0, bounds_error=False)
    api_s1 = f(s1_doys)

    if s1_data.time[0].year == 2017:
        time_mask = (s1_doys >= 80) & (s1_doys <= 273)
    elif s1_data.time[0].year == 2018:
        time_mask = (s1_doys >= 80) & (s1_doys <= 273)
    else:
        print('no time mask')

    if orbit1 != None:
        rel_orbit1 = s1_data.relorbit==orbit1
        if orbit2 != None:
            rel_orbit2 = s1_data.relorbit==orbit2
            xxx = np.logical_and(rel_orbit1,time_mask)
            yyy = np.logical_and(rel_orbit2,time_mask)
            time_mask = np.logical_or(xxx,yyy)

    sar_inference_data = sar_inference_data(s1_data.time, s1_data.lat, s1_data.lon,
                                            s1_data.satellite, s1_data.relorbit,
                                            s1_data.orbitdirection, s1_data.ang,
                                            s1_data.vv, s1_data.vh, vwc_s1, api_s1, time_mask, ndwi_s1)

    return sar_inference_data


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
    state_mask = g.ReadAsArray().astype(np.int)
    state_mask = state_mask > 0

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
    sm_std = sm_all[:,0,0]
    sm_std[:] = 0.2

    b = sm_all[:,0,0]
    b[:] = 0
    b_std = sm_all[:,0,0]
    b_std[:] = 0.5 # not used anyway
    rms = sm_all[:,0,0]
    rms = 0.2
    rms_std = 0.1 # not used anyway

    unc = 1.9
    omega = 0.027

    sm_retrieved = sm_all * np.nan

    if not os.path.exists('/media/tweiss/Work/Paper3_down/'+passes):
        os.makedirs('/media/tweiss/Work/Paper3_down/'+passes)


    np.save('/media/tweiss/Work/Paper3_down/'+passes+'/'+year+version+'_input_vv.npy', vv_all)
    np.save('/media/tweiss/Work/Paper3_down/'+passes+'/'+year+version+'_input_vwc.npy', vwc_all)
    np.save('/media/tweiss/Work/Paper3_down/'+passes+'/'+year+version+'_input_sm_api.npy', sm_all)
    np.save('/media/tweiss/Work/Paper3_down/'+passes+'/'+year+version+'_input_ndwi.npy', ndwi_all)

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

                orbits95[0:np.argmax(vwc)] = False
                orbits117[0:np.argmax(vwc)] = False
                orbits44[0:np.argmax(vwc)] = False
                orbits168[0:np.argmax(vwc)] = False

                # orbits95[0:np.where(time_all == min(time_all, key=lambda x: abs(x-datetime.datetime(2017,5,15))))[0][0]] = False
                # orbits117[0:np.where(time_all == min(time_all, key=lambda x: abs(x-datetime.datetime(2017,5,15))))[0][0]] = False
                # orbits44[0:np.where(time_all == min(time_all, key=lambda x: abs(x-datetime.datetime(2017,5,15))))[0][0]] = False
                # orbits168[0:np.where(time_all == min(time_all, key=lambda x: abs(x-datetime.datetime(2017,5,15))))[0][0]] = False


                b[orbits95] = 0.1
                b[orbits117] = 0.1
                b[orbits44] = 0.2
                b[orbits168] = 0.2


                sm = sm_all[:,z,zz]


                times, svwc, sb, sms, srms, ps, orbit_mask = do_one_pixel_field(vv, vh, vwc, vwc_std, theta, time_all, sm, sm_std, b, b_std, omega, rms, rms_std, orbits,unc=unc)

                vwc_outputs[:,z,zz] = svwc
                sm_outputs[:,z,zz]  = sms
                b_outputs[:,z,zz]  = sb
                rms_outputs[:,z,zz]  = srms

    np.save('/media/tweiss/Work/Paper3_down/'+passes+'/'+year+version+'_sm'+'.npy', sm_outputs)
    np.save('/media/tweiss/Work/Paper3_down/'+passes+'/'+year+version+'_vwc'+'.npy', vwc_outputs)
    np.save('/media/tweiss/Work/Paper3_down/'+passes+'/'+year+version+'_b'+'.npy', b_outputs)
    np.save('/media/tweiss/Work/Paper3_down/'+passes+'/'+year+version+'_rms'+'.npy', rms_outputs)
    np.save('/media/tweiss/Work/Paper3_down/'+passes+'/'+year+version+'_times.npy',times)

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
        sar = get_sar(s1_ncfile, version)
        s1_data = read_sar(sar, self.state_mask)

        vwc_data = read_vwc(s2_vwc, self.state_mask)

        api = get_api(rad_api,year)
        api_data = read_api(api, self.state_mask)

        sar_inference_data = inference_preprocessing(s1_data, vwc_data, api_data, self.state_mask,self.orbit1,self.orbit2)


        xxx = do_inversion(sar_inference_data, self.state_mask, year, version, passes)

        # gg = gdal.Open('NETCDF:"%s":sigma0_vv_multi'%self.s1_ncfile)
        # geo = gg.GetGeoTransform()

        # projction = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.01745329251994328,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]]'

        # time = [i.strftime('%Y-%m-%d') for i in np.array(sar_inference_data.time)[sar_inference_data.time_mask]]

        # sm_name  = self.s1_ncfile.replace('.nc', '_sar_sm.tif')
        # sr_name  = self.s1_ncfile.replace('.nc', '_sar_sr.tif')
        # lai_name = self.s1_ncfile.replace('.nc', '_sar_lai.tif')

        # save_output(sm_name,  sm_outputs,  geo, projction, time)
        # save_output(sr_name,  sr_outputs,  geo, projction, time)
        # save_output(lai_name, lai_outputs, geo, projction, time)




if __name__ == '__main__':


    years = ['2017','2018']
    # years = ['2018']
    versions = ['_multi', '_single']
    versions = ['_multi']
    for year in years:
        for version in versions:
            s1_ncfile = '/media/tweiss/Work/Paper3_down/data/MNI_'+year+'_new_final_paper3.nc'
            state_mask = '/media/tweiss/Work/Paper3_down/GIS/clc_class2.tif'
            state_mask = '/media/tweiss/Work/Paper3_down/GIS/'+year+'_ESU_Field_buffer_30.tif'
            rad_api = '/media/tweiss/Work/Paper3_down/data/RADOLAN_API_v1.0.0.nc'

            s2_vwc = '/media/tweiss/Work/Paper3_down/data/'+year+'/tif1/'

            passes = 'hm'

            sarsar = KaSKASAR(s1_ncfile, state_mask, s2_vwc, rad_api, year, version, passes)

            sarsar.sentinel1_inversion()

