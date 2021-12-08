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

# def reproject_data2(source_img,
#         target_img=None,
#         dstSRS=None,
#         srcSRS=None,
#         srcNodata=np.nan,
#         dstNodata=np.nan,
#         outputType=None,
#         output_format="MEM",
#         verbose=False,
#         xmin=None,
#         xmax=None,
#         ymin=None,
#         ymax=None,
#         xRes=None,
#         yRes=None,
#         xSize=None,
#         ySize=None,
#         resample=1,
#     ):

#     """
#     A method that uses a source and a target images to
#     reproject & clip the source image to match the extent,
#     projection and resolution of the target image.
#     """

#     outputType = (
#         gdal.GDT_Unknown if outputType is None else outputType
#         )
#     if srcNodata is None:
#             try:
#                 srcNodata = " ".join(
#                     [
#                         i.split("=")[1]
#                         for i in gdal.Info(source_img).split("\n")
#                         if " NoData" in i
#                     ]
#                 )
#             except RuntimeError:
#                 srcNodata = None
#     # If the output type is intenger and destination nodata is nan
#     # set it to 0 to avoid warnings
#     if outputType <= 5 and np.isnan(dstNodata):
#         dstNodata = 0

#     if srcSRS is not None:
#         _srcSRS = osr.SpatialReference()
#         try:
#             _srcSRS.ImportFromEPSG(int(srcSRS.split(":")[1]))
#         except:
#             _srcSRS.ImportFromWkt(srcSRS)
#     else:
#         _srcSRS = None


#     if (target_img is None) & (dstSRS is None):
#             raise IOError(
#                 "Projection should be specified ether from "
#                 + "a file or a projection code."
#             )
#     elif target_img is not None:
#             try:
#                 g = gdal.Open(target_img)
#             except RuntimeError:
#                 g = target_img
#             geo_t = g.GetGeoTransform()
#             x_size, y_size = g.RasterXSize, g.RasterYSize

#             if xRes is None:
#                 xRes = abs(geo_t[1])
#             if yRes is None:
#                 yRes = abs(geo_t[5])

#             if xSize is not None:
#                 x_size = 1.0 * xSize * xRes / abs(geo_t[1])
#             if ySize is not None:
#                 y_size = 1.0 * ySize * yRes / abs(geo_t[5])

#             xmin, xmax = (
#                 min(geo_t[0], geo_t[0] + x_size * geo_t[1]),
#                 max(geo_t[0], geo_t[0] + x_size * geo_t[1]),
#             )
#             ymin, ymax = (
#                 min(geo_t[3], geo_t[3] + y_size * geo_t[5]),
#                 max(geo_t[3], geo_t[3] + y_size * geo_t[5]),
#             )
#             dstSRS = osr.SpatialReference()
#             raster_wkt = g.GetProjection()
#             dstSRS.ImportFromWkt(raster_wkt)
#             pdb.set_trace()
#             gg = gdal.Warp(
#                 "",
#                 source_img,
#                 format=output_format,
#                 outputBounds=[xmin, ymin, xmax, ymax],
#                 dstNodata=dstNodata,
#                 warpOptions=["NUM_THREADS=ALL_CPUS"],
#                 xRes=xRes,
#                 yRes=yRes,
#                 dstSRS=dstSRS,
#                 outputType=outputType,
#                 srcNodata=srcNodata,
#                 resampleAlg=resample,
#                 srcSRS=_srcSRS
#             )

#     else:
#             gg = gdal.Warp(
#                 "",
#                 source_img,
#                 format=output_format,
#                 outputBounds=[xmin, ymin, xmax, ymax],
#                 xRes=xRes,
#                 yRes=yRes,
#                 dstSRS=dstSRS,
#                 warpOptions=["NUM_THREADS=ALL_CPUS"],
#                 copyMetadata=True,
#                 outputType=outputType,
#                 dstNodata=dstNodata,
#                 srcNodata=srcNodata,
#                 resampleAlg=resample,
#                 srcSRS=_srcSRS
#             )
#     if verbose:
#         LOG.debug("There are %d bands in this file, use "
#                 + "g.GetRasterBand(<band>) to avoid reading the whole file."
#                 % gg.RasterCount
#             )
#     return gg

def ndwi1_mag(ndwi1):
    vwc = 13.2*ndwi1**2+1.62*ndwi1
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

def get_sar(s1_nc_file):
    s1_data = namedtuple('s1_data', 'time lat lon satellite  relorbit orbitdirection ang_name vv_name, vh_name')
    data = Dataset(s1_nc_file)
    relorbit            = data['relorbit'][:]
    localIncidenceAngle = data['localIncidenceAngle'][:]
    satellite           = data['satellite'][:]
    orbitdirection      = data['orbitdirection'][:]
    time                = data['time'][:]
    lat = data['lat'][:]
    lon = data['lon'][:]

    vv_name = s1_nc_file.replace('.nc', '_vv.tif')
    vh_name = s1_nc_file.replace('.nc', '_vh.tif')
    ang_name = s1_nc_file.replace('.nc', '_ang.tif')

    if not os.path.exists(vv_name):
        gg = gdal.Open('NETCDF:"%s":sigma0_vv_multi'%s1_nc_file)
        geo = gg.GetGeoTransform()
        sigma0_vv_norm_multi = data['sigma0_vv_multi'][:]
        save_to_tif(vv_name, sigma0_vv_norm_multi, geo)

    if not os.path.exists(vh_name):
        gg = gdal.Open('NETCDF:"%s":sigma0_vh_multi'%s1_nc_file)
        geo = gg.GetGeoTransform()
        sigma0_vh_norm_multi = data['sigma0_vh_multi'][:]
        save_to_tif(vh_name, sigma0_vh_norm_multi, geo)

    if not os.path.exists(ang_name):
        gg = gdal.Open('NETCDF:"%s":localIncidenceAngle'%s1_nc_file)
        geo = gg.GetGeoTransform()
        localIncidenceAngle = data['localIncidenceAngle'][:]
        save_to_tif(ang_name, localIncidenceAngle, geo)

    return s1_data(time, lat, lon, satellite, relorbit, orbitdirection, ang_name, vv_name, vh_name)

def get_vwc(vwc_nc_file):
    s1_data = namedtuple('vwc_data', 'time lat lon vwc')
    data = Dataset(vwc_nc_file)

    time = data['time'][:12]
    lat = data['lat'][:]
    lon = data['lon'][:]

    vwc_name = vwc_nc_file.replace('.nc', '_vwc.tif')

    if not os.path.exists(vwc_name):
        gg = gdal.Open('NETCDF:"%s":newBand'%vwc_nc_file)
        geo = gg.GetGeoTransform()
        save_to_tif(vwc_name, data['newBand'][:12,:,:], geo)

    return s1_data(time, lat, lon, vwc_name)

def get_api(api_nc_file):
    api_data = namedtuple('api_data', 'time lat lon api')
    data = Dataset(api_nc_file)

    xxx = date2num(datetime.datetime.strptime('20170301', '%Y%m%d'), units ='hours since 2000-01-01 00:00:00', calendar='gregorian')
    yyy = date2num(datetime.datetime.strptime('20170801', '%Y%m%d'), units ='hours since 2000-01-01 00:00:00', calendar='gregorian')

    time = data['time'][np.where(data['time'][:]==xxx)[0][0]:np.where(data['time'][:]==yyy)[0][0]]
    lat = data['lat'][:]
    lon = data['lon'][:]

    api_name = api_nc_file.replace('.nc', '_api.tif')

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
    s1_data = namedtuple('vwc_data', 'time lat lon vwc')
    vwc = reproject_data(vwc_data.vwc, output_format="MEM", target_img=state_mask)
    time = [datetime.datetime(1970,1,1) + datetime.timedelta(days=float(i)) for i in  vwc_data.time]

    return s1_data(time, vwc_data.lat, vwc_data.lon, vwc)


def read_api(api_data, state_mask):
    s1_data = namedtuple('api_data', 'time lat lon api')

    api = reproject_data(api_data.api, output_format="MEM", target_img=state_mask)
    time = [datetime.datetime(2000,1,1) + datetime.timedelta(hours=float(i)) for i in  api_data.time]

    return s1_data(time, api_data.lat, api_data.lon, api)

# def read_s2_lai(s2_lai, s2_cab, s2_cbrown, state_mask):
#     s2_data = namedtuple('s2_lai', 'time lai cab cbrown')
#     g = gdal.Open(s2_lai)
#     time = []
#     for i in range(g.RasterCount):
#         gg = g.GetRasterBand(i+1)
#         meta = gg.GetMetadata()
#         time.append(datetime.datetime.strptime(meta['DoY'], '%Y%j'))
#     lai  = reproject_data(s2_lai, output_format="MEM", target_img=state_mask)
#     cab  = reproject_data(s2_cab, output_format="MEM", target_img=state_mask)
#     cbrown  = reproject_data(s2_cbrown, output_format="MEM", target_img=state_mask)
#     return s2_data(time, lai, cab, cbrown)

def inference_preprocessing(s1_data, vwc_data, api_data, state_mask, orbit1=None, orbit2=None):
    """Resample S2 smoothed output to match S1 observations
    times"""
    # Move everything to DoY to simplify interpolation

    sar_inference_data = namedtuple('sar_inference_data', 'time lat lon satellite  relorbit orbitdirection ang vv vh vwc api time_mask')


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


    f = interp1d(vwc_doys, vwc_data.vwc.ReadAsArray(), axis=0, bounds_error=False)
    vwc_s1 = f(s1_doys)

    api_s1 = api_data.api.ReadAsArray()[index]
    f = interp1d(api_doys, api_s1, axis=0, bounds_error=False)
    api_s1 = f(s1_doys)

    if s1_data.time[0].year == 2017:
        time_mask = (s1_doys >= 80) & (s1_doys <= 199)
    elif s1_data.time[0].year == 2018:
        print('time mask need to be changed')
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
                                            s1_data.vv, s1_data.vh, vwc_s1, api_s1, time_mask)

    return sar_inference_data


# def get_prior(s1_data, soilMoisture, soilMoisture_std, soilRoughness, soilRoughness_std, state_mask):
#     # this is the function to reading the soil moisture prior
#     # and the soil roughness prior using the satemask
#     # the assumption of inputs are daily data in geotifs
#     prior = namedtuple('prior', 'time sm_prior sm_std sr_prior sr_std')

#     g = gdal.Open(soilMoisture)
#     time = []
#     for i in range(g.RasterCount):
#         gg = g.GetRasterBand(i+1)
#         meta = gg.GetMetadata()
#         time.append(datetime.datetime.strptime(meta['date'], '%Y-%m-%d'))
#     sm_prior  = reproject_data(soilMoisture,     output_format="MEM", target_img=state_mask)
#     sm_std    = reproject_data(soilMoisture_std, output_format="MEM", target_img=state_mask)
#     sr_prior  = reproject_data(soilRoughness,    output_format="MEM", target_img=state_mask)
#     sr_std    = reproject_data(soilRoughness_std,output_format="MEM", target_img=state_mask)

#     prior_doy = np.array([ int(i.strftime('%j')) for i in time])
#     s1_doys = np.array([ int(i.strftime('%j')) for i in s1_data.time])

#     f = interp1d(prior_doy, sm_prior.ReadAsArray(), axis=0, bounds_error=False)

#     sm_s1 = f(s1_doys)
#     f = interp1d(prior_doy,   sm_std.ReadAsArray(), axis=0, bounds_error=False)
#     sm_std_s1 = f(s1_doys)

#     f = interp1d(prior_doy, sr_prior.ReadAsArray(), axis=0, bounds_error=False)
#     sr_s1 = f(s1_doys)
#     f = interp1d(prior_doy,   sr_std.ReadAsArray(), axis=0, bounds_error=False)
#     sr_std_s1 = f(s1_doys)

#     return prior(time, sm_s1, sm_std_s1, sr_s1, sr_std_s1)



# def do_one_pixel_field(data_field, vv, vh, vwc, theta, time, sm, sm_std, b, b_std, omega, rms, orbits, unc):

#     ps   = []
#     vwcs    = []
#     bs = []
#     sms    = []
#     times = []

#     uorbits = np.unique(orbits)
#     uorbits = np.array([95])
#     for orbit in uorbits:
#     # for jj in range(len(vv)):
#         # pdb.set_trace()
#         # orbit_mask = orbits == orbit
#         # orbit_mask = (orbits == 44) | (orbits == 168)
#         orbit_mask = (orbits == 44) | (orbits == 95) | (orbits == 117) | (orbits == 168)
#         # orbit_mask = (orbits == 168)
#         # orbit_mask = (orbits == 44) | (orbits == 95) | (orbits == 117)
#         ovv, ovh, ovwc, otheta, otime = vv[orbit_mask], vh[orbit_mask], vwc[orbit_mask], theta[orbit_mask], time[orbit_mask]
#         osm, osm_std, osb, osb_std  = sm[orbit_mask], sm_std[orbit_mask], b[orbit_mask], b_std[orbit_mask]


#         ovwc_std = np.ones_like(osb)*0.5

#         # alpha     = _calc_eps(osm)
#         # alpha = osm
#         # alpha_std = np.ones_like(alpha)*10
#         # alpha_std = osm_std
#         # pdb.set_trace()

#         prior_mean = np.concatenate([[0,   ]*2, osm,     ovwc,     osb])
#         prior_unc  = np.concatenate([[10., ]*2, osm_std, ovwc_std, osb_std])

#         xvv = np.array([rms, omega])


#         x0 = np.concatenate([xvv, osm, ovwc, osb])

#         bounds = (
#             [[0.013, 0.013]] # s
#           + [[0.0107, 0.0107]] # omega
#           + [[0.01,   0.7]] * osb.shape[0] # mv
#           + [[0,     7.5]] * osb.shape[0] # vwc
#           + [[0.01,       0.6]] * osb.shape[0] # b
#           )


#         gamma = [500, 500]

#         retval = minimize(cost_function_vwc,
#                             x0,
#                             args=(ovh, ovv, otheta, gamma, prior_mean, prior_unc, unc),
#                             jac=True,
#                             bounds = bounds,
#                             options={"disp": False},)


#         posterious_sm   = retval.x[2 : 2+len(osb)]
#         posterious_vwc    = retval.x[2+len(osb)   : 2+2*len(osb)]
#         posterious_b = retval.x[2+2*len(osb)   : 2+3*len(osb)]

#         sms.append(posterious_sm)
#         vwcs.append(posterious_vwc)
#         bs.append(posterious_b)
#         times.append(otime)
#         ps.append(retval.x[:2])

#     order = np.argsort(np.hstack(times))
#     times  = np.hstack(times )[order]
#     vwcs   = np.hstack(vwcs  )[order]
#     bs    = np.hstack(bs   )[order]
#     sms    = np.hstack(sms   )[order].real
#     return times, vwcs, bs, sms, np.array(ps), orbit_mask

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
        # pdb.set_trace()
        # orbit_mask = orbits == orbit
        # orbit_mask = (orbits == 44) | (orbits == 168)
        # orbit_mask = (orbits == 95) | (orbits == 117)
        orbit_mask = (orbits == 44) | (orbits == 95) | (orbits == 117) | (orbits == 168)
        # orbit_mask = (orbits == 168)
        # orbit_mask = (orbits == 44) | (orbits == 95) | (orbits == 117)
        ovv, ovh, ovwc, ovwc_std, otheta, otime = vv[orbit_mask], vh[orbit_mask], vwc[orbit_mask], vwc_std[orbit_mask], theta[orbit_mask], time[orbit_mask]
        osm, osm_std, osb, osb_std = sm[orbit_mask], sm_std[orbit_mask], b[orbit_mask], b_std[orbit_mask]




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


        prior_mean = np.concatenate([[0,   ], [rms], osm,     ovwc,     osb])
        prior_unc  = np.concatenate([[10., ], [rms_std], osm_std, ovwc_std, osb_std])


        x0 = np.concatenate([np.array([omega]), np.array([rms]), osm, ovwc, osb])

        bounds = (
            [[0.027, 0.027]] # omega
          + [[0.005, 0.03]]  # s=rms
          + [[0.01,   0.7]] * osb.shape[0] # mv
          + [[0,     7.5]] * osb.shape[0] # vwc
          + [[0.01,       0.6]] * osb.shape[0] # b
          )



        data = osb

        gamma = [10, 10]

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




def do_inversion(sar_inference_data, state_mask, segment=False):

    orbits = sar_inference_data.relorbit[sar_inference_data.time_mask]
    uorbits = np.unique(orbits)
    if segment:

        out_shape   = sar_inference_data.vwc[sar_inference_data.time_mask].shape
        vwc_outputs = np.zeros(out_shape )
        sm_outputs  = np.zeros(out_shape )
        b_outputs  = np.zeros(out_shape )
        rms_outputs  = np.zeros(out_shape )

        pixel = ['_Field_buffer_30','','_buffer_30','_buffer_50','_buffer_100']
        pixel = ['_Field_buffer_30']
        fields = ['301','508','542']
        # fields = ['508']
        # ESU names
        esus = ['high', 'low', 'med', 'mean']
        esus = ['mean']
        for pixels in pixel:
            print(pixels)
            path_ESU = '/media/tweiss/Work/z_final_mni_data_2017/'
            name_shp = 'ESU'+pixels+'.shp'
            name_ESU = 'ESU'+pixels+'.tif'

            for esu in esus:
                for field in fields:
                    field2 = field + '_' + esu
                    g = gdal.Open(os.path.join(path_ESU, name_ESU))
                    state_mask = g.ReadAsArray().astype(np.int)

                    if pixels == '_Field_buffer_30':
                        if field == '515':
                            mask_value = 4
                            state_mask = state_mask==mask_value
                        elif field == '508':
                            mask_value = 27
                            state_mask = state_mask==mask_value
                        elif field == '542':
                            mask_value = 8
                            state_mask = state_mask==mask_value
                        elif field == '319':
                            mask_value = 67
                            state_mask = state_mask==mask_value
                        elif field == '301':
                            mask_value = 87
                            state_mask = state_mask==mask_value
                    else:
                        pass

                    field_mask = state_mask

                    vv_all = sar_inference_data.vv.ReadAsArray()[sar_inference_data.time_mask]
                    vh_all    = sar_inference_data.vh.ReadAsArray()[sar_inference_data.time_mask]
                    theta_all = sar_inference_data.ang.ReadAsArray()[sar_inference_data.time_mask]
                    time_all = np.array(sar_inference_data.time)[sar_inference_data.time_mask]

                    vwc_all = sar_inference_data.vwc[sar_inference_data.time_mask]
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

                    for z in range(len(state_mask)):
                        for zz in range(len(state_mask[0])):
                            if state_mask[z,zz] == False:
                                pass
                            else:
                                vv = vv_all[:,z,zz]
                                vh = vh_all[:,z,zz]
                                theta = theta_all[:,z,zz]
                                vwc = ndwi1_mag(vwc_all[:,z,zz])
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

                    np.save('/media/tweiss/Work/Paper3_plot/'+field+'_sm'+'.npy', sm_outputs)
                    np.save('/media/tweiss/Work/Paper3_plot/'+field+'_vwc'+'.npy', vwc_outputs)
                    np.save('/media/tweiss/Work/Paper3_plot/'+field+'_b'+'.npy', b_outputs)
                    np.save('/media/tweiss/Work/Paper3_plot/'+field+'_rms'+'.npy', rms_outputs)


                    for u in range(len(sm_retrieved)):

                        fig = plt.gcf()
                        ax = fig.add_subplot(111)

                        if field == '508':
                            quadmesh = ax.imshow(sm_outputs[u,650:750,400:500])
                        elif field == '301':
                            quadmesh = ax.imshow(sm_outputs[u,0:100,200:250])
                        elif field == '542':
                            quadmesh = ax.imshow(sm_outputs[u,250:350,580:630])
                        else:
                            pass
                        plt.colorbar(quadmesh)
                        quadmesh.set_clim(vmin=0.15, vmax=0.35)

                        plt.savefig('/media/tweiss/Daten/data_AGU/test_kaska/down3/'+field+'_'+times[u].strftime("%Y%m%d"), bbox_inches = 'tight')
                        plt.close()

                    # pdb.set_trace()



        pdb.set_trace()

    else:
        mask = gdal.Open(state_mask).ReadAsArray()
        xs, ys = np.where(mask)

        out_shape   = sar_inference_data.vwc[sar_inference_data.time_mask].shape
        time  = np.array(sar_inference_data.time)[sar_inference_data.time_mask]
        vwc_outputs = np.zeros(out_shape )
        sm_outputs  = np.zeros(out_shape )
        b_outputs  = np.zeros(out_shape )

        for i in range(len(xs)):
            indx, indy = xs[i], ys[i]

            # field_mask = slice(None, None), slice(indx, indx+1), slice(indy, indy+1)
            time  = np.array(sar_inference_data.time)[sar_inference_data.time_mask]
            vwc   = sar_inference_data.vwc[sar_inference_data.time_mask][:, indx, indy ]
            api   = sar_inference_data.api[sar_inference_data.time_mask][:, indx, indy ]
            api_std = sar_inference_data.api[sar_inference_data.time_mask][:, indx, indy ]
            pdb.set_trace()
            api_std[:] = 0.2

            # b =
            # b_std =

            # rms =
            # rms_std =
            # sm    = prior.sm_prior[sar_inference_data.time_mask][:, indx, indy ]
            # sm_std= prior.sm_std  [sar_inference_data.time_mask][:, indx, indy ]

            sm[np.isnan(sm)] = 0.2
            sm_std[sm_std==0] = 0.5
            sm_std[np.isnan(sm_std)] = 0.5

            # coef    = prior.sr_prior[sar_inference_data.time_mask][:, indx, indy ]
            # coef_std= prior.sr_std  [sar_inference_data.time_mask][:, indx, indy ]
            # sr[np.isnan(sr)] = 0.1
            # sr_std[np.isnan(sr_std)] = 0.5

            # height = prior.sr_prior[sar_inference_data.time_mask][:, indx, indy ]
            # height[:] = 0.1

            vv    = sar_inference_data.vv.ReadAsArray()[sar_inference_data.time_mask][:, indx, indy ]
            vh    = sar_inference_data.vh.ReadAsArray()[sar_inference_data.time_mask][:, indx, indy ]
            theta = sar_inference_data.ang.ReadAsArray()[sar_inference_data.time_mask][:, indx, indy ]


            vv = np.maximum(vv, 0.0001)
            vv = 10 * np.log10(vv)
            vh = np.maximum(vh, 0.0001)
            vh = 10 * np.log10(vh)

            times, lais, coefs, sms = do_one_pixel_field(sar_inference_data, vv, vh, lai, theta, time, sm, sm_std, coef, coef_std, height)
            times, vwcs, bs, sms, srms, ps, orbit_mask = do_one_pixel_field(data_field, vv, vh, vwc, theta, time, sm, sm_std, b, b_std, omega, rms, rms_std, orbits,unc=unc)

            lai_outputs[:, indx, indy] = lais
            coef_outputs[:, indx, indy]  = coefs
            sm_outputs[:, indx, indy]  = sms

    return lai_outputs, coef_outputs, sm_outputs, uorbits

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

    def __init__(self, s1_ncfile, state_mask, s2_wvc, orbit1=None,orbit2=None):
        self.s1_ncfile = s1_ncfile
        self.state_mask = state_mask
        self.s2_wvc    = s2_vwc
        self.rad_api = rad_api

        self.orbit1     = None
        self.orbit2    = None
        if orbit1 != None:
            self.orbit1     = orbit1
            if orbit2 != None:
                self.orbit2 = orbit2

    def sentinel1_inversion(self, segment=False):
        sar = get_sar(s1_ncfile)
        s1_data = read_sar(sar, self.state_mask)

        vwc_ = get_vwc(s2_vwc)
        vwc_data = read_vwc(vwc_, self.state_mask)

        api = get_api(rad_api)
        api_data = read_api(api, self.state_mask)

        # prior   = get_prior(s1_data, self.sm_prior, self.sm_std, self.sr_prior, self.sr_std, self.state_mask)
        sar_inference_data = inference_preprocessing(s1_data, vwc_data, api_data, self.state_mask,self.orbit1,self.orbit2)


        lai_outputs, sr_outputs, sm_outputs, uorbits = do_inversion(sar_inference_data, self.state_mask, segment)
        pdb.set_trace()
        gg = gdal.Open('NETCDF:"%s":sigma0_vv_multi'%self.s1_ncfile)
        geo = gg.GetGeoTransform()

        projction = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.01745329251994328,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]]'

        time = [i.strftime('%Y-%m-%d') for i in np.array(sar_inference_data.time)[sar_inference_data.time_mask]]

        sm_name  = self.s1_ncfile.replace('.nc', '_sar_sm.tif')
        sr_name  = self.s1_ncfile.replace('.nc', '_sar_sr.tif')
        lai_name = self.s1_ncfile.replace('.nc', '_sar_lai.tif')

        save_output(sm_name,  sm_outputs,  geo, projction, time)
        save_output(sr_name,  sr_outputs,  geo, projction, time)
        save_output(lai_name, lai_outputs, geo, projction, time)




if __name__ == '__main__':
    # s1_ncfile = '/data/nemesis/kaska-sar_quick/S1_LMU_site_2017_new.nc'
    # state_mask = "/home/ucfajlg/Data/python/KaFKA_Validation/LMU/carto/ESU.tif"
    # s2_folder = "/home/ucfajlg/Data/python/KaFKA_Validation/LMU/s2_obs/"
    # s2_lai = f"{s2_folder:s}/outputs/lai.tif"
    # s2_cab = f"{s2_folder:s}/outputs/cab.tif"
    # s2_cbrown = f"{s2_folder:s}/outputs/cbrown.tif"

    # sm_prior = '/data/nemesis/kaska-sar_quick/sm_prior.tif'
    # sm_std   = '/data/nemesis/kaska-sar_quick/sm_std.tif'
    # sr_prior = '/data/nemesis/kaska-sar_quick/sr_prior.tif'
    # sr_std   = '/data/nemesis/kaska-sar_quick/sr_std.tif'
    # sarsar = KaSKASAR(s1_ncfile, state_mask, s2_lai,  s2_cab, s2_cbrown, sm_prior, sm_std, sr_prior ,sr_std)

    # s1_ncfile = '/media/nas_data/Thomas/S1/processed/MNI_2017/MNI_2017.nc'

    # aggregation = '_point'
    aggregation = '_Field_buffer_30'
    # aggregation = '_buffer_100'
    aggregation = '_buffer_100'


    s1_ncfile = '/media/tweiss/Daten/data_AGU/old/'+aggregation+'/MNI_2017_new_final.nc'
    state_mask = '/media/tweiss/Work/z_final_mni_data_2017/ESU'+aggregation+'.tif'

    s2_vwc = "/media/tweiss/Daten/NDWI1/sentinel_2_2017/NDWI1.nc"
    rad_api = '/media/tweiss/Daten/RADOLAN_API_v1.0.0.nc'

    sarsar = KaSKASAR(s1_ncfile, state_mask, s2_vwc, rad_api)

    csv_output_path = '/media/tweiss/Work/paper2/z_dense_s1_time_series_n7multi_Field_buffer_30/csv/'

    add_data = pd.read_csv(csv_output_path+'all_50.csv',header=[0,1,2,3,4,5],index_col=0)

    sarsar.sentinel1_inversion(True)

