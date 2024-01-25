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
from watercloudmodel import cost_function
from scipy.ndimage.filters import gaussian_filter1d
import pdb

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

def read_sar(sar_data, state_mask):
    s1_data = namedtuple('s1_data', 'time lat lon satellite  relorbit orbitdirection ang vv vh')

    ang = reproject_data(sar_data.ang_name, output_format="MEM", target_img=state_mask)
    vv  = reproject_data(sar_data.vv_name, output_format="MEM", target_img=state_mask)
    vh  = reproject_data(sar_data.vh_name, output_format="MEM", target_img=state_mask)

    time = [datetime.datetime(1970,1,1) + datetime.timedelta(days=float(i)) for i in  sar_data.time]
    return s1_data(time, sar_data.lat, sar_data.lon, sar_data.satellite, sar_data.relorbit, sar_data.orbitdirection, ang, vv, vh)

def read_s2_lai(s2_lai, s2_cab, s2_cbrown, state_mask):
    s2_data = namedtuple('s2_lai', 'time lai cab cbrown')
    g = gdal.Open(s2_lai)
    time = []
    for i in range(g.RasterCount):
        gg = g.GetRasterBand(i+1)
        meta = gg.GetMetadata()
        time.append(datetime.datetime.strptime(meta['DoY'], '%Y%j'))
    lai  = reproject_data(s2_lai, output_format="MEM", target_img=state_mask)
    cab  = reproject_data(s2_cab, output_format="MEM", target_img=state_mask)
    cbrown  = reproject_data(s2_cbrown, output_format="MEM", target_img=state_mask)
    return s2_data(time, lai, cab, cbrown)

def inference_preprocessing(s1_data, s2_data, state_mask, orbit1=None, orbit2=None):
    """Resample S2 smoothed output to match S1 observations
    times"""
    # Move everything to DoY to simplify interpolation

    sar_inference_data = namedtuple('sar_inference_data', 'time lat lon satellite  relorbit orbitdirection ang vv vh lai cab cbrown time_mask fields')


    s2_doys = np.array([ int(i.strftime('%j')) for i in s2_data.time])
    s1_doys = np.array([ int(i.strftime('%j')) for i in s1_data.time])

    time_mask = (s1_doys >= s2_doys.min()) & (s1_doys <= s2_doys.max())
    if orbit1 != None:
        rel_orbit1 = s1_data.relorbit==orbit1
        if orbit2 != None:
            rel_orbit2 = s1_data.relorbit==orbit2
            xxx = np.logical_and(rel_orbit1,time_mask)
            yyy = np.logical_and(rel_orbit2,time_mask)
            time_mask = np.logical_or(xxx,yyy)


    pdb.set_trace()
    f = interp1d(s2_doys, s2_data.lai.ReadAsArray(), axis=0, bounds_error=False)
    lai_s1 = f(s1_doys)
    f = interp1d(s2_doys, s2_data.cab.ReadAsArray(), axis=0, bounds_error=False)
    cab_s1 = f(s1_doys)
    f = interp1d(s2_doys, s2_data.cbrown.ReadAsArray(), axis=0, bounds_error=False)
    cbrown_s1 = f(s1_doys)
    # segmentation
    lai_max = np.nanmax(s2_data.lai.ReadAsArray(), axis=0)
    patches = sobel(lai_max)>0.001
    fields  = label(patches)[0]
    g = gdal.Open(state_mask)
    gg = g.GetRasterBand(1)
    ggg = gg.ReadAsArray()
    fields[ggg==0]=0
    sar_inference_data = sar_inference_data(s1_data.time, s1_data.lat, s1_data.lon,
                                            s1_data.satellite, s1_data.relorbit,
                                            s1_data.orbitdirection, s1_data.ang,
                                            s1_data.vv, s1_data.vh, lai_s1, cab_s1, cbrown_s1, time_mask, fields)

    return sar_inference_data


def get_prior(s1_data, soilMoisture, soilMoisture_std, soilRoughness, soilRoughness_std, state_mask):
    # this is the function to reading the soil moisture prior
    # and the soil roughness prior using the satemask
    # the assumption of inputs are daily data in geotifs
    prior = namedtuple('prior', 'time sm_prior sm_std sr_prior sr_std')

    g = gdal.Open(soilMoisture)
    time = []
    for i in range(g.RasterCount):
        gg = g.GetRasterBand(i+1)
        meta = gg.GetMetadata()
        time.append(datetime.datetime.strptime(meta['date'], '%Y-%m-%d'))
    sm_prior  = reproject_data(soilMoisture,     output_format="MEM", target_img=state_mask)
    sm_std    = reproject_data(soilMoisture_std, output_format="MEM", target_img=state_mask)
    sr_prior  = reproject_data(soilRoughness,    output_format="MEM", target_img=state_mask)
    sr_std    = reproject_data(soilRoughness_std,output_format="MEM", target_img=state_mask)

    prior_doy = np.array([ int(i.strftime('%j')) for i in time])
    s1_doys = np.array([ int(i.strftime('%j')) for i in s1_data.time])

    f = interp1d(prior_doy, sm_prior.ReadAsArray(), axis=0, bounds_error=False)

    sm_s1 = f(s1_doys)
    f = interp1d(prior_doy,   sm_std.ReadAsArray(), axis=0, bounds_error=False)
    sm_std_s1 = f(s1_doys)

    f = interp1d(prior_doy, sr_prior.ReadAsArray(), axis=0, bounds_error=False)
    sr_s1 = f(s1_doys)
    f = interp1d(prior_doy,   sr_std.ReadAsArray(), axis=0, bounds_error=False)
    sr_std_s1 = f(s1_doys)

    return prior(time, sm_s1, sm_std_s1, sr_s1, sr_std_s1)

def fresnel(eps, theta):
    theta = np.deg2rad(theta)
    num = (eps-1)*(np.sin(theta)**2 - eps*(1+np.sin(theta)**2))
    den = eps*np.cos(theta) + np.sqrt(eps - np.sin(theta)**2)
    den = den**2
    return np.abs(num/den)

def mv2eps(a, b, c, mv):
    eps = a + b * mv + c * mv**2
    return eps

def quad_approx_solver(a, b, c, theta, alphas):
    x = np.arange(0.01, 0.5, 0.01)
    p = np.polyfit(x, fresnel(mv2eps(a, b, c, x),theta.mean()), 2)
    # 2nd order polynomial
    #solve
    solutions = [np.roots([p[0], p[1], p[2]-aa]) for aa in alphas]
    return solutions


def do_one_pixel_field(sar_inference_data, vv, vh, lai, theta, time, sm, sm_std, sr, sr_std):

    orbits = sar_inference_data.relorbit[sar_inference_data.time_mask]

    lais   = []
    srs    = []
    alphas = []
    sms    = []
    ps     = []
    times  = []
    uorbits = np.unique(orbits)
    for orbit in uorbits:
        orbit_mask = orbits == orbit
        ovv, ovh, olai, otheta, otime = vv[orbit_mask], vh[orbit_mask], lai[orbit_mask], theta[orbit_mask], time[orbit_mask]
        osm, osm_std, osro, osro_std  = sm[orbit_mask], sm_std[orbit_mask], sr[orbit_mask], sr_std[orbit_mask]

        olai_std = np.ones_like(olai)*0.05

        alpha     = fresnel(mv2eps(1.99, 38.9, 11.5, osm), otheta)
        alpha_std = np.ones_like(alpha)*0.2

        soil_sigma_mask = olai < 1
        sigma_soil_vv_mu = np.mean(ovv[soil_sigma_mask])
        sigma_soil_vh_mu = np.mean(ovh[soil_sigma_mask])

        xvv = np.array([1, 0.5, sigma_soil_vv_mu])
        xvh = np.array([1, 0.5, sigma_soil_vh_mu])

        prior_mean = np.concatenate([[0,   ]*6, alpha,     osro,     olai])
        prior_unc  = np.concatenate([[10., ]*6, alpha_std, osro_std, olai_std])

        x0 = np.concatenate([xvv, xvh, alpha, osro, olai])

        bounds = (
            [[None, None]] * 6
          + [[0.1,   3.3]] * olai.shape[0]
          + [[0,     .03]] * olai.shape[0]
          + [[0,       8]] * olai.shape[0]
          )
        # pdb.set_trace()
        gamma = [500, 500]
        retval = minimize(cost_function,
                            x0,
                            args=(ovh, ovv, otheta, gamma, prior_mean, prior_unc),
                            jac=True,
                            bounds = bounds,
                            options={"disp": False},)

        posterious_lai   = retval.x[6+2*len(olai) : ]
        posterious_sr    = retval.x[6+len(olai)   : 6+2*len(olai)]
        posterious_alpha = retval.x[6             : 6+len(olai)]
        sols = np.array(quad_approx_solver(1.99, 38.9, 11.5, otheta, posterious_alpha)).min(axis=1)
        lais.append(posterious_lai)
        srs.append(posterious_sr)
        sms.append(sols)
        times.append(otime)
        ps.append(retval.x[:6])

    order = np.argsort(np.hstack(times))
    times  = np.hstack(times )[order]
    lais   = np.hstack(lais  )[order]
    srs    = np.hstack(srs   )[order]
    sms    = np.hstack(sms   )[order].real
    return times, lais, srs, sms, np.array(ps)

def do_inversion(sar_inference_data, prior, state_mask, segment=False):

    orbits = sar_inference_data.relorbit[sar_inference_data.time_mask]
    uorbits = np.unique(orbits)
    if segment:
        out_shape   = sar_inference_data.lai[sar_inference_data.time_mask].shape
        lai_outputs = np.zeros(out_shape )
        sm_outputs  = np.zeros(out_shape )
        sr_outputs  = np.zeros(out_shape )

        ps_shape = (len(uorbits),) +  sar_inference_data.lai.shape[1:]

        Avv_outputs  = np.zeros(ps_shape)
        Bvv_outputs  = np.zeros(ps_shape)
        Cvv_outputs  = np.zeros(ps_shape)

        Avh_outputs  = np.zeros(ps_shape)
        Bvh_outputs  = np.zeros(ps_shape)
        Cvh_outputs  = np.zeros(ps_shape)

        fields = np.unique(sar_inference_data.fields)[1:]
        for field in fields:
            # get per field data
            # with time mask as well
            field_mask = sar_inference_data.fields == field
            time  = np.array(sar_inference_data.time)[sar_inference_data.time_mask]

            lai   = sar_inference_data.lai[sar_inference_data.time_mask][:, field_mask]

            sm    = prior.sm_prior[sar_inference_data.time_mask][:, field_mask]
            sm_std= prior.sm_std  [sar_inference_data.time_mask][:, field_mask]

            sm[np.isnan(sm)] = 0.2
            sm_std[sm_std==0] = 0.5
            sm_std[np.isnan(sm_std)] = 0.5

            sr    = prior.sr_prior[sar_inference_data.time_mask][:, field_mask]
            sr_std= prior.sr_std  [sar_inference_data.time_mask][:, field_mask]
            sr[np.isnan(sr)] = 0.03
            sr_std[np.isnan(sr_std)] = 1

            vv    = sar_inference_data.vv.ReadAsArray()[sar_inference_data.time_mask][:, field_mask]
            vh    = sar_inference_data.vh.ReadAsArray()[sar_inference_data.time_mask][:, field_mask]
            theta = sar_inference_data.ang.ReadAsArray()[sar_inference_data.time_mask][:, field_mask]


            lai   = np.nanmean(lai, axis=1)
            vv    = np.nanmean(vv, axis=1)
            vh    = np.nanmean(vh, axis=1)
            theta = np.nanmean(theta, axis=1)

            sm     = np.nanmean(sm, axis=1)
            sm_std = np.nanmean(sm_std, axis=1)

            sr     = np.nanmean(sr, axis=1)
            sr_std = np.nanmean(sr_std, axis=1)

            vv = np.maximum(vv, 0.0001)
            vv = 10 * np.log10(vv)
            vh = np.maximum(vh, 0.0001)
            vh = 10 * np.log10(vh)

            times, lais, srs, sms, ps = do_one_pixel_field(sar_inference_data, vv, vh, lai, theta, time, sm, sm_std, sr, sr_std)

            lai_outputs[:, field_mask] = lais[...,None]
            sr_outputs[:, field_mask]  = srs [...,None]
            sm_outputs[:, field_mask]  = sms [...,None]

            for i in range(len(uorbits)):
                Avv_outputs[i, field_mask]  = ps[i,0]
                Bvv_outputs[i, field_mask]  = ps[i,1]
                Cvv_outputs[i, field_mask]  = ps[i,2]
                Avh_outputs[i, field_mask]  = ps[i,3]
                Bvh_outputs[i, field_mask]  = ps[i,4]
                Cvh_outputs[i, field_mask]  = ps[i,5]
    else:
        mask = gdal.Open(state_mask).ReadAsArray()
        xs, ys = np.where(mask)

        out_shape   = sar_inference_data.lai[sar_inference_data.time_mask].shape
        lai_outputs = np.zeros(out_shape )
        sm_outputs  = np.zeros(out_shape )
        sr_outputs  = np.zeros(out_shape )

        ps_shape = (len(uorbits),) +  sar_inference_data.lai.shape[1:]

        Avv_outputs  = np.zeros(ps_shape)
        Bvv_outputs  = np.zeros(ps_shape)
        Cvv_outputs  = np.zeros(ps_shape)

        Avh_outputs  = np.zeros(ps_shape)
        Bvh_outputs  = np.zeros(ps_shape)
        Cvh_outputs  = np.zeros(ps_shape)


        for i in range(len(xs)):
            indx, indy = xs[i], ys[i]

            # field_mask = slice(None, None), slice(indx, indx+1), slice(indy, indy+1)
            time  = np.array(sar_inference_data.time)[sar_inference_data.time_mask]
            lai   = sar_inference_data.lai[sar_inference_data.time_mask][:, indx, indy ]

            sm    = prior.sm_prior[sar_inference_data.time_mask][:, indx, indy ]
            sm_std= prior.sm_std  [sar_inference_data.time_mask][:, indx, indy ]

            sm[np.isnan(sm)] = 0.2
            sm_std[sm_std==0] = 0.5
            sm_std[np.isnan(sm_std)] = 0.5

            sr    = prior.sr_prior[sar_inference_data.time_mask][:, indx, indy ]
            sr_std= prior.sr_std  [sar_inference_data.time_mask][:, indx, indy ]
            sr[np.isnan(sr)] = 0.03
            sr_std[np.isnan(sr_std)] = 1

            vv    = sar_inference_data.vv.ReadAsArray()[sar_inference_data.time_mask][:, indx, indy ]
            vh    = sar_inference_data.vh.ReadAsArray()[sar_inference_data.time_mask][:, indx, indy ]
            theta = sar_inference_data.ang.ReadAsArray()[sar_inference_data.time_mask][:, indx, indy ]

            vv = np.maximum(vv, 0.0001)
            vv = 10 * np.log10(vv)
            vh = np.maximum(vh, 0.0001)
            vh = 10 * np.log10(vh)

            times, lais, srs, sms, ps = do_one_pixel_field(sar_inference_data, vv, vh, lai, theta, time, sm, sm_std, sr, sr_std)

            lai_outputs[:, indx, indy] = lais
            sr_outputs[:, indx, indy]  = srs
            sm_outputs[:, indx, indy]  = sms

            for i in range(len(uorbits)):
                Avv_outputs[i, indx, indy]  = ps[i,0]
                Bvv_outputs[i, indx, indy]  = ps[i,1]
                Cvv_outputs[i, indx, indy]  = ps[i,2]
                Avh_outputs[i, indx, indy]  = ps[i,3]
                Bvh_outputs[i, indx, indy]  = ps[i,4]
                Cvh_outputs[i, indx, indy]  = ps[i,5]
    return lai_outputs, sr_outputs, sm_outputs, Avv_outputs, Bvv_outputs, Cvv_outputs, Avh_outputs, Bvh_outputs, Cvh_outputs, uorbits

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

    def __init__(self, s1_ncfile, state_mask, s2_lai,  s2_cab, s2_cbrown, sm_prior, sm_std, sr_prior ,sr_std,orbit1=None,orbit2=None):
        self.s1_ncfile = s1_ncfile
        self.state_mask = state_mask
        self.s2_lai    = s2_lai
        self.s2_cab    = s2_cab
        self.s2_cbrown = s2_cbrown
        self.sm_prior  = sm_prior
        self.sm_std    = sm_std
        self.sr_prior  = sr_prior
        self.sr_std    = sr_std
        self.orbit1    = None
        self.orbit2    = None
        if orbit1 != None:
            self.orbit1     = orbit1
            if orbit2 != None:
                self.orbit2 = orbit2

    def sentinel1_inversion(self, segment=False):
        sar = get_sar(s1_ncfile)
        s1_data = read_sar(sar, self.state_mask)
        s2_data = read_s2_lai(self.s2_lai, self.s2_cab, self.s2_cbrown, self.state_mask)
        prior   = get_prior(s1_data, self.sm_prior, self.sm_std, self.sr_prior, self.sr_std, self.state_mask)
        sar_inference_data = inference_preprocessing(s1_data, s2_data, self.state_mask,self.orbit1,self.orbit2)

        lai_outputs, sr_outputs, sm_outputs, \
        Avv_outputs, Bvv_outputs, Cvv_outputs, \
        Avh_outputs, Bvh_outputs, Cvh_outputs, uorbits = do_inversion(sar_inference_data, prior, self.state_mask, segment)

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

        Avv_name = self.s1_ncfile.replace('.nc', '_Avv.tif')
        Bvv_name = self.s1_ncfile.replace('.nc', '_Bvv.tif')
        Cvv_name = self.s1_ncfile.replace('.nc', '_Cvv.tif')

        Avh_name = self.s1_ncfile.replace('.nc', '_Avh.tif')
        Bvh_name = self.s1_ncfile.replace('.nc', '_Bvh.tif')
        Cvh_name = self.s1_ncfile.replace('.nc', '_Cvh.tif')

        save_ps_output(Avv_name, Avv_outputs, geo, projction, uorbits)
        save_ps_output(Bvv_name, Bvv_outputs, geo, projction, uorbits)
        save_ps_output(Cvv_name, Cvv_outputs, geo, projction, uorbits)
        save_ps_output(Avh_name, Avh_outputs, geo, projction, uorbits)
        save_ps_output(Bvh_name, Bvh_outputs, geo, projction, uorbits)
        save_ps_output(Cvh_name, Cvh_outputs, geo, projction, uorbits)


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

    aggregation = '_point'
    # aggregation = '_Field_buffer_30'
    # aggregation = '_buffer_100'
    # aggregation = '_buffer_50'
    aggregation = '_buffer_30'

    s1_ncfile = '/media/tweiss/Daten/data_AGU/'+aggregation+'/MNI_2017_new_final.nc'
    state_mask = '/media/tweiss/Work/z_final_mni_data_2017/ESU'+aggregation+'.tif'
    s2_folder = "/media/tweiss/Daten/test_kaska/data/"
    s2_lai = f"{s2_folder:s}/lai.tif"
    s2_cab = f"{s2_folder:s}/cab.tif"
    s2_cbrown = f"{s2_folder:s}/cbrown.tif"

    sm_prior = f'{s2_folder:s}/sm_prior.tif'
    sm_std   = f'{s2_folder:s}/sm_std.tif'
    sr_prior = f'{s2_folder:s}/sr_prior.tif'
    sr_std   = f'{s2_folder:s}/sr_std.tif'
    sarsar = KaSKASAR(s1_ncfile, state_mask, s2_lai, s2_cab, s2_cbrown, sm_prior, sm_std, sr_prior ,sr_std)




    sarsar.sentinel1_inversion(True)

