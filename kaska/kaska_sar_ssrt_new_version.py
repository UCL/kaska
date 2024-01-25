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
from watercloudmodel import cost_function2
from scipy.ndimage.filters import gaussian_filter1d
import pdb
from z_helper import *







def do_one_pixel_field(sar_inference_data, vv, vh, lai, theta, time, sm, sm_std, sr, sr_std, height):

    orbits = sar_inference_data.relorbit[sar_inference_data.time_mask]

    lais   = []
    coefs    = []
    alphas = []
    sms    = []
    ps     = []
    times  = []
    uorbits = np.unique(orbits)
    segmentation_by_orbit = 1

    if segmentation_by_orbit == 1:
        for orbit in uorbits:
            orbit_mask = orbits == orbit
            ovv, ovh, olai, otheta, otime = vv[orbit_mask], vh[orbit_mask], lai[orbit_mask], theta[orbit_mask], time[orbit_mask]
            osm, osm_std, osro, osro_std  = sm[orbit_mask], sm_std[orbit_mask], sr[orbit_mask], sr_std[orbit_mask]

            oheight = height[orbit_mask]

            olai_std = np.ones_like(olai)*0.05

            alpha = osm
            alpha_std = osm_std
            mv = alpha * 1
            coef = osro
            # coef[:] = 0.5

            # prior_mean = np.concatenate([alpha,     coef,     olai, oheight])
            # prior_unc  = np.concatenate([alpha_std, osro_std, olai_std, oheight])
            # x0 = np.concatenate([mv, coef, olai, oheight])

            # bounds = (
            #   [[0.01,   0.5]] * olai.shape[0]
            #   + [[0.01,     1.5]] * olai.shape[0]
            #   + [[0,       8]] * olai.shape[0]
            #   + [[0,       1]] * olai.shape[0]
            #   )

            prior_mean = np.concatenate([alpha,coef])
            prior_unc  = np.concatenate([alpha_std,osro_std])
            x0 = np.concatenate([mv,coef])
            data =  np.concatenate([oheight,olai])
            bounds = (
              [[0.1,   0.5]] * olai.shape[0]
              + [[0.01,     1.5]] * olai.shape[0]
              )

            gamma = [500, 500]

            retval = minimize(cost_function2,
                                x0,
                                args=(ovh, ovv, otheta, gamma, prior_mean, prior_unc, data),
                                jac=True,
                                bounds = bounds,
                                options={"disp": False},)

            # posterious_lai   = retval.x[2*len(olai) : 3*len(olai)]
            posterious_coef    = retval.x[len(olai)   : +2*len(olai)]
            posterious_mv = retval.x[             : +len(olai)]
            # lais.append(posterious_lai)
            coefs.append(posterious_coef)
            sms.append(posterious_mv)

            times.append(otime)

        order = np.argsort(np.hstack(times))
        times  = np.hstack(times )[order]
        # lais   = np.hstack(lais  )[order]
        lais=0
        coefs    = np.hstack(coefs   )[order]
        # coefs=0
        sms    = np.hstack(sms   )[order].real
    else:
        ovv, ovh, olai, otheta, otime = vv, vh, lai, theta, time
        osm, osm_std, osro, osro_std  = sm, sm_std, sr, sr_std

        oheight = height

        olai_std = np.ones_like(olai)*0.05

        alpha = osm
        alpha_std = osm_std
        mv = alpha * 1
        coef = osro

        prior_mean = np.concatenate([alpha,coef])
        prior_unc  = np.concatenate([alpha_std,osro_std])
        x0 = np.concatenate([mv,coef])
        data =  np.concatenate([oheight,olai])
        bounds = (
          [[0.1,   0.5]] * olai.shape[0]
          + [[0.01,     1.5]] * olai.shape[0]
          )

        gamma = [500, 500]

        retval = minimize(cost_function2,
                            x0,
                            args=(ovh, ovv, otheta, gamma, prior_mean, prior_unc, data),
                            jac=True,
                            bounds = bounds,
                            options={"disp": False},)

        # posterious_lai   = retval.x[2*len(olai) : 3*len(olai)]
        posterious_coef    = retval.x[len(olai)   : +2*len(olai)]
        posterious_mv = retval.x[             : +len(olai)]
        # lais.append(posterious_lai)
        coefs.append(posterious_coef)
        sms.append(posterious_mv)

        times.append(otime)


        order = np.argsort(np.hstack(times))
        times  = np.hstack(times )[order]
        # lais   = np.hstack(lais  )[order]
        lais=0
        coefs    = np.hstack(coefs   )[order]
        # coefs=0
        sms    = np.hstack(sms   )[order].real

    return times, lais, coefs, sms

def do_inversion(sar_inference_data, prior, state_mask, segment=False):

    orbits = sar_inference_data.relorbit[sar_inference_data.time_mask]
    uorbits = np.unique(orbits)
    if segment:
        out_shape   = sar_inference_data.lai[sar_inference_data.time_mask].shape
        lai_outputs = np.zeros(out_shape )
        sm_outputs  = np.zeros(out_shape )
        coef_outputs  = np.zeros(out_shape )

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

            coef    = prior.sr_prior[sar_inference_data.time_mask][:, field_mask]
            coef_std= prior.sr_std  [sar_inference_data.time_mask][:, field_mask]

            height = prior.sm_prior[sar_inference_data.time_mask][:, field_mask]
            height[:] = 0.1

            # coef[:] = 0.2
            coef_std[:] = 0.5

            coef[np.isnan(coef)] = 0.1
            coef_std[np.isnan(coef_std)] = 0.5

            vv    = sar_inference_data.vv.ReadAsArray()[sar_inference_data.time_mask][:, field_mask]
            vh    = sar_inference_data.vh.ReadAsArray()[sar_inference_data.time_mask][:, field_mask]
            theta = sar_inference_data.ang.ReadAsArray()[sar_inference_data.time_mask][:, field_mask]


            for jj in range(len(time)):
                time[jj] = time[jj].replace(microsecond=0).replace(second=0).replace(minute=0).replace(hour=0)

            start_date = pd.to_datetime(add_data.index)[0].to_pydatetime().replace(microsecond=0).replace(second=0).replace(minute=0).replace(hour=0)
            end_date =  pd.to_datetime(add_data.index)[-1].to_pydatetime().replace(microsecond=0).replace(second=0).replace(minute=0).replace(hour=0)
            if field == 1:
                add_lai = add_data.filter(like='LAI_insitu').filter(like='301_high').filter(like='Oh04').filter(like='turbid_isotropic').values.flatten()
                add_coef =  add_data.filter(like='coef').filter(like='301_high').filter(like='Oh04').filter(like='turbid_isotropic').values.flatten()
                add_height =  add_data.filter(like='height').filter(like='301_high').filter(like='Oh04').filter(like='turbid_isotropic').values.flatten()
            elif field == 4:
                add_lai = add_data.filter(like='LAI_insitu').filter(like='542_high').filter(like='Oh04').filter(like='turbid_isotropic').values.flatten()
                add_coef =  add_data.filter(like='coef').filter(like='542_high').filter(like='Oh04').filter(like='turbid_isotropic').values.flatten()
                add_height =  add_data.filter(like='height').filter(like='542_high').filter(like='Oh04').filter(like='turbid_isotropic').values.flatten()
            elif field == 5:
                add_lai = add_data.filter(like='LAI_insitu').filter(like='508_high').filter(like='Oh04').filter(like='turbid_isotropic').values.flatten()
                add_coef =  add_data.filter(like='coef').filter(like='508_high').filter(like='Oh04').filter(like='turbid_isotropic').values.flatten()
                add_height =  add_data.filter(like='height').filter(like='508_high').filter(like='Oh04').filter(like='turbid_isotropic').values.flatten()
            else:
                pass
            # elif field == 3:
            #     add_lai = add_data.filter(like='LAI_insitu').filter(like='515_high').filter(like='Oh04').filter(like='turbid_isotropic').values.flatten()
            #     add_coef =  add_data.filter(like='coef').filter(like='515_high').filter(like='Oh04').filter(like='turbid_isotropic').values.flatten()
            # else:
            #     add_lai = add_data.filter(like='LAI_insitu').filter(like='319_high').filter(like='Oh04').filter(like='turbid_isotropic').values.flatten()
            #     add_coef =  add_data.filter(like='coef').filter(like='319_high').filter(like='Oh04').filter(like='turbid_isotropic').values.flatten()


            lai   = np.nanmean(lai, axis=1)
            lai[(start_date <= time) & (end_date >= time)] = add_lai
            vv    = np.nanmean(vv, axis=1)
            vh    = np.nanmean(vh, axis=1)
            theta = np.nanmean(theta, axis=1)

            sm     = np.nanmean(sm, axis=1)
            sm_std = np.nanmean(sm_std, axis=1)

            coef     = np.nanmean(coef, axis=1)
            coef[(start_date <= time) & (end_date >= time)] = add_coef

            coef_std = np.nanmean(coef_std, axis=1)

            height = coef + 1
            height[(start_date <= time) & (end_date >= time)] = add_height

            vv = np.maximum(vv, 0.0001)
            vv = 10 * np.log10(vv)
            vh = np.maximum(vh, 0.0001)
            vh = 10 * np.log10(vh)

            times, lais, coefs, sms = do_one_pixel_field(sar_inference_data, vv, vh, lai, theta, time, sm, sm_std, coef, coef_std, height)

            lai_outputs[:, field_mask] = sms[...,None]

            coef_outputs[:, field_mask]  = coefs [...,None]
            sm_outputs[:, field_mask]  = sms [...,None]

    else:
        mask = gdal.Open(state_mask).ReadAsArray()
        xs, ys = np.where(mask)

        out_shape   = sar_inference_data.lai[sar_inference_data.time_mask].shape
        time  = np.array(sar_inference_data.time)[sar_inference_data.time_mask]
        lai_outputs = np.zeros(out_shape )
        sm_outputs  = np.zeros(out_shape )
        coef_outputs  = np.zeros(out_shape )

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

            coef    = prior.sr_prior[sar_inference_data.time_mask][:, indx, indy ]
            coef_std= prior.sr_std  [sar_inference_data.time_mask][:, indx, indy ]
            sr[np.isnan(sr)] = 0.1
            sr_std[np.isnan(sr_std)] = 0.5

            height = prior.sr_prior[sar_inference_data.time_mask][:, indx, indy ]
            height[:] = 0.1

            vv    = sar_inference_data.vv.ReadAsArray()[sar_inference_data.time_mask][:, indx, indy ]
            vh    = sar_inference_data.vh.ReadAsArray()[sar_inference_data.time_mask][:, indx, indy ]
            theta = sar_inference_data.ang.ReadAsArray()[sar_inference_data.time_mask][:, indx, indy ]


            vv = np.maximum(vv, 0.0001)
            vv = 10 * np.log10(vv)
            vh = np.maximum(vh, 0.0001)
            vh = 10 * np.log10(vh)

            times, lais, coefs, sms = do_one_pixel_field(sar_inference_data, vv, vh, lai, theta, time, sm, sm_std, coef, coef_std, height)

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
        self.orbit1     = None
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

        lai_outputs, sr_outputs, sm_outputs, uorbits = do_inversion(sar_inference_data, prior, self.state_mask, segment)

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
    aggregation = '_buffer_50'


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

    csv_output_path = '/media/tweiss/Work/paper2/z_dense_s1_time_series_n7multi_Field_buffer_30/csv/'

    add_data = pd.read_csv(csv_output_path+'all_50.csv',header=[0,1,2,3,4,5],index_col=0)

    sarsar.sentinel1_inversion(True)

