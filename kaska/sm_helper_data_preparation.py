
import os
import numpy as np
import datetime
from osgeo import gdal
from netCDF4 import Dataset
from netCDF4 import date2num
from collections import namedtuple
from utils import reproject_data
import glob
from scipy.interpolate import interp1d

def save_to_tif(fname, Array, GeoT):
    """
    save array as tif file

    :param fname: str
        name of output (tif) file
    :param Array: array
        contains geographic information
    :param GeoT: ???
        GeoTransform information
    :return: str
        name of output (tif) file
    """

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
    """
    convert self processed netcdf4 file stack of S1 images (SenSARP) to single tif images

    :param s1_nc_file: str
        name of netcdf4 file
    :param version: str
        layer extension based on naming during SenSARP processing
    :return: tuple
        information about time, lat, lon, satellite, relorbit, orbitdirection, ang_name, vv_name, vh_name


    Problem: Getting GeoTranformation from netcdf file!!!!!!!!!!!!!!!!!!!!! problem related to version of gdal???
    """

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
        geo = (12.878782781795174,0.000124866097153945,0.0,53.62977739815832,0.0,-0.0001248646760142416)
        sigma0_vv = data['sigma0_vv'+version][:]
        save_to_tif(vv_name, sigma0_vv, geo)


    if not os.path.exists(vh_name):
        gg = gdal.Open('NETCDF:"%s":sigma0_vh"%s"'%(s1_nc_file,version))
        geo = gg.GetGeoTransform()
        geo = (12.878782781795174, 0.000124866097153945, 0.0, 53.62977739815832, 0.0, -0.0001248646760142416)
        sigma0_vh = data['sigma0_vh'+version][:]
        save_to_tif(vh_name, sigma0_vh, geo)

    if not os.path.exists(ang_name):
        gg = gdal.Open('NETCDF:"%s":theta'%s1_nc_file)
        geo = gg.GetGeoTransform()
        geo = (12.878782781795174, 0.000124866097153945, 0.0, 53.62977739815832, 0.0, -0.0001248646760142416)
        localIncidenceAngle = data['theta'][:]
        save_to_tif(ang_name, localIncidenceAngle, geo)

    return s1_data(time, lat, lon, satellite, relorbit, orbitdirection, ang_name, vv_name, vh_name)

def read_sar(sar_data, state_mask):
    """
    get/reproject sar data and mask on same extent (grid)

    :param sar_data: tuple
        information from function get_sar (time, lat, lon, satellite, relorbit, orbitdirection, ang_name, vv_name, vh_name)
    :param state_mask: array ????
        mask
    :return: tuple
         information about time, lat, long, satellite, relorbit, orbitdirection, ang, vv_name, vh_name
    """

    s1_data = namedtuple('s1_data', 'time lat lon satellite  relorbit orbitdirection ang vv vh')
    ang = reproject_data(sar_data.ang_name, output_format="MEM", target_img=state_mask)
    vv  = reproject_data(sar_data.vv_name, output_format="MEM", target_img=state_mask)
    vh  = reproject_data(sar_data.vh_name, output_format="MEM", target_img=state_mask)
    time = [datetime.datetime(1970,1,1) + datetime.timedelta(days=float(i)) for i in  sar_data.time]
    return s1_data(time, sar_data.lat, sar_data.lon, sar_data.satellite, sar_data.relorbit, sar_data.orbitdirection, ang, vv, vh)

def get_api(api_nc_file,year):
    """
    extract one year and convert RADOLAN netcdf4 file information to tif

    :param api_nc_file: str
        name of netcdf4 file with RADOLAN sm prior information
    :param year: str
        year of interest
    :return: tuple
        information about time, lat, lon, api_name (tif file)
    """

    api_data = namedtuple('api_data', 'time lat lon api')
    data = Dataset(api_nc_file)

    xxx = date2num(datetime.datetime.strptime(year+'0201', '%Y%m%d'), units ='hours since 2000-01-01 00:00:00', calendar='gregorian')
    # yyy = date2num(datetime.datetime.strptime(year+'1001', '%Y%m%d'), units ='hours since 2000-01-01 00:00:00', calendar='gregorian')
    yyy = date2num(datetime.datetime.strptime(year+'0628', '%Y%m%d'), units ='hours since 2000-01-01 00:00:00', calendar='gregorian')

    time = data['time'][np.where(data['time'][:]==xxx)[0][0]:np.where(data['time'][:]==yyy)[0][0]]
    lat = data['lat'][:]
    lon = data['lon'][:]

    api_name = api_nc_file.replace('.nc', '_api'+year+'.tif')

    if not os.path.exists(api_name):
        gg = gdal.Open('NETCDF:"%s":api'%api_nc_file)
        geo = gg.GetGeoTransform()
        save_to_tif(api_name, data['api'][np.where(data['time'][:]==xxx)[0][0]:np.where(data['time'][:]==yyy)[0][0],:,:], geo)

    return api_data(time, lat, lon, api_name)

def read_api(api_data, state_mask):
    """
    get/reproject api data and mask on same extent (grid)

    :param api_data: tuple
        information from funtion get_api (time, lat, lon, api_name (tif file))
    :param state_mask: array ???
        mask
    :return: tuple
        information about time, lat, lon, api_name (tif file)
    """
    s1_data = namedtuple('api_data', 'time lat lon api')

    api = reproject_data(api_data.api, output_format="MEM", target_img=state_mask)
    time = [datetime.datetime(2000,1,1) + datetime.timedelta(hours=float(i)) for i in  api_data.time]

    return s1_data(time, api_data.lat, api_data.lon, api)

def read_vwc(vwc_data, state_mask):
    """
    get/reproject ndwi1 data and mask on same extent (grid)
    calculate vwc from ndwi1 data (current usage of empirical function Maggioni et al. 2006)

    :param vwc_data: str
        path to stored vwc tif files
    :param state_mask: array ???
        mask
    :return: tuple
        information about time, vwc content, ndwi content
    """
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

def ndwi1_mag(ndwi1):
    """
    formula to calculate vwc from ndwi1 after Maggioni et al. 2006

    :param ndwi1: array/float
        ndwi1 information
    :return: array/float
        vwc information
    """
    vwc = 13.2*ndwi1**2+1.62*ndwi1
    return vwc

def ndwi1_cos_maize(ndwi1):
    """
    formula to calculate vwc from ndwi1 after Cosh et al. 2006

    :param ndwi1: array/float
        ndwi1 information
    :return: array/float
        vwc information
    """
    vwc = 9.39*ndwi1+1.26
    return vwc

def get_sm_input(s1_ncfile, s1_version, s2_vwc, rad_api, state_mask, year):
    """
    load all input data for sm retrieval

    :param s1_ncfile: str
        name of netcdf4 file with stack of S1 images (pre processed with SenSARP package)
    :param s2_vwc: str
        name of folder with S2 tif files (pre processed with Google Earth Engine script)
    :param rad_api: str
        name of RADOLAN file with sm prior information
    :param state_mask: array
        mask information
    :param year: str
        year of interest (important for RADOLAN extraction)
    :return: arrays
        input data for retrieval (s1, vwc, api)
    """
    sar = get_sar(s1_ncfile, s1_version)
    print('get_sar:'+str(datetime.datetime.now()))
    s1_data = read_sar(sar, state_mask)
    print('read_sar:' + str(datetime.datetime.now()))
    vwc_data = read_vwc(s2_vwc, state_mask)
    print('read_vwc:' + str(datetime.datetime.now()))
    api = get_api(rad_api, year)
    print('get_api:' + str(datetime.datetime.now()))
    api_data = read_api(api, state_mask)
    print('read_api:' + str(datetime.datetime.now()))

    return s1_data, vwc_data, api_data

def inference_preprocessing(s1_data, vwc_data, api_data, output_folder, orbit1=None, orbit2=None):
    """
    Resample S2 smoothed output to match S1 observations times

    :param s1_data: tuple ????
        ....
    :param vwc_data: tuple ????
        ....
    :param api_data: tuple ????
        ....
    :param orbit1: str
        orbit number
    :param orbit2: str
        orbit number
    :return: tuple
        .....
    """
    # Move everything to DoY to simplify interpolation
    sar_inference_data = namedtuple('sar_inference_data', 'time lat lon satellite  relorbit orbitdirection ang vv vh vwc api time_mask ndwi')

    vwc_doys = np.array([ int(i.strftime('%j')) for i in vwc_data.time])
    s1_doys = np.array([ int(i.strftime('%j')) for i in s1_data.time])


    time = np.array(s1_data.time)
    for jj in range(len(s1_data.time)):
        time[jj] = s1_data.time[jj].replace(microsecond=0).replace(second=0).replace(minute=0)
        time[jj] = time[jj].replace(hour=0)

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
        time_mask = (s1_doys >= 0) & (s1_doys <= 365)

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

    print('inference_processing:' + str(datetime.datetime.now()))

    if output_folder != None:
        np.save(os.path.join(output_folder, 'time.npy'),s1_data.time)
        s1_data.lat.dump(os.path.join(output_folder, 'lat.npy'))
        s1_data.lon.dump(os.path.join(output_folder, 'lon.npy'))
        s1_data.satellite.dump(os.path.join(output_folder, 'satellite.npy'))
        s1_data.relorbit.dump(os.path.join(output_folder, 'relorbit.npy'))
        s1_data.orbitdirection.dump(os.path.join(output_folder, 'orbitdirection.npy'))
        #s1_data.ang.dump(os.path.join(output_folder, 'ang.npy'))
        #s1_data.vv.dump(os.path.join(output_folder, 'vv.npy'))
        #s1_data.vh.dump(os.path.join(output_folder, 'vh.npy'))
        vwc_s1.dump(os.path.join(output_folder, 'vwc.npy'))
        api_s1.dump(os.path.join(output_folder, 'api.npy'))
        time_mask.dump(os.path.join(output_folder, 'time_mask.npy'))
        ndwi_s1.dump(os.path.join(output_folder, 'ndwi.npy'))

    return sar_inference_data
