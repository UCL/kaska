import os
import osr
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
import matplotlib
import subprocess

def reproject_data2(source_img,
        target_img=None,
        dstSRS=None,
        srcSRS=None,
        srcNodata=np.nan,
        dstNodata=np.nan,
        outputType=None,
        output_format="MEM",
        verbose=False,
        xmin=None,
        xmax=None,
        ymin=None,
        ymax=None,
        xRes=None,
        yRes=None,
        xSize=None,
        ySize=None,
        resample=0,
    ):

    """
    A method that uses a source and a target images to
    reproject & clip the source image to match the extent,
    projection and resolution of the target image.

    """

    outputType = (
        gdal.GDT_Unknown if outputType is None else outputType
        )
    if srcNodata is None:
            try:
                srcNodata = " ".join(
                    [
                        i.split("=")[1]
                        for i in gdal.Info(source_img).split("\n")
                        if " NoData" in i
                    ]
                )
            except RuntimeError:
                srcNodata = None
    # If the output type is intenger and destination nodata is nan
    # set it to 0 to avoid warnings
    if outputType <= 5 and np.isnan(dstNodata):
        dstNodata = 0

    if srcSRS is not None:
        _srcSRS = osr.SpatialReference()
        try:
            _srcSRS.ImportFromEPSG(int(srcSRS.split(":")[1]))
        except:
            _srcSRS.ImportFromWkt(srcSRS)
    else:
        _srcSRS = None


    if (target_img is None) & (dstSRS is None):
            raise IOError(
                "Projection should be specified ether from "
                + "a file or a projection code."
            )
    elif target_img is not None:
            try:
                g = gdal.Open(target_img)
            except RuntimeError:
                g = target_img
            geo_t = g.GetGeoTransform()
            x_size, y_size = g.RasterXSize, g.RasterYSize

            if xRes is None:
                xRes = abs(geo_t[1])
            if yRes is None:
                yRes = abs(geo_t[5])

            if xSize is not None:
                x_size = 1.0 * xSize * xRes / abs(geo_t[1])
            if ySize is not None:
                y_size = 1.0 * ySize * yRes / abs(geo_t[5])

            xmin, xmax = (
                min(geo_t[0], geo_t[0] + x_size * geo_t[1]),
                max(geo_t[0], geo_t[0] + x_size * geo_t[1]),
            )
            ymin, ymax = (
                min(geo_t[3], geo_t[3] + y_size * geo_t[5]),
                max(geo_t[3], geo_t[3] + y_size * geo_t[5]),
            )
            dstSRS = osr.SpatialReference()
            raster_wkt = g.GetProjection()
            dstSRS.ImportFromWkt(raster_wkt)
            gg = gdal.Warp(
                "",
                source_img,
                format=output_format,
                outputBounds=[xmin, ymin, xmax, ymax],
                dstNodata=dstNodata,
                warpOptions=["NUM_THREADS=ALL_CPUS"],
                xRes=xRes,
                yRes=yRes,
                dstSRS=dstSRS,
                outputType=outputType,
                srcNodata=srcNodata,
                resampleAlg=resample,
                srcSRS=_srcSRS
            )

    else:
            gg = gdal.Warp(
                "",
                source_img,
                format=output_format,
                outputBounds=[xmin, ymin, xmax, ymax],
                xRes=xRes,
                yRes=yRes,
                dstSRS=dstSRS,
                warpOptions=["NUM_THREADS=ALL_CPUS"],
                copyMetadata=True,
                outputType=outputType,
                dstNodata=dstNodata,
                srcNodata=srcNodata,
                resampleAlg=resample,
                srcSRS=_srcSRS
            )
    if verbose:
        LOG.debug("There are %d bands in this file, use "
                + "g.GetRasterBand(<band>) to avoid reading the whole file."
                % gg.RasterCount
            )
    return gg



def plot(input,min,max,name,path,times,mask=None):

    for i in range(len(input)):
        fig, ax = plt.subplots(figsize=(20, 15))
        input = np.ma.masked_where(input == 0.,input)
        current_cmap = matplotlib.cm.get_cmap()
        current_cmap.set_bad(color='white')

        try:
            hm = input[i]
            hm[mask] = np.nan
            quadmesh = ax.imshow(hm)
            plt.colorbar(quadmesh)
            quadmesh.set_clim(vmin=min, vmax=max)
            plt.savefig(os.path.join(path,name+'_mask',name+'_'+times[i].strftime("%Y%m%d")), bbox_inches = 'tight')
        except TypeError:
            quadmesh = ax.imshow(input[i])
            plt.colorbar(quadmesh)
            quadmesh.set_clim(vmin=min, vmax=max)
            plt.savefig(os.path.join(path,name,name+'_'+times[i].strftime("%Y%m%d")), bbox_inches = 'tight')
        plt.close()

def scatterplot(input1,input2,fields,esus):
    for field in fields:
        aaa = input1.filter(like=field)
        bbb = input2.filter(like=field)

        fig, ax = plt.subplots(figsize=(20, 15))
        colors=['blue','green','red']
        for u, esu in enumerate(esus):
            ccc = aaa.filter(like=esu).values.flatten()
            ddd = bbb.filter(like=esu).values.flatten()
            ax.plot(ccc,ddd,marker='o',color=colors[u], linestyle='')
            ax.set_xlim(0.05,0.4)
            ax.set_ylim(0.05,0.4)
        x = [0, 1]
        y = [0, 1]
        ax.plot(x,y)
        plt.ylabel('SM model')
        plt.xlabel('SM insitu')
        plt.title(field)
        plt.savefig('/media/tweiss/Work/Paper3_down/2017/plot/scatterplot/scatterplot_'+field,bbox_inches='tight')
        plt.close()

def scatterplot_bias(input1,input2,fields,esus):
    for field in fields:
        aaa = input1.filter(like=field)
        bbb = input2.filter(like=field)

        fig, ax = plt.subplots(figsize=(20, 15))
        colors=['blue','green','red']
        for u, esu in enumerate(esus):
            ccc = aaa.filter(like=esu).values.flatten()
            ddd = bbb.filter(like=esu).values.flatten()
            bias = np.nanmean(ccc - ddd)
            ax.plot(ccc,ddd+bias,marker='o',color=colors[u], linestyle='')

        ax.set_xlim(0.05,0.4)
        ax.set_ylim(0.05,0.4)
        x = [0, 1]
        y = [0, 1]
        ax.plot(x,y)
        plt.ylabel('SM model')
        plt.xlabel('SM insitu')
        plt.title(field+' bias corrected')
        plt.savefig('/media/tweiss/Work/Paper3_down/2017/plot/scatterplot/scatterplot_bias_'+field,bbox_inches='tight')
        plt.close()

# subprocess.call('gdal_rasterize -at -of GTiff -a field -te 694748 5345900 703600 5354600 -tr 10 10 -ot Byte -co \"COMPRESS=DEFLATE\" '+'/media/tweiss/Work/Paper3_down/GIS/sgm2017.shp'+' /media/tweiss/Work/Paper3_down/GIS/sgm2017_frucht.tif', shell=True)

# state_mask = '/media/tweiss/Work/Paper3_down/clc5_class2xx_2018.tif'

# mask_frucht = reproject_data('/media/tweiss/Work/Paper3_down/GIS/sgm2017_frucht.tif', output_format="MEM", target_img=state_mask)
# mask_frucht = mask_frucht.ReadAsArray().astype(np.int)


# mask = (mask_frucht != 2) & (mask_frucht != 3)

path = '/media/tweiss/Work/Paper3_down/2017/plot'
times = np.load('/media/tweiss/Work/Paper3_down/2017/times.npy',allow_pickle=True)

# plot(np.load('/media/tweiss/Work/Paper3_down/2017/input_sm_api.npy'),0.1,0.4,'input_sm',path,times,mask)
# plot(10*np.log10(np.load('/media/tweiss/Work/Paper3_down/2017/input_vv.npy')),-25,-5,'input_vv',path,times,mask)
# plot(np.load('/media/tweiss/Work/Paper3_down/2017/input_vwc.npy'),0.0,5,'input_vwc',path,times,mask)

# plot(np.load('/media/tweiss/Work/Paper3_down/2017/output_sm.npy'),0.1,0.4,'output_sm',path,times,mask)
# plot(np.load('/media/tweiss/Work/Paper3_down/2017/output_rms.npy'),0.005,0.03,'output_rms',path,times,mask)
# plot(np.load('/media/tweiss/Work/Paper3_down/2017/output_vwc.npy'),0.0,5,'output_vwc',path,times,mask)
# plot(np.load('/media/tweiss/Work/Paper3_down/2017/output_b.npy'),0.0,0.6,'output_b',path,times,mask)


mask_default = '/media/tweiss/Work/Paper3_down/clc5_class2xx_2018.tif'

pixel = ['_buffer_50']
pixel = ['_buffer_100']

processed_sentinel = ['multi']

fields = ['301', '508', '542', '319', '515']
fields = ['508','301','542']
# ESU names
esus = ['high', 'low', 'med']

df_model = pd.DataFrame()
df_insitu = pd.DataFrame()


for processed_sentinel_data in processed_sentinel:

    for pixels in pixel:
        print(pixels)
        path_ESU = '/media/tweiss/Work/z_final_mni_data_2017/'
        name_shp = 'ESU'+pixels+'.shp'
        name_ESU = 'ESU'+pixels+'.tif'
        mask_esu = reproject_data2(path_ESU+name_ESU, output_format="MEM", target_img=mask_default)



        for field in fields:
            for esu in esus:
                state_mask = mask_esu.ReadAsArray().astype(np.float)

                if field == '515' and esu == 'high':
                    mask_value = 1
                    state_mask = state_mask==mask_value
                elif field == '515' and esu == 'med':
                    mask_value = 2
                    state_mask = state_mask==mask_value
                elif field == '515' and esu == 'low':
                    mask_value = 3
                    state_mask = state_mask==mask_value
                elif field == '508' and esu == 'high':
                    mask_value = 4
                    state_mask = state_mask==mask_value
                elif field == '508' and esu == 'med':
                    mask_value = 5
                    state_mask = state_mask==mask_value
                elif field == '508' and esu == 'low':
                    mask_value = 6
                    state_mask = state_mask==mask_value
                elif field == '542' and esu == 'high':
                    mask_value = 7
                    state_mask = state_mask==mask_value
                elif field == '542' and esu == 'med':
                    mask_value = 8
                    state_mask = state_mask==mask_value
                elif field == '542' and esu == 'low':
                    mask_value = 9
                    state_mask = state_mask==mask_value
                elif field == '319' and esu == 'high':
                    mask_value = 10
                    state_mask = state_mask==mask_value
                elif field == '319' and esu == 'med':
                    mask_value = 11
                    state_mask = state_mask==mask_value
                elif field == '319' and esu == 'low':
                    mask_value = 12
                    state_mask = state_mask==mask_value
                elif field == '301' and esu == 'high':
                    mask_value = 13
                    state_mask = state_mask==mask_value
                elif field == '301' and esu == 'med':
                    mask_value = 14
                    state_mask = state_mask==mask_value
                elif field == '301' and esu == 'low':
                    mask_value = 15
                    state_mask = state_mask==mask_value
                else:
                    state_mask = 0



                sm = np.load('/media/tweiss/Work/Paper3_down/2017/output_sm.npy')
                sm = sm[:,state_mask]

                # sm = np.mean(sm,axis=1)
                sm = sm[:,0]

                file = '/media/tweiss/Work/z_final_mni_data_2017/new_in_situ_s1multi_buffer_100_2017_paper3.csv'
                data = pd.read_csv(file,header=[0,1],index_col=1)
                data_field = data.filter(like=field).filter(like=esu).filter(like='SM')
                data_field.index = pd.to_datetime(data_field.index)
                date = data_field.index
                sm_insitu= data_field[times[0]:times[-1]].values.flatten()

                df_insitu[field+'_'+esu+'_sminsitu'] = sm_insitu
                df_model[field+'_'+esu+'_smmodel'] = sm[1:-1]


scatterplot(df_insitu,df_model,fields,esus)
scatterplot_bias(df_insitu,df_model,fields,esus)


