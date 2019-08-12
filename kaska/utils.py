#!/usr/bin/env python
import logging

import gdal
import osr
import numpy as np

from pathlib import Path

LOG = logging.getLogger(__name__ + ".Sentinel2_Observations")
LOG.setLevel(logging.INFO)
if not LOG.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - ' +
                                  '%(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    LOG.addHandler(ch)
LOG.propagate = False

def read_emulator(emulator_file="/home/ucfafyi/DATA/Prosail/prosail_2NN.npz"):
    f = np.load(str(emulator_file))
    emulator = Two_NN(Hidden_Layers=f.f.Hidden_Layers,
                      Output_Layers=f.f.Output_Layers)
    return emulator

def reproject_data(source_img,
        target_img=None,
        dstSRS=None,
        srcNodata=np.nan,
        dstNodata=np.nan,
        outputType=None,
        verbose=False,
        xmin=None,
        xmax=None,
        ymin=None,
        ymax=None,
        xRes=None,
        yRes=None,
        xSize=None,
        ySize=None,
        resample=1,
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
                format="MEM",
                outputBounds=[xmin, ymin, xmax, ymax],
                dstNodata=dstNodata,
                warpOptions=["NUM_THREADS=ALL_CPUS"],
                xRes=xRes,
                yRes=yRes,
                dstSRS=dstSRS,
                outputType=outputType,
                srcNodata=srcNodata,
                resampleAlg=resample,
            )

    else:
            gg = gdal.Warp(
                "",
                source_img,
                format="MEM",
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
            )
    if verbose:
        print("There are %d bands in this file, use "
                + "g.GetRasterBand(<band>) to avoid reading the whole file."
                % gg.RasterCount
            )
    return gg



def save_output_parameters(time_grid, observations, output_folder, parameter_names,
                           output_data, output_format="GTiff",
                           chunk=None,
                           options=['COMPRESS=DEFLATE',
                                    'BIGTIFF=YES',
                                    'PREDICTOR=1',
                                    'TILED=YES']):
    """Saving the output parameters as (probably all times) GeoTIFFs
    """
    output_folder = Path(output_folder)
    assert len(parameter_names) == len(output_data)
    nt = output_data[0].shape[0]
    assert len(time_grid) == nt
    projection, geo_transform, nx, ny = observations.define_output()
    drv = gdal.GetDriverByName(output_format)
    for (param, data) in zip(parameter_names, output_data):
        if chunk is None:
            outfile = output_folder/f"s2_{param:s}.tif"
        else:
            outfile = output_folder/f"s2_{param:s}_{chunk:s}.tif"
        if outfile.exists():
            outfile.unlink()
        LOG.info(f"Saving file {str(outfile):s}...")
        dst_ds = drv.Create(str(outfile), nx, ny, nt,
                        gdal.GDT_Float32, options)
        dst_ds.SetProjection(projection)
        dst_ds.SetGeoTransform(geo_transform)
        for band in range(nt):
            x = dst_ds.GetRasterBand(band+1)
            x.WriteArray(data[band, :, :].astype(np.float32))
            x.SetMetadata({'parameter': param,
                           'date': time_grid[band].strftime("%Y-%m-%d")})
        dst_ds = None
