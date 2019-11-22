#!/usr/bin/env python
import datetime as dt
import logging

from osgeo import gdal,ogr
from osgeo import osr

import numpy as np

from pathlib import Path

LOG = logging.getLogger(__name__)

def define_temporal_grid(start_date, end_date, temporal_grid_space):
    """Creates a temporal grid"""
    temporal_grid = [start_date + i*dt.timedelta(days=temporal_grid_space)
                    for i in range(int(np.ceil(366/temporal_grid_space)))
                    if start_date + i*dt.timedelta(days=temporal_grid_space)
                                    <= end_date]
    return temporal_grid

def read_emulator(emulator_file="/home/ucfafyi/DATA/Prosail/prosail_2NN.npz"):
    f = np.load(str(emulator_file))
    emulator = Two_NN(Hidden_Layers=f.f.Hidden_Layers,
                      Output_Layers=f.f.Output_Layers)
    return emulator

def reproject_data(source_img,
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



def save_output_parameters(time_grid, observations, output_folder, parameter_names,
                           output_data, output_format="GTiff",
                           chunk=None, fname_pattern="s2",
                           options=['COMPRESS=DEFLATE',
                                    'BIGTIFF=YES',
                                    'PREDICTOR=1',
                                    'TILED=YES']):
    """Saving the output parameters as (probably all times) GeoTIFFs
    """
    output_folder = Path(output_folder)
    if not output_folder.exists():
        output_folder.mkdir(parents=True, exist_ok=True)
    assert len(parameter_names) == len(output_data)
    nt = output_data[0].shape[0]
    assert len(time_grid) == nt, f"time_grid length = {len(time_grid)}, " + \
         f"data length = {nt}. output_data[0].shape = {output_data[0].shape}"
    projection, geo_transform, nx, ny = observations.define_output()
    drv = gdal.GetDriverByName(output_format)
    for (param, data) in zip(parameter_names, output_data):
        for band, tstep in enumerate(time_grid):
            this_date = tstep.strftime("%Y%j")
            
            # Compose the output file name from pattern (satellite),
            # parameters, dates and, optionally, chunk
            extra = ""
            if chunk:
                extra = f"_{chunk:s}"
            
            outfile = output_folder/f"{fname_pattern}_{param}_A{this_date}{extra}.tif"

            if outfile.exists():
                outfile.unlink()
            LOG.info(f"Saving file {str(outfile):s}...")
            dst_ds = drv.Create(str(outfile), nx, ny, 1,
                            gdal.GDT_Float32, options)
            dst_ds.SetProjection(projection)
            dst_ds.SetGeoTransform(geo_transform)
            x = dst_ds.GetRasterBand(1)
            x.WriteArray(data[band, :, :].astype(np.float32))
            x.SetMetadata({'parameter': param,
                            'date': time_grid[band].strftime("%Y-%m-%d"),
                            'doy':this_date})
            dst_ds = None
            g = gdal.Open(str(outfile))
            g.BuildOverviews("average", np.power(2, np.arange(6)))


def get_chunks(nx, ny, block_size= [256, 256]):
    """An iterator to provide square chunks for an image. Basically,
    you pass this function the size of an array (doesn't need to be
    square!), the block size you want the cuncks to have, and it will
    return an iterator with the window of interest.
    
    Parameters
    ----------
    nx : int
        `x` size of the array in pixels.
    ny : int
        `y` size of the array in pixels.
    block_size : list, optional
        Size of the blocks in `x` and `y` in pixels, by default [256, 256].

    Returns
    -------
    An iterator with `this_X`, `this_Y` (upper corner of selection window),
    `nx_valid`, `ny_valid` (number of valid pixels. Should be equal to
    `block_size` most of the time except it'll be smaller to cope with edges)
    and `chunk_no`, the chunk number.
    """
    blocks = []
    nx_blocks = (int)((nx + block_size[0] - 1) / block_size[0])
    ny_blocks = (int)((ny + block_size[1] - 1) / block_size[1])
    nx_valid, ny_valid = block_size
    chunk_no = 0
    for X in range(nx_blocks):
        # change the block size of the final piece
        if X == nx_blocks - 1:
            nx_valid = nx - X * block_size[0]
            buf_size = nx_valid * ny_valid

        # find X offset
        this_X = X * block_size[0]

        # reset buffer size for start of Y loop
        ny_valid = block_size[1]
        buf_size = nx_valid * ny_valid

        # loop through Y lines
        for Y in range(ny_blocks):
            # change the block size of the final piece
            if Y == ny_blocks - 1:
                ny_valid = ny - Y * block_size[1]
                buf_size = nx_valid * ny_valid
            chunk_no += 1
            # find Y offset
            this_Y = Y * block_size[1]
            yield this_X, this_Y, nx_valid, ny_valid, chunk_no


def rasterise_vector(vector_f, sample_f=None,  pixel_size=20):
    """Raterise a vector. Basically, pixels inside a polygon are set to 1,
    and those outside to 0. Two ways of going around this: either you 
    provide a sample raster file to define spatial extent and projection
    via `sample_f`, or you use the the extent of the vector file, and 
    provide the `pixel_size` (in vector file projection units). Note that
    if the vector dataset is a point or line vector file, it'll probably
    also work, but we expect most users to use polygons rather than e.g.
    points.
    
    Parameters
    ----------
    vector_f : str
        An OGR-readable vector dataset path. Pixels inside features
        will be set to 1.
    sample_f : str, optional
        A GDAL-readable raster dataset. If given, this is a sample
        dataset that defines rows, columns, extent and pixel spacing.
    pixel_size : int, optional
        The pixel size if not using, by default 20
    
    Returns
    -------
    GDAL object
        A GDAL object
    """
    source_ds = ogr.Open(vector_f)
    source_layer = source_ds.GetLayer()

    if sample_f is not None:
        g = gdal.Open(sample_f)
        geoT = g.GetGeoTransform()
        rows, cols = g.RasterXSize, g.RasterYSize
        target_dsSRS = g.GetProjectionRef()
    else:
        x_min, x_max, y_min, y_max = source_layer.GetExtent()
        cols = int((x_max - x_min) / pixel_size)
        rows = int((y_max - y_min) / pixel_size)
        geoT = [x_min, pixel_size, 0, y_max, 0, -pixel_size]
        target_dsSRS = osr.SpatialReference()
        target_dsSRS.ImportFromEPSG(4326)
        target_dsSRS = target_dsSRS.ExportToWkt()

    target_ds = gdal.GetDriverByName("MEM").Create(
                "", cols, rows, 1, gdal.GDT_Byte) 
    target_ds.SetGeoTransform(geoT)
    target_ds.SetProjection(target_dsSRS)

    band = target_ds.GetRasterBand(1) 
    band.SetNoDataValue(0) 

    gdal.RasterizeLayer(target_ds, [1], source_layer, burn_values=[1])
    
    return target_ds
