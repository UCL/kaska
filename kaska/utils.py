#!/usr/bin/env python
"""A set of useful helper functions.
"""

import datetime as dt
import logging
from pathlib import Path

from osgeo import gdal, ogr
from osgeo import osr

import numpy as np

from .TwoNN import Two_NN

from .constants import DEFAULT_BLOCK_SIZE

LOG = logging.getLogger(__name__)


def define_temporal_grid(start_date, end_date, temporal_grid_space):
    """Creates a temporal grid"""
    temporal_grid = [start_date + i*dt.timedelta(days=temporal_grid_space)
                     for i in range(int(np.ceil(366/temporal_grid_space)))
                     if start_date + i*dt.timedelta(days=temporal_grid_space)
                     <= end_date]
    return temporal_grid


def read_emulator(emulator_file="/home/ucfafyi/DATA/Prosail/prosail_2NN.npz"):
    """ Read file and get emulator object.
    """
    npz_file = np.load(str(emulator_file))
    emulator = Two_NN(Hidden_Layers=npz_file.f.Hidden_Layers,
                      Output_Layers=npz_file.f.Output_Layers)
    return emulator


def reproject_data(source_img,
                   target_img=None,
                   dst_srs=None,
                   src_srs=None,
                   src_no_data=np.nan,
                   dst_no_data=np.nan,
                   output_type=None,
                   output_format="MEM",
                   verbose=False,
                   xmin=None,
                   xmax=None,
                   ymin=None,
                   ymax=None,
                   x_res=None,
                   y_res=None,
                   x_size=None,
                   y_size=None,
                   resample=1,
                   ):

    """
    A method that uses a source image and a target image to
    reproject & clip the source image to match the extent,
    projection and resolution of the target image.

    """

    output_type = (
        gdal.GDT_Unknown if output_type is None else output_type
        )
    if src_no_data is None:
        try:
            src_no_data = " ".join(
                [
                    i.split("=")[1]
                    for i in gdal.Info(source_img).split("\n")
                    if " NoData" in i
                ]
            )
        except RuntimeError:
            src_no_data = None
    # If the output type is integer and destination nodata is nan
    # set it to 0 to avoid warnings
    if output_type <= 5 and np.isnan(dst_no_data):
        dst_no_data = 0

    if src_srs is not None:
        _my_src_srs = osr.SpatialReference()
        try:
            _my_src_srs.ImportFromEPSG(int(src_srs.split(":")[1]))
        except:
            _my_src_srs.ImportFromWkt(src_srs)
    else:
        _my_src_srs = None

    if (target_img is None) & (dst_srs is None):
        raise IOError(
            "Projection should be specified ether from "
            + "a file or a projection code."
        )
    if target_img is not None:
        try:
            dataset = gdal.Open(target_img)
        except RuntimeError:
            dataset = target_img
        geo_t = dataset.GetGeoTransform()
        x_size, y_size = dataset.RasterXSize, dataset.RasterYSize

        if x_res is None:
            x_res = abs(geo_t[1])
        if y_res is None:
            y_res = abs(geo_t[5])

        if x_size is not None:
            x_size = 1.0 * x_size * x_res / abs(geo_t[1])
        if y_size is not None:
            y_size = 1.0 * y_size * y_res / abs(geo_t[5])

        xmin, xmax = (
            min(geo_t[0], geo_t[0] + x_size * geo_t[1]),
            max(geo_t[0], geo_t[0] + x_size * geo_t[1]),
        )
        ymin, ymax = (
            min(geo_t[3], geo_t[3] + y_size * geo_t[5]),
            max(geo_t[3], geo_t[3] + y_size * geo_t[5]),
        )
        dst_srs = osr.SpatialReference()
        raster_wkt = dataset.GetProjection()
        dst_srs.ImportFromWkt(raster_wkt)
        warped = gdal.Warp(
            "",
            source_img,
            format=output_format,
            outputBounds=[xmin, ymin, xmax, ymax],
            dstNodata=dst_no_data,
            warpOptions=["NUM_THREADS=ALL_CPUS"],
            xRes=x_res,
            yRes=y_res,
            dstSRS=dst_srs,
            outputType=output_type,
            srcNodata=src_no_data,
            resampleAlg=resample,
            srcSRS=_my_src_srs
            )

    else:
        warped = gdal.Warp(
            "",
            source_img,
            format=output_format,
            outputBounds=[xmin, ymin, xmax, ymax],
            xRes=x_res,
            yRes=y_res,
            dstSRS=dst_srs,
            warpOptions=["NUM_THREADS=ALL_CPUS"],
            copyMetadata=True,
            outputType=output_type,
            dstNodata=dst_no_data,
            srcNodata=src_no_data,
            resampleAlg=resample,
            srcSRS=_my_src_srs
            )
    if verbose:
        LOG.debug("There are %d bands in this file, use "
                  "g.GetRasterBand(<band>) to avoid reading the whole file.",
                  warped.RasterCount
                  # % gg.RasterCount
                  )
    return warped


def save_output_parameters(time_grid, observations, output_folder,
                           parameter_names,
                           output_data, output_format="GTiff",
                           chunk=None, fname_pattern="s2",
                           options=None):
    """Saving the output parameters as (probably all times) GeoTIFFs
    """
    if options is None:
        options = ['COMPRESS=DEFLATE',
                   'BIGTIFF=YES',
                   'PREDICTOR=1',
                   'TILED=YES']

    output_folder = Path(output_folder)
    if not output_folder.exists():
        output_folder.mkdir(parents=True, exist_ok=True)
    assert len(parameter_names) == len(output_data)
    num_t = output_data[0].shape[0]
    assert len(time_grid) == num_t, f"time_grid length = {len(time_grid)}, " \
        + f"data length = {num_t}. output_data[0].shape = {output_data[0].shape}"
    projection, geo_transform, num_x, num_y = observations.define_output()
    drv = gdal.GetDriverByName(output_format)
    for (param, data) in zip(parameter_names, output_data):
        for band, tstep in enumerate(time_grid):
            this_date = tstep.strftime("%Y%j")

            # Compose the output file name from pattern (satellite),
            # parameters, dates and, optionally, chunk
            extra = ""
            if chunk:
                extra = f"_{chunk:s}"

            outfile = output_folder/f"{fname_pattern}_{param}_A{this_date} \
                {extra}.tif"

            if outfile.exists():
                outfile.unlink()
            # LOG.info(f"Saving file {str(outfile):s}...")
            LOG.info("Saving file %s...", str(outfile))
            dst_ds = drv.Create(str(outfile), num_x, num_y, 1,
                                gdal.GDT_Float32, options)
            dst_ds.SetProjection(projection)
            dst_ds.SetGeoTransform(geo_transform)
            band_x = dst_ds.GetRasterBand(1)
            band_x.WriteArray(data[band, :, :].astype(np.float32))
            band_x.SetMetadata({'parameter': param,
                                'date': time_grid[band].strftime("%Y-%m-%d"),
                                'doy': this_date})
            dst_ds = None
            dataset = gdal.Open(str(outfile))
            dataset.BuildOverviews("average", np.power(2, np.arange(6)))


def get_chunks(num_x, num_y, block_size=None):
    """An iterator to provide square chunks for an image. Basically,
    you pass this function the size of an array (doesn't need to be
    square!), the block size you want the chunks to have, and it will
    return an iterator with the window of interest.

    Parameters
    ----------
    num_x : int
        `x` size of the array in pixels.
    num_y : int
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
    # blocks = []

    if block_size is None:
        block_size = [DEFAULT_BLOCK_SIZE, DEFAULT_BLOCK_SIZE]  # [256, 256]

    nx_blocks = (int)((num_x + block_size[0] - 1) / block_size[0])
    ny_blocks = (int)((num_y + block_size[1] - 1) / block_size[1])
    nx_valid, ny_valid = block_size
    chunk_no = 0
    for my_x in range(nx_blocks):
        # change the block size of the final piece
        if my_x == nx_blocks - 1:
            nx_valid = num_x - my_x * block_size[0]
            # buf_size = nx_valid * ny_valid

        # find X offset
        this_x = my_x * block_size[0]

        # reset buffer size for start of Y loop
        ny_valid = block_size[1]
        # buf_size = nx_valid * ny_valid

        # loop through Y lines
        for my_y in range(ny_blocks):
            # change the block size of the final piece
            if my_y == ny_blocks - 1:
                ny_valid = num_y - my_y * block_size[1]
                # buf_size = nx_valid * ny_valid
            chunk_no += 1
            # find Y offset
            this_y = my_y * block_size[1]
            yield this_x, this_y, nx_valid, ny_valid, chunk_no


def rasterise_vector(vector_f, sample_f=None, pixel_size=20):
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
    # Reducing no. of locals to stop pylint complaint.
    # But that causes a seg fault when run pytest
    # source_layer = ogr.Open(vector_f).GetLayer()

    if sample_f is not None:
        dataset = gdal.Open(sample_f)
        geo_transform = dataset.GetGeoTransform()
        rows, cols = dataset.RasterXSize, dataset.RasterYSize
        target_ds_srs = dataset.GetProjectionRef()
    else:
        x_min, x_max, y_min, y_max = source_layer.GetExtent()
        cols = int((x_max - x_min) / pixel_size)
        rows = int((y_max - y_min) / pixel_size)
        geo_transform = [x_min, pixel_size, 0, y_max, 0, -pixel_size]
        target_ds_srs = osr.SpatialReference()
        target_ds_srs.ImportFromEPSG(4326)
        target_ds_srs = target_ds_srs.ExportToWkt()

    target_ds = gdal.GetDriverByName("MEM").Create(
                "", cols, rows, 1, gdal.GDT_Byte)
    target_ds.SetGeoTransform(geo_transform)
    target_ds.SetProjection(target_ds_srs)

    band = target_ds.GetRasterBand(1)
    band.SetNoDataValue(0)

    gdal.RasterizeLayer(target_ds, [1], source_layer, burn_values=[1])

    return target_ds
