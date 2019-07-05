#!/usr/bin/env python

import gdal
import osr
import numpy as np


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
