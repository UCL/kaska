#!/usr/bin/env python
"""Dealing with S2 observations"""


import datetime as dt
import os
import sys

import numpy as np
import scipy.sparse as sp # Required for unc
import gdal
import osr

import xml.etree.ElementTree as ET

from collections import namedtuple

from pathlib import Path

import logging

from Two_NN import Two_NN

LOG = logging.getLogger(__name__+".Sentinel2_Observations")

# A SIAC data storage type
S2MSIdata = namedtuple('S2MSIdata',
                     'observations uncertainty mask metadata emulator')

def parse_xml(filename):
    """Parses the XML metadata file to extract view/incidence 
    angles. The file has grids and all sorts of stuff, but
    here we just average everything, and you get 
    1. SZA
    2. SAA 
    3. VZA
    4. VAA.
    """
    with open(filename, 'r') as f:
        tree = ET.parse(filename)
        root = tree.getroot()

        vza = []
        vaa = []
        for child in root:
            for x in child.findall("Tile_Angles"):
                for y in x.find("Mean_Sun_Angle"):
                    if y.tag == "ZENITH_ANGLE":
                        sza = float(y.text)
                    elif y.tag == "AZIMUTH_ANGLE":
                        saa = float(y.text)
                for s in x.find("Mean_Viewing_Incidence_Angle_List"):
                    for r in s:
                        if r.tag == "ZENITH_ANGLE":
                            vza.append(float(r.text))
                            
                        elif r.tag == "AZIMUTH_ANGLE":
                            vaa.append(float(r.text))
                            
    return sza, saa, np.mean(vza), np.mean(vaa)


class reproject_data(object):
    '''
    A function uses a source and a target images and 
    and clip the source image to match the extend, 
    projection and resolution as the target image.

    '''
    def __init__(self, source_img,
                 target_img   = None,
                 dstSRS       = None,
                 srcNodata    = np.nan,
                 dstNodata    = np.nan,
                 outputType   = None,
                 verbose      = False,
                 xmin         = None,
                 xmax         = None,
                 ymin         = None, 
                 ymax         = None,
                 xRes         = None,
                 yRes         = None,
                 xSize        = None,
                 ySize        = None,
                 resample     = 1
                 ):

        self.source_img = source_img
        self.target_img = target_img
        self.verbose    = verbose
        self.dstSRS     = dstSRS
        self.srcNodata  = srcNodata
        self.dstNodata  = dstNodata
        self.outputType = gdal.GDT_Unknown if outputType is None else outputType
        self.xmin       = xmin
        self.xmax       = xmax
        self.ymin       = ymin
        self.ymax       = ymax
        self.xRes       = xRes
        self.yRes       = yRes
        self.xSize      = xSize
        self.ySize      = ySize
        self.resample   = resample
        if self.srcNodata is None:
            try:                           
                self.srcNodata = ' '.join([i.split("=")[1] for i in gdal.Info(self.source_img).split('\n') if' NoData' in i])
            except:                        
                self.srcNodata = None
        if (self.target_img is None) & (self.dstSRS is None):
            raise IOError('Projection should be specified ether from a file or a projection code.')
        elif self.target_img is not None:
            try:
                g     = gdal.Open(self.target_img)
            except:
                g     = target_img
            geo_t = g.GetGeoTransform()
            x_size, y_size = g.RasterXSize, g.RasterYSize     

            if self.xRes is None:
                self.xRes = abs(geo_t[1])
            if self.yRes is None:
                self.yRes = abs(geo_t[5])

            if self.xSize is not None: 
                x_size = 1. * self.xSize * self.xRes / abs(geo_t[1])
            if self.ySize is not None: 
                y_size = 1. * self.ySize * self.yRes / abs(geo_t[5])

            xmin, xmax = min(geo_t[0], geo_t[0] + x_size * geo_t[1]), \
                         max(geo_t[0], geo_t[0] + x_size * geo_t[1])  
            ymin, ymax = min(geo_t[3], geo_t[3] + y_size * geo_t[5]), \
                         max(geo_t[3], geo_t[3] + y_size * geo_t[5])
            dstSRS     = osr.SpatialReference( )
            raster_wkt = g.GetProjection()
            dstSRS.ImportFromWkt(raster_wkt)
            self.g = gdal.Warp('', self.source_img, format = 'MEM', 
                                outputBounds = [xmin, ymin, xmax, ymax], dstNodata=self.dstNodata, warpOptions = ['NUM_THREADS=ALL_CPUS'],\
                                xRes = self.xRes, yRes = self.yRes, dstSRS = dstSRS, outputType = self.outputType, srcNodata = self.srcNodata, resampleAlg = self.resample)
            
        else:
            self.g = gdal.Warp('', self.source_img, format = 'MEM', outputBounds = [self.xmin, self.ymin, \
                               self.xmax, self.ymax], xRes = self.xRes, yRes = self.yRes, dstSRS = self.dstSRS, warpOptions = ['NUM_THREADS=ALL_CPUS'],\
                               copyMetadata=True, outputType = self.outputType, dstNodata=self.dstNodata, srcNodata = self.srcNodata, resampleAlg = self.resample)
        if self.g.RasterCount <= 3:
            self.data = self.g.ReadAsArray()
            #return self.data
        elif self.verbose:
            print('There are %d bands in this file, use g.GetRasterBand(<band>) to avoid reading the whole file.'%self.g.RasterCount)




class Sentinel2Observations(object):
    def __init__(self, parent_folder, emulator_file, state_mask, 
                 band_prob_threshold=20, chunk=None):
        self.band_prob_threshold = band_prob_threshold
        parent_folder = Path(parent_folder)
        emulator_file = Path(emulator_file)
        if not parent_folder.exists():
            LOG.info(f"S2 data folder: {parent_folder}")
            raise IOError("S2 data folder doesn't exist")

        if not emulator_file.exists():
            LOG.info(f"Emulator file: {emulator_file}")
            raise IOError("Emulator file doesn't exist")
        self.band_map = ['02', '03','04','05','06','07',
                         '8A','11','12']
        #self.band_map = ['05', '08']
        
        self.parent = parent_folder
        self.emulator_folder = emulator_file
        self.original_mask = state_mask
        self.state_mask = state_mask
        self._find_granules(self.parent)
        f = np.load(str(emulator_file))
        self.emulator = Two_NN(Hidden_Layers=f.f.Hidden_Layers,
                               Output_Layers=f.f.Output_Layers)
        self.chunk = chunk

    def apply_roi(self, ulx, uly, lrx, lry):
        self.ulx = ulx
        self.uly = uly
        self.lrx = lrx
        self.lry = lry
        width = lrx - ulx
        height = uly - lry
        
        self.state_mask = gdal.Translate("", self.original_mask,
                                         srcWin=[ulx, uly, width, 
                                         abs(height)], format="MEM")

    def define_output(self):
        try:
            g = gdal.Open(self.state_mask)
            proj = g.GetProjection()
            geoT = np.array(g.GetGeoTransform())

        except:
            proj = self.state_mask.GetProjection()
            geoT = np.array(self.state_mask.GetGeoTransform())

        #new_geoT = geoT*1.
        #new_geoT[0] = new_geoT[0] + self.ulx*new_geoT[1]
        #new_geoT[3] = new_geoT[3] + self.uly*new_geoT[5]
        return proj, geoT.tolist() #new_geoT.tolist()


    def _find_granules(self, parent_folder):
        """Finds granules. Currently does so by checking for
        Feng's AOT file."""
        # this is needed to follow symlinks
        test_files = [x for f in parent_folder.iterdir() 
                      for x in f.rglob("**/*_aot.tif") ]
        try:
            self.dates = [ dt.datetime(*(list(map(int, f.parts[-5:-2]))))
                    for f in test_files]
        except ValueError:
            self.dates = [dt.datetime.strptime(f.parts[-1].split(
                "_")[1], "%Y%m%dT%H%M%S") for f in test_files]
        self.date_data = dict(zip(self.dates, 
                                 [f.parent for f in test_files]))
        self.bands_per_observation = {}
        LOG.info(f"Found {len(test_files):d} S2 granules")
        LOG.info(f"First granule: " + 
                 f"{sorted(self.dates)[0].strftime('%Y-%m-%d'):s}")
        LOG.info(f"Last granule: " + 
                 f"{sorted(self.dates)[-1].strftime('%Y-%m-%d'):s}")
                              
        for the_date in self.dates:
            self.bands_per_observation[the_date] = len(self.band_map)


    def read_granule(self, timestep):
        current_folder = self.date_data[timestep]
        
        fname_prefix = [f.name.split("B02")[0]
                        for f in current_folder.glob("*B02_sur.tif")][0]
        cloud_mask = current_folder.parent/f"cloud.tif"
        cloud_mask = reproject_data(str(cloud_mask),
                                target_img=self.state_mask).data
        mask = cloud_mask <= self.band_prob_threshold
        if mask.sum() == 0:
            # No pixels! Pointless to carry on reading!
            print("Stuff")
        rho_surface = []
        for the_band in self.band_map:
            original_s2_file = current_folder/(f"{fname_prefix:s}" +
                                              f"B{the_band:s}_sur.tif")
            LOG.info(f"Original file {str(original_s2_file):s}")
            rho = reproject_data(str(original_s2_file),
                                        target_img=self.state_mask).data
            mask = mask * (rho > 0)
            rho_surface.append(rho)
        rho_surface = np.array(rho_surface)/10000.
        rho_surface[:, ~mask] = np.nan
        LOG.info(f"Total of {mask.sum():d} clear pixels " + 
                 f"({100.*mask.sum()/np.prod(mask.shape):f}%)")
        return rho_surface, mask



if __name__ == "__main__":
    s2_obs = Sentinel2Observations(
                "/home/ucfajlg/Data/python/KaFKA_Validation/LMU/s2_obs/",
                "/home/ucfafyi/DATA/Prosail/prosail_2NN.npz", 
                "/home/ucfajlg/Data/python/KaFKA_Validation/LMU/carto/ESU.tif",
                band_prob_threshold=20, chunk=None)
