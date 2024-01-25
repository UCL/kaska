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


lai = '/media/tweiss/Daten/data_AGU/lai.tif'

g = gdal.Open(lai)
for i in range(g.RasterCount):
    gg = g.GetRasterBand(i+1)
    meta = gg.GetMetadata()

    pdb.set_trace()

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
