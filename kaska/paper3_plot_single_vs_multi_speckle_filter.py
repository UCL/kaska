import numpy as np
import pdb
from osgeo import gdal
import matplotlib.pyplot as plt
from z_helper import *
import datetime
import seaborn as sns
from matplotlib.colors import ListedColormap


""" Inspection of processed data for Paper 3 """


years = ['2017','2018']
# versions = ['_multi', '_single']


def mask_fields(data,field,state_mask):
    if field == 301:
        mask_value = 87
    elif field == 319:
        mask_value = 67
    elif field == 542:
        mask_value = 8
    elif field == 508:
        mask_value = 27
    elif field == 515:
        mask_value = 4
    elif field == 317:
        mask_value = 65
    elif field == 410:
        mask_value = 113
    elif field == 525:
        mask_value = 30
    else:
        print("field not found")

    mask = state_mask == mask_value
    xxx = np.copy(data)
    xxx[:,~mask]=np.nan

    pos = np.argwhere(np.isfinite(xxx[0]))
    x1 = np.min(pos[:,0])
    x2 = np.max(pos[:,0])
    y1 = np.min(pos[:,1])
    y2 = np.max(pos[:,1])

    field_data = xxx[:,x1:x2,y1:y2]
    return field_data


# multi temporal speckle vs spatial speckle filter

param = ['sm', 'vwc', 'b', 'rms']
ymin = [0.1, 0, 0, 0.005]
ymax = [0.4, 5, 0.9, 0.03]

for i, par in enumerate(param):

    for year in years:
        g = gdal.Open('/media/tweiss/Work/Paper3_down/GIS/'+year+'_esu_field_buffer_30.tif')
        state_mask = g.ReadAsArray().astype(np.int)

        sm_multi = np.load('/media/tweiss/Work/Paper3_plot/all'+year+'_multi_'+par+'.npy')
        sm_single = np.load('/media/tweiss/Work/Paper3_plot/all'+year+'_single_'+par+'.npy')
        time = np.load('/media/tweiss/Work/Paper3_down/2017/'+year+'_multitimes.npy',allow_pickle=True)

        if year == '2017':
            fields = [301,319,542,508,515]
        if year == '2018':
            fields = [317,410,525,508]

        for field in fields:

            multi = mask_fields(sm_multi,field,state_mask)
            single = mask_fields(sm_single,field,state_mask)

            plt.plot(time,np.nanmean(multi, axis=(1,2)),label='multi')
            plt.plot(time,np.nanmean(single, axis=(1,2)),label='single')
            plt.title(par+' mean Field'+str(field)+' '+str(year))
            plt.legend()
            plt.savefig('/media/tweiss/Work/Paper3_down/analysis_b/speckle_temp_vs_spatial/'+par+'/mean_field_'+str(field)+'_'+str(year), bbox_inches='tight')
            plt.close()

            for t in np.arange(len(time)):

                f, ax = plt.subplots(1,2)
                im1 = ax[0].imshow(multi[t],vmin=ymin[i], vmax=ymax[i], cmap='viridis_r', aspect='auto')
                ax[0].set_title('multi temporal')
                im2 = ax[1].imshow(single[t],vmin=ymin[i], vmax=ymax[i], cmap='viridis_r', aspect='auto')
                ax[1].set_title('spatial')
                f.subplots_adjust(right=0.85)
                cbar_ax = f.add_axes([0.88, 0.15, 0.04, 0.7])
                f.colorbar(im2, cax=cbar_ax)
                plt.savefig('/media/tweiss/Work/Paper3_down/analysis_b/speckle_temp_vs_spatial/'+par+'/field_'+str(field)+'_'+str(time[t])[:10], bbox_inches='tight')
                plt.close()
