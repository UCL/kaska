from osgeo import gdal
import os
import pdb
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import scipy.stats
import matplotlib.dates as dates

def rmse_prediction(predictions, targets):
    """ calculation of RMSE """
    return np.sqrt(np.nanmean((predictions - targets) ** 2))

def linregress(predictions, targets):
    """ Calculate a linear least-squares regression for two sets of measurements """
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(predictions, targets)
    return slope, intercept, r_value, p_value, std_err

def get_dataset(dataset,mask_301,mask_319,mask_508,mask_515,mask_542):
    stack_date = []
    stack_data = []
    stack_301 = []
    stack_319 = []
    stack_508 = []
    stack_515 = []
    stack_542 = []
    for x in range(1, dataset.RasterCount + 1):
        band = dataset.GetRasterBand(x)
        array = band.ReadAsArray()
        try:
            stack_date.append(datetime.datetime.strptime(band.GetMetadata()['date'], '%Y-%m-%d'))
        except:
            pass
        stack_301.append(np.nanmean(array[mask_301>0]))
        stack_319.append(np.nanmean(array[mask_319>0]))
        stack_508.append(np.nanmean(array[mask_508>0]))
        stack_515.append(np.nanmean(array[mask_515>0]))
        stack_542.append(np.nanmean(array[mask_542>0]))
    return stack_date, stack_301, stack_319, stack_508, stack_515, stack_542

def read_data(path, file_name, extension, field, path_agro, file_name_agro, extension_agro):
    # Read MNI data
    df = read_mni_data(path, file_name, extension, field)

    # Read agro-meteorological station
    # df_agro = read_agrometeo(path_agro, file_name_agro, extension_agro)
    df_agro = 0

    # filter for field
    field_data = df.filter(like=field)

    # filter for relativorbit
    field_data_orbit = filter_relativorbit(field_data, field, 95, 168)
    # field_data = field_data_orbit

    # get rid of NaN values
    parameter_nan = 'LAI'
    field_data = field_data[~np.isnan(field_data.filter(like=parameter_nan).values)]

    # available auxiliary data
    theta_field = np.deg2rad(field_data.filter(like='theta'))
    # theta_field[:] = 45
    sm_field = field_data.filter(like='SM')
    height_field = field_data.filter(like='Height')/100
    lai_field = field_data.filter(like='LAI')
    vwc_field = field_data.filter(like='VWC')
    pol_field = field_data.filter(like='sigma_sentinel_'+pol)
    return df, df_agro, field_data, field_data_orbit, theta_field, sm_field, height_field, lai_field, vwc_field, pol_field

def filter_relativorbit(data, field, orbit1, orbit2=None, orbit3=None, orbit4=None):
    """ data filter for relativ orbits """
    output = data[[(check == orbit1 or check == orbit2 or check == orbit3 or check == orbit4) for check in data[(field,'relativeorbit')]]]
    return output

def read_mni_data(path, file_name, extention, field, sep=';'):
    """ read MNI campaign data """
    df = pd.io.parsers.read_csv(os.path.join(path, file_name + extension), header=[0, 1], sep=sep)
    df = df.set_index(pd.to_datetime(df[field]['date']))
    df = df.drop(df.filter(like='date'), axis=1)
    return df

### mask for fields

# field names
fields = ['301', '508', '542', '319', '515']
# fields = ['508']
# ESU names
esus = ['high', 'low', 'med', 'mean']
esus = ['high', 'low', 'med', 'mean']
esus = ['high']

# Save output path
save_path = '/media/tweiss/Work/z_final_mni_data_2017'

#------------------------------------------------------------------------------
pixel = ['_Field_buffer_30','','_buffer_30','_buffer_50','_buffer_100']
# pixel = ['_Field_buffer_30']
# pixel = ['_buffer_30']
pixel = ['_Field_buffer_30','_buffer_50','_buffer_100']

# processed_sentinel = ['multi','norm_multi']
# processed_sentinel = ['mulit']


path = '/media/tweiss/Daten/new_data'
file_name = 'multi10' # theta needs to be changed to for norm multi
extension = '.csv'

path_agro = '/media/nas_data/2017_MNI_campaign/field_data/meteodata/agrarmeteorological_station'
file_name_agro = 'Eichenried_01012017_31122017_hourly'
extension_agro = '.csv'

# df, df_agro, field_data, field_data_orbit, theta_field, sm_field, height_field, lai_field, vwc_field, pol_field = read_data(path, file_name, extension, field, path_agro, file_name_agro, extension_agro)


plt.rcParams["figure.figsize"] = (12,10)



file_sm = 'MNI_2017_sar_sm.tif'
file_vv = 'MNI_2017_vv.tif'
file_vh = 'MNI_2017_vh.tif'
file_lai = 'MNI_2017_sar_lai.tif'
file_sr = 'MNI_2017_sar_sr.tif'
file_sm_prior = 'sm_prior.tif'
file_sm_std = 'sm_std.tif'
file_lai_prior = 'lai.tif'




pol = 'vv'

for pixels in pixel:
    print(pixels)
    path_ESU = '/media/tweiss/Work/z_final_mni_data_2017/'
    name_shp = 'ESU'+pixels+'.shp'
    name_ESU = 'ESU'+pixels+'.tif'

    path = '/media/tweiss/Daten/data_AGU/'+pixels
    datapath = '/media/tweiss/Daten/data_AGU'
    dataset_sm = gdal.Open(os.path.join(path,file_sm))

    dataset_sm_prior = gdal.Open(os.path.join(datapath,file_sm_prior))
    band1 = dataset_sm_prior.GetRasterBand(1)
    mask = band1.ReadAsArray()

    df_output = pd.DataFrame(columns=pd.MultiIndex(levels=[[],[]], codes=[[],[]]))


    for esu in esus:
        for field in fields:
            g = gdal.Open(os.path.join(path_ESU, name_ESU))
            state_mask = g.ReadAsArray().astype(np.int)

            if pixels == '_Field_buffer_30':
                if field == '515':
                    mask_value = 4
                    state_mask = state_mask==mask_value
                    mask_515 = state_mask
                elif field == '508':
                    mask_value = 27
                    state_mask = state_mask==mask_value
                    mask_508 = state_mask
                elif field == '542':
                    mask_value = 8
                    state_mask = state_mask==mask_value
                    mask_542 = state_mask
                elif field == '319':
                    mask_value = 67
                    state_mask = state_mask==mask_value
                    mask_319 = state_mask
                elif field == '301':
                    mask_value = 87
                    state_mask = state_mask==mask_value
                    mask_301 = state_mask
            else:
                if field == '515' and esu == 'high':
                    mask_value = 1
                    state_mask = state_mask==mask_value
                    mask_515 = state_mask
                elif field == '515' and esu == 'med':
                    mask_value = 2
                    state_mask = state_mask==mask_value
                    mask_515 = state_mask
                elif field == '515' and esu == 'low':
                    mask_value = 3
                    state_mask = state_mask==mask_value
                    mask_515 = state_mask
                elif field == '508' and esu == 'high':
                    mask_value = 4
                    state_mask = state_mask==mask_value
                    mask_508 = state_mask
                elif field == '508' and esu == 'med':
                    mask_value = 5
                    state_mask = state_mask==mask_value
                    mask_508 = state_mask
                elif field == '508' and esu == 'low':
                    mask_value = 6
                    state_mask = state_mask==mask_value
                    mask_508 = state_mask
                elif field == '542' and esu == 'high':
                    mask_value = 7
                    state_mask = state_mask==mask_value
                    mask_542 = state_mask
                elif field == '542' and esu == 'med':
                    mask_value = 8
                    state_mask = state_mask==mask_value
                    mask_542 = state_mask
                elif field == '542' and esu == 'low':
                    mask_value = 9
                    state_mask = state_mask==mask_value
                    mask_542 = state_mask
                elif field == '319' and esu == 'high':
                    mask_value = 10
                    state_mask = state_mask==mask_value
                    mask_319 = state_mask
                elif field == '319' and esu == 'med':
                    mask_value = 11
                    state_mask = state_mask==mask_value
                    mask_319 = state_mask
                elif field == '319' and esu == 'low':
                    mask_value = 12
                    state_mask = state_mask==mask_value
                    mask_319 = state_mask
                elif field == '301' and esu == 'high':
                    mask_value = 13
                    state_mask = state_mask==mask_value
                    mask_301 = state_mask
                elif field == '301' and esu == 'med':
                    mask_value = 14
                    # state_mask = state_mask==mask_value
                    mask_301 = state_mask
                elif field == '301' and esu == 'low':
                    mask_value = 15
                    state_mask = state_mask==mask_value
                    mask_301 = state_mask
                elif field == '515' and esu == 'mean':
                    m = np.ma.array(state_mask,mask=((state_mask==1) | (state_mask==2) | (state_mask==3)))
                    state_mask = m.mask
                    mask_515 = state_mask
                elif field == '508' and esu == 'mean':
                    m = np.ma.array(state_mask,mask=((state_mask==4) | (state_mask==5) | (state_mask==6)))
                    state_mask = m.mask
                    mask_508 = state_mask
                elif field == '542' and esu == 'mean':
                    m = np.ma.array(state_mask,mask=((state_mask==7) | (state_mask==8) | (state_mask==9)))
                    state_mask = m.mask
                    mask_542 = state_mask
                elif field == '319' and esu == 'mean':
                    m = np.ma.array(state_mask,mask=((state_mask==10) | (state_mask==11) | (state_mask==12)))
                    state_mask = m.mask
                    mask_319 = state_mask
                elif field == '301' and esu == 'mean':
                    m = np.ma.array(state_mask,mask=((state_mask==13) | (state_mask==14) | (state_mask==15)))
                    state_mask = m.mask
                    mask_301 = state_mask

    sm_date, sm_301, sm_319, sm_508, sm_515, sm_542 = get_dataset(dataset_sm,mask_301,mask_319,mask_508,mask_515,mask_542)

    dataset_lai = gdal.Open(os.path.join(path,file_lai))
    lai_date, lai_301, lai_319, lai_508, lai_515, lai_542 = get_dataset(dataset_lai,mask_301,mask_319,mask_508,mask_515,mask_542)

    dataset_sr = gdal.Open(os.path.join(path,file_sr))
    sr_date, sr_301, sr_319, sr_508, sr_515, sr_542 = get_dataset(dataset_sr,mask_301,mask_319,mask_508,mask_515,mask_542)

    dataset_lai_prior = gdal.Open(os.path.join(datapath,file_lai_prior))
    lai_prior_date, lai_prior_301, lai_prior_319, lai_prior_508, lai_prior_515, lai_prior_542 = get_dataset(dataset_lai_prior,mask_301,mask_319,mask_508,mask_515,mask_542)

    # dataset_vv = gdal.Open(os.path.join(path,file_vv))
    # vv_date, vv_301, vv_319, vv_508, vv_515, vv_542 = get_dataset(dataset_vv,mask_301,mask_319,mask_508,mask_515,mask_542)

    # dataset_vh = gdal.Open(os.path.join(path,file_vh))
    # vh_date, vh_301, vh_319, vh_508, vh_515, vh_542 = get_dataset(dataset_vh,mask_301,mask_319,mask_508,mask_515,mask_542)

    sm_prior_date, sm_prior_301, sm_prior_319, sm_prior_508, sm_prior_515, sm_prior_542 = get_dataset(dataset_sm_prior,mask,mask,mask,mask,mask)

    dataset_sm_std = gdal.Open(os.path.join(datapath,file_sm_std))
    sm_std_date, sm_std_301, sm_std_319, sm_std_508, sm_std_515, sm_std_542 = get_dataset(dataset_sm_std,mask,mask,mask,mask,mask)

    pathx = '/media/tweiss/Daten/new_data'
    field = '508_high'

    df, df_agro, field_data, field_data_orbit, theta_field, sm_field, height_field, lai_field, vwc_field, pol_field = read_data(pathx, file_name, extension, field, path_agro, file_name_agro, extension_agro)

    start = datetime.datetime.combine(sm_field.index.date[0], datetime.datetime.min.time())
    end = datetime.datetime.combine(sm_field.index.date[len(sm_field.index)-1], datetime.datetime.min.time())

    start2 = np.argwhere(np.array(sm_date)==start)[0][0]
    end2 = np.argwhere(np.array(sm_date)==end)[0][0]

    rmse = rmse_prediction(sm_508[start2:end2],sm_field['508_high']['SM'])
    rmse_prior = rmse_prediction(sm_prior_508[start2:end2],sm_field['508_high']['SM'])
    slope, intercept, r_value, p_value, std_err = linregress(sm_508[start2:end2],sm_field['508_high']['SM'])
    slope_p, intercept_p, r_value_p, p_value_p, std_err_p = linregress(sm_prior_508[start2:end2],sm_field['508_high']['SM'])

    # plt.title('SM 508, RMSE [Vol/%]:'+str(rmse*100)[0:4]+' R2:'+str(r_value)[0:4]+' RMSE Prior,insitu:'+str(rmse_prior*100)[0:4])


    if esu == 'high' or esu == 'mean':
        plt.plot(sm_date, sm_508, label='SM retrieved point; RMSE [Vol%]: '+str(rmse*100)[:4]+'; $R^2$: '+str(r_value)[:4], linewidth=2, color='Red')

plt.plot(sm_field,label='SM field measurement',color='Black', linewidth=3)
plt.plot(sm_prior_date, sm_prior_508, label='SM prior; RMSE [Vol%]: '+str(rmse_prior*100)[:4]+'; $R^2$: '+str(r_value_p)[:4])

dstart = datetime.datetime(2017,3,25)
dend = datetime.datetime(2017,7,16)
plt.tick_params(labelsize=12)
plt.xlim(dstart, dend)
plt.legend(prop={'size': 14})
plt.ylim(0.0,0.7)
plt.grid()
plt.ylabel('Soil Moisture [$m^3/m^3$]', fontsize=15)
plt.xlabel('Date', fontsize=15)
plt.savefig(datapath+'/sm_5082_p2.png')
plt.close()
pdb.set_trace()



# path = '/media/tweiss/Daten/new_data'
# file_name = 'multi10' # theta needs to be changed to for norm multi
# extension = '.csv'

# path_agro = '/media/nas_data/2017_MNI_campaign/field_data/meteodata/agrarmeteorological_station'
# file_name_agro = 'Eichenried_01012017_31122017_hourly'
# extension_agro = '.csv'

# field = '508_high'
# field_plot = ['508_high', '508_low', '508_med']
# pol = 'vv'
# pol = 'vh'

# # output path
# plot_output_path = '/media/tweiss/Daten/plots/paper/'

# df, df_agro, field_data, field_data_orbit, theta_field, sm_field, height_field, lai_field, vwc_field, pol_field = read_data(path, file_name, extension, field, path_agro, file_name_agro, extension_agro)


# plt.rcParams["figure.figsize"] = (12,10)

# path = '/media/tweiss/Daten/data_AGU/ucl'

# datapath = '/media/tweiss/Daten/data_AGU'

# file_sm = 'MNI_2017_sar_sm.tif'
# file_vv = 'MNI_2017_vv.tif'
# file_vh = 'MNI_2017_vh.tif'
# file_lai = 'MNI_2017_sar_lai.tif'
# file_sr = 'MNI_2017_sar_sr.tif'
# file_sm_prior = 'sm_prior.tif'
# file_sm_std = 'sm_std.tif'
# file_lai_prior = 'lai.tif'

# dataset_sm = gdal.Open(os.path.join(path,file_sm))
# band1 = dataset_sm.GetRasterBand(1)


# sm_date, sm_301, sm_319, sm_508, sm_515, sm_542 = get_dataset(dataset_sm,mask_301,mask_319,mask_508,mask_515,mask_542)

# dataset_lai = gdal.Open(os.path.join(path,file_lai))
# lai_date, lai_301, lai_319, lai_508, lai_515, lai_542 = get_dataset(dataset_lai,mask_301,mask_319,mask_508,mask_515,mask_542)

# dataset_sr = gdal.Open(os.path.join(path,file_sr))
# sr_date, sr_301, sr_319, sr_508, sr_515, sr_542 = get_dataset(dataset_sr,mask_301,mask_319,mask_508,mask_515,mask_542)

# dataset_lai_prior = gdal.Open(os.path.join(datapath,file_lai_prior))
# lai_prior_date, lai_prior_301, lai_prior_319, lai_prior_508, lai_prior_515, lai_prior_542 = get_dataset(dataset_lai_prior,mask_301,mask_319,mask_508,mask_515,mask_542)

# # dataset_vv = gdal.Open(os.path.join(path,file_vv))
# # vv_date, vv_301, vv_319, vv_508, vv_515, vv_542 = get_dataset(dataset_vv,mask_301,mask_319,mask_508,mask_515,mask_542)

# # dataset_vh = gdal.Open(os.path.join(path,file_vh))
# # vh_date, vh_301, vh_319, vh_508, vh_515, vh_542 = get_dataset(dataset_vh,mask_301,mask_319,mask_508,mask_515,mask_542)



# dataset_sm_prior = gdal.Open(os.path.join(datapath,file_sm_prior))
# band1 = dataset_sm_prior.GetRasterBand(1)
# mask = band1.ReadAsArray()


# sm_prior_date, sm_prior_301, sm_prior_319, sm_prior_508, sm_prior_515, sm_prior_542 = get_dataset(dataset_sm_prior,mask,mask,mask,mask,mask)

# dataset_sm_std = gdal.Open(os.path.join(datapath,file_sm_std))
# sm_std_date, sm_std_301, sm_std_319, sm_std_508, sm_std_515, sm_std_542 = get_dataset(dataset_sm_std,mask,mask,mask,mask,mask)

# path = '/media/tweiss/Daten/new_data'
# field = '508_high'

# df, df_agro, field_data, field_data_orbit, theta_field, sm_field, height_field, lai_field, vwc_field, pol_field = read_data(path, file_name, extension, field, path_agro, file_name_agro, extension_agro)

# start = datetime.datetime.combine(sm_field.index.date[0], datetime.datetime.min.time())
# end = datetime.datetime.combine(sm_field.index.date[len(sm_field.index)-1], datetime.datetime.min.time())

# start2 = np.argwhere(np.array(sm_date)==start)[0][0]
# end2 = np.argwhere(np.array(sm_date)==end)[0][0]

# rmse = rmse_prediction(sm_508[start2:end2],sm_field['508_high']['SM'])
# rmse_prior = rmse_prediction(sm_prior_508[start2:end2],sm_field['508_high']['SM'])
# slope, intercept, r_value, p_value, std_err = linregress(sm_508[start2:end2],sm_field['508_high']['SM'])
# slope_p, intercept_p, r_value_p, p_value_p, std_err_p = linregress(sm_prior_508[start2:end2],sm_field['508_high']['SM'])

# # plt.title('SM 508, RMSE [Vol/%]:'+str(rmse*100)[0:4]+' R2:'+str(r_value)[0:4]+' RMSE Prior,insitu:'+str(rmse_prior*100)[0:4])
# plt.plot(sm_field,label='SM field measurement',color='Black', linewidth=3)
# plt.plot(sm_prior_date, sm_prior_508, label='SM prior; RMSE [Vol%]: '+str(rmse_prior*100)[:4]+'; $R^2$: '+str(r_value_p)[:4])
# plt.plot(sm_date, sm_508, label='SM retrieved point; RMSE [Vol%]: '+str(rmse*100)[:4]+'; $R^2$: '+str(r_value)[:4], linewidth=2, color='Red')





# path = '/media/tweiss/Daten/data_AGU/ucl'

# datapath = '/media/tweiss/Daten/data_AGU'
# file_sm = 'MNI_2017_sar_sm.tif'

# dataset_sm = gdal.Open(os.path.join(path,file_sm))
# band1 = dataset_sm.GetRasterBand(1)
# mask = band1.ReadAsArray()


# #field
# field_301 = 0.21249178
# field_319 = 0.20654242
# field_508 = 0.23555766
# field_515 = 0.21090584
# field_542 = 0.21022798


# # 30m
# field_301 = 0.21076469
# field_319 = 0.2052274
# field_508 = 0.20654558
# field_515 = 0.21090584
# field_542 = 0.21518409


# mask_301 = band1.ReadAsArray()
# mask_301[mask_301!=field_301] = 0

# mask_319 = band1.ReadAsArray()
# mask_319[mask_319!=field_319] = 0
# mask_508 = band1.ReadAsArray()
# mask_508[mask_508!=field_508] = 0
# mask_515 = band1.ReadAsArray()
# mask_515[mask_515!=field_515] = 0
# mask_542 = band1.ReadAsArray()
# mask_542[mask_542!=field_542] = 0


sm_date, sm_301, sm_319, sm_508, sm_515, sm_542 = get_dataset(dataset_sm,mask_301,mask_319,mask_508,mask_515,mask_542)

dataset_lai = gdal.Open(os.path.join(path,file_lai))
lai_date, lai_301, lai_319, lai_508, lai_515, lai_542 = get_dataset(dataset_lai,mask_301,mask_319,mask_508,mask_515,mask_542)

dataset_sr = gdal.Open(os.path.join(path,file_sr))
sr_date, sr_301, sr_319, sr_508, sr_515, sr_542 = get_dataset(dataset_sr,mask_301,mask_319,mask_508,mask_515,mask_542)

dataset_lai_prior = gdal.Open(os.path.join(datapath,file_lai_prior))
lai_prior_date, lai_prior_301, lai_prior_319, lai_prior_508, lai_prior_515, lai_prior_542 = get_dataset(dataset_lai_prior,mask_301,mask_319,mask_508,mask_515,mask_542)

rmse = rmse_prediction(sm_508[start2:end2],sm_field['508_high']['SM'])
rmse_prior = rmse_prediction(sm_prior_508[start2:end2],sm_field['508_high']['SM'])
slope, intercept, r_value, p_value, std_err = linregress(sm_508[start2:end2],sm_field['508_high']['SM'])


plt.plot(sm_date, sm_508, label='SM retrieved 30m; RMSE [Vol%]: '+str(rmse*100)[:4]+'; $R^2$: '+str(r_value)[:4], linewidth=2, color='Orange')



# path = '/media/tweiss/Daten/data_AGU/ucl'

# datapath = '/media/tweiss/Daten/data_AGU'
# file_sm = 'MNI_2017_sar_sm.tif'

# dataset_sm = gdal.Open(os.path.join(path,file_sm))
# band1 = dataset_sm.GetRasterBand(1)
# # mask = band1.ReadAsArray()


# #field
# field_301 = 0.21249178
# field_319 = 0.20654242
# field_508 = 0.23555766
# field_515 = 0.21090584
# field_542 = 0.21022798



# mask_301 = band1.ReadAsArray()
# mask_301[mask_301!=field_301] = 0

# mask_319 = band1.ReadAsArray()
# mask_319[mask_319!=field_319] = 0
# mask_508 = band1.ReadAsArray()
# mask_508[mask_508!=field_508] = 0
# mask_515 = band1.ReadAsArray()
# mask_515[mask_515!=field_515] = 0
# mask_542 = band1.ReadAsArray()
# mask_542[mask_542!=field_542] = 0


sm_date, sm_301, sm_319, sm_508, sm_515, sm_542 = get_dataset(dataset_sm,mask_301,mask_319,mask_508,mask_515,mask_542)

dataset_lai = gdal.Open(os.path.join(path,file_lai))
lai_date, lai_301, lai_319, lai_508, lai_515, lai_542 = get_dataset(dataset_lai,mask_301,mask_319,mask_508,mask_515,mask_542)

dataset_sr = gdal.Open(os.path.join(path,file_sr))
sr_date, sr_301, sr_319, sr_508, sr_515, sr_542 = get_dataset(dataset_sr,mask_301,mask_319,mask_508,mask_515,mask_542)

dataset_lai_prior = gdal.Open(os.path.join(datapath,file_lai_prior))
lai_prior_date, lai_prior_301, lai_prior_319, lai_prior_508, lai_prior_515, lai_prior_542 = get_dataset(dataset_lai_prior,mask_301,mask_319,mask_508,mask_515,mask_542)

rmse = rmse_prediction(sm_508[start2:end2],sm_field['508_high']['SM'])
rmse_prior = rmse_prediction(sm_prior_508[start2:end2],sm_field['508_high']['SM'])
slope, intercept, r_value, p_value, std_err = linregress(sm_508[start2:end2],sm_field['508_high']['SM'])

plt.plot(sm_date, sm_508, label='SM retrieved field; RMSE [Vol%]: '+str(rmse*100)[:4]+'; $R^2$: '+str(r_value)[:4], linewidth=2, color='Green')












dstart = datetime.datetime(2017,3,25)
dend = datetime.datetime(2017,7,16)
plt.tick_params(labelsize=12)
plt.xlim(dstart, dend)
plt.legend(prop={'size': 14})
plt.ylim(0.1,0.4)
plt.grid()
plt.ylabel('Soil Moisture [$m^3/m^3$]', fontsize=15)
plt.xlabel('Date', fontsize=15)
plt.savefig(datapath+'/sm_5082_p2.png')
plt.close()
pdb.set_trace()












### soil moisture time


path = '/media/tweiss/Work/Jose/new_backscatter/field'

file_sm = 'MNI_2017_sar_sm.tif'

dataset_sm = gdal.Open(os.path.join(path,file_sm))

for x in range(1, dataset_sm.RasterCount + 1):
    band = dataset_sm.GetRasterBand(x)
    array = band.ReadAsArray()
    datum = datetime.datetime.strptime(band.GetMetadata()['date'], '%Y-%m-%d')
    plt.imshow(array, cmap='Blues', vmin=0.0, vmax=0.4)
    #legend
    cbar = plt.colorbar()
    cbar.set_label('Soil Moisture [$m^3/m^3$]', rotation=270, labelpad=20)
    plt.text(450,40,datum.date())
    plt.savefig(path+'/gif/sm_'+str(datum.date())+'.png')
    plt.close()























































path = '/media/tweiss/Daten/new_data'
file_name = 'multi10' # theta needs to be changed to for norm multi
extension = '.csv'

path_agro = '/media/nas_data/2017_MNI_campaign/field_data/meteodata/agrarmeteorological_station'
file_name_agro = 'Eichenried_01012017_31122017_hourly'
extension_agro = '.csv'

field = '508_high'
field_plot = ['508_high', '508_low', '508_med']
pol = 'vv'
pol = 'vh'

# output path
plot_output_path = '/media/tweiss/Daten/plots/paper/'

df, df_agro, field_data, field_data_orbit, theta_field, sm_field, height_field, lai_field, vwc_field, pol_field = read_data(path, file_name, extension, field, path_agro, file_name_agro, extension_agro)


plt.rcParams["figure.figsize"] = (12,10)

path = '/home/tweiss/Desktop/LRZ Sync+Share/Jose/new_backscatter/point'

datapath = '/media/tweiss/Daten/test_kaska/data'
file_sm = 'MNI_2017_sar_sm.tif'

dataset_sm = gdal.Open(os.path.join(path,file_sm))
band1 = dataset_sm.GetRasterBand(1)
mask = band1.ReadAsArray()


#field
field_301 = 0.21249178
field_319 = 0.20654242
field_508 = 0.23555766
field_515 = 0.21090584
field_542 = 0.21022798


# 30m
field_301 = 0.21076469
field_319 = 0.2052274
field_508 = 0.20654558
field_515 = 0.21090584
field_542 = 0.21518409

# 1m
field_301 = 0.21167806
field_319 = 0.20519826
field_508 = 0.22907102
field_515 = 0.2043577
field_542 = 0.21640626


mask_301 = band1.ReadAsArray()
mask_301[mask_301!=field_301] = 0
mask_319 = band1.ReadAsArray()
mask_319[mask_319!=field_319] = 0
mask_508 = band1.ReadAsArray()
mask_508[mask_508!=field_508] = 0
mask_515 = band1.ReadAsArray()
mask_515[mask_515!=field_515] = 0
mask_542 = band1.ReadAsArray()
mask_542[mask_542!=field_542] = 0


file_sm = 'MNI_2017_sar_sm.tif'
file_vv = 'MNI_2017_vv.tif'
file_vh = 'MNI_2017_vh.tif'
file_lai = 'MNI_2017_sar_lai.tif'
file_sr = 'MNI_2017_sar_sr.tif'
file_sm_prior = 'sm_prior.tif'
file_sm_std = 'sm_std.tif'
file_lai_prior = 'lai.tif'


sm_date, sm_301, sm_319, sm_508, sm_515, sm_542 = get_dataset(dataset_sm,mask_301,mask_319,mask_508,mask_515,mask_542)

dataset_lai = gdal.Open(os.path.join(path,file_lai))
lai_date, lai_301, lai_319, lai_508, lai_515, lai_542 = get_dataset(dataset_lai,mask_301,mask_319,mask_508,mask_515,mask_542)

dataset_sr = gdal.Open(os.path.join(path,file_sr))
sr_date, sr_301, sr_319, sr_508, sr_515, sr_542 = get_dataset(dataset_sr,mask_301,mask_319,mask_508,mask_515,mask_542)

dataset_lai_prior = gdal.Open(os.path.join(datapath,file_lai_prior))
lai_prior_date, lai_prior_301, lai_prior_319, lai_prior_508, lai_prior_515, lai_prior_542 = get_dataset(dataset_lai_prior,mask_301,mask_319,mask_508,mask_515,mask_542)

# dataset_vv = gdal.Open(os.path.join(path,file_vv))
# vv_date, vv_301, vv_319, vv_508, vv_515, vv_542 = get_dataset(dataset_vv,mask_301,mask_319,mask_508,mask_515,mask_542)

# dataset_vh = gdal.Open(os.path.join(path,file_vh))
# vh_date, vh_301, vh_319, vh_508, vh_515, vh_542 = get_dataset(dataset_vh,mask_301,mask_319,mask_508,mask_515,mask_542)



dataset_sm_prior = gdal.Open(os.path.join(datapath,file_sm_prior))
band1 = dataset_sm_prior.GetRasterBand(1)
mask = band1.ReadAsArray()


sm_prior_date, sm_prior_301, sm_prior_319, sm_prior_508, sm_prior_515, sm_prior_542 = get_dataset(dataset_sm_prior,mask,mask,mask,mask,mask)

dataset_sm_std = gdal.Open(os.path.join(datapath,file_sm_std))
sm_std_date, sm_std_301, sm_std_319, sm_std_508, sm_std_515, sm_std_542 = get_dataset(dataset_sm_std,mask,mask,mask,mask,mask)

path = '/media/tweiss/Daten/new_data'
field = '508_high'

df, df_agro, field_data, field_data_orbit, theta_field, sm_field, height_field, lai_field, vwc_field, pol_field = read_data(path, file_name, extension, field, path_agro, file_name_agro, extension_agro)

start = datetime.datetime.combine(sm_field.index.date[0], datetime.datetime.min.time())
end = datetime.datetime.combine(sm_field.index.date[len(sm_field.index)-1], datetime.datetime.min.time())

start2 = np.argwhere(np.array(sm_date)==start)[0][0]
end2 = np.argwhere(np.array(sm_date)==end)[0][0]

rmse = rmse_prediction(sm_508[start2:end2],sm_field['508_high']['SM'])
rmse_prior = rmse_prediction(sm_prior_508[start2:end2],sm_field['508_high']['SM'])
slope, intercept, r_value, p_value, std_err = linregress(sm_508[start2:end2],sm_field['508_high']['SM'])
slope_p, intercept_p, r_value_p, p_value_p, std_err_p = linregress(sm_prior_508[start2:end2],sm_field['508_high']['SM'])

# plt.title('SM 508, RMSE [Vol/%]:'+str(rmse*100)[0:4]+' R2:'+str(r_value)[0:4]+' RMSE Prior,insitu:'+str(rmse_prior*100)[0:4])
plt.plot(sm_field,label='SM field measurement',color='Black', linewidth=3)
plt.plot(sm_prior_date, sm_prior_508, label='SM prior; RMSE [Vol%]: '+str(rmse_prior*100)[:4]+'; $R^2$: '+str(r_value_p)[:4])
# plt.plot(sm_date, sm_508, label='SM retrieved point; RMSE [Vol%]: '+str(rmse*100)[:4]+'; $R^2$: '+str(r_value)[:4], linewidth=2, color='Red')





path = '/home/tweiss/Desktop/LRZ Sync+Share/Jose/new_backscatter/30m'

datapath = '/media/tweiss/Daten/test_kaska/data'
file_sm = 'MNI_2017_sar_sm.tif'

dataset_sm = gdal.Open(os.path.join(path,file_sm))
band1 = dataset_sm.GetRasterBand(1)
mask = band1.ReadAsArray()


#field
field_301 = 0.21249178
field_319 = 0.20654242
field_508 = 0.23555766
field_515 = 0.21090584
field_542 = 0.21022798


# 30m
field_301 = 0.21076469
field_319 = 0.2052274
field_508 = 0.20654558
field_515 = 0.21090584
field_542 = 0.21518409


mask_301 = band1.ReadAsArray()
mask_301[mask_301!=field_301] = 0

mask_319 = band1.ReadAsArray()
mask_319[mask_319!=field_319] = 0
mask_508 = band1.ReadAsArray()
mask_508[mask_508!=field_508] = 0
mask_515 = band1.ReadAsArray()
mask_515[mask_515!=field_515] = 0
mask_542 = band1.ReadAsArray()
mask_542[mask_542!=field_542] = 0


sm_date, sm_301, sm_319, sm_508, sm_515, sm_542 = get_dataset(dataset_sm,mask_301,mask_319,mask_508,mask_515,mask_542)

dataset_lai = gdal.Open(os.path.join(path,file_lai))
lai_date, lai_301, lai_319, lai_508, lai_515, lai_542 = get_dataset(dataset_lai,mask_301,mask_319,mask_508,mask_515,mask_542)

dataset_sr = gdal.Open(os.path.join(path,file_sr))
sr_date, sr_301, sr_319, sr_508, sr_515, sr_542 = get_dataset(dataset_sr,mask_301,mask_319,mask_508,mask_515,mask_542)

dataset_lai_prior = gdal.Open(os.path.join(datapath,file_lai_prior))
lai_prior_date, lai_prior_301, lai_prior_319, lai_prior_508, lai_prior_515, lai_prior_542 = get_dataset(dataset_lai_prior,mask_301,mask_319,mask_508,mask_515,mask_542)

rmse = rmse_prediction(sm_508[start2:end2],sm_field['508_high']['SM'])
rmse_prior = rmse_prediction(sm_prior_508[start2:end2],sm_field['508_high']['SM'])
slope, intercept, r_value, p_value, std_err = linregress(sm_508[start2:end2],sm_field['508_high']['SM'])


plt.plot(sm_date, sm_508, label='SM retrieved 30m; RMSE [Vol%]: '+str(rmse*100)[:4]+'; $R^2$: '+str(r_value)[:4], linewidth=2, color='Orange')



path = '/home/tweiss/Desktop/LRZ Sync+Share/Jose/new_backscatter/field'

datapath = '/media/tweiss/Daten/test_kaska/data'
file_sm = 'MNI_2017_sar_sm.tif'

dataset_sm = gdal.Open(os.path.join(path,file_sm))
band1 = dataset_sm.GetRasterBand(1)
mask = band1.ReadAsArray()


#field
field_301 = 0.21249178
field_319 = 0.20654242
field_508 = 0.23555766
field_515 = 0.21090584
field_542 = 0.21022798



mask_301 = band1.ReadAsArray()
mask_301[mask_301!=field_301] = 0

mask_319 = band1.ReadAsArray()
mask_319[mask_319!=field_319] = 0
mask_508 = band1.ReadAsArray()
mask_508[mask_508!=field_508] = 0
mask_515 = band1.ReadAsArray()
mask_515[mask_515!=field_515] = 0
mask_542 = band1.ReadAsArray()
mask_542[mask_542!=field_542] = 0


sm_date, sm_301, sm_319, sm_508, sm_515, sm_542 = get_dataset(dataset_sm,mask_301,mask_319,mask_508,mask_515,mask_542)

dataset_lai = gdal.Open(os.path.join(path,file_lai))
lai_date, lai_301, lai_319, lai_508, lai_515, lai_542 = get_dataset(dataset_lai,mask_301,mask_319,mask_508,mask_515,mask_542)

dataset_sr = gdal.Open(os.path.join(path,file_sr))
sr_date, sr_301, sr_319, sr_508, sr_515, sr_542 = get_dataset(dataset_sr,mask_301,mask_319,mask_508,mask_515,mask_542)

dataset_lai_prior = gdal.Open(os.path.join(datapath,file_lai_prior))
lai_prior_date, lai_prior_301, lai_prior_319, lai_prior_508, lai_prior_515, lai_prior_542 = get_dataset(dataset_lai_prior,mask_301,mask_319,mask_508,mask_515,mask_542)

rmse = rmse_prediction(sm_508[start2:end2],sm_field['508_high']['SM'])
rmse_prior = rmse_prediction(sm_prior_508[start2:end2],sm_field['508_high']['SM'])
slope, intercept, r_value, p_value, std_err = linregress(sm_508[start2:end2],sm_field['508_high']['SM'])

plt.plot(sm_date, sm_508, label='SM retrieved field; RMSE [Vol%]: '+str(rmse*100)[:4]+'; $R^2$: '+str(r_value)[:4], linewidth=2, color='Green')












dstart = datetime.datetime(2017,3,25)
dend = datetime.datetime(2017,7,16)
plt.tick_params(labelsize=12)
plt.xlim(dstart, dend)
plt.legend(prop={'size': 14})
plt.ylim(0.1,0.4)
plt.grid()
plt.ylabel('Soil Moisture [$m^3/m^3$]', fontsize=15)
plt.xlabel('Date', fontsize=15)
plt.savefig(datapath+'/sm_5082_p2.png')
plt.close()


path = '/home/tweiss/Desktop/LRZ Sync+Share/Jose/new_backscatter/point'

datapath = '/media/tweiss/Daten/test_kaska/data'
file_sm = 'MNI_2017_sar_sm.tif'

dataset_sm = gdal.Open(os.path.join(path,file_sm))
band1 = dataset_sm.GetRasterBand(1)
mask = band1.ReadAsArray()


#field
field_301 = 0.21249178
field_319 = 0.20654242
field_508 = 0.23555766
field_515 = 0.21090584
field_542 = 0.21022798


# 30m
field_301 = 0.21076469
field_319 = 0.2052274
field_508 = 0.20654558
field_515 = 0.21090584
field_542 = 0.21518409

# 1m
field_301 = 0.21167806
field_319 = 0.20519826
field_508 = 0.22907102
field_515 = 0.2043577
field_542 = 0.21640626


mask_301 = band1.ReadAsArray()
mask_301[mask_301!=field_301] = 0
mask_319 = band1.ReadAsArray()
mask_319[mask_319!=field_319] = 0
mask_508 = band1.ReadAsArray()
mask_508[mask_508!=field_508] = 0
mask_515 = band1.ReadAsArray()
mask_515[mask_515!=field_515] = 0
mask_542 = band1.ReadAsArray()
mask_542[mask_542!=field_542] = 0


file_sm = 'MNI_2017_sar_sm.tif'
file_vv = 'MNI_2017_vv.tif'
file_vh = 'MNI_2017_vh.tif'
file_lai = 'MNI_2017_sar_lai.tif'
file_sr = 'MNI_2017_sar_sr.tif'
file_sm_prior = 'sm_prior.tif'
file_sm_std = 'sm_std.tif'
file_lai_prior = 'lai.tif'


sm_date, sm_301, sm_319, sm_508, sm_515, sm_542 = get_dataset(dataset_sm,mask_301,mask_319,mask_508,mask_515,mask_542)

dataset_lai = gdal.Open(os.path.join(path,file_lai))
lai_date, lai_301, lai_319, lai_508, lai_515, lai_542 = get_dataset(dataset_lai,mask_301,mask_319,mask_508,mask_515,mask_542)

dataset_sr = gdal.Open(os.path.join(path,file_sr))
sr_date, sr_301, sr_319, sr_508, sr_515, sr_542 = get_dataset(dataset_sr,mask_301,mask_319,mask_508,mask_515,mask_542)

dataset_lai_prior = gdal.Open(os.path.join(datapath,file_lai_prior))
lai_prior_date, lai_prior_301, lai_prior_319, lai_prior_508, lai_prior_515, lai_prior_542 = get_dataset(dataset_lai_prior,mask_301,mask_319,mask_508,mask_515,mask_542)

# dataset_vv = gdal.Open(os.path.join(path,file_vv))
# vv_date, vv_301, vv_319, vv_508, vv_515, vv_542 = get_dataset(dataset_vv,mask_301,mask_319,mask_508,mask_515,mask_542)

# dataset_vh = gdal.Open(os.path.join(path,file_vh))
# vh_date, vh_301, vh_319, vh_508, vh_515, vh_542 = get_dataset(dataset_vh,mask_301,mask_319,mask_508,mask_515,mask_542)



dataset_sm_prior = gdal.Open(os.path.join(datapath,file_sm_prior))
band1 = dataset_sm_prior.GetRasterBand(1)
mask = band1.ReadAsArray()


sm_prior_date, sm_prior_301, sm_prior_319, sm_prior_508, sm_prior_515, sm_prior_542 = get_dataset(dataset_sm_prior,mask,mask,mask,mask,mask)

dataset_sm_std = gdal.Open(os.path.join(datapath,file_sm_std))
sm_std_date, sm_std_301, sm_std_319, sm_std_508, sm_std_515, sm_std_542 = get_dataset(dataset_sm_std,mask,mask,mask,mask,mask)





path = '/media/tweiss/Daten/new_data'
field = '301_high'

df, df_agro, field_data, field_data_orbit, theta_field, sm_field, height_field, lai_field, vwc_field, pol_field = read_data(path, file_name, extension, field, path_agro, file_name_agro, extension_agro)

start = datetime.datetime.combine(sm_field.index.date[0], datetime.datetime.min.time())
end = datetime.datetime.combine(sm_field.index.date[len(sm_field.index)-1], datetime.datetime.min.time())

start2 = np.argwhere(np.array(sm_date)==start)[0][0]
end2 = np.argwhere(np.array(sm_date)==end)[0][0]

sm_301[start2:end2]

rmse = rmse_prediction(sm_301[start2:end2],sm_field['301_high']['SM'])
rmse_prior = rmse_prediction(sm_prior_301[start2:end2],sm_field['301_high']['SM'])
slope, intercept, r_value, p_value, std_err = linregress(sm_301[start2:end2],sm_field['301_high']['SM'])
slope_p, intercept_p, r_value_p, p_value_p, std_err_p = linregress(sm_prior_301[start2:end2],sm_field['301_high']['SM'])

# plt.title('SM 301, RMSE [Vol/%]: '+str(rmse*100)[0:4]+' R2: '+str(r_value)[0:4]+' RMSE Prior,insitu:'+str(rmse_prior*100)[0:4])
plt.plot(sm_field,label='SM field measurement', color='Black', linewidth=3)
plt.plot(sm_prior_date, sm_prior_301, label='SM prior; RMSE [Vol%]: '+str(rmse_prior*100)[:4]+'; $R^2$: '+str(r_value_p)[:4])
# plt.plot(sm_date, sm_301, label='SM retrieved point; RMSE [Vol%]: '+str(rmse*100)[:4]+'; $R^2$: '+str(r_value)[:4], linewidth=2, color='Red')




path = '/home/tweiss/Desktop/LRZ Sync+Share/Jose/new_backscatter/30m'

datapath = '/media/tweiss/Daten/test_kaska/data'
file_sm = 'MNI_2017_sar_sm.tif'

dataset_sm = gdal.Open(os.path.join(path,file_sm))
band1 = dataset_sm.GetRasterBand(1)
mask = band1.ReadAsArray()


#field
field_301 = 0.21249178
field_319 = 0.20654242
field_508 = 0.23555766
field_515 = 0.21090584
field_542 = 0.21022798


# 30m
field_301 = 0.21076469
field_319 = 0.2052274
field_508 = 0.20654558
field_515 = 0.21090584
field_542 = 0.21518409


mask_301 = band1.ReadAsArray()
mask_301[mask_301!=field_301] = 0

mask_319 = band1.ReadAsArray()
mask_319[mask_319!=field_319] = 0
mask_508 = band1.ReadAsArray()
mask_508[mask_508!=field_508] = 0
mask_515 = band1.ReadAsArray()
mask_515[mask_515!=field_515] = 0
mask_542 = band1.ReadAsArray()
mask_542[mask_542!=field_542] = 0


sm_date, sm_301, sm_319, sm_508, sm_515, sm_542 = get_dataset(dataset_sm,mask_301,mask_319,mask_508,mask_515,mask_542)

dataset_lai = gdal.Open(os.path.join(path,file_lai))
lai_date, lai_301, lai_319, lai_508, lai_515, lai_542 = get_dataset(dataset_lai,mask_301,mask_319,mask_508,mask_515,mask_542)

dataset_sr = gdal.Open(os.path.join(path,file_sr))
sr_date, sr_301, sr_319, sr_508, sr_515, sr_542 = get_dataset(dataset_sr,mask_301,mask_319,mask_508,mask_515,mask_542)

dataset_lai_prior = gdal.Open(os.path.join(datapath,file_lai_prior))
lai_prior_date, lai_prior_301, lai_prior_319, lai_prior_508, lai_prior_515, lai_prior_542 = get_dataset(dataset_lai_prior,mask_301,mask_319,mask_508,mask_515,mask_542)


rmse = rmse_prediction(sm_301[start2:end2],sm_field['301_high']['SM'])
rmse_prior = rmse_prediction(sm_prior_301[start2:end2],sm_field['301_high']['SM'])
slope, intercept, r_value, p_value, std_err = linregress(sm_301[start2:end2],sm_field['301_high']['SM'])

plt.plot(sm_date, sm_301, label='SM retrieved 30m; RMSE [Vol%]: '+str(rmse*100)[:4]+'; $R^2$: '+str(r_value)[:4], linewidth=2, color='Orange')


path = '/home/tweiss/Desktop/LRZ Sync+Share/Jose/new_backscatter/field'

datapath = '/media/tweiss/Daten/test_kaska/data'
file_sm = 'MNI_2017_sar_sm.tif'

dataset_sm = gdal.Open(os.path.join(path,file_sm))
band1 = dataset_sm.GetRasterBand(1)
mask = band1.ReadAsArray()


#field
field_301 = 0.21249178
field_319 = 0.20654242
field_508 = 0.23555766
field_515 = 0.21090584
field_542 = 0.21022798



mask_301 = band1.ReadAsArray()
mask_301[mask_301!=field_301] = 0

mask_319 = band1.ReadAsArray()
mask_319[mask_319!=field_319] = 0
mask_508 = band1.ReadAsArray()
mask_508[mask_508!=field_508] = 0
mask_515 = band1.ReadAsArray()
mask_515[mask_515!=field_515] = 0
mask_542 = band1.ReadAsArray()
mask_542[mask_542!=field_542] = 0


sm_date, sm_301, sm_319, sm_508, sm_515, sm_542 = get_dataset(dataset_sm,mask_301,mask_319,mask_508,mask_515,mask_542)

dataset_lai = gdal.Open(os.path.join(path,file_lai))
lai_date, lai_301, lai_319, lai_508, lai_515, lai_542 = get_dataset(dataset_lai,mask_301,mask_319,mask_508,mask_515,mask_542)

dataset_sr = gdal.Open(os.path.join(path,file_sr))
sr_date, sr_301, sr_319, sr_508, sr_515, sr_542 = get_dataset(dataset_sr,mask_301,mask_319,mask_508,mask_515,mask_542)

dataset_lai_prior = gdal.Open(os.path.join(datapath,file_lai_prior))
lai_prior_date, lai_prior_301, lai_prior_319, lai_prior_508, lai_prior_515, lai_prior_542 = get_dataset(dataset_lai_prior,mask_301,mask_319,mask_508,mask_515,mask_542)

rmse = rmse_prediction(sm_301[start2:end2],sm_field['301_high']['SM'])
rmse_prior = rmse_prediction(sm_prior_301[start2:end2],sm_field['301_high']['SM'])
slope, intercept, r_value, p_value, std_err = linregress(sm_301[start2:end2],sm_field['301_high']['SM'])


plt.plot(sm_date, sm_301, label='SM retrieved field; RMSE [Vol%]: '+str(rmse*100)[:4]+'; $R^2$: '+str(r_value)[:4], linewidth=2, color='Green')

dstart = datetime.datetime(2017,3,25)
dend = datetime.datetime(2017,7,16)
plt.tick_params(labelsize=12)
plt.xlim(dstart, dend)
plt.legend(prop={'size': 14})
plt.ylim(0.1,0.3)
plt.grid()
plt.ylabel('Soil Moisture [$m^3/m^3$]', fontsize=15)
plt.xlabel('Date', fontsize=15)
plt.savefig(datapath+'/sm_301_field2.png')
plt.close()












path = '/media/tweiss/Daten/new_data'
field = '515_med'

df, df_agro, field_data, field_data_orbit, theta_field, sm_field, height_field, lai_field, vwc_field, pol_field = read_data(path, file_name, extension, field, path_agro, file_name_agro, extension_agro)

start1 = datetime.datetime.combine(sm_prior_date[0], datetime.datetime.min.time())
end1 = datetime.datetime.combine(sm_prior_date[len(sm_prior_date)-1], datetime.datetime.min.time())
start = datetime.datetime.combine(sm_field.index.date[0], datetime.datetime.min.time())
end = datetime.datetime.combine(sm_field.index.date[len(sm_field.index)-1], datetime.datetime.min.time())

start2 = np.argwhere(np.array(sm_date)==start)[0][0]
end2 = np.argwhere(np.array(sm_date)==end)[0][0]

sm_515[start2:end2]

rmse = rmse_prediction(sm_515[start2:end2],sm_field['515_med']['SM'][:-1])
rmse_prior = rmse_prediction(sm_prior_515[start2:end2],sm_field['515_med']['SM'][:-1])
slope, intercept, r_value, p_value, std_err = linregress(sm_515[start2:end2],sm_field['515_med']['SM'][:-1])
slope_p, intercept_p, r_value_p, p_value_p, std_err_p = linregress(sm_prior_515[start2:end2],sm_field['515_med']['SM'][:-1])

# plt.title('SM 515, RMSE [Vol/%]:'+str(rmse*100)[0:4]+' R2:'+str(r_value)[0:4]+' RMSE Prior,insitu:'+str(rmse_prior*100)[0:4])
plt.plot(sm_field,label='SM field measurement')
plt.plot(sm_prior_date, sm_prior_515, label='SM prior; RMSE [Vol%]: '+str(rmse_prior*100)[:4]+'; $R^2$: '+str(r_value_p)[:4])
plt.plot(sm_date, sm_515, label='SM retrieved field; RMSE [Vol%]: '+str(rmse*100)[:4]+'; $R^2$: '+str(r_value)[:4], linewidth=2, color='Green')
plt.legend(prop={'size': 14})
plt.grid()


plt.xlim(start1, end1)
plt.ylim(0.1, 0.35)
plt.ylabel('Soil Moisture [$m^3/m^3$]', fontsize=15)
plt.xlabel('Date', fontsize=15)
plt.tick_params(labelsize=15)
# plt.gca().xaxis.set_major_locator(dates.DayLocator())
plt.gca().xaxis.set_major_formatter(dates.DateFormatter('%Y-%m-%d'))
plt.tight_layout()
plt.savefig(datapath+'/sm_515_field.png',dpi=600)
plt.close()


path = '/media/tweiss/Daten/new_data'
field = '508_high'

df, df_agro, field_data, field_data_orbit, theta_field, sm_field, height_field, lai_field, vwc_field, pol_field = read_data(path, file_name, extension, field, path_agro, file_name_agro, extension_agro)

start1 = datetime.datetime.combine(sm_prior_date[0], datetime.datetime.min.time())
end1 = datetime.datetime.combine(sm_prior_date[len(sm_prior_date)-1], datetime.datetime.min.time())
start = datetime.datetime.combine(sm_field.index.date[0], datetime.datetime.min.time())
end = datetime.datetime.combine(sm_field.index.date[len(sm_field.index)-1], datetime.datetime.min.time())

start2 = np.argwhere(np.array(sm_date)==start)[0][0]
end2 = np.argwhere(np.array(sm_date)==end)[0][0]

sm_508[start2:end2]

rmse = rmse_prediction(sm_508[start2:end2],sm_field['508_high']['SM'][:])
rmse_prior = rmse_prediction(sm_prior_508[start2:end2],sm_field['508_high']['SM'][:])
slope, intercept, r_value, p_value, std_err = linregress(sm_508[start2:end2],sm_field['508_high']['SM'][:])
slope_p, intercept_p, r_value_p, p_value_p, std_err_p = linregress(sm_prior_508[start2:end2],sm_field['508_high']['SM'][:])

# plt.title('SM 508, RMSE [Vol/%]:'+str(rmse*100)[0:4]+' R2:'+str(r_value)[0:4]+' RMSE Prior,insitu:'+str(rmse_prior*100)[0:4])
plt.plot(sm_field,label='SM field measurement')
plt.plot(sm_prior_date, sm_prior_508, label='SM prior; RMSE [Vol%]: '+str(rmse_prior*100)[:4]+'; $R^2$: '+str(r_value_p)[:4])
plt.plot(sm_date, sm_508, label='SM retrieved field; RMSE [Vol%]: '+str(rmse*100)[:4]+'; $R^2$: '+str(r_value)[:4], linewidth=2, color='Green')
plt.legend(prop={'size': 14})
plt.grid()

plt.xlim(start1, end1)
plt.ylim(0.1, 0.35)
plt.ylabel('Soil Moisture [$m^3/m^3$]', fontsize=15)
plt.xlabel('Date', fontsize=15)
plt.tick_params(labelsize=15)
plt.gca().xaxis.set_major_formatter(dates.DateFormatter('%Y-%m-%d'))
plt.tight_layout()
plt.savefig(datapath+'/sm_508_field.png',dpi=600)
plt.close()


path = '/media/tweiss/Daten/new_data'
field = '542_high'

df, df_agro, field_data, field_data_orbit, theta_field, sm_field, height_field, lai_field, vwc_field, pol_field = read_data(path, file_name, extension, field, path_agro, file_name_agro, extension_agro)

start1 = datetime.datetime.combine(sm_prior_date[0], datetime.datetime.min.time())
end1 = datetime.datetime.combine(sm_prior_date[len(sm_prior_date)-1], datetime.datetime.min.time())
start = datetime.datetime.combine(sm_field.index.date[0], datetime.datetime.min.time())
end = datetime.datetime.combine(sm_field.index.date[len(sm_field.index)-1], datetime.datetime.min.time())
start2 = np.argwhere(np.array(sm_date)==start)[0][0]
end2 = np.argwhere(np.array(sm_date)==end)[0][0]

sm_542[start2:end2]

rmse = rmse_prediction(sm_542[start2:end2],sm_field['542_high']['SM'][:])
rmse_prior = rmse_prediction(sm_prior_542[start2:end2],sm_field['542_high']['SM'][:])
slope, intercept, r_value, p_value, std_err = linregress(sm_542[start2:end2],sm_field['542_high']['SM'][:])
slope_p, intercept_p, r_value_p, p_value_p, std_err_p = linregress(sm_prior_542[start2:end2],sm_field['542_high']['SM'][:])

# plt.title('SM 542, RMSE [Vol/%]:'+str(rmse*100)[0:4]+' R2:'+str(r_value)[0:4]+' RMSE Prior,insitu:'+str(rmse_prior*100)[0:4])
plt.plot(sm_field,label='SM field measurement')
plt.plot(sm_prior_date, sm_prior_542, label='SM prior; RMSE [Vol%]: '+str(rmse_prior*100)[:4]+'; $R^2$: '+str(r_value_p)[:4])
plt.plot(sm_date, sm_542, label='SM retrieved field; RMSE [Vol%]: '+str(rmse*100)[:4]+'; $R^2$: '+str(r_value)[:4], linewidth=2, color='Green')
plt.legend(prop={'size': 14})
plt.grid()

plt.xlim(start1, end1)
plt.ylim(0.1, 0.35)
plt.ylabel('Soil Moisture [$m^3/m^3$]', fontsize=15)
plt.xlabel('Date', fontsize=15)
plt.tick_params(labelsize=15)
plt.gca().xaxis.set_major_formatter(dates.DateFormatter('%Y-%m-%d'))
plt.tight_layout()
plt.savefig(datapath+'/sm_542_field.png',dpi=600)
plt.close()

path = '/media/tweiss/Daten/new_data'
field = '301_high'

df, df_agro, field_data, field_data_orbit, theta_field, sm_field, height_field, lai_field, vwc_field, pol_field = read_data(path, file_name, extension, field, path_agro, file_name_agro, extension_agro)

start1 = datetime.datetime.combine(sm_prior_date[0], datetime.datetime.min.time())
end1 = datetime.datetime.combine(sm_prior_date[len(sm_prior_date)-1], datetime.datetime.min.time())
start = datetime.datetime.combine(sm_field.index.date[0], datetime.datetime.min.time())
end = datetime.datetime.combine(sm_field.index.date[len(sm_field.index)-1], datetime.datetime.min.time())
start2 = np.argwhere(np.array(sm_date)==start)[0][0]
end2 = np.argwhere(np.array(sm_date)==end)[0][0]

sm_301[start2:end2]

rmse = rmse_prediction(sm_301[start2:end2],sm_field['301_high']['SM'][:])
rmse_prior = rmse_prediction(sm_prior_301[start2:end2],sm_field['301_high']['SM'][:])
slope, intercept, r_value, p_value, std_err = linregress(sm_301[start2:end2],sm_field['301_high']['SM'][:])
slope_p, intercept_p, r_value_p, p_value_p, std_err_p = linregress(sm_prior_301[start2:end2],sm_field['301_high']['SM'][:])

# plt.title('SM 301, RMSE [Vol/%]:'+str(rmse*100)[0:4]+' R2:'+str(r_value)[0:4]+' RMSE Prior,insitu:'+str(rmse_prior*100)[0:4])
plt.plot(sm_field,label='SM field measurement')
plt.plot(sm_prior_date, sm_prior_301, label='SM prior; RMSE [Vol%]: '+str(rmse_prior*100)[:4]+'; $R^2$: '+str(r_value_p)[:4])
plt.plot(sm_date, sm_301, label='SM retrieved field; RMSE [Vol%]: '+str(rmse*100)[:4]+'; $R^2$: '+str(r_value)[:4], linewidth=2, color='Green')
plt.legend(prop={'size': 14})
plt.grid()

plt.xlim(start1, end1)
plt.ylim(0.1, 0.35)
plt.ylabel('Soil Moisture [$m^3/m^3$]', fontsize=15)
plt.xlabel('Date', fontsize=15)
plt.tick_params(labelsize=15)
plt.gca().xaxis.set_major_formatter(dates.DateFormatter('%Y-%m-%d'))
plt.tight_layout()
plt.savefig(datapath+'/sm_301_field.png',dpi=600)
plt.close()
