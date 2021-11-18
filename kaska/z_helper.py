
import pandas as pd
import numpy as np
import scipy.stats
import os
import pdb
### Helper functions for plots###
#-------------------------------


# Helper functions for statistical parameters
#--------------------------------------------
def rmse_prediction(predictions, targets):
    """ calculation of RMSE """
    return np.sqrt(np.nanmean((predictions - targets) ** 2))

def bias_prediction(predictions, targets):
    """ calculation of bias """
    return np.nanmean(predictions - targets)

def bias_advanced(predictions, targets):
    xxx = predictions.values - targets.values
    xxx[xxx>1.5]=np.nan
    xxx[xxx<(-1.5)]=np.nan
    length = int(len(xxx)/2)

    return np.nanmean(xxx[:length]), np.nanmean(xxx[length:])

def ubrmse_prediction(rmse,bias):
    """ calculation of unbiased RMSE """
    return np.sqrt(rmse ** 2 - bias ** 2)

def linregress(predictions, targets):
    """ Calculate a linear least-squares regression for two sets of measurements """

    # get rid of NaN values
    predictions_new, targets_new = nan_values(predictions, targets)

    # linregress calculation
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(predictions_new, targets_new)
    return slope, intercept, r_value, p_value, std_err

def nan_values(predictions, targets):
    """ get rid of nan values"""
    predictions2 = predictions[~np.isnan(predictions)]
    targets2 = targets[~np.isnan(predictions)]
    predictions3 = predictions2[~np.isnan(targets2)]
    targets3 = targets2[~np.isnan(targets2)]
    return predictions3, targets3

# provide in-situ data
#-------------------------------
def read_mni_data(path, file_name, extension, field, sep=','):
    """ read MNI campaign data """
    df = pd.io.parsers.read_csv(os.path.join(path, file_name + extension), header=[0, 1], sep=sep)
    df = df.set_index(pd.to_datetime(df[field]['date']))
    df = df.drop(df.filter(like='date'), axis=1)
    return df

def read_agrometeo(path, file_name, extension, sep=';', decimal=','):
    """ read agro-meteorological station (hourly data) """
    df = pd.read_csv(os.path.join(path, file_name + extension), sep=sep, decimal=decimal)

    # df['SUM_NN050'] = df['SUM_NN050'].str.replace(',','.')
    # df['SUM_NN050'] = df['SUM_NN050'].str.replace('-','0').astype(float)

    # df['date'] = df['Tag'] + ' ' + df['Stunde']
    df['date'] = df['Tag']
    # df = df.set_index(pd.to_datetime(df['date'], format='%d.%m.%Y %H:%S'))
    df = df.set_index(pd.to_datetime(df['date'], format='%d.%m.%Y'))
    return df

def filter_relativorbit(data, field, orbit1, orbit2=None, orbit3=None, orbit4=None):
    """ data filter for relativ orbits """
    output = data[[(check == orbit1 or check == orbit2 or check == orbit3 or check == orbit4) for check in data[(field,'relativeorbit')]]]
    return output

def read_data(path, file_name, extension, field, path_agro, file_name_agro, extension_agro, pol, orbit1=None, orbit2=None, orbit3=None, orbit4=None):
    """ return all in-situ data """

    # Read MNI data
    df = read_mni_data(path, file_name, extension, field)

    # Read agro-meteorological station
    df_agro = read_agrometeo(path_agro, file_name_agro, extension_agro)

    # filter for field
    field_data = df.filter(like=field)

    # filter for relativorbit
    if orbit1 != None:
        field_data_orbit = filter_relativorbit(field_data, field, orbit1, orbit2, orbit3, orbit4)
        field_data = field_data_orbit
    else:
        field_data_orbit = None

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
    vv_field = field_data.filter(like='sigma_sentinel_vv')
    vh_field = field_data.filter(like='sigma_sentinel_vh')
    relativeorbit = field_data.filter(like='relativeorbit')
    vwcpro_field = field_data.filter(like='watercontentpro')
    return df, df_agro, field_data, field_data_orbit, theta_field, sm_field, height_field, lai_field, vwc_field, pol_field, vv_field, vh_field, relativeorbit, vwcpro_field

# Hanning smoother
#---------------------------------------------------------
def smooth(x,window_len=11,window='hanning'):
        if x.ndim != 1:
                raise ValueError #, "smooth only accepts 1 dimension arrays."
        if x.size < window_len:
                raise ValueError #, "Input vector needs to be bigger than window size."
        if window_len<3:
                return x
        if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
                raise ValueError #, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
        s=np.r_[2*x[0]-x[window_len-1::-1],x,2*x[-1]-x[-1:-window_len:-1]]
        if window == 'flat': #moving average
                w=np.ones(window_len,'d')
        else:
                w=eval('np.'+window+'(window_len)')
        y=np.convolve(w/w.sum(),s,mode='same')
        return y[window_len:-window_len+1]

