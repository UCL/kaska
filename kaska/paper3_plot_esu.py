import numpy as np
import pdb
from osgeo import gdal
import matplotlib.pyplot as plt
from z_helper import *
import datetime
import seaborn as sns
from matplotlib.colors import ListedColormap
from watercloudmodel_vwc_rms import cost_function_vwc, ssrt_jac_vwc, ssrt_vwc
from matplotlib import gridspec
from pandas.plotting import register_matplotlib_converters
import matplotlib.dates as mdates

class plot_esu(object):
    """Plotting scatterplots"""

    def __init__(self, years, esus, passes, esu_size_tiff):
        self.esus = esus
        self.years = years
        self.passes = passes

        if not os.path.exists('/media/tweiss/Work/Paper3_down/'+passes+'/esu'):
            os.makedirs('/media/tweiss/Work/Paper3_down/'+passes+'/esu')
        if not os.path.exists('/media/tweiss/Work/Paper3_down/'+passes+'/boxplot'):
            os.makedirs('/media/tweiss/Work/Paper3_down/'+passes+'/boxplot')

        if not os.path.exists('/media/tweiss/Work/Paper3_down/'+passes+'/spatial'):
            os.makedirs('/media/tweiss/Work/Paper3_down/'+passes+'/spatial')

        self.plot(years, esus, passes, esu_size_tiff)

    def extraction(self,var,state_mask,mask_time):

        xxx = np.copy(var)
        xxx[:,~state_mask]=np.nan
        xxx = xxx[mask_time,:]
        mean = np.nanmean(xxx,axis=(1,2))
        return mean

    def extraction2(self,var,state_mask,mask_time):

        xxx = np.copy(var)
        xxx[:,~state_mask]=np.nan
        xxx = xxx[mask_time,:]
        return xxx

    def plot(self, years, esus, passes, esu_size_tiff):
        """
        years = ['2017', '2018']

        esus = ['high', 'med', 'low']

        esu_size_tiff = '_ESU_buffer_100.tif'
        """

        fig, ax = plt.subplots(figsize=(20, 15))

        insitu_all_years = []
        mean_all_years = []
        mean_all_bias_years = []
        for year in years:
            var_sm = np.load('/media/tweiss/Work/Paper3_down/'+passes+'/'+year+'_multi_'+'sm'+'.npy')
            var_sm_api = np.load('/media/tweiss/Work/Paper3_down/'+passes+'/'+year+'_multi_'+'input_sm_api'+'.npy')
            var_vwc_input = np.load('/media/tweiss/Work/Paper3_down/'+passes+'/'+year+'_multi_'+'input_vwc'+'.npy')
            var_vwc_output = np.load('/media/tweiss/Work/Paper3_down/'+passes+'/'+year+'_multi_'+'vwc'+'.npy')
            var_b = np.load('/media/tweiss/Work/Paper3_down/'+passes+'/'+year+'_multi_'+'b'+'.npy')
            var_rms = np.load('/media/tweiss/Work/Paper3_down/'+passes+'/'+year+'_multi_'+'rms'+'.npy')
            var_vv_input = np.load('/media/tweiss/Work/Paper3_down/'+passes+'/'+year+'_multi_'+'input_vv'+'.npy')
            var_theta_input = np.load('/media/tweiss/Work/Paper3_down/'+passes+'/'+year+'_multi_'+'input_theta'+'.npy')

            time = np.load('/media/tweiss/Work/Paper3_down/'+passes+'/'+year+'_multi_times.npy',allow_pickle=True)

            file = '/media/tweiss/Work/z_final_mni_data_2017/new_in_situ_s1multi_buffer_100_'+year+'_paper3.csv'

            data = pd.read_csv(file,header=[0,1],index_col=1)

            if year == '2017':
                fields = ['301','319','542','508','515']
                # fields = ['301','319','508']
            if year == '2018':
                fields = ['317','410','525','508']

            yy = 0.1
            insitu_all = []
            mean_all = []
            mean_all_bias = []
            rf = []
            bf = []
            uf = []

            meteo = pd.read_csv('/media/tweiss/Work/Paper3_down/GIS/Eichenried_0101'+year+'_3112'+year+'_hourly.csv', sep=';', decimal=',')
            if year == '2017':
                meteo2 = meteo.stack().str.replace(',','.').unstack()
                meteo2['SUM']= pd.to_numeric(meteo2['SUM_NN050'],errors='coerce')
                meteo2['date'] = pd.to_datetime(meteo2['Tag']+' '+meteo2['Stunde'])
                s = meteo2.resample('d', on='date')['SUM'].sum()

            elif year == '2018':
                meteo['date'] = pd.to_datetime(meteo['Tag']+' '+meteo['Stunde'])
                s = meteo.resample('d', on='date')['SUM_NN050'].sum()
            else:
                s = None

            fig, ax = plt.subplots(figsize=(20, 15))

            self.plot_spatial(var_sm,year,passes,time,s=s,par='sm')
            # self.plot_spatial2(var_sm,year,passes,time,s=s,par='sm')
            self.plot_spatial(var_sm_api,year,passes,time,s=s,par='sm_api')

            self.boxplot_sm_area(var_sm,year,time,passes,meteo=s)

            for field in fields:

                insitu_field = []
                mean_field = []
                for esu in esus:

                    g = gdal.Open('/media/tweiss/Work/Paper3_down/GIS/'+year+esu_size_tiff)
                    state_mask = g.ReadAsArray().astype(np.int)

                    state_mask = self.state_mask(year,field,esu,state_mask)
                    data_field = data.filter(like=field).filter(like=esu)
                    if year == '2018':
                        if (field == '410') or (field == '508'):
                            data_field = data_field.filter([(field+'_'+esu,'SM')]).dropna()
                        else:
                            data_field = data_field.filter([(field+'_'+esu,'SM'),(field+'_'+esu,'VWC')]).dropna()
                    data_field.index = pd.to_datetime(data_field.index)
                    # data_field = data_field.dropna()
                    date = data_field.index

                    time2 = pd.to_datetime(time)
                    time2 = time2.strftime('%Y-%m-%d')
                    date2 = date.strftime('%Y-%m-%d')
                    mask_time = np.isin(time2,date2)
                    times = pd.to_datetime(date2)

                    sm = self.extraction(var_sm,state_mask,mask_time)
                    sm_api = self.extraction(var_sm_api,state_mask,mask_time)
                    vwc_input = self.extraction(var_vwc_input,state_mask,mask_time)
                    vwc_output = self.extraction(var_vwc_output,state_mask,mask_time)
                    b = self.extraction(var_b,state_mask,mask_time)
                    rms = self.extraction(var_rms,state_mask,mask_time)
                    vv = self.extraction(var_vv_input,state_mask,mask_time)
                    theta = self.extraction(var_theta_input,state_mask,mask_time)

                    sm_insitu = data_field.filter(like='SM').values.flatten()
                    if year == '2018':
                        if (field == '410') or (field == '508'):
                            pass
                        else:
                            vwc_insitu = data_field.filter(like='VWC').values.flatten()


                    fig, ax = plt.subplots(figsize=(17, 13))
                    gs = gridspec.GridSpec(5, 1, height_ratios=[5, 5, 5, 5, 5])
                    ax = plt.subplot(gs[0])

                    ax.plot(times,sm_insitu, label='insitu')

                    rmse_vv = rmse_prediction(sm_insitu,sm_api)
                    bias_vv = bias_prediction(sm_insitu,sm_api)
                    ubrmse_vv = ubrmse_prediction(rmse_vv,bias_vv)

                    slope, intercept, r_value, p_value, std_err= linregress(sm_insitu,sm)
                    ax.plot(times,sm_api, label='prior RMSE:'+str(rmse_vv)[0:6]+' ubRMSE:'+str(ubrmse_vv)[0:6])

                    rmse_vv = rmse_prediction(sm_insitu,sm)
                    bias_vv = bias_prediction(sm_insitu,sm)
                    ubrmse_vv = ubrmse_prediction(rmse_vv,bias_vv)
                    ax.plot(times,sm, label='model RMSE:'+str(rmse_vv)[0:6]+' ubRMSE:'+str(ubrmse_vv)[0:6] + ' R2:'+str(r_value)[0:4])
                    plt.ylabel('Soil moisture', fontsize=18)
                    plt.ylim(0.05,0.45)
                    plt.grid(linestyle='dotted')
                    plt.legend()
                    plt.subplots_adjust(hspace=.0)
                    plt.setp(ax.get_xticklabels(), visible=False)

                    ax1 = plt.subplot(gs[1])

                    if year == '2018':
                        if (field == '410') or (field == '508'):
                            pass
                        else:
                            ax1.plot(times,vwc_insitu,label='insitu')
                    ax1.plot(times,vwc_input,label='input vwc')
                    ax1.plot(times,vwc_output,label='model vwc')
                    plt.ylabel('VWC', fontsize=18)
                    plt.ylim(0,6)
                    plt.grid(linestyle='dotted')
                    plt.legend()
                    plt.subplots_adjust(hspace=.0)
                    plt.setp(ax1.get_xticklabels(), visible=False)

                    ax2 = plt.subplot(gs[2])

                    ax2.plot(times,b,label='model b')

                    plt.ylabel('b', fontsize=18)
                    plt.ylim(0,1)
                    plt.grid(linestyle='dotted')
                    plt.legend()
                    plt.subplots_adjust(hspace=.0)
                    plt.setp(ax1.get_xticklabels(), visible=False)

                    ax3 = plt.subplot(gs[4])
                    omega = 0.027
                    sigma_vv, vv_g, vv_c = ssrt_vwc(sm, vwc_output, rms, omega, b, theta)

                    ax3.plot(times,10*np.log10(vv),label='S1')
                    ax3.plot(times,10*np.log10(sigma_vv),label='model')
                    ax3.plot(times,10*np.log10(vv_g),label='ground')
                    ax3.plot(times,10*np.log10(vv_c),label='canopy')
                    plt.ylabel('VV [dB]', fontsize=18)
                    plt.ylim(-30,-5)
                    plt.grid(linestyle='dotted')
                    plt.legend()
                    # plt.setp(ax1.get_xticklabels(), visible=False)

                    ax4 = plt.subplot(gs[3])

                    ax4.plot(times,rms,label='model rms')
                    # ax4.plot(times,b_old,label='b calibrated')

                    plt.ylabel('rms', fontsize=18)
                    plt.ylim(0.005,0.03)
                    plt.grid(linestyle='dotted')
                    plt.legend()
                    plt.subplots_adjust(hspace=.0)
                    plt.setp(ax4.get_xticklabels(), visible=False)

                    ax3.set_xlabel('Date', fontsize=18)
                    # plt.tick_params(labelsize=17)
                    plt.subplots_adjust(hspace=.0)
                    rmse_vv = rmse_prediction(sm_insitu,sm)
                    bias_vv = bias_prediction(sm_insitu,sm)
                    ubrmse_vv = ubrmse_prediction(rmse_vv,bias_vv)

                    slope, intercept, r_value, p_value, std_err= linregress(sm_insitu,sm)

                    plt.savefig('/media/tweiss/Work/Paper3_down/'+passes+'/esu/'+year+'_'+field+'_'+esu+'.png', bbox_inches = 'tight')

                    plt.close()


                    vwc_insitu = data_field.filter(like='VWC').values.flatten()

                    if year == '2018':
                        if (field == '410') or (field == '508'):
                            pass
                        else:
                            self.plot_vwc_b(times, vwc_output, vwc_insitu, b, year, field, esu, passes)
                    else:
                        self.plot_vwc_b(times, vwc_output, vwc_insitu, b, year, field, esu, passes)

                    box_sm = self.extraction2(var_sm,state_mask,mask_time)
                    box_api = self.extraction2(var_sm_api,state_mask,mask_time)

                    self.boxplot3(box_sm,'sm',field,esu,year,times,passes,box_api,sm_insitu,s)


    def plot_vwc_b(self, times, vwc_output,vwc_insitu, b, year, field, esu, passes):

        fig, ax = plt.subplots(figsize=(20, 7))
        gs = gridspec.GridSpec(2, 1, height_ratios=[5, 5])
        ax = plt.subplot(gs[0])

        mask_x = np.isnan(vwc_insitu)

        ax.plot(times[~mask_x],vwc_output[~mask_x],label='VWC - Sentinel-2')
        ax.plot(times[~mask_x],vwc_insitu[~mask_x],label='VWC - in-situ measurements')
        plt.ylabel('VWC [kg/m²]', fontsize=18)
        plt.ylim(0,7)
        plt.grid(linestyle='dotted')
        plt.legend()
        plt.subplots_adjust(hspace=.0)
        plt.setp(ax.get_xticklabels(), visible=False)

        ax1 = plt.subplot(gs[1])

        ax1.plot(times[~mask_x],b[~mask_x], label="b'", color='green')

        plt.ylabel("b'", fontsize=18)
        plt.ylim(0,0.95)
        plt.grid(linestyle='dotted')
        plt.legend()

        slope, intercept, r_value, p_value, std_err = linregress(vwc_output[~mask_x],vwc_insitu[~mask_x])

        plt.savefig('/media/tweiss/Work/Paper3_down/'+passes+'/esu/vwcb_'+year+'_'+field+'_'+esu+str(r_value)[0:4]+'.png', bbox_inches = 'tight')

        plt.close()



    def state_mask(self,year,field,esu,state_mask):

        if year == '2017':
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
        elif year == '2018':
            if field == '317' and esu == 'high':
                mask_value = 4
                state_mask = state_mask==mask_value
            elif field == '317' and esu == 'med':
                mask_value = 6
                state_mask = state_mask==mask_value
            elif field == '317' and esu == 'low':
                mask_value = 5
                state_mask = state_mask==mask_value
            elif field == '410' and esu == 'high':
                mask_value = 7
                state_mask = state_mask==mask_value
            elif field == '410' and esu == 'med':
                mask_value = 9
                state_mask = state_mask==mask_value
            elif field == '410' and esu == 'low':
                mask_value = 8
                state_mask = state_mask==mask_value
            elif field == '508' and esu == 'high':
                mask_value = 10
                state_mask = state_mask==mask_value
            elif field == '508' and esu == 'med':
                mask_value = 12
                state_mask = state_mask==mask_value
            elif field == '508' and esu == 'low':
                mask_value = 11
                state_mask = state_mask==mask_value
            elif field == '525' and esu == 'high':
                mask_value = 13
                state_mask = state_mask==mask_value
            elif field == '525' and esu == 'med':
                mask_value = 15
                state_mask = state_mask==mask_value
            elif field == '525' and esu == 'low':
                mask_value = 14
                state_mask = state_mask==mask_value
            else:
                state_mask = 0
        else:
            state_mask = 0

        return state_mask


    def boxplot3(self,var_multi,par,field,esu,year,time,passes,sm_api,sm_insitu,meteo=None):
        f, ax = plt.subplots(figsize=(20, 15))

        xx = var_multi.reshape(var_multi.shape[0], (var_multi.shape[1]*var_multi.shape[2]))
        sns.boxplot(np.repeat(np.arange(len(xx)), len(xx[0])), xx.flatten(), color='skyblue')

        sm_api2 = np.nanmean(sm_api,axis=(1,2))
        ax.plot(sm_api2,'r-o',linewidth=4, label='SM Api')
        ax.plot(sm_insitu,'b-o',linewidth=4, label = 'SM insitu')
        ind = list(range(1,len(time)+1))
        time2 = [i.strftime('%d-%m') for i in time]
        plt.xticks(ind,time2, rotation=45)
        ax.set_ylabel('SM')
        plt.legend()
        if (year == '2017') or (year == '2018'):
            ax2 = ax.twinx()
            mask_time2 = np.isin(meteo.index,time)

            ax2.bar(np.arange(len(meteo[mask_time2])),meteo[mask_time2])
            ax2.set_ylim(0,150)
            ax2.set_xticks([])
            ax2.set_ylabel('Precipitation')
        plt.xticks(ind,time2, rotation=45)
        plt.savefig('/media/tweiss/Work/Paper3_down/'+passes+'/boxplot/'+par+str(field)+'_'+esu+'_'+str(year), bbox_inches='tight')
        plt.close()



    def plot_spatial(self, sm, year, passes, id, s=None, par='xx'):

        id2 = pd.to_datetime(id)
        id2 = id2.strftime('%Y-%m-%d')
        id2 = pd.to_datetime(id2)

        f, ax = plt.subplots(figsize=(20, 15))
        time2 = [i.strftime('%d-%m') for i in id2]


        sm[sm == 0] = 'nan'

        sm_mean = np.mean(sm,axis=(1,2))
        xxx = np.arange(len(sm_mean))
        sm_std = np.std(sm,axis=(1,2))
        ax.errorbar(xxx,sm_mean,sm_std,fmt='-o')
        ax.set_xticks([])
        ax2 = ax.twinx()
        mask_time2 = np.isin(s.index,id2)

        ax2.bar(np.arange(len(s[mask_time2])),s[mask_time2])
        ax2.set_ylim(0,150)
        ax2.set_xticks([])
        ax2.set_ylabel('Precipitation')
        ind = list(range(1,len(time2)+1))
        plt.xticks(ind,time2, rotation=45)
        plt.savefig('/media/tweiss/Work/Paper3_down/'+passes+'/spatial/'+par+'_'+year, bbox_inches='tight')
        plt.close()

        for iii in np.arange(len(sm)):

            f, ax = plt.subplots(figsize=(20, 15))
            cmap = plt.cm.viridis_r
            # label = 'Soil Moisture [m$^3$/m$^3$]'
            label = 'Bodenfeuchte [m$^3$/m$^3$]'
            cmap.set_bad(color='white')
            plt.rcParams['axes.labelsize'] = 20

            im1 = ax.imshow(sm[iii,:,:], vmin=0.1, vmax=0.4, cmap=cmap, aspect='auto')
            # ax.set_title('sm_'+year, fontsize=20)
            f.subplots_adjust(right=0.85)
            cbar_ax = f.add_axes([0.88, 0.15, 0.04, 0.7])
            ticklabs = cbar_ax.get_yticklabels()
            cbar_ax.set_yticklabels(ticklabs, fontsize=20)
            f.colorbar(im1, cax=cbar_ax, label=label)

            ax.xaxis.set_tick_params(labelsize=20)
            ax.yaxis.set_tick_params(labelsize=20)

            plt.setp(ax, xticks=[22*6.1, 2*22*6.1, 3*22*6.1, 4*22*6.1, 5*22*6.1, 6*22*6.1], xticklabels=['11.64°E', '11.66°E', '11.68°E', '11.70°E', '11.72°E', '11.74°E'], yticks=[22*6.12, (22+39)*6.12, (22+39*2)*6.12, (22+39*3)*6.12], yticklabels=['48.30°N', '48.28°N', '48.26°N', '48.24°N'])
            ax.set_ylim(len(sm[0]),0)
            mean = np.nanmean(sm[iii,:,:])

            plt.savefig('/media/tweiss/Work/Paper3_down/'+passes+'/spatial/'+par+'_'+year+'_'+str(id[iii])[:10]+'_'+str(mean)[0:5]+'.png', bbox_inches='tight')

        #     plt.close()

    def plot_spatial2(self, sm, year, passes, id, s=None, par='xx'):

        id2 = pd.to_datetime(id)
        id2 = id2.strftime('%Y-%m-%d')
        id2 = pd.to_datetime(id2)

        # f, ax = plt.subplots(figsize=(20, 15))
        # time2 = [i.strftime('%d-%m') for i in id2]


        sm[sm == 0] = 'nan'

        # sm_mean = np.mean(sm,axis=(1,2))
        # xxx = np.arange(len(sm_mean))
        # sm_std = np.std(sm,axis=(1,2))
        # ax.errorbar(xxx,sm_mean,sm_std,fmt='-o')
        # ax.set_xticks([])
        # ax2 = ax.twinx()
        # mask_time2 = np.isin(s.index,id2)

        # ax2.bar(np.arange(len(s[mask_time2])),s[mask_time2])
        # ax2.set_ylim(0,150)
        # ax2.set_xticks([])
        # ax2.set_ylabel('Precipitation')
        # ind = list(range(1,len(time2)+1))
        # plt.xticks(ind,time2, rotation=45)
        # plt.savefig('/media/tweiss/Work/Paper3_down/'+passes+'/spatial/'+par+'_'+year, bbox_inches='tight')
        # plt.close()

        for iii in np.arange(len(sm)):

            f, ax = plt.subplots(figsize=(20, 15))
            cmap = plt.cm.viridis_r
            # label = 'Soil Moisture [m$^3$/m$^3$]'
            cmap.set_bad(color='white')
            # plt.rcParams['axes.labelsize'] = 20

            im1 = ax.imshow(sm[iii,:,:], vmin=0.1, vmax=0.4, cmap=cmap, aspect='auto')
            plt.axis('off')

            plt.savefig('/media/tweiss/Work/Paper3_down/'+passes+'/spatial2/'+par+'_'+year+'_'+str(id[iii])[:10]+'_'+'.png', bbox_inches='tight')

        #     plt.close()


    def boxplot_sm_area(self,var_multi,year,time,passes,meteo=None):

        f, ax = plt.subplots(figsize=(20, 15))

        if year == '2017':
            time_final = pd.date_range(start='2017-03-21',end='2017-09-30')
        elif year == '2018':
            time_final = pd.date_range(start='2018-03-21',end='2018-09-30')
        else:
            pass

        new_array = np.zeros((len(time_final),len(var_multi[0,:,0])*len(var_multi[0,0,:])))
        new_array[:] = np.nan

        time3 = pd.to_datetime(time).strftime('%Y-%m-%d')
        time3 = pd.to_datetime(time3)
        mask_time_final = np.isin(time_final,time3)

        xx = var_multi.reshape(var_multi.shape[0], (var_multi.shape[1]*var_multi.shape[2]))
        new_array[mask_time_final] = xx

        new_array2 = new_array.flatten()
        id = np.repeat(np.arange(0,len(new_array)),len(new_array[0]))
        df = pd.DataFrame({'id':id,'value':new_array2},columns=['id','value'])
        df2 = df.head(new_array[0].shape[0]*25)

        sns.boxplot(data=df,x='id',y='value', color='skyblue', showfliers = False)
        # sns.violinplot(data=df, y='value', x='id', color='skyblue')

        ind = list(range(1,len(time)+1))
        time2 = [i.strftime('%d-%m') for i in time]
        # plt.xticks(ind,time2, rotation=45)
        ax.set_ylabel('SM')
        ax.set_ylim(0.1,0.45)
        freq = int(10)
        ax.set_xticklabels(time_final[::freq].strftime('%d-%m'))
        xtix = ax.get_xticks()
        ax.set_xticks(xtix[::freq])
        # f.autofmt_xdate()
        # plt.legend()



        if (year == '2017') or (year == '2018'):
            ax2 = ax.twinx()
            mask_time2 = np.isin(meteo.index,time_final)

            ax2.bar(np.arange(len(meteo[mask_time2])),meteo[mask_time2])
            ax2.set_ylim(0,150)
            ax2.set_xticks([])
            ax2.set_ylabel('Precipitation [mm]')

        ax.set_xticks(xtix[::freq])
        plt.savefig('/media/tweiss/Work/Paper3_down/'+passes+'/boxplot/'+'sm_area_'+str(year), bbox_inches='tight')
        plt.close()

        if year == '2017':
            # time_invest = pd.date_range(start="2017-05-29",end="2017-06-06")
            time_invest = pd.date_range(start="2017-03-28",end="2017-07-31")
            time_mask_invest = np.isin(time_final,time_invest)
            data_invest = new_array[time_mask_invest]

            id_invest = np.repeat(np.arange(0,len(data_invest)),len(data_invest[0]))
            df_invest = pd.DataFrame({'id':id_invest,'value':data_invest.flatten()},columns=['id','value'])


            f, ax = plt.subplots(figsize=(30, 10))

            gs = gridspec.GridSpec(2, 1, height_ratios=[5,5])
            ax = plt.subplot(gs[0])
            plt.tick_params(labelsize=16)

            sns.boxplot(data=df_invest,x='id',y='value', color='skyblue', showfliers = False)
            ax.set_ylabel('SM',fontsize=16)
            ax.set_ylim(0.1,0.4)
            ax.set_xlabel('')
            ax.set_xticklabels(time_invest.strftime('%Y-%m-%d'))
            xtix = ax.get_xticks()
            ax.set_xticks(xtix)
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.subplots_adjust(hspace=.0)
            ax1 = plt.subplot(gs[1])
            plt.tick_params(labelsize=16)
            plt.rcParams.update({'font.size': 16})
            meteo2 = self.import_meteo()


            mask_time3 = np.isin(meteo2.index,time_invest)
            hm = meteo2[mask_time3]
            # hm = meteo2.iloc[145:160]
            ax0 = hm.plot.bar(ax=ax1,rot=0, fontsize=16)
            xxx = [pd_datetime.strftime("%Y-%m-%d") for pd_datetime in hm.index]
            # ax0.set_xticklabels([pd_datetime.strftime("%Y-%m-%d") for pd_datetime in hm.index])
            ax0.set_xticklabels(xxx)
            ax0.set_ylabel('Precipitation [mm]',fontsize=16)
            ax0.set_xlabel('Date')
            plt.xticks([4, 34, 65, 95])
            ax0.set_xticklabels([xxx[4],xxx[34],xxx[65],xxx[95]])
            plt.savefig('/media/tweiss/Work/Paper3_down/'+passes+'/boxplot/'+'precip_2017', bbox_inches='tight')
            plt.close()

        if year == '2018':
            # time_invest = pd.date_range(start="2018-07-16",end="2018-07-26")
            time_invest = pd.date_range(start="2018-03-28",end="2018-07-31")
            time_mask_invest = np.isin(time_final,time_invest)
            data_invest = new_array[time_mask_invest]

            id_invest = np.repeat(np.arange(0,len(data_invest)),len(data_invest[0]))
            df_invest = pd.DataFrame({'id':id_invest,'value':data_invest.flatten()},columns=['id','value'])


            f, ax = plt.subplots(figsize=(30, 10))

            gs = gridspec.GridSpec(2, 1, height_ratios=[5,5])
            ax = plt.subplot(gs[0])
            plt.tick_params(labelsize=16)

            sns.boxplot(data=df_invest,x='id',y='value', color='skyblue', showfliers = False)
            ax.set_ylabel('SM',fontsize=16)
            ax.set_ylim(0.1,0.4)
            ax.set_xlabel('')
            ax.set_xticklabels(time_invest.strftime('%Y-%m-%d'))
            xtix = ax.get_xticks()
            ax.set_xticks(xtix)
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.subplots_adjust(hspace=.0)
            ax1 = plt.subplot(gs[1])
            plt.tick_params(labelsize=16)
            plt.rcParams.update({'font.size': 16})
            meteo2 = self.import_meteo()


            mask_time3 = np.isin(meteo2.index,time_invest)
            hm = meteo2[mask_time3]
            # hm = meteo2.iloc[145:160]
            ax0 = hm.plot.bar(ax=ax1,rot=0, fontsize=16)
            xxx = [pd_datetime.strftime("%Y-%m-%d") for pd_datetime in hm.index]
            # ax0.set_xticklabels([pd_datetime.strftime("%Y-%m-%d") for pd_datetime in hm.index])
            ax0.set_ylabel('Precipitation [mm]',fontsize=16)
            ax0.set_xlabel('Date')
            plt.xticks([4, 34, 65, 95])
            ax0.set_xticklabels([xxx[4],xxx[34],xxx[65],xxx[95]])
            plt.savefig('/media/tweiss/Work/Paper3_down/'+passes+'/boxplot/'+'precip_2018', bbox_inches='tight')
            plt.close()


    def import_meteo(self):
        eichenried = pd.read_csv('/media/tweiss/Work/Paper3_down/Tag_Eichenried.csv', sep=';', decimal=',')
        # freising = pd.read_csv('/media/tweiss/Work/Paper3_down/Tag_Freising.csv', sep=';', decimal=',')
        grub = pd.read_csv('/media/tweiss/Work/Paper3_down/Tag_Grub.csv', sep=';', decimal=',')

        meteo = pd.merge(eichenried,grub,on='Tag',how='inner')

        meteo.columns = ['date','Eichenried','Grub']
        meteo['date'] = pd.to_datetime(meteo['date'],format='%d.%m.%Y')
        meteo.index=meteo['date']
        meteo = meteo.drop(columns=['date'])

        return meteo



if __name__ == '__main__':

    years = ['2017','2018']
    years = ['2017']
    versions = ['_multi', '_single']
    versions = ['_multi']

    esus = ['high', 'med', 'low']

    esu_size_tiff = '_ESU_buffer_100.tif' # buffer around ESU 100, 50, 30 etc
    passes = 'hm'


    plot_esu(years, esus, passes, esu_size_tiff)

