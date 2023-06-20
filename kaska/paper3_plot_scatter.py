import numpy as np
import pdb
import gdal
import matplotlib.pyplot as plt
from z_helper import *
import datetime
import seaborn as sns
from matplotlib.colors import ListedColormap
import skill_metrics as sm
from matplotlib.lines import Line2D

class plot_scatter(object):
    """Plotting scatterplots"""

    def __init__(self, years, esus, passes, esu_size_tiff):
        self.esus = esus
        self.years = years
        self.passes = passes

        if not os.path.exists('/media/tweiss/Work/Paper3_down/'+passes+'/taylor'):
            os.makedirs('/media/tweiss/Work/Paper3_down/'+passes+'/taylor')

        self.plot(years, esus, passes, esu_size_tiff)
        self.plot(years, esus, passes, esu_size_tiff, 'wheat')
        self.plot(years, esus, passes, esu_size_tiff, 'maize')
        self.plot2(years, esus, passes, esu_size_tiff)

    def plot(self, years, esus, passes, esu_size_tiff,crop=None):
        """
        years = ['2017', '2018']

        esus = ['high', 'med', 'low']

        esu_size_tiff = '_ESU_buffer_100.tif'
        """

        fig, ax = plt.subplots(figsize=(20, 15))

        insitu_all_years = []
        mean_all_years = []
        mean_all_bias_years = []

        ccoef_field = []
        crmsd_field = []
        sdev_field = []
        labels_field = []
        pppp = 0

        for year in years:
            var_multi = np.load('/media/tweiss/Work/Paper3_down/'+passes+'/'+year+'_multi_'+'sm'+'.npy')

            time = np.load('/media/tweiss/Work/Paper3_down/'+passes+'/'+year+'_multi_times.npy',allow_pickle=True)

            file = '/media/tweiss/Work/z_final_mni_data_2017/new_in_situ_s1multi_buffer_100_'+year+'_paper3.csv'

            data = pd.read_csv(file,header=[0,1],index_col=1)

            if year == '2017':
                if crop == 'wheat':
                    fields = ['301','542','508']
                elif crop == 'maize':
                    fields = ['319','515']
                else:
                    fields = ['301','319','542','508','515']
                    crop=''
            if year == '2018':
                if crop == 'wheat':
                    fields = ['317','525']
                elif crop == 'maize':
                    fields = ['410','508']
                else:
                    fields = ['317','410','525','508']
                    crop=''

            yy = 0.1
            insitu_all = []
            mean_all = []
            mean_all_bias = []
            bias_collection = []
            rf = []
            bf = []
            uf = []
            ccoef = []
            crmsd = []
            sdev = []
            labels = []

            ppp = 0


            fig, ax = plt.subplots(figsize=(20, 15))
            for field in fields:

                insitu_field = []
                mean_field = []
                for esu in esus:
                    g = gdal.Open('/media/tweiss/Work/Paper3_down/GIS/'+year+esu_size_tiff)
                    state_mask = g.ReadAsArray().astype(np.int)
                    
                    state_mask = self.state_mask(year,field,esu,state_mask)

                    data_field = data.filter(like=field).filter(like=esu).filter(like='SM')
                    data_field.index = pd.to_datetime(data_field.index)
                    data_field = data_field.dropna()
                    date = data_field.index
                    xxx = np.copy(var_multi)
                    xxx[:,~state_mask]=np.nan
                    time2 = pd.to_datetime(time)
                    time2 = time2.strftime('%Y-%m-%d')
                    date2 = date.strftime('%Y-%m-%d')
                    mask_time = np.isin(time2,date2)
                    yyy = xxx[mask_time,:]
                    mean_rt = np.nanmean(yyy,axis=(1,2))

                    # sm_insitu = np.repeat(data_field.values.flatten(),len(yyy[0][~np.isnan(yyy[0])]))
                    # yyy = yyy[~np.isnan(yyy)]

                    if field == '301':
                        color = 'green'
                    elif field == '319':
                        color = 'red'
                    elif field == '508':
                        color = 'blue'
                    elif field == '515':
                        color = 'orange'
                    elif field == '542':
                        color = 'black'
                    elif field == '317':
                        color = 'grey'
                    elif field == '410':
                        color = 'brown'
                    elif field == '525':
                        color = 'yellow'
                    else:
                        pass

                    insitu = data_field.values.flatten()
                    bias = np.nanmean(insitu - mean_rt)

                    ax.plot(insitu,mean_rt+bias,marker='o',color=color, linestyle='')

                    insitu_all = np.append(insitu_all,insitu)
                    mean_all = np.append(mean_all,mean_rt)
                    insitu_field = np.append(insitu_field,insitu)
                    mean_field = np.append(mean_field,mean_rt)
                    mean_all_bias = np.append(mean_all_bias,mean_rt+bias)
                    bias_collection = np.append(bias_collection,bias)
                    insitu_all_years = np.append(insitu_all_years,insitu)
                    mean_all_years = np.append(mean_all_years,mean_rt)
                    mean_all_bias_years = np.append(mean_all_bias_years,mean_rt+bias)


                    stats = sm.taylor_statistics(mean_rt+bias,insitu,'data')

                    if ppp == 0:
                        ccoef = np.append(ccoef,stats['ccoef'][0])
                        crmsd = np.append(crmsd,stats['crmsd'][0])
                        sdev = np.append(sdev,stats['sdev'][0])
                        labels = np.append(labels,'initial')

                    ccoef = np.append(ccoef,stats['ccoef'][1])
                    crmsd = np.append(crmsd,stats['crmsd'][1])
                    sdev = np.append(sdev,stats['sdev'][1])
                    labels = np.append(labels,field+esu)

                    ppp = ppp+1

                rmse_field = rmse_prediction(insitu_field,mean_field)
                bias_field = bias_prediction(insitu_field,mean_field)
                ubrmse_field = ubrmse_prediction(rmse_field,bias_field)

                stats = sm.taylor_statistics(mean_field+bias_field,insitu_field,'data')

                if pppp == 0:
                    ccoef_field = np.append(ccoef_field,stats['ccoef'][0])
                    crmsd_field = np.append(crmsd_field,stats['crmsd'][0])
                    sdev_field = np.append(sdev_field,stats['sdev'][0])
                    labels_field = np.append(labels_field,'initial')

                    pppp = pppp+1


                ccoef_field = np.append(ccoef_field,stats['ccoef'][1])
                crmsd_field = np.append(crmsd_field,stats['crmsd'][1])
                sdev_field = np.append(sdev_field,stats['sdev'][1])
                labels_field = np.append(labels_field,year + '-' + field)


                yy = yy + 0.02

                plt.text(0.4,yy,field+' rmse:'+str(rmse_field)[0:4]+' ubrmse:'+str(ubrmse_field)[0:4], color=color)


                rf.append(rmse_field)
                bf.append(bias_field)
                uf.append(ubrmse_field)


            ax.set_xlim(0,0.5)
            ax.set_ylim(0,0.5)


            x = [0, 1]
            y = [0, 1]
            ax.plot(x,y)
            plt.ylabel('SM model', fontsize=20)
            plt.xlabel('SM insitu', fontsize=20)

            ax.xaxis.set_tick_params(labelsize=20)
            ax.yaxis.set_tick_params(labelsize=20)

            rmse = rmse_prediction(insitu_all,mean_all)
            bias = bias_prediction(insitu_all,mean_all)
            ubrmse = ubrmse_prediction(rmse,bias)

            yy = yy +0.02

            plt.text(0.4,yy,'rmse all:'+str(rmse_field)[0:4]+' ubrmse all:'+str(ubrmse_field)[0:4], color='black')

            if not os.path.exists('/media/tweiss/Work/Paper3_down/'+passes+'/scatterplot'):
                os.makedirs('/media/tweiss/Work/Paper3_down/'+passes+'/scatterplot')

            plt.savefig('/media/tweiss/Work/Paper3_down/'+passes+'/scatterplot/'+year+crop,bbox_inches='tight')
            plt.close()

            self.plot_scat(insitu_all, mean_all_bias, year, passes, mean_all, bias_collection,crop)

            # self.plot_taylor(ccoef, crmsd, sdev, labels, passes, year)
            # pdb.set_trace()
        if crop == '':
            self.plot_taylor(ccoef_field, crmsd_field, sdev_field, labels_field, passes, year)

        # self.plot_scat(insitu_all_years, mean_all_bias_years, '2017-2018', passes, mean_all, bin_a=40,bin_b=30)

    def plot2(self, years, esus, passes, esu_size_tiff):
        """
        years = ['2017', '2018']

        esus = ['high', 'med', 'low']

        esu_size_tiff = '_ESU_buffer_100.tif'
        """
        crop=''
        fig, ax = plt.subplots(figsize=(20, 15))

        insitu_all_years = []
        mean_all_years = []
        mean_all_bias_years = []

        ccoef_field = []
        crmsd_field = []
        sdev_field = []
        labels_field = []
        pppp = 0

        for year in years:
            var_multi = np.load('/media/tweiss/Work/Paper3_down/'+passes+'/'+year+'_multi_'+'sm'+'.npy')

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
            ccoef = []
            crmsd = []
            sdev = []
            labels = []

            ppp = 0


            fig, ax = plt.subplots(figsize=(20, 15))
            for field in fields:

                insitu_field = []
                mean_field = []

                g = gdal.Open('/media/tweiss/Work/Paper3_down/GIS/'+year+'_ESU_Field_buffer_30.tif')
                state_mask = g.ReadAsArray().astype(np.int)

                state_mask = self.state_mask2(year,field,state_mask)

                data_field = data.filter(like=field).filter(like='SM')
                data_field.index = pd.to_datetime(data_field.index)
                data_field = data_field.dropna()
                date = data_field.index
                xxx = np.copy(var_multi)
                xxx[:,~state_mask]=np.nan
                time2 = pd.to_datetime(time)
                time2 = time2.strftime('%Y-%m-%d')
                date2 = date.strftime('%Y-%m-%d')
                mask_time = np.isin(time2,date2)
                yyy = xxx[mask_time,:]
                mean_rt = np.nanmean(yyy,axis=(1,2))

                # sm_insitu = np.repeat(data_field.values.flatten(),len(yyy[0][~np.isnan(yyy[0])]))
                # yyy = yyy[~np.isnan(yyy)]

                if field == '301':
                    color = 'green'
                elif field == '319':
                    color = 'red'
                elif field == '508':
                    color = 'blue'
                elif field == '515':
                    color = 'orange'
                elif field == '542':
                    color = 'black'
                elif field == '317':
                    color = 'grey'
                elif field == '410':
                    color = 'brown'
                elif field == '525':
                    color = 'yellow'
                else:
                    pass



                if year == '2017':
                    bbch = pd.read_csv('/media/tweiss/Work/z_final_mni_data_2017/bbch_2017.csv',header=[0,1])
                elif year == '2018':
                    bbch = pd.read_csv('/media/tweiss/Work/z_final_mni_data_2017/bbch_2018.csv',header=[0,1])
                else:
                    pass

                bbch_value = 37
                print(year)
                print(field)
                bbch.index = pd.to_datetime(bbch['None','None'])
                bbch_field = bbch.filter(like=field)
                lower37 = bbch_field.loc[bbch_field[field,'BBCH median']>bbch_value]



                pos37 = data_field.index.get_loc(lower37.index[-1],method='nearest')

                if field == '515':
                    data_field = data_field.drop(data_field.columns[0],1)

                mean_rt = mean_rt[0:pos37]
                insitu = data_field.mean(axis=1).values.flatten()[0:pos37]
                # insitu = data_field.mean(axis=1).values.flatten()


                bias = np.nanmean(insitu - mean_rt)

                ax.plot(insitu,mean_rt+bias,marker='o',color=color, linestyle='')

                insitu_all = np.append(insitu_all,insitu)
                mean_all = np.append(mean_all,mean_rt)
                insitu_field = np.append(insitu_field,insitu)
                mean_field = np.append(mean_field,mean_rt)
                mean_all_bias = np.append(mean_all_bias,mean_rt+bias)
                insitu_all_years = np.append(insitu_all_years,insitu)
                mean_all_years = np.append(mean_all_years,mean_rt)
                mean_all_bias_years = np.append(mean_all_bias_years,mean_rt+bias)


                stats = sm.taylor_statistics(mean_rt+bias,insitu,'data')

                rmse_field = rmse_prediction(insitu,mean_rt)
                bias_field = bias_prediction(insitu,mean_rt)
                ubrmse_field = ubrmse_prediction(rmse_field,bias_field)

                stats = sm.taylor_statistics(mean_rt+bias_field,insitu,'data')


                if pppp == 0:
                    ccoef_field = np.append(ccoef_field,stats['ccoef'][0])
                    crmsd_field = np.append(crmsd_field,stats['crmsd'][0])
                    sdev_field = np.append(sdev_field,stats['sdev'][0])
                    labels_field = np.append(labels_field,'initial')

                    pppp = pppp+1


                ccoef_field = np.append(ccoef_field,stats['ccoef'][1])
                crmsd_field = np.append(crmsd_field,stats['crmsd'][1])
                sdev_field = np.append(sdev_field,stats['sdev'][1])
                labels_field = np.append(labels_field,year + '-' + field)


                yy = yy + 0.02

                plt.text(0.4,yy,field+' rmse:'+str(rmse_field)[0:4]+' ubrmse:'+str(ubrmse_field)[0:4], color=color)


                rf.append(rmse_field)
                bf.append(bias_field)
                uf.append(ubrmse_field)


            ax.set_xlim(0,0.5)
            ax.set_ylim(0,0.5)


            x = [0, 1]
            y = [0, 1]
            ax.plot(x,y)
            plt.ylabel('SM model', fontsize=20)
            plt.xlabel('SM insitu', fontsize=20)

            ax.xaxis.set_tick_params(labelsize=20)
            ax.yaxis.set_tick_params(labelsize=20)

            rmse = rmse_prediction(insitu_all,mean_all)
            bias = bias_prediction(insitu_all,mean_all)
            ubrmse = ubrmse_prediction(rmse,bias)

            yy = yy +0.02

            plt.text(0.4,yy,'rmse all:'+str(rmse_field)[0:4]+' ubrmse all:'+str(ubrmse_field)[0:4], color='black')

            if not os.path.exists('/media/tweiss/Work/Paper3_down/'+passes+'/scatterplot'):
                os.makedirs('/media/tweiss/Work/Paper3_down/'+passes+'/scatterplot')

            plt.savefig('/media/tweiss/Work/Paper3_down/'+passes+'/scatterplot/'+year+'_v2',bbox_inches='tight')
            plt.close()

            # self.plot_taylor(ccoef, crmsd, sdev, labels, passes, year)
        # self.plot_taylor(ccoef_field, crmsd_field, sdev_field, labels_field, passes, year, name_ex='_v4_lower'+str(bbch_value))
        self.plot_taylor(ccoef_field, crmsd_field, sdev_field, labels_field, passes, year, name_ex='_v4_higher'+str(bbch_value))
        pdb.set_trace()
        # self.plot_taylor(ccoef_field, crmsd_field, sdev_field, labels_field, passes, year, name_ex='_v4')


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

    def state_mask2(self,year,field,state_mask):

        if year == '2017':
            if field == '515':
                mask_value = 4
                state_mask = state_mask==mask_value
            elif field == '508':
                mask_value = 27
                state_mask = state_mask==mask_value
            elif field == '542':
                mask_value = 8
                state_mask = state_mask==mask_value
            elif field == '319':
                mask_value = 67
                state_mask = state_mask==mask_value
            elif field == '301':
                mask_value = 87
                state_mask = state_mask==mask_value
            else:
                state_mask = 0
        elif year == '2018':
            if field == '317':
                mask_value = 65
                state_mask = state_mask==mask_value
            elif field == '410':
                mask_value = 113
                state_mask = state_mask==mask_value
            elif field == '508':
                mask_value = 27
                state_mask = state_mask==mask_value
            elif field == '525':
                mask_value = 30
                state_mask = state_mask==mask_value
            else:
                state_mask = 0
        else:
            state_mask = 0

        return state_mask


    def plot_scat(self,a,b,year,passes,c,d,crop,bin_a=40,bin_b=25):
        """ """

        if year == '2018':
            bin_a=42
            bin_b=32
        fig, ax = plt.subplots(figsize=(20, 15))
        hhh = ax.hist2d(a, b, bins=(bin_a, bin_b), cmap=plt.cm.jet, cmin=1, vmax=8)
        ax.set_xlim(0.0,0.5)
        ax.set_ylim(0.0,0.5)
        x = [0, 1]
        y = [0, 1]
        ax.plot(x,y)
        plt.ylabel('SM model [m³/m³]',fontsize=20)
        plt.xlabel('SM insitu [m³/m³]',fontsize=20)
        ax.xaxis.set_tick_params(labelsize=20)
        ax.yaxis.set_tick_params(labelsize=20)
        # plt.title('bias corrected')
        plt.rcParams.update({'font.size': 20})
        plt.colorbar(hhh[3],ax=ax).set_label(label='Density Distribution',size=20)

        rmse_field = rmse_prediction(a,c)
        # bias_field = bias_prediction(a,c)
        ubrmse_field = rmse_prediction(a,b)

        plt.text(0.02,0.48,'RMSE: '+str(rmse_field)[0:5]+' m³/m³', fontsize=20)
        plt.text(0.02,0.46,'ubRMSE: '+str(ubrmse_field)[0:5]+' m³/m³', fontsize=20)
        plt.text(0.02,0.44,'Min bias: '+str(np.min(d))[0:5]+' m³/m³'+'; Max bias: '+str(np.max(d))[0:4]+' m³/m³', fontsize=20)
        plt.text(0.02,0.42,'Min model: '+str(min(b))[0:4]+' m³/m³'+'; Max model: '+str(max(b))[0:4]+' m³/m³', fontsize=20)
        plt.text(0.02,0.4,'Min insitu: '+str(min(a))[0:4]+' m³/m³'+'; Max insitu: '+str(max(a))[0:4]+' m³/m³', fontsize=20)

        plt.savefig('/media/tweiss/Work/Paper3_down/'+passes+'/scatterplot/scatterplot_bias_'+year+crop,bbox_inches='tight')
        plt.close()

    def plot_taylor(self, ccoef, crmsd, sdev, labels, passes, year, name_ex=''):

        ### Taylor plot
        #------------------
        # Info: Made some changes within skill_metrics package (rename of RMSD to ubRMSE!)

        # field_short = ['508_high','508_low','508_med','301_high','301_low','301_med','542_high','542_low','542_med']
        marker = ['P','o','X','s','d','^','v','p','h']
        colors = ['b', 'r', 'y', 'm', 'g', 'y']
        fig, ax = plt.subplots(figsize=(14, 10))

        # sm.taylor_diagram(sdev,crmsd,ccoef, styleOBS = '-', colOBS = 'r', markerobs = 'o', titleOBS = 'observation')
        # y = 0
        # yy = 0
        # for k, kk in enumerate(labels):

            # sm.taylor_diagram(np.array(sdev), np.array(crmsd), np.array(ccoef), alpha = 1.0, markercolor=colors[yy], markerSize=8, markerLabel = labels, markerLabelColor = 'b', markerLegend = 'on', colCOR = 'k', colRMS='k', styleOBS = '-', colOBS = 'r', markerobs = 'o', titleOBS = 'Ref')
            # plt.scatter(crmsd[k],ccoef[k],s=80,c=colors[yy],marker=marker[y])
        sdev[0] = 0.042
        sm.taylor_diagram(sdev[0:6],crmsd[0:6],ccoef[0:6], markerLabel = labels[0:6].tolist(),markerLabelColor = 'r', markerColor = 'r',  tickRMS = range(0,60,10), colRMS = 'm', styleRMS = ':', widthRMS = 2.0, titleRMS = 'on', titleRMSDangle = 40.0, colSTD = 'b', styleSTD = '-.', widthSTD = 1.0, titleSTD = 'on', colCOR = 'k', styleCOR = '--', widthCOR = 1.0, titleCOR = 'on', markerSize = 12, markerLegend = 'on')

        sm.taylor_diagram(np.append(sdev[0],sdev[6:]),np.append(crmsd[0],crmsd[6:]),np.append(ccoef[0],ccoef[6:]), overlay = 'on', markerLabel = labels.tolist(), markerColor = 'b', markerLegend = 'on', markerSize = 12)

        # pdb.set_trace()

            # if y == 2:
            #     y = 0
            #     yy = yy+1
            # else:
            #     y = y+1



        # for kk in canopy_list:
        #     for kkk in opt_mod:
        #         fig, ax = plt.subplots(figsize=(8, 6))

                # s1_vv = df_taylor.filter(like=kk).filter(like=kkk).filter(like='S1_vv').values.flatten()
                # model_vv = df_taylor.filter(like=kk).filter(like=kkk).filter(like='biasedmodel_').values.flatten()
                # model_vv_ub = df_taylor.filter(like=kk).filter(like=kkk).filter(like='unbiasedmodeldb').values.flatten()

                # s1_vv = 10*np.log10(s1_vv)
                # model_vv_ub = model_vv_ub

                # predictions = model_vv_ub[~np.isnan(model_vv_ub)]
                # targets = s1_vv[~np.isnan(model_vv_ub)]
                # predictions = predictions[~np.isnan(targets)]
                # targets = targets[~np.isnan(targets)]

                # stats = sm.taylor_statistics(predictions,targets,'data')

                # ccoef = stats['ccoef'][0]
                # crmsd = stats['crmsd'][0]
                # sdev = stats['sdev'][0]
                # label = ['']
                # y=0
                # for k in surface_list:
                #     yy=0
                    # for kkkk in field_short:
                        # s1_vv = df_taylor.filter(like=k).filter(like=kk).filter(like=kkk).filter(like='S1_vv').filter(like=kkkk).values.flatten()
                        # model_vv = df_taylor.filter(like=k).filter(like=kk).filter(like=kkk).filter(like='biasedmodel_').filter(like=kkkk).values.flatten()
                        # model_vv_ub = df_taylor.filter(like=k).filter(like=kk).filter(like=kkk).filter(like='unbiasedmodeldb').filter(like=kkkk).values.flatten()

                        # s1_vv = 10*np.log10(s1_vv)
                        # model_vv_ub = model_vv_ub

                        # predictions = model_vv_ub[~np.isnan(model_vv_ub)]
                        # targets = s1_vv[~np.isnan(model_vv_ub)]
                        # predictions = predictions[~np.isnan(targets)]
                        # targets = targets[~np.isnan(targets)]

                        # stats = sm.taylor_statistics(predictions,targets,'data')
                        # plt.scatter(stats['crmsd'][1],stats['ccoef'][1],s=80,c=colors[y],marker=marker[yy])
                        # ccoef = np.append(ccoef,stats['ccoef'][1])
                        # crmsd = np.append(crmsd,stats['crmsd'][1])
                        # sdev = np.append(sdev,stats['sdev'][1])
                    #     if kkkk == 'I2EM':
                    #         label.append('IEM_B')
                    #     elif kkkk == 'WaterCloud':
                    #         label.append('WCM')
                    #     else:
                    #         label.append(kkkk)

                    #     yy=yy+1

                    # y=y+1


        legend_elements = [Line2D([0], [0], color='w', lw=4, label=labels[1]+' wheat', marker='P',markerfacecolor='r', markerSize=12), Line2D([0], [0], color='w', lw=4, label=labels[2]+' maize', marker='o',markerfacecolor='r', markerSize=12), Line2D([0], [0], color='w', lw=4, label=labels[3]+' wheat', marker='X',markerfacecolor='r', markerSize=12), Line2D([0], [0], color='w', lw=4, label=labels[4]+' wheat', marker='s',markerfacecolor='r', markerSize=12), Line2D([0], [0], color='w', lw=4, label=labels[5]+' maize', marker='d',markerfacecolor='r', markerSize=12), Line2D([0], [0], color='w', lw=4, label=labels[6]+' wheat', marker='P',markerfacecolor='b', markerSize=12), Line2D([0], [0], color='w', lw=4, label=labels[7]+' maize', marker='o',markerfacecolor='b', markerSize=12), Line2D([0], [0], color='w', lw=4, label=labels[8]+' wheat', marker='X',markerfacecolor='b', markerSize=12), Line2D([0], [0], color='w', lw=4, label=labels[9]+' maize', marker='s',markerfacecolor='b', markerSize=12)]

        # legend_elements2 = [mpatches.Patch(color=colors[0], label=surface_list[0]),mpatches.Patch(color=colors[1], label=surface_list[1]),mpatches.Patch(color=colors[2], label=surface_list[2]),mpatches.Patch(color=colors[3], label=surface_list[3]),mpatches.Patch(color=colors[4], label='IEM_B')]

        leg = ax.legend(handles=legend_elements, prop={'size': 20}, bbox_to_anchor=(0.78, 0.37, 0.6, 0.8))
        # leg1 = ax.legend(handles=legend_elements2, prop={'size': 14},loc='lower left')
        # ax.add_artist(leg)
        # plt.grid(linestyle='dotted')
        # plt.xlabel('ubRMSE',fontsize=16)
        # plt.ylabel('$R^2$', fontsize=16)
        # plt.xlim(1.35,2.75)
        # plt.ylim(0.31,0.85)
        # plt.tick_params(labelsize=17)
        plt.savefig('/media/tweiss/Work/Paper3_down/'+passes+'/taylor/taylor_'+year+name_ex+'.png')
        plt.close()


