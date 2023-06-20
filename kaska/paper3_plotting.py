import numpy as np
import pdb
import gdal
import matplotlib.pyplot as plt
from z_helper import *
import datetime
import seaborn as sns
from matplotlib.colors import ListedColormap
from pandas.plotting import register_matplotlib_converters
from osgeo.osr import SpatialReference, CoordinateTransformation
import pyproj

class plot_paper_3(object):

    def __init__(self, years, esus, passes,time_contrainst):

        """
        time_contrainst = ['no']
        """

        if not os.path.exists('/media/tweiss/Work/Paper3_down/'+passes+'/rgb'):
            os.makedirs('/media/tweiss/Work/Paper3_down/'+passes+'/rgb')

        if not os.path.exists('/media/tweiss/Work/Paper3_down/'+passes+'/field'):
            os.makedirs('/media/tweiss/Work/Paper3_down/'+passes+'/field')

        if not os.path.exists('/media/tweiss/Work/Paper3_down/'+passes+'/spatial_calculations'):
            os.makedirs('/media/tweiss/Work/Paper3_down/'+passes+'/spatial_calculations')
        if not os.path.exists('/media/tweiss/Work/Paper3_down/'+passes+'/boxplot'):
            os.makedirs('/media/tweiss/Work/Paper3_down/'+passes+'/boxplot')

        plt.rcParams["figure.figsize"] = (20,15)

        self.plot_model_param(years,esus,passes,time_contrainst)

    def plot_model_param(self,years,esus,passes,time_contrainst):
        """
        plot model output sm, vwc, b, rms
        plot model input vv, sm_api, vwc

        """

        param = ['sm', 'vwc', 'b', 'rms', 'input_vv', 'input_sm_api', 'input_vwc']
        ymin_mean = [0.2, 0, 0, 0.005,  -5, 0.23, 0]
        ymax_mean = [0.3, 5, 0.6, 0.03,  -16, 0.27, 5]
        ymin_std = [0.0, 0, 0.1, 0.0, None, 0.0, 0]
        ymax_std = [0.25, 3, 0.25, 1e-16, None, 0.25, 3]
        ymin_var = [0, None, None, None, None, 0, None]
        ymax_var = [0.4, None, None, None, None, 0.8, None]

        for i, par in enumerate(param):

            for year in years:

                if year == '2017':
                    fields = [0,301,319,542,508,515]
                if year == '2018':
                    fields = [0,317,410,525,508]

                for time_con in time_contrainst:

                    for field in fields:



                        g = gdal.Open('/media/tweiss/Work/Paper3_down/GIS/'+year+'_esu_field_buffer_30.tif')
                        state_mask = g.ReadAsArray().astype(np.int)
                        g = gdal.Open('/media/tweiss/Work/Paper3_down/GIS/clc_class2.tif')
                        state_mask_2 = g.ReadAsArray().astype(np.int)

                        var_multi = np.load('/media/tweiss/Work/Paper3_down/'+passes+'/'+year+'_multi_'+par+'.npy')

                        time = np.load('/media/tweiss/Work/Paper3_down/'+passes+'/'+year+'_multi_times.npy',allow_pickle=True)

                        if time_con == 'yes':
                            m = time < datetime.datetime(int(year),7,15)
                            var_multi = var_multi[m]
                            time = time[m]
                            name_ex = year+'0715'
                        else:
                            name_ex = ''
                            pass

                        if field > 0.:
                            var_multi = self.mask_fields(var_multi,field,state_mask)
                            # for t, tt in enumerate(time):
                            #     if par == 'input_vv':
                            #         self.plot(10*np.log10(var_multi[t]), vmin=ymin_mean[i], vmax=ymax_mean[i], name='field/'+par+'_'+str(field)+'_'+str(tt)[:10], mask=state_mask_2,par=par, passes=passes)
                            #     else:
                            #         self.plot(var_multi[t], vmin=ymin_mean[i], vmax=ymax_mean[i], name='field/'+par+'_'+str(field)+'_'+str(tt)[:10], mask=state_mask_2,par=par, passes=passes)

                            if par == 'sm':
                                file = '/media/tweiss/Work/z_final_mni_data_2017/new_in_situ_s1multi_buffer_100_'+year+'_paper3.csv'

                                data = pd.read_csv(file,header=[0,1],index_col=1)

                                data_field = data.filter(like=str(field)).filter(like='SM')
                                data_field.index = pd.to_datetime(data_field.index)
                                sm_insitu = data_field.mean(axis=1).values.flatten()

                                date = data_field.index

                                time2 = pd.to_datetime(time)
                                time2 = time2.strftime('%Y-%m-%d')
                                date2 = date.strftime('%Y-%m-%d')
                                mask_time = np.isin(time2,date2)
                                times = pd.to_datetime(date2)

                                var_api = np.load('/media/tweiss/Work/Paper3_down/'+passes+'/'+year+'_multi_'+'input_sm_api'+'.npy')
                                var_api = self.mask_fields(var_api,field,state_mask)

                                sm = self.extraction_xxx(var_multi,state_mask,mask_time)
                                sm_api = self.extraction_xxx(var_api,state_mask,mask_time)

                                if year == '2017':
                                    meteo = pd.read_csv('/media/tweiss/Work/Paper3_down/GIS/Eichenried_01012017_31122017_hourly.csv', sep=';', decimal=',')
                                    meteo2 = meteo.stack().str.replace(',','.').unstack()
                                    meteo2['date'] = pd.to_datetime(meteo2['Tag']+' '+meteo2['Stunde'])
                                    meteo2['SUM']= pd.to_numeric(meteo2['SUM_NN050'],errors='coerce')
                                    s = meteo2.resample('d', on='date')['SUM'].sum()
                                else:
                                    s = None



                                self.boxplot2(sm,par,field,year,times,passes,sm_api,sm_insitu,s)
                            else:
                                self.boxplot(var_multi,par,field,year,time,passes)
                        else:
                            pass

                        value_mean, value_std, value_var = calc_pix(var_multi)

                        if par == 'input_vv':
                            value_mean = 10*np.log10(value_mean)
                            value_std = 10*np.log10(value_std)
                            value_var = 10*np.log10(value_var)

                            self.plot_rgb(var_multi[1],var_multi[20],var_multi[40],mask=state_mask_2,name='rgb/rgb_'+year+'_'+str(field),passes=passes)

                            self.plot_rgb(var_multi[0],var_multi[int(len(var_multi)/2.)],var_multi[-1],mask=state_mask_2,name='rgb/rgb_bme'+year+'_'+str(field),passes=passes)

                            self.plot_rgb(var_multi[1],var_multi[45],var_multi[85],mask=state_mask_2,name='rgb/rgb_0323_0530_0729'+year+'_'+str(field),passes=passes)
                            self.plot_rgb(var_multi[1],var_multi[45],var_multi[75],mask=state_mask_2,name='rgb/rgb_0323_0530_0715'+year+'_'+str(field),passes=passes)
                            self.plot_rgb(var_multi[45],var_multi[75],var_multi[85],mask=state_mask_2,name='rgb/rgb_0530_0715_0729'+year+'_'+str(field),passes=passes)
                            self.plot_rgb(var_multi[55],var_multi[92],var_multi[-1],mask=state_mask_2,name='rgb/rgb_0615_0809_0928'+year+'_'+str(field),passes=passes)


                        self.plot(value_mean, vmin=ymin_mean[i], vmax=ymax_mean[i], name='spatial_calculations/'+par+year+'value_mean'+name_ex+'_'+str(field), mask=state_mask_2, par=par, passes=passes, year=year)
                        self.plot(value_std, vmin=ymin_std[i], vmax=ymax_std[i], name='spatial_calculations/'+par+year+'value_std'+name_ex+'_'+str(field), mask=state_mask_2, par=par, passes=passes, year=year)
                        self.plot(value_var, vmin=ymin_var[i], vmax=ymax_var[i], name='spatial_calculations/'+par+year+'value_var'+name_ex+'_'+str(field), mask=state_mask_2, par=par, passes=passes, year=year)
                        self.plot(value_var, name='spatial_calculations/'+par+year+'value_var2'+name_ex+'_'+str(field), par=par, passes=passes, year=year)

    def extraction_xxx(self,var,state_mask,mask_time):

        xxx = np.copy(var)
        xxx = xxx[mask_time,:]
        return xxx


    def plot(self,input,vmin=None,vmax=None,name=None,mask=None,par=None,passes=None,year=None):

        f, ax = plt.subplots(1,1)

        try:
            input = np.ma.masked_where(mask == 0.,input)
        except IndexError:
            pass

        if par == 'input_vv':
            cmap = plt.cm.Greys_r
            label = 'VV [dB]'
        elif par == 'vwc':
            cmap = plt.cm.YlGn
            label = 'VWC [kg/m$^2$]'
        elif par == 'input_vwc':
            cmap = plt.cm.RdYlGn
            label = 'kg/m$^2$'
        else:
            cmap = plt.cm.viridis_r
            label = 'Soil Moisture [m$^3$/m$^3$]'
        cmap.set_bad(color='white')
        plt.rcParams['axes.labelsize'] = 20

        im1 = ax.imshow(input,vmin=vmin, vmax=vmax, cmap=cmap, aspect='auto')
        # ax.set_title(name, fontsize=20)
        f.subplots_adjust(right=0.85)
        cbar_ax = f.add_axes([0.8, 0.15, 0.04, 0.7])
        ticklabs = cbar_ax.get_yticklabels()
        cbar_ax.set_yticklabels(ticklabs, fontsize=20)
        f.colorbar(im1, cax=cbar_ax, label=label)



        if len(input) > 200:
            if year == '2017':

                g = gdal.Open('/media/tweiss/Work/Paper3_down/GIS/2017_line2.tif')
            elif year == '2018':
                g = gdal.Open('/media/tweiss/Work/Paper3_down/GIS/2018_line2.tif')

            # # Define the wgs84 system (EPSG 4326)
            # epsg4326 = SpatialReference()
            # epsg4326.ImportFromEPSG(4326)

            # # Define ...
            # epsg32632 = SpatialReference()
            # epsg32632.ImportFromEPSG(32632)


            # rd2latlon = CoordinateTransformation(epsg32632, epsg4326)
            # ulx, xres, xskew, uly, yskew, yres  = g.GetGeoTransform()
            # lrx = ulx + (g.RasterXSize * xres)
            # lry = uly + (g.RasterYSize * yres)
            # lonlatmin = rd2latlon.TransformPoint(ulx, lry)
            # lonlatmax = rd2latlon.TransformPoint(lrx, uly)
            # p = pyproj.Proj(proj='utm', zone=32, ellps='WGS84')
            # ulx1, lry1 = p(ulx, lry,inverse=True)
            # lrx1, uly1 = p(lrx, uly,inverse=True)

            # BBox = ((uly,  lry, lrx, ulx))

            # BBox2 = ((lonlatmax[1], lonlatmin[1], lonlatmax[0], lonlatmin[0]))

            im1 = ax.imshow(input,vmin=vmin, vmax=vmax, cmap=cmap, aspect='auto')
            # ax.set_title(name, fontsize=20)
            f.subplots_adjust(right=0.85)
            cbar_ax = f.add_axes([0.8, 0.15, 0.04, 0.7])
            ticklabs = cbar_ax.get_yticklabels()
            cbar_ax.set_yticklabels(ticklabs, fontsize=20)
            cbar_ax.set_label(label)
            f.colorbar(im1, cax=cbar_ax, label=label)


            state_mask_3 = g.ReadAsArray().astype(np.int)
            state_mask_3=state_mask_3.astype('float')
            state_mask_3[state_mask_3==0]=np.nan
            state_mask_3[state_mask_3>0]=1


            # Let's also design our color mapping: 1s should be plotted in blue, 2s in red, etc...
            if year == '2017':
                col_dict={1:"magenta"}
            elif year == '2018':
                col_dict={1:"orange"}
            else:
                col_dict={1:"black"}


            # We create a colormar from our list of colors
            cm = ListedColormap([col_dict[x] for x in col_dict.keys()])


            # cbar_ax = f.add_axes([0.8, 0.15, 0.04, 0.7])
            # cbar = f.colorbar(im2, cax=cbar_ax, ticks=[1,2,3,4,5,6,7,8,9,10,11,12])
            # cbar.ax.set_yticks([0.2, 0.4, 0.6, 0.8])
            # cbar.ax.set_yticklabels(['WTriticale', 'WWeizen', 'WGerste', 'Wiese', 'Wiese', 'Wiese', 'Wiese', 'Mais', 'SHafer', 'Luzerne', 'Gemuese', 'Bohne'])



            im2 = ax.imshow(state_mask_3,cmap=cm)
        else:
            im1 = ax.imshow(input,vmin=vmin, vmax=vmax, cmap=cmap, aspect='auto')
            # ax.set_title(name, fontsize=20)
            f.subplots_adjust(right=0.85)
            cbar_ax = f.add_axes([0.8, 0.15, 0.04, 0.7])
            ticklabs = cbar_ax.get_yticklabels()
            cbar_ax.set_yticklabels(ticklabs, fontsize=20)
            f.colorbar(im1, cax=cbar_ax, label=label)

        ax.xaxis.set_tick_params(labelsize=20)
        ax.yaxis.set_tick_params(labelsize=20)

        plt.setp(ax, xticks=[22*6.1, 2*22*6.1, 3*22*6.1, 4*22*6.1, 5*22*6.1, 6*22*6.1], xticklabels=['11.64°E', '11.66°E', '11.68°E', '11.70°E', '11.72°E', '11.74°E'], yticks=[22*6.12, (22+39)*6.12, (22+39*2)*6.12, (22+39*3)*6.12], yticklabels=['48.30°N', '48.28°N', '48.26°N', '48.24°N'])
        ax.set_ylim(len(input),0)

        plt.savefig('/media/tweiss/Work/Paper3_down/'+passes+'/'+name+str(input.mean())[0:5]+'.png', bbox_inches='tight')
        # xtick = []
        # ytick = []
        # for t in ax.get_xticks():
        #     xx, yy = p(t, ax.get_yticks()[0],inverse=True)
        #     xtick.append(xx)
        # for t in ax.get_yticks():
        #     xx, yy = p(ax.get_xticks()[0], t,inverse=True)
        #     ytick.append(yy)

        # pdb.set_trace()
        plt.close()


    def plot_rgb(self,rrr,ggg,bbb,mask=None,name=None,passes=None):


        rrr = 10*np.log10(rrr)
        ggg = 10*np.log10(ggg)
        bbb = 10*np.log10(bbb)

        try:
            rrr = np.ma.masked_where(mask == 0.,rrr)
            ggg = np.ma.masked_where(mask == 0.,ggg)
            bbb = np.ma.masked_where(mask == 0.,bbb)
        except IndexError:
            pass

        OldMin = -20
        OldMax = -5
        NewMin = 0
        NewMax = 255

        OldRange = (OldMax - OldMin)
        NewRange = (NewMax - NewMin)
        rrr2 = ((((rrr - OldMin) * NewRange) / OldRange) + NewMin).astype(int)
        ggg2 = ((((ggg - OldMin) * NewRange) / OldRange) + NewMin).astype(int)
        bbb2 = ((((bbb - OldMin) * NewRange) / OldRange) + NewMin).astype(int)
        rgb = np.dstack((rrr2,ggg2,bbb2))
        plt.imshow(rgb)
        plt.savefig('/media/tweiss/Work/Paper3_down/'+passes+'/'+name, bbox_inches='tight')
        plt.close()

    def boxplot(self,var_multi,par,field,year,time,passes):
        xx = var_multi.reshape(var_multi.shape[0], (var_multi.shape[1]*var_multi.shape[2]))
        if par == 'input_vv':
            sns.boxplot(np.repeat(np.arange(len(time)), len(xx[0])), 10*np.log10(xx.flatten()))
        else:
            sns.boxplot(np.repeat(np.arange(len(time)), len(xx[0])), xx.flatten())
        ind = list(range(1,len(time)+1))
        time2 = [i.strftime('%d-%m') for i in time]
        plt.xticks(ind,time2, rotation=45)
        plt.savefig('/media/tweiss/Work/Paper3_down/'+passes+'/boxplot/'+par+str(field)+'_'+str(year), bbox_inches='tight')
        plt.close()

    def boxplot2(self,var_multi,par,field,year,time,passes,sm_api,sm_insitu,meteo=None):
        f, ax = plt.subplots(1,1)

        xx = var_multi.reshape(var_multi.shape[0], (var_multi.shape[1]*var_multi.shape[2]))

        sns.boxplot(np.repeat(np.arange(len(time)), len(xx[0])), xx.flatten(), color='skyblue')

        sm_api2 = np.nanmean(sm_api,axis=(1,2))
        ax.plot(sm_api2,'r-o',linewidth=4, label='SM Api')
        ax.plot(sm_insitu,'b-o',linewidth=4, label = 'SM insitu')
        ind = list(range(1,len(time)+1))
        time2 = [i.strftime('%d-%m') for i in time]
        plt.xticks(ind,time2, rotation=45)
        ax.set_ylabel('SM')
        plt.legend()
        if year == '2017':
            ax2 = ax.twinx()
            mask_time2 = np.isin(meteo.index,time)

            ax2.bar(np.arange(len(meteo[mask_time2])),meteo[mask_time2])
            ax2.set_ylim(0,150)
            ax2.set_xticks([])
            ax2.set_ylabel('Precipitation')
        plt.xticks(ind,time2, rotation=45)
        plt.savefig('/media/tweiss/Work/Paper3_down/'+passes+'/boxplot/'+par+str(field)+'_'+str(year), bbox_inches='tight')
        plt.close()


    def mask_fields(self,data,field,state_mask):
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




