
import subprocess

# Rasterize Stadtgüter Fruchtfelder 2017
# 1 = Mais
# 2 = Winterweizen
# 3 = Triticale
# 4 = Rest
subprocess.call('gdal_rasterize -at -of GTiff -a field -te 694748 5345900 703600 5354600 -tr 10 10 -ot Byte -co \"COMPRESS=DEFLATE\" '+'/media/tweiss/Work/Paper3_down/GIS/sgm/sgm2017.shp'+' /media/tweiss/Work/Paper3_down/GIS/sgm2017_frucht.tif', shell=True)

# Rasterize Corine Land Cover 5 ha version 2018
# 211 = nicht bewässertes Ackerland
# 231 = Wiesen und Weiden
# 0 = Rest
subprocess.call('gdal_rasterize -at -of GTiff -a clc18 -te 694748 5345900 703600 5354600 -tr 10 10 -ot Byte -co \"COMPRESS=DEFLATE\" '+'/media/tweiss/Work/Paper3_down/GIS/clc5_2018.utm32s.shape/clc5/clc5_class2xx.shp'+' /media/tweiss/Work/Paper3_down/GIS/clc_class2.tif', shell=True)

# Rasterize 2017 fields ESU_Field_buffer_30.shp
# Field 301 = 87 (triti)
# Field 319 = 67 (maize)
# Field 542 = 8 (triti)
# Field 508 = 27 (wheat)
# Field 515 = 4 (maize)
subprocess.call('gdal_rasterize -at -of GTiff -a ID -te 694748 5345900 703600 5354600 -tr 10 10 -ot Byte -co \"COMPRESS=DEFLATE\" '+'/media/tweiss/Work/z_final_mni_data_2017/ESU_Field_buffer_30.shp'+' /media/tweiss/Work/Paper3_down/GIS/2017_ESU_Field_buffer_30.tif', shell=True)

# Rasterize 2018 fields ESU_2018_Field_buffer_30.shp
# Field 317 = 65 (triti)
# Field 410 = 113 (maize)
# Field 525 = 30 (wheat)
# Field 508 = 27 (maize)
subprocess.call('gdal_rasterize -at -of GTiff -a ID -te 694748 5345900 703600 5354600 -tr 10 10 -ot Byte -co \"COMPRESS=DEFLATE\" '+'/media/tweiss/Work/z_final_mni_data_2017/ESU_2018_Field_buffer_30.shp'+' /media/tweiss/Work/Paper3_down/GIS/2018_ESU_Field_buffer_30.tif', shell=True)


# Rasterize Stadtgüter Fruchtfelder 2017
#if(Frucht_lan='Wintertriticale',1,if(Frucht_lan='Winterweizen',2,if(Frucht_lan='Wintergerste',3,if(Frucht_lan='Wiesengras',4,if(Frucht_lan='Kleegras',5,if(Frucht_lan='Ackergras',6,if(Frucht_lan='Weidegras',7,if(Frucht_lan='Mais',8,if(frucht_lan='Sommerhafer',9,if(Frucht_lan='Luzerne',10,if(Frucht_lan='Feldgemuese',11,if(Frucht_lan='Ackerbohnen',12,13))))))))))))
subprocess.call('gdal_rasterize -at -of GTiff -a field_id -te 694748 5345900 703600 5354600 -tr 20 20 -ot Byte -co \"COMPRESS=DEFLATE\" '+'/media/tweiss/Work/Paper3_down/GIS/2017_field_line.shp'+' /media/tweiss/Work/Paper3_down/GIS/sgm2017_line.tif', shell=True)

subprocess.call('gdalwarp -of GTiff -te 694748 5345900 703600 5354600 -tr 10 10 -ot Byte -co \"COMPRESS=DEFLATE\" '+'/media/tweiss/Work/Paper3_down/GIS/sgm2017_line.tif'+' /media/tweiss/Work/Paper3_down/GIS/sgm2017_line2.tif', shell=True)

# Rasterize Field boundaries line
subprocess.call('gdal_rasterize -at -of GTiff -a ID -te 694748 5345900 703600 5354600 -tr 20 20 -ot Byte -co \"COMPRESS=DEFLATE\" '+'/media/tweiss/Work/Paper3_down/GIS/2017_line.shp'+' /media/tweiss/Work/Paper3_down/GIS/2017_line.tif', shell=True)

subprocess.call('gdalwarp -of GTiff -te 694748 5345900 703600 5354600 -tr 10 10 -ot Byte -co \"COMPRESS=DEFLATE\" '+'/media/tweiss/Work/Paper3_down/GIS/2017_line.tif'+' /media/tweiss/Work/Paper3_down/GIS/2017_line2.tif', shell=True)

subprocess.call('gdal_rasterize -at -of GTiff -a ID -te 694748 5345900 703600 5354600 -tr 20 20 -ot Byte -co \"COMPRESS=DEFLATE\" '+'/media/tweiss/Work/Paper3_down/GIS/2018_line.shp'+' /media/tweiss/Work/Paper3_down/GIS/2018_line.tif', shell=True)

subprocess.call('gdalwarp -of GTiff -te 694748 5345900 703600 5354600 -tr 10 10 -ot Byte -co \"COMPRESS=DEFLATE\" '+'/media/tweiss/Work/Paper3_down/GIS/2018_line.tif'+' /media/tweiss/Work/Paper3_down/GIS/2018_line2.tif', shell=True)


# Rasterize ESU buffer 100 m 2017
subprocess.call('gdal_rasterize -at -of GTiff -a FID_ -te 694748 5345900 703600 5354600 -tr 10 10 -ot Byte -co \"COMPRESS=DEFLATE\" '+'/media/tweiss/Work/z_final_mni_data_2017/ESU_buffer_100.shp'+' /media/tweiss/Work/Paper3_down/GIS/2017_ESU_buffer_100.tif', shell=True)

subprocess.call('gdal_rasterize -at -of GTiff -a ID -te 694748 5345900 703600 5354600 -tr 10 10 -ot Byte -co \"COMPRESS=DEFLATE\" '+'/media/tweiss/Work/z_final_mni_data_2017/ESU_2018_buffer_100.shp'+' /media/tweiss/Work/Paper3_down/GIS/2018_ESU_buffer_100.tif', shell=True)

# buffer 30
subprocess.call('gdal_rasterize -at -of GTiff -a FID_ -te 694748 5345900 703600 5354600 -tr 10 10 -ot Byte -co \"COMPRESS=DEFLATE\" '+'/media/tweiss/Work/z_final_mni_data_2017/ESU_buffer_30.shp'+' /media/tweiss/Work/Paper3_down/GIS/2017_ESU_buffer_30.tif', shell=True)

# esu 2017
subprocess.call('gdal_rasterize -at -of GTiff -a FID_ -te 694748 5345900 703600 5354600 -tr 10 10 -ot Byte -co \"COMPRESS=DEFLATE\" '+'/media/tweiss/Work/z_final_mni_data_2017/ESU.shp'+' /media/tweiss/Work/Paper3_down/GIS/2017_ESU.tif', shell=True)

# agvolution
subprocess.call('gdal_rasterize -at -of GTiff -a fid -te 758967 5937147 768056 5945451 -tr 10 10 -ot Byte -co \"COMPRESS=DEFLATE\" '+'/media/tweiss/data/Arbeit_einordnen/agvolution/sensor_data_agvolution/farm_one/test.shp'+' /media/AUF/userdata/agvolution/fields_1.tif', shell=True)

