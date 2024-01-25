import os
import yaml
from sar_pre_processing.sar_pre_processor import *
import warnings
warnings.filterwarnings("ignore")
class run_SenSARP(object):
    """
    Class to run SenSARP default mode
    """
    def __init__(self, path_s1data, output_folder, sample_config_file, name_tag, gpt_location='~/snap/bin/gpt', year=None, lr_lat=None,lr_lon=None,ul_lat=None,ul_lon=None,multi_speck=None,norm_angle=None):
        self.input_folder = path_s1data
        self.output_folder = output_folder
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        self.gpt_location = os.path.expanduser(gpt_location)
        self.sample_config_file = sample_config_file
        self.name_tag = name_tag
        self.config_file = self.name_tag + '_config_file.yaml'
        self.year = year
        self.lr_lat = lr_lat #todo: add test if lr_lon, ul_lat, ul_lon are specified as well!!!!
        self.lr_lon = lr_lon
        self.ul_lat = ul_lat
        self.ul_lon = ul_lon
        self.multi_speck = multi_speck
        self.norm_angle = norm_angle

        self.open_yaml()
        self.add_options()
        self.run()

    def open_yaml(self):

        with open(self.sample_config_file) as stream:
           data = yaml.safe_load(stream)

        data['input_folder'] = self.input_folder
        data['output_folder'] = self.output_folder
        data['gpt'] = self.gpt_location

        with open(self.config_file, 'wb') as stream:
           yaml.safe_dump(data, stream, default_flow_style=False,
                          explicit_start=True, allow_unicode=True, encoding='utf-8')

    def add_options(self):

        with open(self.config_file) as stream:
           data = yaml.safe_load(stream)

        # Filter option
        ## Filter via year of interest
        if self.year != None:
            data['year'] = self.year

        if self.lr_lat != None:
            ## Define region of interest
            data['region']['lr']['lat'] = self.lr_lat # lower right latitude
            data['region']['lr']['lon'] = self.lr_lon # lower right longitude
            data['region']['ul']['lat'] = self.ul_lat # upper left latitude
            data['region']['ul']['lon'] = self.ul_lon # upper left longitude
            data['region']['subset'] = 'yes'

        if self.multi_speck != None:
            ## Define multi-temporal filtering properties
            data['speckle_filter']['multi_temporal']['apply'] = 'yes'
            data['speckle_filter']['multi_temporal']['files'] = self.multi_speck # Number of files used for multi temporal filtering

        if self.norm_angle != None:
            ## Define incidence angle for normalization
            data['normalization_angle'] = self.norm_angle

        with open('test_config_file.yaml', 'wb') as stream:
           yaml.safe_dump(data, stream, default_flow_style=False,
                          explicit_start=True, allow_unicode=True, encoding='utf-8')

    def run(self):

        processing = SARPreProcessor(config=self.config_file)
        processing.create_processing_file_list()
        print('start step 1')
        processing.pre_process_step1()
        print('start step 2')
        processing.pre_process_step2()
        print('start step 3')
        processing.pre_process_step3()
        print('start add netcdf information')
        processing.add_netcdf_information()
        print('start create netcdf stack')
        processing.create_netcdf_stack()
