"""
 This file reads the input data (model forcing) and write the output to netcdf file
"""

import os
import sys
import xarray as xr
import pandas as pd
import numpy as np
import time
import logging
from modules.radCor import correctRadiation
from constants import *
from config import * 


class IOClass:

    def __init__(self, DATA=None):
        """ Init IO Class"""

        # start module logging
        self.logger = logging.getLogger(__name__)

        # Initialize data
        self.DATA = DATA
        self.RESTART = None
        self.RESULT = None

    def create_data_file(self):
        """ Returns the data xarray """
    
        if (restart==True):
            print('--------------------------------------------------------------')
            print('\t RESTART FROM PREVIOUS STATE')
            print('-------------------------------------------------------------- \n')
            
            # Load the restart file
            timestamp = pd.to_datetime(time_start).strftime('%Y-%m-%dT%H-%M-%S')
            if (os.path.isfile(os.path.join(data_path, 'restart', 'restart_'+timestamp+'.nc')) & (time_start != time_end)):
                self.GRID_RESTART = xr.open_dataset(os.path.join(data_path, 'restart', 'restart_'+timestamp+'.nc'))
                self.restart_date = self.GRID_RESTART.time     # Get time of the last calculation
                self.init_data_dataset()                       # Read data from the last date to the end of the data file
            else:
                self.logger.error('No restart file available for the given date %s' % (timestamp))  # if there is a problem kill the program
                self.logger.error('OR start date %s equals end date %s \n' % (time_start, time_end))
                sys.exit(1)
        else:
            self.restart_date = None
            self.init_data_dataset()  # If no restart, read data according to the dates defined in the config.py
        
        return self.DATA


   
    def create_result_file(self):
        """ Returns the data xarray """
        self.init_result_dataset()
        return self.RESULT


         
    def create_restart_file(self):
        """ Returns the data xarray """
        self.init_restart_dataset()
        return self.RESTART



    def create_grid_restart(self):
        return self.GRID_RESTART



    def init_data_dataset(self):
        """     
        PRES        ::   Air Pressure [hPa]
        N           ::   Cloud cover  [fraction][%/100]
        RH2         ::   Relative humidity (2m over ground)[%]
        RRR         ::   Precipitation per time step [mm]
        SNOWFALL    ::   Snowfall per time step [m]
        G           ::   Solar radiation at each time step [W m-2]
        T2          ::   Air temperature (2m over ground) [K]
        U2          ::   Wind speed (magnitude) [m/s]
        HGT         ::   Elevation [m]
        """
    
        # Open input dataset
        self.DATA = xr.open_dataset(os.path.join(data_path,'input',input_netcdf))
        self.DATA['time'] = np.sort(self.DATA['time'].values)

        start_interval=str(self.DATA.time.values[0])[0:16]
        end_interval = str(self.DATA.time.values[-1])[0:16]
        time_steps = str(len(self.DATA.time))
        print('\n Maximum available time interval from %s until %s. Time steps: %s \n\n' % (start_interval, end_interval, time_steps))

        # Check if restart
        if self.restart_date is None:
            print('--------------------------------------------------------------')
            print('\t Integration from %s to %s' % (time_start, time_end))
            print('--------------------------------------------------------------\n')
            self.DATA = self.DATA.sel(time=slice(time_start, time_end))   # Select dates from config.py
        else:
            # There is nothing to do if the dates are equal
            if (self.restart_date==time_end):
                print('Start date equals end date ... no new data ... EXIT')
                sys.exit(1)
            else:
                # otherwise, run the model from the restart date to the defined end date
                print('Starting from %s (from restart file) to %s (from config.py) \n' % (self.restart_date.values, time_end))
                self.DATA = self.DATA.sel(time=slice(self.restart_date, time_end))

        if time_start < start_interval:
            print('\n WARNING! Selected startpoint before first timestep of input data\n')
        if time_end > end_interval:
            print('\n WARNING! Selected endpoint after last timestep of input data\n')
        if time_start > end_interval or time_end < start_interval:
            print('\n ERROR! Selected period not availalbe in input data\n')


        print('--------------------------------------------------------------')
        print('Checking input data .... \n')
        
        def check(field, max, min):
            '''Check the validity of the input data '''
            if np.nanmax(field) > max or np.nanmin(field) < min:
                print('Please check the input data, its seems they are out of range %s MAX: %.2f MIN: %.2f \n' % (str.capitalize(field.name), np.nanmax(field), np.nanmin(field)))
        
        if ('T2' in self.DATA):
            print('Temperature data (T2) ... ok ')
            check(self.DATA.T2, 313.16, 243.16)
        if ('RH2' in self.DATA):
            print('Relative humidity data (RH2) ... ok ')
            check(self.DATA.RH2, 100.0, 0.0)
        if ('G' in self.DATA):
            print('Shortwave data (G) ... ok ')
            check(self.DATA.G, 1600.0, 0.0)
        if ('U2' in self.DATA):
            print('Wind velocity data (U2) ... ok ')
            check(self.DATA.U2, 50.0, 0.0)
        if ('RRR' in self.DATA):
            print('Precipitation data (RRR) ... ok ')
            check(self.DATA.RRR, 20.0, 0.0)
        if ('N' in self.DATA):
            print('Cloud cover data (N) ... ok ')
            check(self.DATA.N, 1.0, 0.0)
        if ('PRES' in self.DATA):
            print('Pressure data (PRES) ... ok ')
            check(self.DATA.PRES, 1080.0, 400.0)
        if ('LWin' in self.DATA):
            print('Incoming longwave data (LWin) ... ok ')
            check(self.DATA.LWin, 400.0, 200.0)
        if ('SNOWFALL' in self.DATA):
            print('Snowfall data (SNOWFALL) ... ok ')
            check(self.DATA.SNOWFALL, 0.1, 0.0)

        print('\n Glacier gridpoints: %s \n\n' %(np.nansum(self.DATA.MASK>=1)))

 
    def init_result_dataset(self):
        """ This function creates the result file 
        Args:
            
            self.DATA    ::  self.DATA structure 
            
        Returns:
            
            self.RESULT  ::  one-dimensional self.RESULT structure"""
        
        self.RESULT = xr.Dataset()
        self.RESULT.coords['time'] = self.DATA.coords['time']
        self.RESULT.coords['lat'] = self.DATA.coords['lat']
        self.RESULT.coords['lon'] = self.DATA.coords['lon']

        self.RESULT.attrs['force_use_TP'] = str(force_use_TP)
        self.RESULT.attrs['Time_step_input_file_seconds'] = dt
        self.RESULT.attrs['merging'] = str(merging)
        self.RESULT.attrs['density_threshold_merging'] = density_threshold_merging
        self.RESULT.attrs['temperature_threshold_merging'] = temperature_threshold_merging
        self.RESULT.attrs['merge_max'] = merge_max
        self.RESULT.attrs['split_max'] = split_max
        self.RESULT.attrs['c_stab'] = c_stab

        self.RESULT.attrs['albedo_method'] = albedo_method
        self.RESULT.attrs['densification_method'] = densification_method
        self.RESULT.attrs['penetrating_method'] = penetrating_method
        self.RESULT.attrs['roughness_method'] = roughness_method
        self.RESULT.attrs['saturation_water_vapour_method'] = saturation_water_vapour_method
        self.RESULT.attrs['initial_snowheight'] = initial_snowheight
        self.RESULT.attrs['initial_snow_layer_heights'] = initial_snow_layer_heights
        self.RESULT.attrs['initial_glacier_layer_heights'] = initial_glacier_layer_heights
        self.RESULT.attrs['initial_glacier_height'] = initial_glacier_height
        self.RESULT.attrs['initial_top_density_snowpack'] = initial_top_density_snowpack
        self.RESULT.attrs['initial_botton_density_snowpack'] = initial_botton_density_snowpack
        self.RESULT.attrs['temperature_top'] = temperature_top
        self.RESULT.attrs['temperature_bottom'] = temperature_bottom
        self.RESULT.attrs['const_init_temp'] = const_init_temp
        self.RESULT.attrs['merge_snow_threshold'] = merge_snow_threshold
        self.RESULT.attrs['minimum_snow_height'] = minimum_snow_height
        self.RESULT.attrs['minimum_snow_to_reset_albedo'] = minimum_snow_to_reset_albedo
        self.RESULT.attrs['density_fresh_snow'] = density_fresh_snow
        self.RESULT.attrs['albedo_fresh_snow'] = albedo_fresh_snow
        self.RESULT.attrs['albedo_firn'] = albedo_firn
        self.RESULT.attrs['albedo_ice'] = albedo_ice
        self.RESULT.attrs['albedo_mod_snow_aging'] = albedo_mod_snow_aging
        self.RESULT.attrs['albedo_mod_snow_depth'] = albedo_mod_snow_depth
        self.RESULT.attrs['roughness_fresh_snow'] = roughness_fresh_snow
        self.RESULT.attrs['roughness_ice'] = roughness_ice
        self.RESULT.attrs['roughness_firn'] = roughness_firn
        self.RESULT.attrs['aging_factor_roughness'] = aging_factor_roughness
        self.RESULT.attrs['surface_emission_coeff'] = surface_emission_coeff
        self.RESULT.attrs['snow_ice_threshold'] = snow_ice_threshold
        self.RESULT.attrs['snow_firn_threshold'] = snow_firn_threshold

        self.add_variable_along_latlon(self.RESULT, self.DATA.HGT, 'HGT', 'm', 'Elevation')
        self.add_variable_along_latlon(self.RESULT, self.DATA.MASK, 'MASK', 'boolean', 'Glacier mask')
        self.add_variable_along_latlontime(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'T2', 'K', 'Air temperature at 2 m')
        self.add_variable_along_latlontime(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'RH2', '%', 'Relative humidity at 2 m')
        self.add_variable_along_latlontime(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'U2', 'm s\u207b\xb9', 'Wind velocity at 2 m')
        self.add_variable_along_latlontime(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'RRR', 'mm','Total precipiation')
        self.add_variable_along_latlontime(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'RAIN', 'mm', 'Liquid precipitation')
        self.add_variable_along_latlontime(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'SNOWFALL', 'm', 'Snowfall')
        self.add_variable_along_latlontime(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'PRES', 'hPa', 'Atmospheric pressure')
        self.add_variable_along_latlontime(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'N', '-', 'Cloud fraction')
        self.add_variable_along_latlontime(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'G', 'W m\u207b\xb2', 'Incoming shortwave radiation')
        self.add_variable_along_latlontime(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'LWin', 'W m\u207b\xb2', 'Incoming longwave radiation')
        self.add_variable_along_latlontime(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'LWout', 'W m\u207b\xb2', 'Outgoing longwave radiation')
        self.add_variable_along_latlontime(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'H', 'W m\u207b\xb2', 'Sensible heat flux')
        self.add_variable_along_latlontime(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'LE', 'W m\u207b\xb2', 'Latent heat flux')
        self.add_variable_along_latlontime(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'B', 'W m\u207b\xb2', 'Ground heat flux')
        self.add_variable_along_latlontime(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'ME', 'W m\u207b\xb2', 'Available melt energy')
        self.add_variable_along_latlontime(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'MB', 'm w.e.', 'Total mass balance')
        self.add_variable_along_latlontime(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'surfMB', 'm w.e.', 'Surface mass balance')
        self.add_variable_along_latlontime(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'intMB', 'm w.e.', 'Internal mass balance')
        self.add_variable_along_latlontime(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'EVAPORATION', 'm w.e.', 'Evaporation')
        self.add_variable_along_latlontime(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'SUBLIMATION', 'm w.e.', 'Sublimation')
        self.add_variable_along_latlontime(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'CONDENSATION', 'm w.e.', 'Condensation')
        self.add_variable_along_latlontime(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'DEPOSITION', 'm w.e.', 'Depostion')
        self.add_variable_along_latlontime(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'surfM', 'm w.e.', 'Surface melt')
        self.add_variable_along_latlontime(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'subM', 'm w.e.', 'Subsurface melt')
        self.add_variable_along_latlontime(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'Q', 'm w.e.', 'Runoff')
        self.add_variable_along_latlontime(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'REFREEZE', 'm w.e.', 'Refreezing')
        self.add_variable_along_latlontime(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'SNOWHEIGHT', 'm', 'Snowheight') 
        self.add_variable_along_latlontime(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'TOTALHEIGHT', 'm', 'Total domain height')
        self.add_variable_along_latlontime(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'TS', 'K', 'Surface temperature')
        self.add_variable_along_latlontime(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'Z0', 'm', 'Roughness length')
        self.add_variable_along_latlontime(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'ALBEDO', '-', 'Albedo')
        self.add_variable_along_latlontime(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'NLAYERS', '-', 'Number of layers')
       
        if (full_field):
            self.RESULT.coords['layer'] = np.arange(max_layers)
             
            self.add_variable_along_latlonlayertime(self.RESULT, np.full((self.RESULT.coords['time'].shape[0], self.DATA.T2.shape[1], self.DATA.T2.shape[2], self.RESULT.coords['layer'].shape[0]),
                                                                np.nan), 'LAYER_HEIGHT', 'm', 'Height of each layer')
            self.add_variable_along_latlonlayertime(self.RESULT, np.full((self.RESULT.coords['time'].shape[0], self.DATA.T2.shape[1], self.DATA.T2.shape[2], self.RESULT.coords['layer'].shape[0]),
                                                                np.nan), 'LAYER_RHO', 'kg m^-3', 'Layer density')
            self.add_variable_along_latlonlayertime(self.RESULT, np.full((self.RESULT.coords['time'].shape[0], self.DATA.T2.shape[1], self.DATA.T2.shape[2], self.RESULT.coords['layer'].shape[0]),
                                                                np.nan), 'LAYER_T', 'K', 'Layer temperature')
            self.add_variable_along_latlonlayertime(self.RESULT, np.full((self.RESULT.coords['time'].shape[0], self.DATA.T2.shape[1], self.DATA.T2.shape[2], self.RESULT.coords['layer'].shape[0]),
                                                                np.nan), 'LAYER_LWC', '-', 'Liquid water content of layer')
            self.add_variable_along_latlonlayertime(self.RESULT, np.full((self.RESULT.coords['time'].shape[0], self.DATA.T2.shape[1], self.DATA.T2.shape[2], self.RESULT.coords['layer'].shape[0]),
                                                                np.nan), 'LAYER_CC', 'J m^-2', 'Cold content of each layer')
            self.add_variable_along_latlonlayertime(self.RESULT, np.full((self.RESULT.coords['time'].shape[0], self.DATA.T2.shape[1], self.DATA.T2.shape[2], self.RESULT.coords['layer'].shape[0]),
                                                                np.nan), 'LAYER_POROSITY', '-', 'Porosity of each layer')
            self.add_variable_along_latlonlayertime(self.RESULT, np.full((self.RESULT.coords['time'].shape[0], self.DATA.T2.shape[1], self.DATA.T2.shape[2], self.RESULT.coords['layer'].shape[0]),
                                                                np.nan), 'LAYER_LW', 'm w.e.q.', 'Liquid water of each layer')
            self.add_variable_along_latlonlayertime(self.RESULT, np.full((self.RESULT.coords['time'].shape[0], self.DATA.T2.shape[1], self.DATA.T2.shape[2], self.RESULT.coords['layer'].shape[0]),
                                                                np.nan), 'LAYER_ICE_FRACTION', '-', 'Volumetric ice fraction of each layer')
            self.add_variable_along_latlonlayertime(self.RESULT, np.full((self.RESULT.coords['time'].shape[0], self.DATA.T2.shape[1], self.DATA.T2.shape[2], self.RESULT.coords['layer'].shape[0]),
                                                                np.nan), 'LAYER_IRREDUCIBLE_WATER', '-', 'Irreducible water of each layer')
            self.add_variable_along_latlonlayertime(self.RESULT, np.full((self.RESULT.coords['time'].shape[0], self.DATA.T2.shape[1], self.DATA.T2.shape[2], self.RESULT.coords['layer'].shape[0]),
                                                                np.nan), 'LAYER_REFREEZE', 'm w.e.q.', 'Refreezing of each layer')
        
        print('\n') 
        print('Output dataset ... ok')
        return self.RESULT
    
    def init_restart_dataset(self):
        """ This function creates the result file 
        Args:
            
            self.DATA    ::  self.DATA structure 
            
        Returns:
            
            self.RESULT  ::  one-dimensional self.RESULT structure"""
        
        self.RESTART = xr.Dataset()
        self.RESTART.coords['time'] = self.DATA.coords['time'][-1]
        self.RESTART.coords['lat'] = self.DATA.coords['lat']
        self.RESTART.coords['lon'] = self.DATA.coords['lon']
        self.RESTART.coords['layer'] = np.arange(max_layers)
    
        self.add_variable_along_latlon(self.RESTART, np.full((1), np.nan), 'NLAYERS', '-', 'Number of layers')
        
        self.add_variable_along_latlonlayer(self.RESTART, np.full((self.DATA.T2.shape[1], self.DATA.T2.shape[2], self.RESTART.coords['layer'].shape[0]),
                                                            np.nan), 'LAYER_HEIGHT', 'm', 'Height of each layer')
        self.add_variable_along_latlonlayer(self.RESTART, np.full((self.DATA.T2.shape[1], self.DATA.T2.shape[2], self.RESTART.coords['layer'].shape[0]),
                                                            np.nan), 'LAYER_RHO', 'kg m^-3', 'Layer density')
        self.add_variable_along_latlonlayer(self.RESTART, np.full((self.DATA.T2.shape[1], self.DATA.T2.shape[2], self.RESTART.coords['layer'].shape[0]),
                                                            np.nan), 'LAYER_T', 'K', 'Layer temperature')
        self.add_variable_along_latlonlayer(self.RESTART, np.full((self.DATA.T2.shape[1], self.DATA.T2.shape[2], self.RESTART.coords['layer'].shape[0]),
                                                            np.nan), 'LAYER_LW', 'm w.e.q.', 'Liquid water of layer')
        
        print('Restart ddataset ... ok \n')
        print('--------------------------------------------------------------\n')
    
        return self.RESTART
   

 
    def create_local_result_dataset(self):
        """ This function creates the result dataset for a grid point 
        Args:
            
            self.DATA    ::  self.DATA structure 
            
        Returns:
            
            self.RESULT  ::  one-dimensional self.RESULT structure"""
    
        self.RESULT = xr.Dataset()
        self.RESULT.coords['time'] = self.DATA.coords['time']
        self.RESULT.coords['lat'] = self.DATA.coords['lat']
        self.RESULT.coords['lon'] = self.DATA.coords['lon']

        self.add_variable_along_time(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'T2', 'K', 'Air temperature at 2 m')
        self.add_variable_along_time(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'RH2', '%', 'Relative humidity at 2 m')
        self.add_variable_along_time(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'U2', 'm s^-1', 'Wind velocity at 2 m')
        self.add_variable_along_time(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'RRR', 'mm', 'Total precipitation')
        self.add_variable_along_time(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'RAIN', 'mm', 'Liquid precipitation')
        self.add_variable_along_time(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'SNOWFALL', 'm', 'Snowfall')
        self.add_variable_along_time(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'PRES', 'hPa', 'Atmospheric pressure')
        self.add_variable_along_time(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'N', '-', 'Cloud fraction')
        self.add_variable_along_time(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'G', 'W m\u207b\xb2', 'Incoming shortwave radiation')
        self.add_variable_along_time(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'LWin', 'W m\u207b\xb2', 'Incoming longwave radiation')
        self.add_variable_along_time(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'LWout', 'W m\u207b\xb2', 'Outgoing longwave radiation')
        self.add_variable_along_time(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'H', 'W m\u207b\xb2', 'Sensible heat flux')
        self.add_variable_along_time(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'LE', 'W m\u207b\xb2', 'Latent heat flux')
        self.add_variable_along_time(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'B', 'W m\u207b\xb2', 'Ground heat flux')
        self.add_variable_along_time(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'ME', 'W m\u207b\xb2', 'Available melt energy')
        self.add_variable_along_time(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'MB', 'm w.e.', 'Total mass balance')
        self.add_variable_along_time(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'surfMB', 'm w.e.', 'Surface mass balance')
        self.add_variable_along_time(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'intMB', 'm w.e.', 'Internal mass balance')
        self.add_variable_along_time(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'EVAPORATION', 'm w.e.', 'Evaporation')
        self.add_variable_along_time(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'SUBLIMATION', 'm w.e.', 'Sublimation')
        self.add_variable_along_time(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'CONDENSATION', 'm w.e.', 'Condensation')
        self.add_variable_along_time(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'DEPOSITION', 'm w.e.', 'Depostion')
        self.add_variable_along_time(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'surfM', 'm w.e.', 'Surface melt')
        self.add_variable_along_time(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'subM', 'm w.e.', 'Subsurface melt')
        self.add_variable_along_time(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'Q', 'm w.e.', 'Runoff')
        self.add_variable_along_time(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'REFREEZE', 'm w.e.', 'Refreezing')
        self.add_variable_along_time(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'SNOWHEIGHT', 'm', 'Snowheight') 
        self.add_variable_along_time(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'TOTALHEIGHT', 'm', 'Total domain height') 
        self.add_variable_along_time(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'TS', 'K', 'Surface temperature')
        self.add_variable_along_time(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'Z0', 'm', 'Roughness length')
        self.add_variable_along_time(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'ALBEDO', '-', 'Albedo')
        self.add_variable_along_time(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'NLAYERS', '-', 'Number of layers')
 
        if (full_field):
            self.RESULT.coords['layer'] = np.arange(max_layers)
            self.add_variable_along_layertime(self.RESULT, np.full((self.RESULT.coords['time'].shape[0], self.RESULT.coords['layer'].shape[0]), np.nan), 'LAYER_HEIGHT', 'm', 'Layer height')
            self.add_variable_along_layertime(self.RESULT, np.full((self.RESULT.coords['time'].shape[0], self.RESULT.coords['layer'].shape[0]), np.nan), 'LAYER_RHO', 'kg m^-3', 'Density of layer')
            self.add_variable_along_layertime(self.RESULT, np.full((self.RESULT.coords['time'].shape[0], self.RESULT.coords['layer'].shape[0]), np.nan), 'LAYER_T', 'K', 'Layer temperature')
            self.add_variable_along_layertime(self.RESULT, np.full((self.RESULT.coords['time'].shape[0], self.RESULT.coords['layer'].shape[0]), np.nan), 'LAYER_LWC', 'kg m^-2', 'LWC of each layer')
            self.add_variable_along_layertime(self.RESULT, np.full((self.RESULT.coords['time'].shape[0], self.RESULT.coords['layer'].shape[0]), np.nan), 'LAYER_CC', 'J m^-2', 'Cold content of each layer')
            self.add_variable_along_layertime(self.RESULT, np.full((self.RESULT.coords['time'].shape[0], self.RESULT.coords['layer'].shape[0]), np.nan), 'LAYER_POROSITY', '-', 'Porosity of each layer')
            self.add_variable_along_layertime(self.RESULT, np.full((self.RESULT.coords['time'].shape[0], self.RESULT.coords['layer'].shape[0]), np.nan), 'LAYER_LW', 'm w.e.q.', 'Liquid water of each layer')
            self.add_variable_along_layertime(self.RESULT, np.full((self.RESULT.coords['time'].shape[0], self.RESULT.coords['layer'].shape[0]), np.nan), 'LAYER_ICE_FRACTION', '-', 'Ice fraction of each layer')
            self.add_variable_along_layertime(self.RESULT, np.full((self.RESULT.coords['time'].shape[0], self.RESULT.coords['layer'].shape[0]), np.nan), 'LAYER_IRREDUCIBLE_WATER', '-', 'Irreducible water of each layer')
            self.add_variable_along_layertime(self.RESULT, np.full((self.RESULT.coords['time'].shape[0], self.RESULT.coords['layer'].shape[0]), np.nan), 'LAYER_REFREEZE', 'm w.e.q.', 'Refreezing of each layer')
    
        
        return self.RESULT
     

    def create_local_restart_dataset(self):
        """ This function creates the result dataset for a grid point 
        Args:
            
            self.DATA    ::  self.DATA structure 
            
        Returns:
            
            self.RESTART  ::  one-dimensional self.RESULT structure"""
    
        self.RESTART = xr.Dataset()
        self.RESTART.coords['time'] = self.DATA.coords['time'][-1]
        self.RESTART.coords['lat'] = self.DATA.coords['lat']
        self.RESTART.coords['lon'] = self.DATA.coords['lon']
        self.RESTART.coords['layer'] = np.arange(max_layers)
       
        self.add_variable_along_latlon(self.RESTART, np.full((1), np.nan), 'NLAYERS', '-', 'Number of layers')
        self.add_variable_along_layer(self.RESTART, np.full((self.RESTART.coords['layer'].shape[0]), np.nan), 'LAYER_HEIGHT', 'm', 'Layer height')
        self.add_variable_along_layer(self.RESTART, np.full((self.RESTART.coords['layer'].shape[0]), np.nan), 'LAYER_RHO', 'kg m^-3', 'Density of layer')
        self.add_variable_along_layer(self.RESTART, np.full((self.RESTART.coords['layer'].shape[0]), np.nan), 'LAYER_T', 'K', 'Layer temperature')
        self.add_variable_along_layer(self.RESTART, np.full((self.RESTART.coords['layer'].shape[0]), np.nan), 'LAYER_LW', 'm w.e.q.', 'Liquid water of each layer')
    
        return self.RESTART


    def write_results_future(self, results):
        """ This function aggregates the point result 
        
        results         ::  List with the result from COSIPI
        """
        self.RESULT.T2.loc[dict(lon=results.lon.values, lat=results.lat.values)] = results.T2
        self.RESULT.RH2.loc[dict(lon=results.lon.values, lat=results.lat.values)] = results.RH2
        self.RESULT.U2.loc[dict(lon=results.lon.values, lat=results.lat.values)] = results.U2
        self.RESULT.RRR.loc[dict(lon=results.lon.values, lat=results.lat.values)] = results.RRR
        self.RESULT.RAIN.loc[dict(lon=results.lon.values, lat=results.lat.values)] = results.RAIN
        self.RESULT.SNOWFALL.loc[dict(lon=results.lon.values, lat=results.lat.values)] = results.SNOWFALL
        self.RESULT.PRES.loc[dict(lon=results.lon.values, lat=results.lat.values)] = results.PRES
        self.RESULT.N.loc[dict(lon=results.lon.values, lat=results.lat.values)] = results.N
        self.RESULT.G.loc[dict(lon=results.lon.values, lat=results.lat.values)] = results.G
        self.RESULT.LWin.loc[dict(lon=results.lon.values, lat=results.lat.values)] = results.LWin
        self.RESULT.LWout.loc[dict(lon=results.lon.values, lat=results.lat.values)] = results.LWout
        self.RESULT.H.loc[dict(lon=results.lon.values, lat=results.lat.values)] = results.H
        self.RESULT.LE.loc[dict(lon=results.lon.values, lat=results.lat.values)] = results.LE
        self.RESULT.B.loc[dict(lon=results.lon.values, lat=results.lat.values)] = results.B
        self.RESULT.ME.loc[dict(lon=results.lon.values, lat=results.lat.values)] = results.ME
        self.RESULT.MB.loc[dict(lon=results.lon.values, lat=results.lat.values)] = results.MB
        self.RESULT.surfMB.loc[dict(lon=results.lon.values, lat=results.lat.values)] = results.surfMB
        self.RESULT.intMB.loc[dict(lon=results.lon.values, lat=results.lat.values)] = results.intMB
        self.RESULT.EVAPORATION.loc[dict(lon=results.lon.values, lat=results.lat.values)] = results.EVAPORATION
        self.RESULT.SUBLIMATION.loc[dict(lon=results.lon.values, lat=results.lat.values)] = results.SUBLIMATION
        self.RESULT.CONDENSATION.loc[dict(lon=results.lon.values, lat=results.lat.values)] = results.CONDENSATION
        self.RESULT.DEPOSITION.loc[dict(lon=results.lon.values, lat=results.lat.values)] = results.DEPOSITION
        self.RESULT.surfM.loc[dict(lon=results.lon.values, lat=results.lat.values)] = results.surfM
        self.RESULT.subM.loc[dict(lon=results.lon.values, lat=results.lat.values)] = results.subM
        self.RESULT.Q.loc[dict(lon=results.lon.values, lat=results.lat.values)] = results.Q
        self.RESULT.REFREEZE.loc[dict(lon=results.lon.values, lat=results.lat.values)] = results.REFREEZE
        self.RESULT.SNOWHEIGHT.loc[dict(lon=results.lon.values, lat=results.lat.values)] = results.SNOWHEIGHT
        self.RESULT.TOTALHEIGHT.loc[dict(lon=results.lon.values, lat=results.lat.values)] = results.TOTALHEIGHT
        self.RESULT.TS.loc[dict(lon=results.lon.values, lat=results.lat.values)] = results.TS
        self.RESULT.Z0.loc[dict(lon=results.lon.values, lat=results.lat.values)] = results.Z0
        self.RESULT.ALBEDO.loc[dict(lon=results.lon.values, lat=results.lat.values)] = results.ALBEDO
        self.RESULT.NLAYERS.loc[dict(lon=results.lon.values, lat=results.lat.values)] = results.NLAYERS
          
        if full_field:
            self.RESULT.LAYER_HEIGHT.loc[dict(lon=results.lon.values, lat=results.lat.values, layer=np.arange(max_layers))] = results.LAYER_HEIGHT
            self.RESULT.LAYER_RHO.loc[dict(lon=results.lon.values, lat=results.lat.values, layer=np.arange(max_layers))] = results.LAYER_RHO
            self.RESULT.LAYER_T.loc[dict(lon=results.lon.values, lat=results.lat.values, layer=np.arange(max_layers))] = results.LAYER_T
            self.RESULT.LAYER_LWC.loc[dict(lon=results.lon.values, lat=results.lat.values, layer=np.arange(max_layers))] = results.LAYER_LWC
            self.RESULT.LAYER_CC.loc[dict(lon=results.lon.values, lat=results.lat.values, layer=np.arange(max_layers))] = results.LAYER_CC
            self.RESULT.LAYER_POROSITY.loc[dict(lon=results.lon.values, lat=results.lat.values, layer=np.arange(max_layers))] = results.LAYER_POROSITY
            self.RESULT.LAYER_LW.loc[dict(lon=results.lon.values, lat=results.lat.values, layer=np.arange(max_layers))] = results.LAYER_LW
            self.RESULT.LAYER_ICE_FRACTION.loc[dict(lon=results.lon.values, lat=results.lat.values, layer=np.arange(max_layers))] = results.LAYER_ICE_FRACTION
            self.RESULT.LAYER_IRREDUCIBLE_WATER.loc[dict(lon=results.lon.values, lat=results.lat.values, layer=np.arange(max_layers))] = results.LAYER_IRREDUCIBLE_WATER
            self.RESULT.LAYER_REFREEZE.loc[dict(lon=results.lon.values, lat=results.lat.values, layer=np.arange(max_layers))] = results.LAYER_REFREEZE


    def write_restart_future(self, results):
        """ Writes the restart file 
        
        RESULT      :: RESULT dataset
        
        """

        self.RESTART['NLAYERS'] = results.NLAYERS.values
        self.RESTART.LAYER_HEIGHT.loc[dict(lon=results.lon.values, lat=results.lat.values, layer=np.arange(max_layers))] = results.LAYER_HEIGHT
        self.RESTART.LAYER_RHO.loc[dict(lon=results.lon.values, lat=results.lat.values, layer=np.arange(max_layers))] = results.LAYER_RHO
        self.RESTART.LAYER_T.loc[dict(lon=results.lon.values, lat=results.lat.values, layer=np.arange(max_layers))] = results.LAYER_T
        self.RESTART.LAYER_LW.loc[dict(lon=results.lon.values, lat=results.lat.values, layer=np.arange(max_layers))] = results.LAYER_LW
    

    def get_result(self):
        return self.RESULT

    def get_restart(self):
        return self.RESTART

    def get_grid_restart(self):
        return self.GRID_RESTART

    #---------------------------------------------------
    # Auxiliary functions for writing variables
    #--------------------------------------------------- 
    def add_variable_along_latlon(self, ds, var, name, units, long_name):
        """ This function self.adds missing variables to the self.DATA class """
        ds[name] = var
        ds[name].attrs['units'] = units
        ds[name].attrs['long_name'] = long_name
        ds[name].encoding['_FillValue'] = -9999
        return ds
    
    def add_variable_along_time(self, ds, var, name, units, long_name):
        """ This function self.adds missing variables to the self.DATA class """
        ds[name] = xr.DataArray(var, coords=[('time', ds.time)])
        ds[name].attrs['units'] = units
        ds[name].attrs['long_name'] = long_name
        ds[name].encoding['_FillValue'] = -9999
        return ds
    
    def add_variable_along_latlontime(self, ds, var, name, units, long_name):
        """ This function self.adds missing variables to the self.DATA class """
        ds[name] = (('time','lat','lon'), var)
        ds[name].attrs['units'] = units
        ds[name].attrs['long_name'] = long_name
        ds[name].encoding['_FillValue'] = -9999
        return ds
    
    def add_variable_along_latlonlayertime(self, ds, var, name, units, long_name):
        """ This function self.adds missing variables to the self.DATA class """
        ds[name] = (('time','lat','lon','layer'), var)
        ds[name].attrs['units'] = units
        ds[name].attrs['long_name'] = long_name
        ds[name].encoding['_FillValue'] = -9999
        return ds
    
    def add_variable_along_latlonlayer(self, ds, var, name, units, long_name):
        """ This function self.adds missing variables to the self.DATA class """
        ds[name] = (('lat','lon','layer'), var)
        ds[name].attrs['units'] = units
        ds[name].attrs['long_name'] = long_name
        ds[name].encoding['_FillValue'] = -9999
        return ds
    
    def add_variable_along_layertime(self, ds, var, name, units, long_name):
        """ This function self.adds missing variables to the self.DATA class """
        ds[name] = (('time','layer'), var)
        ds[name].attrs['units'] = units
        ds[name].attrs['long_name'] = long_name
        ds[name].encoding['_FillValue'] = -9999
        return ds
    
    def add_variable_along_layer(self, ds, var, name, units, long_name):
        """ This function self.adds missing variables to the self.DATA class """
        ds[name] = (('layer'), var)
        ds[name].attrs['units'] = units
        ds[name].attrs['long_name'] = long_name
        ds[name].encoding['_FillValue'] = -9999
        return ds
