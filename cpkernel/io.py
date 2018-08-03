"""
 This file reads the input data (model forcing) and write the output to netcdf file
"""

import os
import sys
import xarray as xr
import pandas as pd
import numpy as np
import time

from config import * 
from modules.radCor import correctRadiation

class IOClass:

    def __init__(self, DATA=None):
        """ Init IO Class"""
        self.DATA = DATA
        self.RESTART = None
        self.RESULT = None

    def create_data_file(self):
        """ Returns the data xarray """
    
        if (restart==True):
            print('--------------------------------------------------------------')
            print('\t self.RESTART FROM PREVIOUS STATE')
            print('-------------------------------------------------------------- \n')
            
            # Load the restart file
            if os.path.isfile(restart_netcdf):
                self.GRID_RESTART = xr.open_dataset(restart_netcdf)
                self.restart_date = self.GRID_RESTART.time     # Get time of the last calculation
                self.init_data_dataset()       # Read data from the last date to the end of the data file
            else:
                print('No restart file available! \n')  # if there is a problem kill the program
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
        RRR         ::   Precipitation per time step [m]
        G           ::   Solar radiation at each time step [W m-2]
        T2          ::   Air temperature (2m over ground) [K]
        U2          ::   Wind speed (magnitude) m/s
        """
    
        # Open input dataset
        self.DATA = xr.open_dataset(input_netcdf)
        self.DATA['time'] = np.sort(self.DATA['time'].values)
        
        # Check if restart
        if self.restart_date is None:
            print('--------------------------------------------------------------')
            print('\t Integration from %s to %s' % (time_start, time_end))
            print('--------------------------------------------------------------\n')
            self.DATA = self.DATA.sel(time=slice(time_start, time_end))   # Select dates from config.py
        else:
            # Get end date from the input data
            end_date = self.DATA.time[-1]
    
            # There is nothing to do if the dates are equal
            if (self.restart_date==end_date):
                print('Start date equals end date ... no new data ... EXIT')
                sys.exit(1)
            else:
                # otherwise, run the model from the restart date to the end of the input data
                print('Starting from %s (from restart file) to %s (from config.py) \n' % (self.restart_date.values, end_date.values))
                self.DATA = self.DATA.sel(time=slice(self.restart_date, end_date))
    
        print('--------------------------------------------------------------')
        print('Checking input data .... \n')
        
        if ('T2' in self.DATA):
            print('Temperature data (T2) ... ok ')
        if ('RH2' in self.DATA):
            print('Relative humidity data (RH2) ... ok ')
        if ('G' in self.DATA):
            print('Shortwave data (G) ... ok ')
        if ('U2' in self.DATA):
            print('Wind velocity data (U2) ... ok ')
        if ('RRR' in self.DATA):
            print('Precipitation data (RRR) ... ok ')
        if ('N' in self.DATA):
            print('Cloud cover data (N) ... ok ')
        if ('PRES' in self.DATA):
            print('Pressure data (PRES) ... ok ')
        if ('LWin' in self.DATA):
            print('Incoming longwave data (LWin) ... ok ')
   
    def init_result_dataset(self):
        """ This function creates the result file 
        Args:
            
            self.DATA    ::  self.DATA structure 
            
        Returns:
            
            self.RESULT  ::  one-dimensional self.RESULT structure"""
        
        self.RESULT = xr.Dataset()
        self.RESULT.coords['lat'] = self.DATA.coords['lat']
        self.RESULT.coords['lon'] = self.DATA.coords['lon']
        self.RESULT.coords['time'] = self.DATA.coords['time']
    
        self.add_variable_along_latlontime(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'SNOWHEIGHT', 'm', 'Snowheight')
        self.add_variable_along_latlontime(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'EVAPORATION', 'm w.e.q.', 'Evaporation')
        self.add_variable_along_latlontime(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'SUBLIMATION', 'm w.e.q.', 'Sublimation')
        self.add_variable_along_latlontime(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'MELT', 'm w.e.q.', 'Surface melt')
        self.add_variable_along_latlontime(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'LWin', 'W m^-2', 'Incoming longwave radiation')
        self.add_variable_along_latlontime(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'LWout', 'W m^-2', 'Outgoing longwave radiation')
        self.add_variable_along_latlontime(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'H', 'W m^-2', 'Sensible heat flux')
        self.add_variable_along_latlontime(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'LE', 'W m^-2', 'Latent heat flux')
        self.add_variable_along_latlontime(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'B', 'W m^-2', 'Ground heat flux')
        self.add_variable_along_latlontime(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'TS', 'K', 'Surface temperature')
        self.add_variable_along_latlontime(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'RH2', '%', 'Relative humidity at 2 m')
        self.add_variable_along_latlontime(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'T2', 'K', 'Air temperature at 2 m')
        self.add_variable_along_latlontime(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'G', 'W m^-2', 'Incoming shortwave radiation')
        self.add_variable_along_latlontime(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'U2', 'm s^-1', 'Wind velocity at 2 m')
        self.add_variable_along_latlontime(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'N', '-', 'Cloud fraction')
        self.add_variable_along_latlontime(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'Z0', 'm', 'Roughness length')
        self.add_variable_along_latlontime(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'ALBEDO', '-', 'Albedo')
        self.add_variable_along_latlontime(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'RUNOFF', 'm w.e.q.', 'Runoff')
       
        if (full_field):
            self.RESULT.coords['layer'] = np.arange(max_layers)
            
            self.add_variable_along_latlontime(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'NLAYERS', '-', 'Number of layers')
            
            self.add_variable_along_latlonlayertime(self.RESULT, np.full((self.DATA.T2.shape[0], self.DATA.T2.shape[1], self.RESULT.coords['layer'].shape[0],self.RESULT.coords['time'].shape[0]), 
                                                                np.nan), 'LAYER_HEIGHT', 'm', 'Height of each layer')
            self.add_variable_along_latlonlayertime(self.RESULT, np.full((self.DATA.T2.shape[0], self.DATA.T2.shape[1], self.RESULT.coords['layer'].shape[0],self.RESULT.coords['time'].shape[0]), 
                                                                np.nan), 'LAYER_RHO', 'kg m^-3', 'Layer density')
            self.add_variable_along_latlonlayertime(self.RESULT, np.full((self.DATA.T2.shape[0], self.DATA.T2.shape[1], self.RESULT.coords['layer'].shape[0],self.RESULT.coords['time'].shape[0]), 
                                                                np.nan), 'LAYER_T', 'K', 'Layer temperature')
            self.add_variable_along_latlonlayertime(self.RESULT, np.full((self.DATA.T2.shape[0], self.DATA.T2.shape[1], self.RESULT.coords['layer'].shape[0],self.RESULT.coords['time'].shape[0]), 
                                                                np.nan), 'LAYER_LWC', 'kg m^-2', 'Liquid water content of layer')
            self.add_variable_along_latlonlayertime(self.RESULT, np.full((self.DATA.T2.shape[0], self.DATA.T2.shape[1], self.RESULT.coords['layer'].shape[0],self.RESULT.coords['time'].shape[0]), 
                                                                np.nan), 'LAYER_CC', 'J m^-2', 'Cold content of each layer')
            self.add_variable_along_latlonlayertime(self.RESULT, np.full((self.DATA.T2.shape[0], self.DATA.T2.shape[1], self.RESULT.coords['layer'].shape[0],self.RESULT.coords['time'].shape[0]), 
                                                                np.nan), 'LAYER_POROSITY', '-', 'Porosity of each layer')
            self.add_variable_along_latlonlayertime(self.RESULT, np.full((self.DATA.T2.shape[0], self.DATA.T2.shape[1], self.RESULT.coords['layer'].shape[0],self.RESULT.coords['time'].shape[0]), 
                                                                np.nan), 'LAYER_VOL', '-', 'Volumetic ice content of each layer')
            self.add_variable_along_latlonlayertime(self.RESULT, np.full((self.DATA.T2.shape[0], self.DATA.T2.shape[1], self.RESULT.coords['layer'].shape[0],self.RESULT.coords['time'].shape[0]), 
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
        self.RESTART.coords['lat'] = self.DATA.coords['lat']
        self.RESTART.coords['lon'] = self.DATA.coords['lon']
        self.RESTART.coords['time'] = self.DATA.coords['time'][-1] 
        self.RESTART.coords['layer'] = np.arange(max_layers)
    
        self.add_variable_along_latlon(self.RESTART, np.full((1), np.nan), 'NLAYERS', '-', 'Number of layers')
        
        self.add_variable_along_latlonlayer(self.RESTART, np.full((self.DATA.T2.shape[0], self.DATA.T2.shape[1], self.RESTART.coords['layer'].shape[0]), 
                                                            np.nan), 'LAYER_HEIGHT', 'm', 'Height of each layer')
        self.add_variable_along_latlonlayer(self.RESTART, np.full((self.DATA.T2.shape[0], self.DATA.T2.shape[1], self.RESTART.coords['layer'].shape[0]), 
                                                            np.nan), 'LAYER_RHO', 'kg m^-3', 'Layer density')
        self.add_variable_along_latlonlayer(self.RESTART, np.full((self.DATA.T2.shape[0], self.DATA.T2.shape[1], self.RESTART.coords['layer'].shape[0]), 
                                                            np.nan), 'LAYER_T', 'K', 'Layer temperature')
        self.add_variable_along_latlonlayer(self.RESTART, np.full((self.DATA.T2.shape[0], self.DATA.T2.shape[1], self.RESTART.coords['layer'].shape[0]), 
                                                            np.nan), 'LAYER_LWC', 'kg m^-2', 'Liquid water content of layer')
        self.add_variable_along_latlonlayer(self.RESTART, np.full((self.DATA.T2.shape[0], self.DATA.T2.shape[1], self.RESTART.coords['layer'].shape[0]), 
                                                            np.nan), 'LAYER_CC', 'J m^-2', 'Cold content of each layer')
        self.add_variable_along_latlonlayer(self.RESTART, np.full((self.DATA.T2.shape[0], self.DATA.T2.shape[1], self.RESTART.coords['layer'].shape[0]), 
                                                            np.nan), 'LAYER_POROSITY', '-', 'Porosity of each layer')
        self.add_variable_along_latlonlayer(self.RESTART, np.full((self.DATA.T2.shape[0], self.DATA.T2.shape[1], self.RESTART.coords['layer'].shape[0]), 
                                                            np.nan), 'LAYER_VOL', '-', 'Volumetic ice content of each layer')
        self.add_variable_along_latlonlayer(self.RESTART, np.full((self.DATA.T2.shape[0], self.DATA.T2.shape[1], self.RESTART.coords['layer'].shape[0]), 
                                                            np.nan), 'LAYER_REFREEZE', 'm w.e.q.', 'Refreezing of each layer')
        
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
        self.RESULT.coords['lat'] = self.DATA.coords['lat']
        self.RESULT.coords['lon'] = self.DATA.coords['lon']
        self.RESULT.coords['time'] = self.DATA.coords['time']
    
        self.add_variable_along_time(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'SNOWHEIGHT', 'm', 'Snowheight')
        self.add_variable_along_time(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'EVAPORATION', 'm w.e.q.', 'Evaporation')
        self.add_variable_along_time(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'SUBLIMATION', 'm w.e.q.', 'Sublimation')
        self.add_variable_along_time(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'MELT', 'm w.e.q.', 'Surface melt')
        self.add_variable_along_time(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'LWin', 'W m^-2', 'Incoming longwave radiation')
        self.add_variable_along_time(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'LWout', 'W m^-2', 'Outgoing longwave radiation')
        self.add_variable_along_time(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'H', 'W m^-2', 'Sensible heat flux')
        self.add_variable_along_time(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'LE', 'W m^-2', 'Latent heat flux')
        self.add_variable_along_time(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'B', 'W m^-2', 'Ground heat flux')
        self.add_variable_along_time(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'TS', 'K', 'Surface temperature')
        self.add_variable_along_time(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'RH2', '%', 'Relative humidity at 2 m')
        self.add_variable_along_time(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'T2', 'K', 'Air temperature at 2 m')
        self.add_variable_along_time(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'G', 'W m^-2', 'Incoming shortwave radiation')
        self.add_variable_along_time(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'U2', 'm s^-1', 'Wind velocity at 2 m')
        self.add_variable_along_time(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'N', '-', 'Cloud fraction')
        self.add_variable_along_time(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'Z0', 'm', 'Roughness length')
        self.add_variable_along_time(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'ALBEDO', '-', 'Albedo')
        self.add_variable_along_time(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'RUNOFF', 'm w.e.q.', 'Runoff')
        
        if (full_field):
            self.RESULT.coords['layer'] = np.arange(max_layers)
            self.add_variable_along_time(self.RESULT, np.full_like(self.DATA.T2, np.nan), 'NLAYERS', '-', 'Number of layers')
            self.add_variable_along_layertime(self.RESULT, np.full((self.RESULT.coords['layer'].shape[0], self.RESULT.coords['time'].shape[0]), np.nan), 'LAYER_HEIGHT', 'm', 'Layer height')
            self.add_variable_along_layertime(self.RESULT, np.full((self.RESULT.coords['layer'].shape[0], self.RESULT.coords['time'].shape[0]), np.nan), 'LAYER_RHO', 'kg m^-3', 'Density of layer')
            self.add_variable_along_layertime(self.RESULT, np.full((self.RESULT.coords['layer'].shape[0], self.RESULT.coords['time'].shape[0]), np.nan), 'LAYER_T', 'K', 'Layer temperature')
            self.add_variable_along_layertime(self.RESULT, np.full((self.RESULT.coords['layer'].shape[0], self.RESULT.coords['time'].shape[0]), np.nan), 'LAYER_LWC', 'kg m^-2', 'LWC of each layer')
            self.add_variable_along_layertime(self.RESULT, np.full((self.RESULT.coords['layer'].shape[0], self.RESULT.coords['time'].shape[0]), np.nan), 'LAYER_CC', 'J m^-2', 'Cold content of each layer')
            self.add_variable_along_layertime(self.RESULT, np.full((self.RESULT.coords['layer'].shape[0], self.RESULT.coords['time'].shape[0]), np.nan), 'LAYER_POROSITY', '-', 'Porosity of each layer')
            self.add_variable_along_layertime(self.RESULT, np.full((self.RESULT.coords['layer'].shape[0], self.RESULT.coords['time'].shape[0]), np.nan), 'LAYER_VOL', '-', 'Volumetric ice content of each layer')
            self.add_variable_along_layertime(self.RESULT, np.full((self.RESULT.coords['layer'].shape[0], self.RESULT.coords['time'].shape[0]), np.nan), 'LAYER_REFREEZE', 'm w.e.q.', 'Refreezing of each layer')
    
        return self.RESULT
      
    def create_local_restart_dataset(self):
        """ This function creates the result dataset for a grid point 
        Args:
            
            self.DATA    ::  self.DATA structure 
            
        Returns:
            
            self.RESTART  ::  one-dimensional self.RESULT structure"""
    
        self.RESTART = xr.Dataset()
        self.RESTART.coords['lat'] = self.DATA.coords['lat']
        self.RESTART.coords['lon'] = self.DATA.coords['lon']
        self.RESTART.coords['time'] = self.DATA.coords['time'][-1] 
        self.RESTART.coords['layer'] = np.arange(max_layers)
       
        self.add_variable_along_latlon(self.RESTART, np.full((1), np.nan), 'NLAYERS', '-', 'Number of layers')
        self.add_variable_along_layer(self.RESTART, np.full((self.RESTART.coords['layer'].shape[0]), np.nan), 'LAYER_HEIGHT', 'm', 'Layer height')
        self.add_variable_along_layer(self.RESTART, np.full((self.RESTART.coords['layer'].shape[0]), np.nan), 'LAYER_RHO', 'kg m^-3', 'Density of layer')
        self.add_variable_along_layer(self.RESTART, np.full((self.RESTART.coords['layer'].shape[0]), np.nan), 'LAYER_T', 'K', 'Layer temperature')
        self.add_variable_along_layer(self.RESTART, np.full((self.RESTART.coords['layer'].shape[0]), np.nan), 'LAYER_LWC', 'kg m^-2', 'LWC of each layer')
        self.add_variable_along_layer(self.RESTART, np.full((self.RESTART.coords['layer'].shape[0]), np.nan), 'LAYER_CC', 'J m^-2', 'Cold content of each layer')
        self.add_variable_along_layer(self.RESTART, np.full((self.RESTART.coords['layer'].shape[0]), np.nan), 'LAYER_POROSITY', '-', 'Porosity of each layer')
        self.add_variable_along_layer(self.RESTART, np.full((self.RESTART.coords['layer'].shape[0]), np.nan), 'LAYER_VOL', '-', 'Volumetric ice content of each layer')
        self.add_variable_along_layer(self.RESTART, np.full((self.RESTART.coords['layer'].shape[0]), np.nan), 'LAYER_REFREEZE', 'm w.e.q.', 'Refreezing of each layer')
    
        return self.RESTART

    def write_results(self, results):
        """ This function aggregates the point result 
        
        results         ::  List with the result from COSIPI
        """
        # Get only the 2D fields from the results list 
        results = [x[0] for x in results]
         
        # Assign point results to the aggregated dataset
        for i in np.arange(len(results)):
            self.RESULT.SNOWHEIGHT.loc[dict(lon=results[i].lon.values, lat=results[i].lat.values)] = results[i].SNOWHEIGHT
            self.RESULT.EVAPORATION.loc[dict(lon=results[i].lon.values, lat=results[i].lat.values)] = results[i].EVAPORATION
            self.RESULT.SUBLIMATION.loc[dict(lon=results[i].lon.values, lat=results[i].lat.values)] = results[i].SUBLIMATION
            self.RESULT.MELT.loc[dict(lon=results[i].lon.values, lat=results[i].lat.values)] = results[i].MELT
            self.RESULT.LWin.loc[dict(lon=results[i].lon.values, lat=results[i].lat.values)] = results[i].LWin
            self.RESULT.LWout.loc[dict(lon=results[i].lon.values, lat=results[i].lat.values)] = results[i].LWout
            self.RESULT.H.loc[dict(lon=results[i].lon.values, lat=results[i].lat.values)] = results[i].H
            self.RESULT.LE.loc[dict(lon=results[i].lon.values, lat=results[i].lat.values)] = results[i].LE
            self.RESULT.B.loc[dict(lon=results[i].lon.values, lat=results[i].lat.values)] = results[i].B
            self.RESULT.TS.loc[dict(lon=results[i].lon.values, lat=results[i].lat.values)] = results[i].TS
            self.RESULT.RH2.loc[dict(lon=results[i].lon.values, lat=results[i].lat.values)] = results[i].RH2
            self.RESULT.T2.loc[dict(lon=results[i].lon.values, lat=results[i].lat.values)] = results[i].T2
            self.RESULT.G.loc[dict(lon=results[i].lon.values, lat=results[i].lat.values)] = results[i].G
            self.RESULT.U2.loc[dict(lon=results[i].lon.values, lat=results[i].lat.values)] = results[i].U2
            self.RESULT.N.loc[dict(lon=results[i].lon.values, lat=results[i].lat.values)] = results[i].N
            self.RESULT.Z0.loc[dict(lon=results[i].lon.values, lat=results[i].lat.values)] = results[i].Z0
            self.RESULT.ALBEDO.loc[dict(lon=results[i].lon.values, lat=results[i].lat.values)] = results[i].ALBEDO
            self.RESULT.RUNOFF.loc[dict(lon=results[i].lon.values, lat=results[i].lat.values)] = results[i].RUNOFF
           
            if full_field:
                self.RESULT.NLAYERS.loc[dict(lon=results[i].lon.values, lat=results[i].lat.values)] = results[i].NLAYERS
                self.RESULT.LAYER_HEIGHT.loc[dict(lon=results[i].lon.values, lat=results[i].lat.values, layer=np.arange(max_layers))] = results[i].LAYER_HEIGHT
                self.RESULT.LAYER_RHO.loc[dict(lon=results[i].lon.values, lat=results[i].lat.values, layer=np.arange(max_layers))] = results[i].LAYER_RHO
                self.RESULT.LAYER_T.loc[dict(lon=results[i].lon.values, lat=results[i].lat.values, layer=np.arange(max_layers))] = results[i].LAYER_T
                self.RESULT.LAYER_LWC.loc[dict(lon=results[i].lon.values, lat=results[i].lat.values, layer=np.arange(max_layers))] = results[i].LAYER_LWC
                self.RESULT.LAYER_CC.loc[dict(lon=results[i].lon.values, lat=results[i].lat.values, layer=np.arange(max_layers))] = results[i].LAYER_CC
                self.RESULT.LAYER_POROSITY.loc[dict(lon=results[i].lon.values, lat=results[i].lat.values, layer=np.arange(max_layers))] = results[i].LAYER_POROSITY
                self.RESULT.LAYER_VOL.loc[dict(lon=results[i].lon.values, lat=results[i].lat.values, layer=np.arange(max_layers))] = results[i].LAYER_VOL
                self.RESULT.LAYER_REFREEZE.loc[dict(lon=results[i].lon.values, lat=results[i].lat.values, layer=np.arange(max_layers))] = results[i].LAYER_REFREEZE
    
        comp = dict(zlib=True, complevel=9)
        encoding = {var: comp for var in self.RESULT.data_vars}
   
        print('\n')
        print('--------------------------------------------------------------')
        print('\t Writing result file %s \n' % (output_netcdf))
        
        self.RESULT.to_netcdf(output_netcdf, encoding=encoding)

    def write_restart(self, results):
        """ Writes the restart file 
        
        RESULT      :: RESULT dataset
        
        """

        # Close open restart file so that the new file can be written
        if (restart==True):
            self.GRID_RESTART.close()

        # Create restart file (last time step)
        # Get only the restart fields from the results list 
        results = [x[1] for x in results]
        
        for i in np.arange(len(results)):
            self.RESTART['NLAYERS'] = results[i].NLAYERS.values
            self.RESTART.LAYER_HEIGHT.loc[dict(lon=results[i].lon.values, lat=results[i].lat.values, layer=np.arange(max_layers))] = results[i].LAYER_HEIGHT
            self.RESTART.LAYER_RHO.loc[dict(lon=results[i].lon.values, lat=results[i].lat.values, layer=np.arange(max_layers))] = results[i].LAYER_RHO
            self.RESTART.LAYER_T.loc[dict(lon=results[i].lon.values, lat=results[i].lat.values, layer=np.arange(max_layers))] = results[i].LAYER_T
            self.RESTART.LAYER_LWC.loc[dict(lon=results[i].lon.values, lat=results[i].lat.values, layer=np.arange(max_layers))] = results[i].LAYER_LWC
            self.RESTART.LAYER_CC.loc[dict(lon=results[i].lon.values, lat=results[i].lat.values, layer=np.arange(max_layers))] = results[i].LAYER_CC
            self.RESTART.LAYER_POROSITY.loc[dict(lon=results[i].lon.values, lat=results[i].lat.values, layer=np.arange(max_layers))] = results[i].LAYER_POROSITY
            self.RESTART.LAYER_VOL.loc[dict(lon=results[i].lon.values, lat=results[i].lat.values, layer=np.arange(max_layers))] = results[i].LAYER_VOL
            self.RESTART.LAYER_REFREEZE.loc[dict(lon=results[i].lon.values, lat=results[i].lat.values, layer=np.arange(max_layers))] = results[i].LAYER_REFREEZE
    
        comp = dict(zlib=True, complevel=9)
        encoding = {var: comp for var in self.RESTART.data_vars}
    
        print('\t Writing restart file %s \n' % (restart_netcdf))
        print('-------------------------------------------------------------- \n')
        
        self.RESTART.to_netcdf(restart_netcdf, encoding=encoding)



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
        ds[name] = (('lat','lon','time'), var)
        ds[name].attrs['units'] = units
        ds[name].attrs['long_name'] = long_name
        ds[name].encoding['_FillValue'] = -9999
        return ds
    
    def add_variable_along_latlonlayertime(self, ds, var, name, units, long_name):
        """ This function self.adds missing variables to the self.DATA class """
        ds[name] = (('lat','lon','layer','time'), var)
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
        ds[name] = (('layer','time'), var)
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
