"""
 Read the input data (model forcing) and write the output to netCDF file.
"""

import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

from cosipy.config import Config
from cosipy.constants import Constants


class IOClass:

    def __init__(self, DATA=None):
        """Initialise the IO Class.
        
        Attributes:
            atm (str): Meteorological variables selected for output.
            internal (str): Glacier variables selected for output.
            full (str): Full field variables selected for output.
            DATA (xarray.Dataset or None): Meteorological data.
            RESTART (xarray.Dataset or None): Restart file data.
            RESULT (xarray.Dataset or None): Model result data.
        """

        self.atm = self.get_output_variables(Config.output_atm)
        self.internal = self.get_output_variables(Config.output_internal)
        self.full = self.get_output_variables(Config.output_full)

        # Initialize data
        self.DATA = DATA
        self.RESTART = None
        self.RESULT = None
      
        # If local IO class is initialized we need to get the dimensions of the dataset
        if DATA is not None:
            self.time = self.DATA.sizes["time"]

    def get_output_variables(self, variables: str) -> list:
        """Get output variable names from configuration string."""
        # Sets are unordered -> same config, different checksums
        return [item for item in variables.split(",")]

    def create_data_file(self):
        """Create the input data and read the restart file if necessary.

        Returns:
            xarray.Dataset: Model input data.
        """
    
        if Config.restart:
            print(f"{'-'*62}\n\tRESTART FROM PREVIOUS STATE\n{'-'*62}\n")
            
            # Load the restart file
            timestamp = pd.to_datetime(Config.time_start).strftime('%Y-%m-%dT%H-%M')
            restart_path = os.path.join(
                Config.data_path,"restart", f"restart_{timestamp}.nc"
            )
            if os.path.isfile(restart_path) & (Config.time_start != Config.time_end):
                self.GRID_RESTART = xr.open_dataset(restart_path)
                self.restart_date = self.GRID_RESTART.time+np.timedelta64(Constants.dt,'s')     # Get time of the last calculation and add one time step
                self.init_data_dataset()                       # Read data from the last date to the end of the data file
            else:
                print(  # if there is a problem kill the program
                    f"No restart file available for the given date {timestamp}\nOR start date {Config.time_start} equals end date {Config.time_end}\n"
                )
                sys.exit(1)
        else:
            self.restart_date = None
            self.init_data_dataset()  # If no restart, read data according to the dates defined in the config.py

        #----------------------------------------------
        # Tile the data is desired
        #----------------------------------------------
        if Config.tile:
            if Config.WRF:
                self.DATA = self.DATA.isel(south_north=slice(Config.ystart,Config.yend), west_east=slice(Config.xstart,Config.xend))
            else:
                self.DATA = self.DATA.isel(lat=slice(Config.ystart,Config.yend), lon=slice(Config.xstart,Config.xend))

        self.ny = self.DATA.sizes[Config.northing]
        self.nx = self.DATA.sizes[Config.easting]
        self.time = self.DATA.sizes['time']

        return self.DATA


    def create_result_file(self) -> xr.Dataset:
        """Create and initialise the RESULT dataset."""
        self.init_result_dataset()
        return self.RESULT
         
    def create_restart_file(self) -> xr.Dataset:
        """Create and initialise the RESTART dataset."""
        self.init_restart_dataset()
        return self.RESTART

    def create_grid_restart(self):
        """Create and initialise the GRID_RESTART structure.

        This contains the layer state of the last time step, which is
        required for the restart option.
        """
        return self.GRID_RESTART

    def create_nan_array(self) -> np.ndarray:
        """Create and fill a NaN array with time, (x,y) dimensions.
        
        Returns:
            Filled array with time and 2D spatial coordinates.
        """

        return np.full((self.time, self.ny, self.nx), np.nan)

    def create_3d_nan_array(self, max_layers: int) -> np.ndarray:
        """Create and fill a NaN array with time, (x,y,z) dimensions.

        Args:
            The maximum number of model layers.

        Returns:
            Filled array with time and 3D spatial coordinates.
        """

        return np.full((self.time, self.ny, self.nx, max_layers), np.nan)

    def init_data_dataset(self):
        """Read and store the input netCDF data.

        The input data should contain the following variables:
            :PRES: Air pressure [hPa].
            :N: Cloud cover fraction [-].
            :RH2: 2m relative humidity [%].
            :RRR: Precipitation per time step [mm].
            :SNOWFALL: Snowfall per time step [m].
            :G: Solar radiation per time step [|W m^-2|].
            :T2: 2m air temperature [K].
            :U2: Wind speed (magnitude) [|m s^-1|].
            :HGT: Elevation [m].
        """
    
        # Open input dataset
        self.DATA = xr.open_dataset(os.path.join(Config.data_path,'input',Config.input_netcdf))
        self.DATA['time'] = np.sort(self.DATA['time'].values)
        start_interval=str(self.DATA.time.values[0])[0:16]
        end_interval = str(self.DATA.time.values[-1])[0:16]
        time_steps = str(self.DATA.sizes['time'])
        print(
            f"\nMaximum available time interval from {start_interval} until {end_interval}. Time steps: {time_steps}\n\n")

        # Check if restart option is set
        if self.restart_date is None:
            print(
                f"{'-'*62}\n\tIntegration from {Config.time_start} to {Config.time_end}\n{'-'*62}\n"
            )
            self.DATA = self.DATA.sel(time=slice(Config.time_start, Config.time_end))   # Select dates from config.py
        else:
            # There is nothing to do if the dates are equal
            if self.restart_date==Config.time_end:
                print('Start date equals end date ... no new data ... EXIT')
                sys.exit(1)
            else:
                # otherwise, run the model from the restart date to the defined end date
                print(f'Starting from {self.restart_date.values} (from restart file) to {Config.time_end} (from config.toml)\n')
                self.DATA = self.DATA.sel(time=slice(self.restart_date, Config.time_end))

        if Config.time_start < start_interval:
            print('\nWARNING! Selected startpoint before first timestep of input data\n')
        if Config.time_end > end_interval:
            print('\nWARNING! Selected endpoint after last timestep of input data\n')
        if Config.time_start > end_interval or Config.time_end < start_interval:
            print('\nERROR! Selected period not available in input data\n')


        print('--------------------------------------------------------------')
        print(f"{'-'*62}\nChecking input data ....\n")
        
        # Define an auxiliary function to check the validity of the data
        def check(field, _max, _min):
            """Check the validity of the input data."""
            if np.nanmax(field) > _max or np.nanmin(field) < _min:
                print(f"Please check the input data, its seems they are out of range {str.capitalize(field.name)} MAX: {np.nanmax(field):.2f} MIN: {np.nanmin(field):.2f} \n")

        # Check if data is within valid bounds
        data_bounds = {
            "T2": (313.16, 243.16),
            "RH2": (100.0, 0.0),
            "G": (1600.0, 0.0),
            "U2": (50.0, 0.0),
            "RRR": (20.0, 0.0),
            "N": (1.0, 0.0),
            "PRES": (1080.0, 400.0),
            "LWin": (400.0, 200.0),
            "SNOWFALL": (0.1, 0.0),
            "SLOPE": (0.0, 90.0),
        }

        for key, bounds in data_bounds.items():
            if key in self.DATA:
                print(f"{key} ... ok")
                check(self.DATA[key], bounds[0], bounds[1])

        print(f"\nGlacier gridpoints: {np.nansum(self.DATA.MASK >= 1)} \n\n")

 
    def get_result_metadata(self) -> tuple:
        """Get variable names and units.
        
        Returns:
            tuple[dict, dict]: Metadata for spatial and spatiotemporal
            variables, which includes netCDF keyname, unit, and long
            name.
        """
        metadata_spatial = {
            "HGT": ("m", "Elevation"),
            "MASK": ("boolean", "Glacier mask"),
            "SLOPE": ("degrees", "Terrain slope"),
            "ASPECT": ("degrees", "Aspect of slope"),
        }
        metadata_spatiotemporal = {
            "T2": ("K", "Air temperature at 2 m"),
            "RH2": ("%", "Relative humidity at 2 m"),
            "U2": ("m s\u207b\xb9", "Wind velocity at 2 m"),
            "PRES": ("hPa", "Atmospheric pressure"),
            "G": ("W m\u207b\xb2", "Incoming shortwave radiation"),
            "RRR": ("mm", "Total precipitation"),
            "SNOWFALL": ("m", "Snowfall"),
            "N": ("-", "Cloud fraction"),
            "LWin": ("W m\u207b\xb2", "Incoming longwave radiation"),
        }

        return metadata_spatial, metadata_spatiotemporal

    def init_result_dataset(self) -> xr.Dataset:
        """Create the final dataset to aggregate and store the results.

        Aggregates results from individual COSIPY runs. After the
        dataset is filled with results from all the workers, the dataset
        is written to disk.

        Returns:
            One-dimensional structure with the model output.
        """
        
        # Coordinates
        self.RESULT = xr.Dataset()
        self.RESULT.coords['time'] = self.DATA.coords['time']
        self.RESULT.coords['lat'] = self.DATA.coords['lat']
        self.RESULT.coords['lon'] = self.DATA.coords['lon']

        # Global attributes from config.py
        self.RESULT.attrs['Start_from_restart_file'] = str(Config.restart)
        self.RESULT.attrs['Stake_evaluation'] = str(Config.stake_evaluation)
        self.RESULT.attrs['WRF_simulation'] = str(Config.WRF)
        self.RESULT.attrs['Compression_level'] = Config.compression_level
        self.RESULT.attrs['Slurm_use'] = str(Config.slurm_use)
        self.RESULT.attrs['Full_field'] = str(Config.full_field)
        self.RESULT.attrs['Force_use_TP'] = str(Config.force_use_TP)
        self.RESULT.attrs['Force_use_N'] = str(Config.force_use_N)
        self.RESULT.attrs['Tile_of_glacier_of_interest'] = str(Config.tile)

        # Global attributes from constants.py
        self.RESULT.attrs['Time_step_input_file_seconds'] = Constants.dt
        self.RESULT.attrs['Max_layers'] = Constants.max_layers
        self.RESULT.attrs['Z_measurement_height'] = Constants.z
        self.RESULT.attrs['Stability_correction'] = Constants.stability_correction
        self.RESULT.attrs['Albedo_method'] = Constants.albedo_method
        self.RESULT.attrs['Densification_method'] = Constants.densification_method
        self.RESULT.attrs['Penetrating_method'] = Constants.penetrating_method
        self.RESULT.attrs['Roughness_method'] = Constants.roughness_method
        self.RESULT.attrs['Saturation_water_vapour_method'] = Constants.saturation_water_vapour_method

        self.RESULT.attrs['Initial_snowheight'] = Constants.initial_snowheight_constant
        self.RESULT.attrs['Initial_snow_layer_heights'] = Constants.initial_snow_layer_heights
        self.RESULT.attrs['Initial_glacier_height'] = Constants.initial_glacier_height
        self.RESULT.attrs['Initial_glacier_layer_heights'] = Constants.initial_glacier_layer_heights
        self.RESULT.attrs['Initial_top_density_snowpack'] = Constants.initial_top_density_snowpack
        self.RESULT.attrs['Initial_bottom_density_snowpack'] = Constants.initial_bottom_density_snowpack
        self.RESULT.attrs['Temperature_bottom'] = Constants.temperature_bottom
        self.RESULT.attrs['Const_init_temp'] = Constants.const_init_temp

        self.RESULT.attrs['Center_snow_transfer_function'] = Constants.center_snow_transfer_function
        self.RESULT.attrs['Spread_snow_transfer_function'] = Constants.spread_snow_transfer_function
        self.RESULT.attrs['Multiplication_factor_for_RRR_or_SNOWFALL'] = Constants.mult_factor_RRR
        self.RESULT.attrs['Minimum_snow_layer_height'] = Constants.minimum_snow_layer_height
        self.RESULT.attrs['Minimum_snowfall'] = Constants.minimum_snowfall

        self.RESULT.attrs['Remesh_method'] = Constants.remesh_method
        self.RESULT.attrs['First_layer_height_log_profile'] = Constants.first_layer_height
        self.RESULT.attrs['Layer_stretching_log_profile'] = Constants.layer_stretching

        self.RESULT.attrs['Merge_max'] = Constants.merge_max
        self.RESULT.attrs['Layer_stretching_log_profile'] = Constants.layer_stretching
        self.RESULT.attrs['Density_threshold_merging'] = Constants.density_threshold_merging
        self.RESULT.attrs['Temperature_threshold_merging'] = Constants.temperature_threshold_merging

        self.RESULT.attrs['Density_fresh_snow'] = Constants.constant_density
        self.RESULT.attrs['Albedo_fresh_snow'] = Constants.albedo_fresh_snow
        self.RESULT.attrs['Albedo_firn'] = Constants.albedo_firn
        self.RESULT.attrs['Albedo_ice'] = Constants.albedo_ice
        self.RESULT.attrs['Albedo_mod_snow_aging'] = Constants.albedo_mod_snow_aging
        self.RESULT.attrs['Albedo_mod_snow_depth'] = Constants.albedo_mod_snow_depth
        self.RESULT.attrs['Roughness_fresh_snow'] = Constants.roughness_fresh_snow
        self.RESULT.attrs['Roughness_ice'] = Constants.roughness_ice
        self.RESULT.attrs['Roughness_firn'] = Constants.roughness_firn
        self.RESULT.attrs['Aging_factor_roughness'] = Constants.aging_factor_roughness
        self.RESULT.attrs['Snow_ice_threshold'] = Constants.snow_ice_threshold

        self.RESULT.attrs['lat_heat_melting'] = Constants.lat_heat_melting
        self.RESULT.attrs['lat_heat_vaporize'] = Constants.lat_heat_vaporize
        self.RESULT.attrs['lat_heat_sublimation'] = Constants.lat_heat_sublimation
        self.RESULT.attrs['spec_heat_air'] = Constants.spec_heat_air
        self.RESULT.attrs['spec_heat_ice'] = Constants.spec_heat_ice
        self.RESULT.attrs['spec_heat_water'] = Constants.spec_heat_water
        self.RESULT.attrs['k_i'] = Constants.k_i
        self.RESULT.attrs['k_w'] = Constants.k_w
        self.RESULT.attrs['k_a'] = Constants.k_a
        self.RESULT.attrs['water_density'] = Constants.water_density
        self.RESULT.attrs['ice_density'] = Constants.ice_density
        self.RESULT.attrs['air_density'] = Constants.air_density
        self.RESULT.attrs['sigma'] = Constants.sigma
        self.RESULT.attrs['zero_temperature'] = Constants.zero_temperature
        self.RESULT.attrs['Surface_emission_coeff'] = Constants.surface_emission_coeff

        # Variables given by the input dataset
        spatial, spatiotemporal = self.get_result_metadata()

        for name, metadata in spatial.items():
            if name in self.DATA:
                self.add_variable_along_latlon(
                    self.RESULT, self.DATA[name], name, metadata[0], metadata[1]
                )
        for name, metadata in spatiotemporal.items():
            if name in self.DATA:
                self.add_variable_along_latlontime(
                    self.RESULT, self.DATA[name], name, metadata[0], metadata[1]
                )

        if "RRR" not in self.DATA:
            self.add_variable_along_latlontime(
                self.RESULT,
                np.full_like(self.DATA.T2, np.nan),
                "RRR",
                spatiotemporal["RRR"][1],
                spatiotemporal["RRR"][0],
            )
        if "N" not in self.DATA:
            self.add_variable_along_latlontime(
                self.RESULT,
                np.full_like(self.DATA.T2, np.nan),
                "N",
                spatiotemporal["N"][1],
                spatiotemporal["N"][0],
            )

        print(f"\nOutput dataset ... ok")

        return self.RESULT
  

    def create_global_result_arrays(self):
        """Create the global numpy arrays to store each output variable.

        Each global array will be filled with local results from the
        workers. The arrays will then be assigned to the RESULT dataset
        and stored to disk (see COSIPY.py).
        """
        
        if "RAIN" in self.atm:
            self.RAIN = self.create_nan_array()
        if "SNOWFALL" in self.atm:
            self.SNOWFALL = self.create_nan_array()
        if "LWin" in self.atm:
            self.LWin = self.create_nan_array()
        if "LWout" in self.atm:
            self.LWout = self.create_nan_array()
        if "H" in self.atm:
            self.H = self.create_nan_array()
        if "LE" in self.atm:
            self.LE = self.create_nan_array()
        if "B" in self.atm:
            self.B = self.create_nan_array()
        if "QRR" in self.atm:
            self.QRR = self.create_nan_array()
        if "TS" in self.atm:
            self.TS = self.create_nan_array()
        if "ALBEDO" in self.atm:
            self.ALBEDO = self.create_nan_array()
        if "Z0" in self.atm:
            self.Z0 = self.create_nan_array()

        if "MB" in self.internal:
            self.MB = self.create_nan_array()
        if "surfMB" in self.internal:
            self.surfMB = self.create_nan_array()
        if "Q" in self.internal:
            self.Q = self.create_nan_array()
        if "SNOWHEIGHT" in self.internal:
            self.SNOWHEIGHT = self.create_nan_array()
        if "TOTALHEIGHT" in self.internal:
            self.TOTALHEIGHT = self.create_nan_array()
        if "LAYERS" in self.internal:
            self.LAYERS = self.create_nan_array()
        if "ME" in self.internal:
            self.ME = self.create_nan_array()
        if "intMB" in self.internal:
            self.intMB = self.create_nan_array()
        if "EVAPORATION" in self.internal:
            self.EVAPORATION = self.create_nan_array()
        if "SUBLIMATION" in self.internal:
            self.SUBLIMATION = self.create_nan_array()
        if "CONDENSATION" in self.internal:
            self.CONDENSATION = self.create_nan_array()
        if "DEPOSITION" in self.internal:
            self.DEPOSITION = self.create_nan_array()
        if "REFREEZE" in self.internal:
            self.REFREEZE = self.create_nan_array()
        if "subM" in self.internal:
            self.subM = self.create_nan_array()
        if "surfM" in self.internal:
            self.surfM = self.create_nan_array()
        if "MOL" in self.internal:
            self.MOL = self.create_nan_array()

        if Config.full_field:
            max_layers = Constants.max_layers  # faster lookup
            if "HEIGHT" in self.full:
                self.LAYER_HEIGHT = self.create_3d_nan_array(max_layers)
            if "RHO" in self.full:
                self.LAYER_RHO = self.create_3d_nan_array(max_layers)
            if "T" in self.full:
                self.LAYER_T = self.create_3d_nan_array(max_layers)
            if "LWC" in self.full:
                self.LAYER_LWC = self.create_3d_nan_array(max_layers)
            if "CC" in self.full:
                self.LAYER_CC = self.create_3d_nan_array(max_layers)
            if "POROSITY" in self.full:
                self.LAYER_POROSITY = self.create_3d_nan_array(max_layers)
            if "ICE_FRACTION" in self.full:
                self.LAYER_ICE_FRACTION = self.create_3d_nan_array(max_layers)
            if "IRREDUCIBLE_WATER" in self.full:
                self.LAYER_IRREDUCIBLE_WATER = self.create_3d_nan_array(max_layers)
            if "REFREEZE" in self.full:
                self.LAYER_REFREEZE = self.create_3d_nan_array(max_layers)
   

    def copy_local_to_global(self,y,x,local_RAIN,local_SNOWFALL,local_LWin,local_LWout,local_H,local_LE,local_B,local_QRR,
                             local_MB, local_surfMB,local_Q,local_SNOWHEIGHT,local_TOTALHEIGHT,local_TS,local_ALBEDO, \
                             local_LAYERS,local_ME,local_intMB,local_EVAPORATION,local_SUBLIMATION,local_CONDENSATION, \
                             local_DEPOSITION,local_REFREEZE,local_subM,local_Z0,local_surfM,local_MOL,local_LAYER_HEIGHT, \
                             local_LAYER_RHO,local_LAYER_T,local_LAYER_LWC,local_LAYER_CC,local_LAYER_POROSITY, \
                             local_LAYER_ICE_FRACTION,local_LAYER_IRREDUCIBLE_WATER,local_LAYER_REFREEZE):
        """Copy the local results from workers to global numpy arrays."""
        if "RAIN" in self.atm:
            self.RAIN[:,y,x] = local_RAIN
        if "SNOWFALL" in self.atm:
            self.SNOWFALL[:,y,x] = local_SNOWFALL
        if "LWin" in self.atm:
            self.LWin[:,y,x] = local_LWin
        if "LWout" in self.atm:
            self.LWout[:,y,x] = local_LWout
        if "H" in self.atm:
            self.H[:,y,x] = local_H
        if "LE" in self.atm:
            self.LE[:,y,x] = local_LE
        if "B" in self.atm:
            self.B[:,y,x] = local_B
        if "QRR" in self.atm:
            self.QRR[:,y,x] = local_QRR
        if "TS" in self.atm:
            self.TS[:,y,x] = local_TS 
        if "ALBEDO" in self.atm:
            self.ALBEDO[:,y,x] = local_ALBEDO 
        if "Z0" in self.atm:
            self.Z0[:,y,x] = local_Z0 
        
        if "MB" in self.internal:
            self.MB[:,y,x] = local_MB
        if "surfMB" in self.internal:
            self.surfMB[:,y,x] = local_surfMB
        if "Q" in self.internal:
            self.Q[:,y,x] = local_Q
        if "SNOWHEIGHT" in self.internal:
            self.SNOWHEIGHT[:,y,x] = local_SNOWHEIGHT
        if "TOTALHEIGHT" in self.internal:
            self.TOTALHEIGHT[:,y,x] = local_TOTALHEIGHT 
        if "LAYERS" in self.internal:
            self.LAYERS[:,y,x] = local_LAYERS 
        if "ME" in self.internal:
            self.ME[:,y,x] = local_ME 
        if "intMB" in self.internal:
            self.intMB[:,y,x] = local_intMB 
        if "EVAPORATION" in self.internal:
            self.EVAPORATION[:,y,x] = local_EVAPORATION 
        if "SUBLIMATION" in self.internal:
            self.SUBLIMATION[:,y,x] = local_SUBLIMATION 
        if "CONDENSATION" in self.internal:
            self.CONDENSATION[:,y,x] = local_CONDENSATION 
        if "DEPOSITION" in self.internal:
            self.DEPOSITION[:,y,x] = local_DEPOSITION 
        if "REFREEZE" in self.internal:
            self.REFREEZE[:,y,x] = local_REFREEZE 
        if "subM" in self.internal:
            self.subM[:,y,x] = local_subM 
        if "surfM" in self.internal:
            self.surfM[:,y,x] = local_surfM 
        if "MOL" in self.internal:
            self.MOL[:,y,x] = local_MOL 

        if Config.full_field:
            if "HEIGHT" in self.full:
                self.LAYER_HEIGHT[:,y,x,:] = local_LAYER_HEIGHT 
            if "RHO" in self.full:
                self.LAYER_RHO[:,y,x,:] = local_LAYER_RHO 
            if "T" in self.full:
                self.LAYER_T[:,y,x,:] = local_LAYER_T 
            if "LWC" in self.full:
                self.LAYER_LWC[:,y,x,:] = local_LAYER_LWC 
            if "CC" in self.full:
                self.LAYER_CC[:,y,x,:] = local_LAYER_CC 
            if "POROSITY" in self.full:
                self.LAYER_POROSITY[:,y,x,:] = local_LAYER_POROSITY 
            if "ICE_FRACTION" in self.full:
                self.LAYER_ICE_FRACTION[:,y,x,:] = local_LAYER_ICE_FRACTION 
            if "IRREDUCIBLE_WATER" in self.full:
                self.LAYER_IRREDUCIBLE_WATER[:,y,x,:] = local_LAYER_IRREDUCIBLE_WATER 
            if "REFREEZE" in self.full:
                self.LAYER_REFREEZE[:,y,x,:] = local_LAYER_REFREEZE 


    def write_results_to_file(self):
        """Add the global numpy arrays to the RESULT dataset."""

        if "RAIN" in self.atm:
            self.add_variable_along_latlontime(self.RESULT, self.RAIN, "RAIN", "mm", "Liquid precipitation") 
        if "SNOWFALL" in self.atm:
            self.add_variable_along_latlontime(self.RESULT, self.SNOWFALL, "SNOWFALL", "m w.e.", "Snowfall") 
        if "LWin" in self.atm:
            self.add_variable_along_latlontime(self.RESULT, self.LWin, "LWin", "W m\u207b\xb2", "Incoming longwave radiation") 
        if "LWout" in self.atm:
            self.add_variable_along_latlontime(self.RESULT, self.LWout, "LWout", "W m\u207b\xb2", "Outgoing longwave radiation") 
        if "H" in self.atm:
            self.add_variable_along_latlontime(self.RESULT, self.H, "H", "W m\u207b\xb2", "Sensible heat flux") 
        if "LE" in self.atm:
            self.add_variable_along_latlontime(self.RESULT, self.LE, "LE", "W m\u207b\xb2", "Latent heat flux") 
        if "B" in self.atm:
            self.add_variable_along_latlontime(self.RESULT, self.B, "B", "W m\u207b\xb2", "Ground heat flux")
        if "QRR" in self.atm:
            self.add_variable_along_latlontime(self.RESULT, self.QRR, "QRR", "W m\u207b\xb2", "Rain heat flux")
        if "TS" in self.atm:
            self.add_variable_along_latlontime(self.RESULT, self.TS, "TS", "K", "Surface temperature") 
        if "ALBEDO" in self.atm:
            self.add_variable_along_latlontime(self.RESULT, self.ALBEDO, "ALBEDO", "-", "Albedo") 
        if "Z0" in self.atm:
            self.add_variable_along_latlontime(self.RESULT, self.Z0, "Z0", "m", "Roughness length") 
        
        if "surfMB" in self.internal:
            self.add_variable_along_latlontime(self.RESULT, self.surfMB, "surfMB", "m w.e.", "Surface mass balance") 
        if "MB" in self.internal:
            self.add_variable_along_latlontime(self.RESULT, self.MB, "MB", "m w.e.", "Mass balance") 
        if "Q" in self.internal:
            self.add_variable_along_latlontime(self.RESULT, self.Q, "Q", "m w.e.", "Runoff") 
        if "SNOWHEIGHT" in self.internal:
            self.add_variable_along_latlontime(self.RESULT, self.SNOWHEIGHT, "SNOWHEIGHT", "m", "Snowheight") 
        if "TOTALHEIGHT" in self.internal:
            self.add_variable_along_latlontime(self.RESULT, self.TOTALHEIGHT, "TOTALHEIGHT", "m", "Total domain height") 
        if "LAYERS" in self.internal:
            self.add_variable_along_latlontime(self.RESULT, self.LAYERS, "LAYERS", "-", "Number of layers") 
        if "ME" in self.internal:
            self.add_variable_along_latlontime(self.RESULT, self.ME, "ME", "W m\u207b\xb2", "Available melt energy") 
        if "intMB" in self.internal:
            self.add_variable_along_latlontime(self.RESULT, self.intMB, "intMB", "m w.e.", "Internal mass balance") 
        if "EVAPORATION" in self.internal:
            self.add_variable_along_latlontime(self.RESULT, self.EVAPORATION, "EVAPORATION", "m w.e.", "Evaporation") 
        if "SUBLIMATION" in self.internal:
            self.add_variable_along_latlontime(self.RESULT, self.SUBLIMATION, "SUBLIMATION", "m w.e.", "Sublimation") 
        if "CONDENSATION" in self.internal:
            self.add_variable_along_latlontime(self.RESULT, self.CONDENSATION, "CONDENSATION", "m w.e.", "Condensation") 
        if "DEPOSITION" in self.internal:
            self.add_variable_along_latlontime(self.RESULT, self.DEPOSITION, "DEPOSITION", "m w.e.", "Deposition") 
        if "REFREEZE" in self.internal:
            self.add_variable_along_latlontime(self.RESULT, self.REFREEZE, "REFREEZE", "m w.e.", "Refreezing") 
        if "subM" in self.internal:
            self.add_variable_along_latlontime(self.RESULT, self.subM, "subM", "m w.e.", "Subsurface melt") 
        if "surfM" in self.internal:
            self.add_variable_along_latlontime(self.RESULT, self.surfM, "surfM", "m w.e.", "Surface melt") 
        if "MOL" in self.internal:
            self.add_variable_along_latlontime(self.RESULT, self.MOL, "MOL", "m", "Monin Obukhov length") 

        if Config.full_field:
            if "HEIGHT" in self.full:
                self.add_variable_along_latlonlayertime(self.RESULT, self.LAYER_HEIGHT, "LAYER_HEIGHT", "m", "Layer height") 
            if "RHO" in self.full:
                self.add_variable_along_latlonlayertime(self.RESULT, self.LAYER_RHO, "LAYER_RHO", "kg m^-3", "Layer density") 
            if "T" in self.full:
                self.add_variable_along_latlonlayertime(self.RESULT, self.LAYER_T, "LAYER_T", "K", "Layer temperature") 
            if "LWC" in self.full:
                self.add_variable_along_latlonlayertime(self.RESULT, self.LAYER_LWC, "LAYER_LWC", "kg m^-2", "layer liquid water content") 
            if "CC" in self.full:
                self.add_variable_along_latlonlayertime(self.RESULT, self.LAYER_CC, "LAYER_CC", "J m^-2", "Cold content") 
            if "POROSITY" in self.full:
                self.add_variable_along_latlonlayertime(self.RESULT, self.LAYER_POROSITY, "LAYER_POROSITY", "-", "Porosity") 
            if "ICE_FRACTION" in self.full:
                self.add_variable_along_latlonlayertime(self.RESULT, self.LAYER_ICE_FRACTION, "LAYER_ICE_FRACTION", "-", "Layer ice fraction") 
            if "IRREDUCIBLE_WATER" in self.full:
                self.add_variable_along_latlonlayertime(self.RESULT, self.LAYER_IRREDUCIBLE_WATER, "LAYER_IRREDUCIBLE_WATER", "-", "Irreducible water") 
            if "REFREEZE" in self.full:
                self.add_variable_along_latlonlayertime(self.RESULT, self.LAYER_REFREEZE, "LAYER_REFREEZE", "m w.e.", "Refreezing") 

    def create_empty_restart(self) -> xr.Dataset:
        """Create an empty dataset for the RESTART attribute.

        Returns:
            Empty xarray dataset with coordinates from self.DATA.
        """
        dataset = xr.Dataset()
        dataset.coords['time'] = self.DATA.coords['time'][-1]
        dataset.coords['lat'] = self.DATA.coords['lat']
        dataset.coords['lon'] = self.DATA.coords['lon']
        dataset.coords['layer'] = np.arange(Constants.max_layers)

        return dataset

    def init_restart_dataset(self) -> xr.Dataset:
        """Initialise the restart dataset.

        Returns:
            The empty restart dataset.
        """
        self.RESTART = self.create_empty_restart()

        print(f"Restart dataset ... ok\n{'-'*62}\n")

        return self.RESTART
  

    def create_global_restart_arrays(self):
        """Initialise the global numpy arrays to store layer profiles.

        Each global array will be filled with local results from the
        workers. The arrays will then be assigned to the RESTART dataset
        and stored to disk (see COSIPY.py).
        """

        max_layers = Constants.max_layers  # faster lookup

        self.RES_NLAYERS = np.full((self.ny,self.nx), np.nan)
        self.RES_NEWSNOWHEIGHT = np.full((self.ny, self.nx), np.nan)
        self.RES_NEWSNOWTIMESTAMP = np.full((self.ny, self.nx), np.nan)
        self.RES_OLDSNOWTIMESTAMP = np.full((self.ny, self.nx), np.nan)
        self.RES_LAYER_HEIGHT = np.full((self.ny,self.nx,max_layers), np.nan)
        self.RES_LAYER_RHO = np.full((self.ny,self.nx,max_layers), np.nan)
        self.RES_LAYER_T = np.full((self.ny,self.nx,max_layers), np.nan)
        self.RES_LAYER_LWC = np.full((self.ny,self.nx,max_layers), np.nan)
        self.RES_LAYER_IF = np.full((self.ny,self.nx,max_layers), np.nan)


    def create_local_restart_dataset(self) -> xr.Dataset:
        """Create the result dataset for a single grid point.
            
        Returns:
            RESTART dataset initialised with layer profiles.
        """
    
        self.RESTART = self.create_empty_restart()
        
        self.add_variable_along_scalar(self.RESTART, np.full((1), np.nan), 'NLAYERS', '-', 'Number of layers')
        self.add_variable_along_scalar(self.RESTART, np.full((1), np.nan), 'NEWSNOWHEIGHT', 'm .w.e', 'New snow height')
        self.add_variable_along_scalar(self.RESTART, np.full((1), np.nan), 'NEWSNOWTIMESTAMP', 's', 'New snow timestamp')
        self.add_variable_along_scalar(self.RESTART, np.full((1), np.nan), 'OLDSNOWTIMESTAMP', 's', 'Old snow timestamp')

        self.add_variable_along_layer(self.RESTART, np.full((self.RESTART.coords['layer'].shape[0]), np.nan), 'LAYER_HEIGHT', 'm', 'Layer height')
        self.add_variable_along_layer(self.RESTART, np.full((self.RESTART.coords['layer'].shape[0]), np.nan), 'LAYER_RHO', 'kg m^-3', 'Layer density')
        self.add_variable_along_layer(self.RESTART, np.full((self.RESTART.coords['layer'].shape[0]), np.nan), 'LAYER_T', 'K', 'Layer temperature')
        self.add_variable_along_layer(self.RESTART, np.full((self.RESTART.coords['layer'].shape[0]), np.nan), 'LAYER_LWC', '-', 'Layer liquid water content')
        self.add_variable_along_layer(self.RESTART, np.full((self.RESTART.coords['layer'].shape[0]), np.nan), 'LAYER_IF', '-', 'Layer ice fraction')


        return self.RESTART
    

    def copy_local_restart_to_global(self,y,x,local_restart):
        """Copy local restart data from workers to global numpy arrays.

        Args:
            y: Latitude index.
            x: Longitude index.
            local_restart: Local RESTART dataset.
        """
        self.RES_NLAYERS[y,x] = local_restart.NLAYERS
        self.RES_NEWSNOWHEIGHT[y,x] = local_restart.NEWSNOWHEIGHT
        self.RES_NEWSNOWTIMESTAMP[y,x] = local_restart.NEWSNOWTIMESTAMP
        self.RES_OLDSNOWTIMESTAMP[y,x] = local_restart.OLDSNOWTIMESTAMP
        self.RES_LAYER_HEIGHT[y,x,:] = local_restart.LAYER_HEIGHT 
        self.RES_LAYER_RHO[y,x,:] = local_restart.LAYER_RHO
        self.RES_LAYER_T[y,x,:] = local_restart.LAYER_T
        self.RES_LAYER_LWC[y,x,:] = local_restart.LAYER_LWC
        self.RES_LAYER_IF[y,x,:] = local_restart.LAYER_IF


    def write_restart_to_file(self):
        """Add global numpy arrays to the RESTART dataset."""
        self.add_variable_along_latlon(self.RESTART, self.RES_NLAYERS, 'NLAYERS', '-', 'Number of layers')
        self.add_variable_along_latlon(self.RESTART, self.RES_NEWSNOWHEIGHT, 'new_snow_height', 'm .w.e', 'New snow height')
        self.add_variable_along_latlon(self.RESTART, self.RES_NEWSNOWTIMESTAMP, 'new_snow_timestamp', 's', 'New snow timestamp')
        self.add_variable_along_latlon(self.RESTART, self.RES_OLDSNOWTIMESTAMP, 'old_snow_timestamp', 's', 'Old snow timestamp')
        self.add_variable_along_latlonlayer(self.RESTART, self.RES_LAYER_HEIGHT, 'LAYER_HEIGHT', 'm', 'Layer height')
        self.add_variable_along_latlonlayer(self.RESTART, self.RES_LAYER_RHO, 'LAYER_RHO', 'kg m^-3', 'Layer density')
        self.add_variable_along_latlonlayer(self.RESTART, self.RES_LAYER_T, 'LAYER_T', 'K', 'Layer temperature')
        self.add_variable_along_latlonlayer(self.RESTART, self.RES_LAYER_LWC, 'LAYER_LWC', '-', 'Layer liquid water content')
        self.add_variable_along_latlonlayer(self.RESTART, self.RES_LAYER_IF, 'LAYER_IF', '-', 'Layer ice fraction')


    # TODO: Make it Pythonian - Finish the getter/setter functions
    @property
    def RAIN(self):
        return self.__RAIN
    @property
    def SNOWFALL(self):
        return self.__SNOWFALL
    @property
    def LWin(self):
        return self.__LWin
    @property
    def LWout(self):
        return self.__LWout
    @property
    def H(self):
        return self.__H
    @property
    def LE(self):
        return self.__LE
    @property
    def B(self):
        return self.__B
    @property
    def QRR(self):
        return self.__QRR
    @property
    def MB(self):
        return self.__MB
    
    
    @RAIN.setter
    def RAIN(self, x):
        self.__RAIN = x
    @SNOWFALL.setter
    def SNOWFALL(self, x):
        self.__SNOWFALL = x
    @LWin.setter
    def LWin(self, x):
        self.__LWin = x
    @LWout.setter
    def LWout(self, x):
        self.__LWout = x
    @H.setter
    def H(self, x):
        self.__H = x
    @LE.setter
    def LE(self, x):
        self.__LE = x
    @B.setter
    def B(self, x):
        self.__B = x
    @QRR.setter
    def QRR(self, x):
        self.__QRR = x
    @MB.setter
    def MB(self, x):
        self.__MB = x


    def get_result(self) -> xr.Dataset:
        """Get the RESULT data structure."""
        return self.RESULT

    def get_restart(self) -> xr.Dataset:
        """Get the RESTART data structure."""
        return self.RESTART

    def get_grid_restart(self) -> xr.Dataset:
        """Get the GRID_RESTART data structure."""
        return self.GRID_RESTART

    # ==================================================================
    # Auxiliary functions for writing variables to NetCDF files
    # ==================================================================
    def add_variable_along_scalar(self, ds, var, name, units, long_name):
        """Add scalar data to a dataset.

        Args:
            ds (xr.Dataset): Target data structure.
            var (np.ndarray): New data.
            name (str): The new variable's abbreviated name.
            units (str): New variable units.
            long_name (str): The new variable's full name.

        Returns:
            xr.Dataset: Target dataset with the new scalar variable.
        """
        ds[name] = var.data
        ds[name].attrs['units'] = units
        ds[name].attrs['long_name'] = long_name
        ds[name].encoding['_FillValue'] = -9999
        return ds

    def add_variable_along_latlon(self, ds, var, name, units, long_name):
        """Add spatial data to a dataset.

        Args:
            ds (xr.Dataset): Target data structure.
            var (np.ndarray): New spatial data.
            name (str): The new variable's abbreviated name.
            units (str): New variable units.
            long_name (str): The new variable's full name.

        Returns:
            xr.Dataset: Target dataset with the new spatial variable.
        """
        ds[name] = ((Config.northing,Config.easting), var.data)
        ds[name].attrs['units'] = units
        ds[name].attrs['long_name'] = long_name
        ds[name].encoding['_FillValue'] = -9999
        return ds
    
    def add_variable_along_time(self, ds, var, name, units, long_name):
        """Add temporal data to a dataset.

        Args:
            ds (xr.Dataset): Target data structure.
            var (np.ndarray): New temporal data.
            name (str): The new variable's abbreviated name.
            units (str): New variable units.
            long_name (str): The new variable's full name.

        Returns:
            xr.Dataset: Target dataset with the new temporal variable.
        """
        ds[name] = xr.DataArray(var.data, coords=[('time', ds.time)])
        ds[name].attrs['units'] = units
        ds[name].attrs['long_name'] = long_name
        ds[name].encoding['_FillValue'] = -9999
        return ds
    
    def add_variable_along_latlontime(self, ds, var, name, units, long_name):
        """Add spatiotemporal data to a dataset.

        Args:
            ds (xr.Dataset): Target data structure.
            var (np.ndarray): New spatiotemporal data.
            name (str): The new variable's abbreviated name.
            units (str): New variable units.
            long_name (str): The new variable's full name.

        Returns:
            xr.Dataset: Target dataset with the new spatiotemporal
            variable.
        """
        ds[name] = (('time',Config.northing,Config.easting), var.data)
        ds[name].attrs['units'] = units
        ds[name].attrs['long_name'] = long_name
        ds[name].encoding['_FillValue'] = -9999
        return ds
    
    def add_variable_along_latlonlayertime(self, ds, var, name, units, long_name):
        """Add a spatiotemporal mesh to a dataset.

        Args:
            ds (xr.Dataset): Target data structure.
            var (np.ndarray): New spatiotemporal mesh data.
            name (str): The new variable's abbreviated name.
            units (str): New variable units.
            long_name (str): The new variable's full name.

        Returns:
            xr.Dataset: Target dataset with the new spatiotemporal mesh.
        """
        ds[name] = (('time',Config.northing,Config.easting,'layer'), var.data)
        ds[name].attrs['units'] = units
        ds[name].attrs['long_name'] = long_name
        ds[name].encoding['_FillValue'] = -9999
        return ds
    
    def add_variable_along_latlonlayer(self, ds, var, name, units, long_name):
        """Add a spatial mesh to a dataset.

        Args:
            ds (xr.Dataset): Target data structure.
            var (np.ndarray): New spatial mesh.
            name (str): The new variable's abbreviated name.
            units (str): New variable units.
            long_name (str): The new variable's full name.

        Returns:
            xr.Dataset: Target dataset with the new spatial mesh.
        """
        ds[name] = ((Config.northing,Config.easting,'layer'), var.data)
        ds[name].attrs['units'] = units
        ds[name].attrs['long_name'] = long_name
        ds[name].encoding['_FillValue'] = -9999
        return ds
    
    def add_variable_along_layertime(self, ds, var, name, units, long_name):
        """Add temporal layer data to a dataset.

        Args:
            ds (xr.Dataset): Target data structure.
            var (np.ndarray): New layer data with a time coordinate.
            name (str): The new variable's abbreviated name.
            units (str): New variable units.
            long_name (str): The new variable's full name.

        Returns:
            xr.Dataset: Target dataset with the new layer data.
        """
        ds[name] = (('time','layer'), var.data)
        ds[name].attrs['units'] = units
        ds[name].attrs['long_name'] = long_name
        ds[name].encoding['_FillValue'] = -9999
        return ds
    
    def add_variable_along_layer(self, ds, var, name, units, long_name):
        """Add layer data to a dataset.

        Args:
            ds (xr.Dataset): Target data structure.
            var (np.ndarray): New layer data.
            name (str): The new variable's abbreviated name.
            units (str): New variable units.
            long_name (str): The new variable's full name.

        Returns:
            xr.Dataset: Target dataset with the new layer data.
        """
        ds[name] = ('layer', var.data)
        ds[name].attrs['units'] = units
        ds[name].attrs['long_name'] = long_name
        ds[name].encoding['_FillValue'] = -9999
        return ds
