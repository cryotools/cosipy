"""
 This is the COSIPY configuration (init) file.
 Please make your changes here.
"""
from pathlib import Path

''' REQUIRED INPUT '''
home = str(Path.home())
folder='/Seafile/diss/io/'

# example Halji 2D Setup
input_example_2D='input_COSIPY2D-mask_values-example.nc'
output_example_2D= 'output_COSIPY2D-mask_values-example.nc'

# example 2D Setup
input_example_1D = 'input_COSIPY-example.nc'
output_example_1D = 'output_COSIPY-example.nc'

# example 1D Setup
input_Halji = 'input_prepro_HAR_Halji_10_2000-10_2011_ohneKorrekturen.nc'
output_Halji = 'output_HAR_Halji_10_2000-10_2011_ohneKorrekturen.nc'

# change last example for different setups!
input_netcdf=home+folder+input_example_1D
output_netcdf=home+folder+output_example_1D

time_start = 0                                      # time index to start
time_end = 7200                                     # len(T2) usually the length of the time series
dt = 3600                                           # 3600, 7200, 10800 [s] length of time step per iteration in seconds

debug_level = 0                                     # DEBUG levels: 0, 10, 20, 30

merging_level = 0                                   # Merge layers with similar properties:
                                                    # 0 = False
                                                    # 1 = 5. [kg m^-3] and 0.05 [K]
                                                    # 2 = 10. and 0.1

merge_snow_threshold = 0.02                         # Minimal height of layer [m]:
                                                    # thin layers rise computational needs
                                                    # of upwind schemes (e.g. heatEquation)

''' required variables '''

temperature_bottom = 268                            # bottom temperature [K]

c_stab = 0.3                                        # cfl criteria


                                                    # ToDo strings with names for parametrisations