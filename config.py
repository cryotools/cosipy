"""
 This is the COSIPY configuration (init) file.
 Please make your changes here.
"""

''' REQUIRED INPUT '''

mat_path = 'input/input_COSIPY-example.mat'   # PATH and filename of model forcing/input file
                                        # see 'inputData.py' file
                                        #  todo should be netCDF (in progress)

nc_path = 'input/input_COSIPY-example.nc'

time_start = 0                      # time index to start
time_end = 3900                     # len(T2) usually the length of the time series
dt = 3600                     # 3600, 7200, 10800 [s] length of time step per iteration in seconds

debug_level = 0                  # DEBUG levels: 0, 10, 20, 30

merging_level = 0                # Merge layers with similar properties:
                                    # 0 = False
                                    # 1 = 5. [kg m^-3] and 0.05 [K]
                                    # 2 = 10. and 0.1

merge_snow_threshold = 0.02    # Minimal height of layer [m]:
                                # thin layers rise computational needs
                                # of upwind schemes (e.g. heatEquation)

''' required variables '''

temperature_bottom = 268                        # bottom temperature [K]

c_stab = 0.3                    # cfl criteria


#### strings with names for parametrisations