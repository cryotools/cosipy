"""
 This is the COSIPY configuration (init) file.
 Please make your changes here.
"""

''' REQUIRED INPUT '''
input_netcdf = '/home/anselm/Seafile/Diss-2/input_data/study_regions-aufbereitet/Halji/prepro_HAR_Halji_10_2000-10_2011_ohneKorrekturen.nc'      # example netcdf 1D input file
mat_path = 'input/input_COSIPY-example.mat'         # same input file as matlab file; obsolete?
output_netcdf = 'output/output_example-1D.nc'       # example output file; in future user have to define external path?

time_start = 0                                      # time index to start
time_end = 96360                                    # len(T2) usually the length of the time series
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