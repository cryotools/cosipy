"""
 This is the COSIPY configuration (init) file.
 Please make your changes here.
"""

''' source code for repository '''
### use follwoing lines if you want to be asked for in- and output file path
# input_netcdf = input("Please type the full path to the netcdf input-file (with the .nc ending)!: ")
# output_netcdf = input("Please type the full path to the netcdf output-file (without the .nc ending)!: ")
### or set them static if they do not change
#input_netcdf=
#output_netcdf

dt = 3600                                           # 3600, 7200, 10800 [s] length of time step per iteration in seconds

debug_level = 0                                     # DEBUG levels: 0, 10, 20, 30

merging_level = 1                                   # Merge layers with similar properties:
                                                    # 0 = False
                                                    # 1 = 5. [kg m^-3] and 0.05 [K]
                                                    # 2 = 10. and 0.1

plots = 1                                           # if 1 create plot of output variables

merge_snow_threshold = 0.01                         # Minimal height of layer [m]:
                                                    # thin layers rise computational needs
                                                    # of upwind schemes (e.g. heatEquation)
''' required variables '''

temperature_bottom = 268                            # bottom temperature [K]

c_stab = 0.3                                        # cfl criteria

### next two lines only for development, in real cases controled by time length of input data
time_start = '2016-06-02'                                      # time index to start
time_end = '2016-06-02'                                       # len(T2) usually the length of the time series


                                                    # ToDo strings with names for parametrisations

## use for test!!!
'''example 1D file'''
input_netcdf= './input/data_amalia_2D.nc'
output_netcdf = 'output/output_amalia.nc'


