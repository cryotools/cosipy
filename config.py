"""
 This is the COSIPY configuration (init) file.
 Please make your changes here.
"""
dt = 3600                                           # 3600, 7200, 10800 [s] length of time step per iteration in seconds

debug_level = 0                                     # DEBUG levels: 0, 10, 20, 30

merging_level = 1                                   # Merge layers with similar properties:
                                                    # 0 = False
                                                    # 1 = 5. [kg m^-3] and 0.05 [K]
                                                    # 2 = 10. and 0.1

plots = 0                                           # if 1 create plot of output variables

merge_snow_threshold = 0.02                         # Minimal height of layer [m]:
                                                    # thin layers rise computational needs
                                                    # of upwind schemes (e.g. heatEquation)
''' required variables '''

temperature_bottom = 268                            # bottom temperature [K]

c_stab = 0.3                                        # cfl criteria; enlarge for time saving, shrink for stability

time_start = '2000-10-01'                              # input data needs time
time_end = '2000-10-02'

                                                    # ToDo strings with names for parametrisations

''' source code for repository '''
### use follwoing lines if you want to be asked for in- and output file path
# input_netcdf = input("Please type the full path to the netcdf input-file (with the .nc ending)!: ")
# output_netcdf = input("Please type the full path to the netcdf output-file (without the .nc ending)!: ")
### or set them static if they do not change
#input_netcdf=
#output_netcdf

## use for test!!! offer online?
#'''example 1D file'''
#input_netcdf= './input/data_amalia_2D.nc'
#output_netcdf = 'output/output_amalia.nc'

'''
    source code for local development - Anselm
    only used for local development
    clean for master branch!!!!!!!!!!
'''
from pathlib import Path
####local
home = str(Path.home())
input_folder=home+'/Seafile/work/io/Halji/input/'
output_folder =home+'/Seafile/work/io/Halji/output/'

####cluster
#input_folder='/data/scratch/arndtans/io/Halji/input/'
#output_folder='/data/scratch/arndtans/io/Halji/output/'

input_Halji = 'input_prepro_Halji-lat_lon_time_180709.nc'
output_Halji = 'output_Halji_'+str(time_start.replace("-",""))+'-'+str(time_end.replace("-",""))

input_netcdf=input_folder+input_Halji
output_netcdf=output_folder+output_Halji