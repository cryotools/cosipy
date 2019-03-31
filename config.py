"""
 This is the COSIPY configuration (init) file.
 Please make your changes here.
"""
## Simulation period
time_start = '2009-05-01T00:00'
time_end   = '2010-04-30T00:00'

time_start_str=(time_start[0:10]).replace('-','')
time_end_str=(time_end[0:10]).replace('-','')

data_path = './data'
input_netcdf= 'Zhadang_ERA5_2009-2010.nc'
#input_netcdf= 'HEF_input.nc'
output_netcdf = 'Zhadang_ERA5_'+time_start_str+'-'+time_end_str+'.nc'

## Set keyword to true if you want to use the job scheduler Slurm (own configuration file slurm_config.py)
slurm_use = False

## Port for local cluster
local_port = 8786

## Write full fields
full_field = False

## Restart
restart = False

## If total precipitation and snowfall in input data use total precipitation!
force_use_TP = False

## Time step in the input files [s]
dt = 3600                                           # 3600, 7200, 10800 [s] length of time step per iteration in seconds

## Properties for debug
debug_level = 0                                     # DEBUG levels: 0, 10, 20, 30

## Merging level
merging = True
density_threshold_merging = 20                      # If merging is true threshold for layer densities difference two layer
                                                    # try: 5-10 (kg m^-3)
temperature_threshold_merging = 0.1                 # If mering is true threshold for layer temperatures to merge
                                                    # try: 0.05-0.1 (K)

# How many mergings and splittings are allowed per time step
merge_max = 2          
split_max = 2 

## Max. number of layers, just for the restart file
max_layers = 200

## Max. height of snow layers
max_snow_layer_height = 0.5
max_glacier_layer_height = 0.8

## CFL criteria
c_stab = 0.3

# Configuration if worker for local cluster (not slurm) Number of workers, if None all cores are used
workers = None
