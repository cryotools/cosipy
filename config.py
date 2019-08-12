"""
 This is the COSIPY configuration (init) file.
 Please make your changes here.
"""
## Simulation period
time_start = '2018-09-17T08:00'
time_end   = '2019-06-06T12:00'

time_start_str=(time_start[0:10]).replace('-','')
time_end_str=(time_end[0:10]).replace('-','')

data_path = './data/'
input_netcdf= 'hef_pit01.nc'
output_netcdf = 'hef_pit01_adapt_'+time_start_str+'-'+time_end_str+'.nc'

## Set keyword to true if you want to use the job scheduler Slurm (own configuration file slurm_config.py)
slurm_use = False

## Port for local cluster
local_port = 8786

# Tile
tile = False
xstart = 0
xend = 20 
ystart = 0
yend = 20

## Remeshing (log_profile or adaptive_profile)
remesh_method = 'log_profile'

## Write full fields
full_field = True 

## Restart, set to true if you want to start from restart file
restart = False

## If total precipitation and snowfall in input data use total precipitation!
force_use_TP = True

## Time step in the input files [s]
dt = 3600                                           # 3600, 7200, 10800 [s] length of time step per iteration in seconds

# How many mergings are allowed per time step
merge_max = 5

density_threshold_merging = 20                      # If merging is true threshold for layer densities difference two layer
                                                    # try: 5-10 (kg m^-3)
temperature_threshold_merging = 0.1                 # If mering is true threshold for layer temperatures to merge
                                                    # try: 0.05-0.1 (K)

## Max. number of layers, just for the restart file
max_layers = 200

# Configuration if worker for local cluster (not slurm) Number of workers, if None all cores are used
workers = 1#None
