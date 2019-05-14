"""
 This is the COSIPY configuration (init) file.
 Please make your changes here.
"""
## Simulation period
time_start = '2014-01-14T00:00'
time_end   = '2014-01-16T00:00'
#time_start = '2009-01-16T00:00'
#time_end   = '2009-01-17T00:00'

time_start_str=(time_start[0:10]).replace('-','')
time_end_str=(time_end[0:10]).replace('-','')

data_path = './data'
input_netcdf= 'Greenland.nc'
#input_netcdf= 'Zhadang_ERA5_2009.nc'
output_netcdf = 'Greenland_'+time_start_str+'-'+time_end_str+'.nc'
#output_netcdf = 'Zhandang_'+time_start_str+'-'+time_end_str+'.nc'

## Set keyword to true if you want to use the job scheduler Slurm (own configuration file slurm_config.py)
slurm_use = False 

## Port for local cluster
local_port = 8786

# Tile
tile = True
xstart = 0
xend = 1 
ystart = 0
yend = 1

## Remeshing (log_profile, adaptive_profile)
remesh_method = 'log_profile'

## Write full fields
full_field = True 

## Restart
restart = False

## If total precipitation and snowfall in input data use total precipitation!
force_use_TP = False

## Time step in the input files [s]
dt = 3600                                           # 3600, 7200, 10800 [s] length of time step per iteration in seconds

## Properties for debug
debug_level = 0                                     # DEBUG levels: 0, 10, 20, 30

## Merging level
# How many mergings and splittings are allowed per time step
merging = True
merge_max = 1          

density_threshold_merging = 20                      # If merging is true threshold for layer densities difference two layer
                                                    # try: 5-10 (kg m^-3)
temperature_threshold_merging = 0.1                 # If mering is true threshold for layer temperatures to merge
                                                    # try: 0.05-0.1 (K)

## Max. number of layers, just for the restart file
max_layers = 200

## CFL criteria
c_stab = 0.3

# Configuration if worker for local cluster (not slurm) Number of workers, if None all cores are used
workers = 1

