"""
 This is the COSIPY configuration (init) file.
 Please make your changes here.
"""

## Set keyword to true if you want to use the job scheduler Slurm (own configuration file slurm_config.py)
slurm_use = False

## Simulation period
time_start = '2014-10-01T12:00'
time_end   = '2014-12-31T23:00'

##
setup_no= '1'

##  Input/Output files 
data_path = '/home/anz/Seafile/work/io/Argog-Co/'
input_netcdf= 'input_ArgogCo_1d.nc'
output_netcdf = 'Argog_1D_setup_'+setup_no+'.nc'

## Snowfall given in input file
snowheight_measurements = False # if snow exists in m

## Write full fields
full_field = False 

## Restart
restart = False 

## Time step in the input files [s]
dt = 3600                                           # 3600, 7200, 10800 [s] length of time step per iteration in seconds

## Debug level
debug_level = 0                                     # DEBUG levels: 0, 10, 20, 30

## Merging level
merging_level = 0                                   # Merge layers with similar properties:
                                                    # 0 = False
                                                    # 1 = <5. [kg m^-3] and <0.05 [K]
                                                    # 2 = <10. and <0.1

## Minimum height of layer [m]
merge_snow_threshold = 0.01    

## Max. number of layers, just for the restart file
max_layers = 100

## Lower boundary condition (temperature [K])
temperature_bottom = 268

## CFL criteria
c_stab = 0.5

## Number of workers, if None all cores are used
workers = None