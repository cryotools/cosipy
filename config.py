"""
 This is the COSIPY configuration (init) file.
 Please make your changes here.
"""

## Set keyword to true if you want to use the job scheduler Slurm (own configuration file slurm_config.py)
slurm_use = False

## Simulation period
time_start = '2018-05-25T00:00'
time_end   = '2018-05-25T23:00'

##  Input/Output files 
data_path = './data'
input_netcdf= 'Hintereisferner_input.nc'
output_netcdf = 'Hintereisferner_output.nc'

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
temperature_bottom = 272                     

## CFL criteria
c_stab = 0.5                                        

## Number of workers, if None all cores are used
workers = None
