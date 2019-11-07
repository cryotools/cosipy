"""
 This is the COSIPY configuration (init) file.
 Please make your changes here.
"""

#-----------------------------------
# SIMULATION PERIOD 
#-----------------------------------
# HEF
#time_start = '2018-09-17T08:00'
#time_end   = '2019-05-01T15:00'

# Greenland
#time_start = '2015-07-01T00:00'
#time_end   = '2015-07-05T00:00'
time_start = '2015-07-05T00:00'
time_end   = '2015-07-10T00:00'

#-----------------------------------
# FILENAMES AND PATHS 
#-----------------------------------
time_start_str=(time_start[0:10]).replace('-','')
time_end_str=(time_end[0:10]).replace('-','')

data_path = './data/'                       
#input_netcdf= 'HEF/HEF_input.nc'
#output_netcdf = 'HEF_'+time_start_str+'-'+time_end_str+'.nc'

input_netcdf= 'Greenland/wrf2cosipy_input.nc'
output_netcdf = 'Greenland_'+time_start_str+'-'+time_end_str+'.nc'

#-----------------------------------
# RESTART 
#-----------------------------------
restart = True             # set to true if you want to start from restart file 

#-----------------------------------
# STAKE DATA 
#-----------------------------------
stakes_loc_file = './data/input/HEF/loc_stakes.csv'        # path to stake location file
stakes_data_file = './data/input/HEF/data_stakes_hef.csv' # path to stake data file
eval_method = 'rmse'                                        # how to evaluate the simulations ('rmse')
obs_type = 'snowheight'                                    # What kind of stake data is used 'mb' or 'snowheight'

#-----------------------------------
# PARALLELIZATION 
#-----------------------------------
slurm_use = True            # use SLURM
workers = None              # number of workers, if local cluster is used
local_port = 8786           # port for local cluster

#-----------------------------------
# WRITE FULL FIELDS 
#-----------------------------------    
full_field = False          # write full fields (2D data) to file 
    
#-----------------------------------
# TOTAL PRECIPITATION  
#-----------------------------------
force_use_TP = False        # If total precipitation and snowfall in input data use total precipitation

#-----------------------------------
# CLOUD COVER FRACTION  
#-----------------------------------
force_use_N = False        # If cloud cover fraction and incoming longwave radiation in input data use cloud cover fraction

#-----------------------------------
# SUBSET  (provide pixel values) 
#-----------------------------------
tile = True 
xstart = 20
xend = 40
ystart = 20
yend = 40

