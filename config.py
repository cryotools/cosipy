"""
 This is the COSIPY configuration (init) file.
 Please make your changes here.
"""

#-----------------------------------
# SIMULATION PERIOD 
#-----------------------------------
# Zhadang
# time_start = '2009-01-01T06:00'
# time_end   = '2009-01-10T00:00'

all_icestupas = ['guttannen22_scheduled', 'guttannen22_unscheduled', 'guttannen21', 'gangles21']
icestupa_name = 'guttannen22_unscheduled'
# icestupa_name = 'guttannen21'
# icestupa_name = 'guttannen20'
# icestupa_name = 'gangles21'

if icestupa_name in ['guttannen20', 'guttannen21', 'guttannen22_scheduled', 'guttannen22_unscheduled']:
    plon = 8.29
    plat = 46.66
    hgt = 1047.6
    cld = 0.5  # Cloudiness factor
    stationName = icestupa_name
    stationAlt = hgt
    stationLat = plat
    timezone_lon = plon

if icestupa_name in ['gangles21']:
    plon = 77.606949
    plat = 34.216638
    hgt = 4009
    cld = 0.1  # Cloudiness factor
    stationName = icestupa_name
    stationAlt = hgt
    stationLat = plat
    timezone_lon = plon

if icestupa_name == 'guttannen20':
    drone_evaluation = True
    thermistor_evaluation = False
    thermalcam_evaluation = False
    time_start = '2020-01-03T16:00'
    time_end   = '2020-04-06T12:00'
    radf = 7.67                                       # Spray radius [m]

if icestupa_name in ['guttannen22_scheduled']:
    thermistor_evaluation = False
    drone_evaluation = True
    time_start = '2021-12-03T12:00'
    time_end   = '2022-04-12T00:00'
    radf = 4.845                                       # Spray radius [m]

if icestupa_name in ['guttannen22_unscheduled']:
    drone_evaluation = True
    thermistor_evaluation = False
    time_start = '2021-12-03T12:00'
    time_end   = '2022-04-12T00:00'
    radf = 4.072                                       # Spray radius [m]

if icestupa_name == 'guttannen21':
    drone_evaluation = True
    thermistor_evaluation = False
    thermalcam_evaluation = False
    time_start = '2020-11-22T15:00'
    time_end   = '2021-05-10T01:00'
    # time_end   = '2021-03-10T01:00'

    radf = 6.913                                       # Spray radius [m]

if icestupa_name == 'gangles21':
    drone_evaluation = True
    thermistor_evaluation = False
    time_start = '2021-01-18'
    time_end   = '2021-06-20'
    # time_end   = '2021-04-10'

    radf = 10.22                                       # Spray radius [m]


# Hintereisferner
#time_start = '2018-09-17T08:00'
#time_end   = '2019-07-03T13:00'

#-----------------------------------
# FILENAMES AND PATHS 
#-----------------------------------
# time_start_str=(time_start[0:10]).replace('-','')
# time_end_str=(time_end[0:10]).replace('-','')

data_path = './data/'

# Zhadang example
# input_netcdf= 'Zhadang/Zhadang_ERA5_2009.nc'
# output_netcdf = 'Zhadang_ERA5_'+time_start_str+'-'+time_end_str+'.nc'

input_netcdf= icestupa_name + '/input.nc'
output_netcdf = icestupa_name + '.nc'

# Hintereisferner example
#input_netcdf = 'HEF/HEF_input.nc'
#output_netcdf = 'hef.nc'

#-----------------------------------
# RESTART 
#-----------------------------------
restart = False                                             # set to true if you want to start from restart file

#-----------------------------------
# STAKE DATA 
#-----------------------------------
# stakes_loc_file = './data/input/HEF/loc_stakes.csv'         # path to stake location file
# stakes_data_file = './data/input/HEF/data_stakes_hef.csv'   # path to stake data file
# eval_method = 'rmse'                                        # how to evaluate the simulations ('rmse')
# obs_type = 'snowheight'                                     # What kind of stake data is used 'mb' or 'snowheight'

stake_evaluation = False
thermalcam_evaluation = False
eval_method = 'rmse'                                        # how to evaluate the simulations ('rmse')

if drone_evaluation:
    observations_data_file = './data/input/' + icestupa_name + '/drone.csv'   # path to stake data file
    obs_type = 'volume'

if thermistor_evaluation:
    observations_data_file = './data/input/' + icestupa_name + '/thermistor.csv'   # path to stake data file
    obs_type = 'bulkTemp'

if thermalcam_evaluation:
    observations_data_file = './data/input/' + icestupa_name + '/thermalcam.csv'   # path to stake data file
    obs_type = 'surfTemp'

#-----------------------------------
# STANDARD LAT/LON or WRF INPUT 
#-----------------------------------
# Dimensions
WRF = False                                                 # Set to True if you use WRF as input

northing = 'lat'	                                    # name of dimension	in in- and -output
easting = 'lon'					                        # name of dimension in in- and -output
if WRF:
    northing = 'south_north'                                # name of dimension in WRF in- and output
    easting = 'west_east'                                   # name of dimension in WRF in- and output

# Interactive simulation with WRF
WRF_X_CSPY = False

#-----------------------------------
# COMPRESSION of output netCDF
#-----------------------------------
compression_level = 2                                       # Choose value between 1 and 9 (highest compression)
                                                            # Recommendation: choose 1, 2 or 3 (higher not worthwhile, because of needed time for writing output)
#-----------------------------------
# PARALLELIZATION 
#-----------------------------------
slurm_use = False                                           # use SLURM
workers = None                                              # number of workers, if local cluster is used
local_port = 8786                                           # port for local cluster

#-----------------------------------
# WRITE FULL FIELDS 
#-----------------------------------    
full_field = False                                          # write full fields (2D data) to file
if WRF_X_CSPY:
    full_field = True
    
#-----------------------------------
# TOTAL PRECIPITATION  
#-----------------------------------
force_use_TP = False                                        # If total precipitation and snowfall in input data;
                                                            # use total precipitation

#-----------------------------------
# CLOUD COVER FRACTION  
#-----------------------------------
force_use_N = False                                         # If cloud cover fraction and incoming longwave radiation
                                                            # in input data use cloud cover fraction

#-----------------------------------
# SUBSET  (provide pixel values) 
#-----------------------------------
tile = False
xstart = 20
xend = 40
ystart = 20
yend = 40
