"""
 This is the configuration (init) file for the utility aws2cosipy_2aws.
 Please make your changes here.
"""

#------------------------
# Declare variable names 
#------------------------

# Pressure
PRES_var = 'PRES'

# Temperature
T2_var = 'T2'
in_K = False

# Relative humidity
RH2_var = 'RH2'

# Incoming shortwave radiation
G_var = 'G'

# Precipitation
RRR_var = 'RRR_BEC_PON'

# Wind velocity
U2_var = 'U2'

# Incoming longwave radiation
LWin_var = 'LWinCor_Avg'

# Snowfall
SNOWFALL_var = 'SNOWFALL'

# Cloud cover fraction
N_var = 'N'

#------------------------
# Aggregation to hourly data
#------------------------
aggregate = False
aggregation_step = 'H'

# Delimiter in csv file
delimiter = ','

# WRF non uniform grid
WRF = False

#------------------------
# Radiation module 
#------------------------
radiationModule = True 

# Time zone
timezone_lon = 10.0

# Zenit threshold (>threshold == zenit): maximum potential solar zenith angle during the whole year, specific for each location
zeni_thld = 85.0            # If you do not know the exact value for your location, set value to 89.0

#------------------------
# Point model 
#------------------------
point_model = False
plon = 9.92935966099034
plat = 46.4054432842672
hgt = 2441

#------------------------
# Interpolation arguments 
#------------------------
stationName_valley = 'SAM'
stationAlt_v = 1708.0
# long = 9°53'
# lat = 46°32'

stationName_mountain = 'COV'
stationAlt_m = 3302.0
# long = 9°49'
# lat = 46°25'

# which weather station should be used as initial value
intialValley = True
 
stationNames_RRR = 'PON_BEC'	# Since precipitation from two other weather stations is used, the altitude of these weather stations has to be set.
stationAlt_RRR_mean = 1899.0	# If you use the same weather stations: use mean altitude of 'station_valley' and 'station_mountain'
# AltBEC = 2090
# AltPON = 1708

# lapes_T, lapse_RH and U_mean are defined hourly out of the data of the 2 aws

U2_constant = True
U2_const = 3		    # constant value for the wind velocity

T2_weighted = True          # to weight the temperature, change the function in "aws2cosipy_2aws.py
T_const = 0.0		    # constant temperature (K) to add to all temperature values

lapse_RRR = (0.4 / 8760)    # default RRR gradient from Schwarb(2000) [mm/(m*h)]
RRR_additional_in_percentage = 0.0

lapse_SNOWFALL  = 0.0000    # Snowfall % per m (0 to 1)
