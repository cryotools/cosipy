"""
 This is the configuration (init) file for the utility aws2cosipy.
 Please make your changes here.
"""

#------------------------
# Declare variable names 
#------------------------

# Pressure
PRES_var = 'PRES'

# Temperature
T2_var = 'T2'
in_K = True

# Relative humidity
RH2_var = 'RH2'

# Incoming shortwave radiation
G_var = 'G'

# Precipitation
RRR_var = 'RRR'

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
aggregate = True
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
timezone_lon = 90.0

# Zenit threshold (>threshold == zenit): maximum potential solar zenith angle during the whole year, specific for each location
zeni_thld = 85.0            # I you do not know the exact value for your location, set value to 89.0

#------------------------
# Point model 
#------------------------
point_model = False
plon = 90.64
plat = 30.47
hgt = 5665.0

#------------------------
# Interpolation arguments 
#------------------------
stationName = 'Zhadang'
stationAlt = 5665.0

lapse_T         = -0.006    # Temp K per  m
lapse_RH        =  0.000    # RH % per  m (0 to 1)
lapse_RRR       =  0.0000   # mm per m
lapse_SNOWFALL  =  0.0000   # Snowfall % per m (0 to 1)
