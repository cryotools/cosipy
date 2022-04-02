"""
 This is the configuration (init) file for the utility aws2cosipy.
 Please make your changes here.
"""

#------------------------
# Declare variable names 
#------------------------

# Pressure
# PRES_var = 'PRES'
PRES_var = 'press'

# Temperature
# T2_var = 'T2'
T2_var = 'temp'
in_K = False

# Relative humidity
# RH2_var = 'RH2'
RH2_var = 'RH'

# Incoming shortwave radiation
# G_var = 'G'
G_var = 'SW_global'

# Precipitation
# RRR_var = 'RRR'
RRR_var = 'ppt'

# Wind velocity
# U2_var = 'U2'
U2_var = 'wind'

# Incoming longwave radiation
# LWin_var = 'LWinCor_Avg'
LWin_var = 'LW_in'

# Snowfall
SNOWFALL_var = 'SNOWFALL'

# Cloud cover fraction
N_var = 'N'

# Fountain discharge
DISCHARGE_var = 'Discharge'

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
radiationModule = 'Wohlfahrt2016' # 'Moelg2009', 'Wohlfahrt2016', 'none'
LUT = False                   # If there is already a Look-up-table for topographic shading and sky-view-factor built for this area, set to True

dtstep = 3600*3               # time step (s)
stationLat = -54.4            # Latitude of station
timezone_lon = 90.0	      # Longitude of station
# tcart = 26                    # Station time correction in hour angle units (1 is 4 min)

# Zenit threshold (>threshold == zenit): maximum potential solar zenith angle during the whole year, specific for each location
zeni_thld = 89.0              # If you do not know the exact value for your location, set value to 89.0

#------------------------
# Point model 
#------------------------
point_model = True
#guttannen21
plon = 8.29
plat = 46.66
hgt = 1047.6
radf = 6.9
name = 'guttannen21'

#gangles21
plon = 77.606949
plat = 34.216638
hgt = 4009
radf = 10.2
name = 'gangles21'

# plon = -54.4
# plat = 90.0
# hgt = 5665.0

#------------------------
# Interpolation arguments 
#------------------------
# stationName = 'guttannen21'
# stationAlt = 1047.6
stationName = 'gangles21'
stationAlt = 4009
# stationName = 'Zhadang'
# stationAlt = 5665.0

lapse_T         = -0.006    # Temp K per  m
lapse_RH        =  0.000    # RH % per  m (0 to 1)
lapse_RRR       =  0.0000   # mm per m
lapse_SNOWFALL  =  0.0000   # Snowfall % per m (0 to 1)
