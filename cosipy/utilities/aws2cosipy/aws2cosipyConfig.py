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
tcart = 26                    # Station time correction in hour angle units (1 is 4 min)
timezone_lon = 90.0	      # Longitude of station

# Zenit threshold (>threshold == zenit): maximum potential solar zenith angle during the whole year, specific for each location
zeni_thld = 85.0              # If you do not know the exact value for your location, set value to 89.0

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
