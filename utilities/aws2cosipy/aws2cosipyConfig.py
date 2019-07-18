"""
 This is the configuration (init) file for the utility cs2posipy.
 Please make your changes here.
"""

#------------------------
# Declare variable names 
#------------------------

# Pressure
PRES_var = 'Press_Avg'   

# Temperature
T2_var = 'Tair_Avg'  
in_K = False

# Cloud cover fraction
N_var = 'tcc'

# Relative humidity
RH2_var = 'Hum_Avg'   

# Incoming shortwave radiation
G_var = 'SWin_Avg'   

# Precipitation
RRR_var = 'Rain_Tot' 

# Wind velocity
U2_var = 'Wspeed'     

# Incoming longwave radiation
LWin_var = 'LWinCor_Avg'

# Snowfall
SNOWFALL_var = 'SNOWFALL'

#------------------------
# Radiation module 
#------------------------
radiationModule = True 

# Time zone
timezone_lon = 15.0

# Zenit threshold (>threshold == zenit)
zeni_thld = 85.0

#------------------------
# Point model 
#------------------------
point_model = True
plon = 10.74643
plat = 46.79241

#------------------------
# Interpolation arguments 
#------------------------
stationName = 'Hintereisferner'
stationAlt = 2880.0

lapse_T         = -0.006  # Temp K per  m
lapse_RH        = 0.001  # RH % per  m (0 to 1)
lapse_RRR       = 0.001   # RRR % per m (0 to 1)
lapse_SNOWFALL  = 0.001   # Snowfall % per m (0 to 1)
