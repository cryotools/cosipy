"""
 This is the configuration (init) file for the utility cs2posipy.
 Please make your changes here.
"""

#------------------------
# Declare variable names 
#------------------------

# Temperature
T2_var = 'T2'

# Relative humidity
RH2_var = 'RH2'

# Wind velocity
U2_var = 'U2'

# Incoming shortwave radiation
G_var = 'G'

# Pressure
PRES_var = 'PRES'

# Precipitation
RRR_var = 'RRR'

# Cloud cover fraction
N_var = 'N'

# Snowfall
SNOWFALL_var = 'SNOWFALL'

#------------------------
# Radiation module 
#------------------------
radiationModule = True 

# Time zone
timezone_lon = 90.0

# Zenit threshold (>threshold == zenit)
zeni_thld = 86.0

#------------------------
# Interpolation arguments 
#------------------------
stationName = 'Zhadang'
stationAlt = 5665

lapse_T    = -0.006     # Temp K per  m
lapse_RH   = 0.002      # RH % per  m (0 to 1)
lapse_RRR  = 0.0001     # RRR % per m (0 to 1)
