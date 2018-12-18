"""
 This is the configuration (init) file for the utility cs2posipy.
 Please make your changes here.
"""

#------------------------
# Declare variable names 
#------------------------

# Temperature
T_var = 'T2'  

# Relative humidity
RH_var = 'RH2'   

# Wind velocity
U_var = 'U2'     

# Precipitation
RRR_var = 'RRR' 

# Incoming shortwave radiation
G_var = 'G'   

# Incoming longwave radiation
LW_var = ''   

# Pressure
P_var = 'PRES'   

# Cloud cover fraction
N_var = 'N'   

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
stationName = 'Argog'
stationAlt = 5263

lapse_T    = -0.006  # Temp K per  m
lapse_RH   = 0.001  # RH % per  m (0 to 1)
lapse_RRR  = 0.001   # RRR % per m (0 to 1)
