"""
 This is the configuration (init) file for the utility cs2posipy.
 Please make your changes here.
"""

#------------------------
# Declare variable names 
#------------------------

# Temperature
T_var = 'AirTC_Avg'  

# Relative humidity
RH_var = 'RH'   

# Wind velocity
U_var = 'WS_ms_S_WVT'     

# Precipitation
RRR_var = 'Rain_mm_Tot' 

# Incoming shortwave radiation
G_var = 'SWin_Avg'   

# Incoming longwave radiation
LW_var = 'LWin_Avg'   

# Pressure
P_var = 'Press_Avg'   


#------------------------
# Radiation module 
#------------------------
radiationModule = True 

# Time zone
timezone_lon = 15.0

# Zenit threshold (>threshold == zenit)
zeni_thld = 85.0

#------------------------
# Interpolation arguments 
#------------------------
stationName = 'Hintereisferner'
stationAlt = 2880.0

lapse_T    = -0.006  # Temp K per  m
lapse_RH   = 0.001  # RH % per  m (0 to 1)
lapse_RRR  = 0.001   # RRR % per m (0 to 1)
