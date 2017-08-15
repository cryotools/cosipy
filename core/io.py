"""
 This file reads in the input data (model forcing)
"""

import scipy.io as sio
from config import mat_path, nc_path

' Load climatic forcing (variables from Matlab file) '

DATA = sio.loadmat(mat_path)

wind_speed = DATA['u2']             # Wind speed (magnitude) m/s
solar_radiation = DATA['G']         # Solar radiation at each time step [W m-2]
temperature_2m = DATA['T2']         # Air temperature (2m over ground) [K]
relative_humidity = DATA['rH2']     # Relative humidity (2m over ground)[%]
snowfall = DATA['snowfall']         # Snowfall per time step [m]
air_pressure = DATA['p']            # Air Pressure [hPa]
cloud_cover = DATA['N']             # Cloud cover  [fraction][%/100]
initial_snow_height = DATA['sh']    # Initial snow height [m]

# todo Load climatic forcing (variables from NetCDF file)
# from netCDF4 import Dataset
# from config import nc_path


####def read
###check if 1 D or 2D

####def write