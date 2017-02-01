#!/usr/bin/env python

"""
 This file reads in the input data (model forcing)
"""

import scipy.io as sio
from config import mat_path, nc_path

''' Load climatic forcing (variables from Matlab file) '''

DATA = sio.loadmat(mat_path)

u2 = DATA['u2']                     # Wind speed (magnitude) m/s
G = DATA['G']                       # Solar radiation at each time step [W m-2]
T2 = DATA['T2']                     # Air temperature (2m over ground) [K]
rH2 = DATA['rH2']                   # Relative humidity (2m over ground)[%]
snowfall = DATA['snowfall']         # Snowfall per time step [m]
p = DATA['p']                       # Air Pressure [hPa]
N = DATA['N']                       # Cloud cover  [fraction][%/100]
sh = DATA['sh']                     # Initial snow height [m]

# todo Load climatic forcing (variables from NetCDF file)
# from netCDF4 import Dataset
# from config import nc_path
