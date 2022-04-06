 
"""
 This file reads the input data (model forcing) and write the output to netcdf file.  There is the create_1D_input
 (point model) and create_2D_input (distirbuted simulations) function. In case of the 1D input, the function works
 without a static file, in that file the static variables are created. For both cases, lapse rates can be determined
 in the aws2cosipyConfig.py file.
"""
import sys
import xarray as xr
import pandas as pd
import numpy as np
import netCDF4 as nc
import time
import dateutil

sys.path.append('../../')

from constants import make_icestupa
from config import time_start, time_end, icestupa_name, obs_type
from utilities.aws2cosipy.aws2cosipyConfig import *
from cosipy.modules.radCor import *

if __name__ == "__main__":
    dfd = pd.read_csv("../../data/input/" + icestupa_name + "/drone.csv")
    dft = pd.read_csv("../../data/input/" + icestupa_name + "/thermistor.csv")
    dfd['TIMESTAMP'] = pd.to_datetime(dfd['TIMESTAMP'])
    dft['TIMESTAMP'] = pd.to_datetime(dft['TIMESTAMP'])
    dfd = dfd.set_index('TIMESTAMP')
    dft = dft.set_index('TIMESTAMP')
    # Intersection
    dfd = dfd[[item for item in dfd.columns if item in obs_type]]
    dfd.to_csv("../../data/input/" + icestupa_name + "/observations.csv")
    dft = dft[[item for item in dft.columns if item in obs_type]]

    df = pd.merge(dfd, dft, on ='TIMESTAMP', how ="outer")
    df = df.sort_index()
    print(dfd)
    # df.to_csv("../../data/input/" + icestupa_name + "/observations.csv")
