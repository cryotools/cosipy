"""
 This file reads the input data (model forcing) and write the output to netcdf file
"""
import sys
import xarray as xr
import pandas as pd
import numpy as np
import time
import dateutil

ds = xr.open_dataset('../data/output/Zhadang_ERA5_20090501-20100430.nc')

ds = ds.sel(lat=30.4665, lon=90.6275)
print(ds['LE'])
ds = ds.resample(time='D').mean()
ds.to_netcdf('mean.nc')
