"""
 This file reads the input data (model forcing) and write the output to netcdf file
"""
import sys
import xarray as xr
import pandas as pd
import numpy as np
import xrspatial as xrs
import time
from itertools import product

sys.path.append('../../')

from utilities.wrf2cosipy.wrf2cosipyConfig import *
from constants import densification_method, constant_density, water_density

import argparse

def create_input(wrf_file, cosipy_file, start_date, end_date):
    """ This function creates an input dataset from WRF data
    """

    print('-------------------------------------------')
    print('Create input \n')
    print('Read input file %s' % (wrf_file))

    # Read WRF file
    ds = xr.open_dataset(wrf_file)
    
    # Rename the time coordinate
    ds = ds.rename({'XTIME':'Time'})
    
    # Re-format timestamp (only hour and minutes, no seconds)
    ds = ds.assign_coords(Time=pd.to_datetime(ds['Time'].values).strftime('%Y-%m-%dT%H-%M'))

    # Select the specified period
    if ((start_date!=None) & (end_date!=None)):
        ds = ds.sel(Time=slice(start_date,end_date))

    # Create COSIPY input file
    dso = xr.Dataset()
    dso.coords['time'] = (('time'), ds.Time.values)
    dso.coords['lat'] = (('south_north', 'west_east'), ds.XLAT[0].values)
    dso.coords['lon'] = (('south_north', 'west_east'), ds.XLONG[0].values)
 
    # Add variables to file 
    dso = add_variable_along_latlon(dso, ds.HGT[0].values, 'HGT', 'm', 'Elevation')
    dso = add_variable_along_timelatlon(dso, ds.T2.values, 'T2', 'm', 'Temperature at 2 m')
    dso = add_variable_along_timelatlon(dso, wrf_rh(ds.T2.values, ds.Q2.values, ds.PSFC.values), 'RH2', '%', 'Relative humidity at 2 m')
    dso = add_variable_along_timelatlon(dso, ds.SWDOWN.values, 'G', 'W m^-2', 'Incoming shortwave radiation')
    dso = add_variable_along_timelatlon(dso, ds.GLW.values, 'LWin', 'W m^-2', 'Incoming longwave radiation')
    dso = add_variable_along_timelatlon(dso, ds.PSFC.values/100.0, 'PRES', 'hPa', 'Atmospheric Pressure')
    dso = add_variable_along_latlon(dso, ds.SNOWH[0], 'SNOWHEIGHT', 'm', 'Initial snowheight')
    dso = add_variable_along_latlon(dso, ds.SNOW[0], 'SWE', 'kg m^-2', 'Snow Water Equivalent')
    dso = add_variable_along_latlon(dso, ds.SNOWC[0], 'SNOWC', '-', 'Flag indicating snow coverage (1 for snow cover)')
    dso = add_variable_along_latlon(dso, ds.LU_INDEX[0], 'LU_INDEX', '-', 'Land use category')
    dso = add_variable_along_latlon(dso, ds.TSK[0], 'TSK', 'K', 'Skin temperature')
    
    # Wind velocity at 2 m (assuming neutral stratification)
    z  = hu     # Height of measurement
    z0 = 0.0040 # Roughness length for momentum
    umag = np.sqrt(ds.V10.values**2+ds.U10.values**2)   # Mean wind velocity
    U2 = umag * (np.log(2 / z0) / np.log(10 / z0))
    dso = add_variable_along_timelatlon(dso, U2, 'U2', 'm s^-1', 'Wind velocity at 2 m')

    # Add glacier mask to file (derived from the land use category)
    mask = ds.LU_INDEX[0].values
    mask[mask!=lu_class] = 0
    mask[mask==lu_class] = 1
    dso = add_variable_along_latlon(dso, mask, 'MASK', '-', 'Glacier mask')
    
    # Derive precipitation from accumulated values
    rrr = np.full_like(ds.T2, 0.)
    for t in np.arange(1,len(dso.time)):
        rrr[t,:,:] = (ds.RAINNC[t,:,:] + ds.RAINC[t,:,:])  - (ds.RAINNC[t-1,:,:] + ds.RAINC[t-1,:,:])    
    dso = add_variable_along_timelatlon(dso, rrr, 'RRR', 'mm', 'Total precipitation')
 
    # Calc fresh snow density
    if (densification_method!='constant'):
        density_fresh_snow = np.maximum(109.0+6.0*(ds.T2.values-273.16)+26.0*np.sqrt(U2), 50.0)
    else:
        density_fresh_snow = constant_density 	
 
    # Derive snowfall from accumulated values
    snowf = np.full_like(ds.T2, 0.)
    for t in np.arange(1,len(dso.time)):
        snowf[t,:,:] = ds.SNOWNC[t,:,:]-ds.SNOWNC[t-1,:,:]   
    snowf = (snowf/1000.0)*(water_density/density_fresh_snow)
    dso = add_variable_along_timelatlon(dso, snowf, 'SNOWFALL', 'm', 'Snowfall')
    
    # Compute slope & aspect from HGT
    DX = ds.attrs.get('DX').astype("float")
    DY = ds.attrs.get('DY').astype("float")
    HGT = ds.HGT[0]
    HGT.attrs['res'] = (DX,DY)
    slope = xrs.slope(HGT)
    aspect = xrs.aspect(HGT)
    dso = add_variable_along_latlon(dso, slope, 'SLOPE', 'degrees', 'Slope')
    dso = add_variable_along_latlon(dso, aspect, 'ASPECT', 'degrees', 'Aspect')

    # Write file
    dso.to_netcdf(cosipy_file)
  
    # Do some checks 
    check(dso.T2,316.16,223.16)
    check(dso.RH2,100.0,0.0)
    check(dso.SNOWFALL,1.0,0.0)
    check(dso.U2,50.0,0.0)
    check(dso.G,1600.0,0.0)
    check(dso.PRES,1080.0,200.0)
    check(dso.LWin,500.0,100.0)


def add_variable_along_latlon(ds, var, name, units, long_name):
    """ This function self.adds missing variables to the self.DATA class """
    ds[name] = (('south_north','west_east'), var)
    ds[name].attrs['units'] = units
    ds[name].attrs['long_name'] = long_name
    ds[name].encoding['_FillValue'] = -9999
    return ds
    
def add_variable_along_timelatlon(ds, var, name, units, long_name):
    """ This function adds missing variables to the DATA class """
    ds[name] = (('time','south_north','west_east'), var)
    ds[name].attrs['units'] = units
    ds[name].attrs['long_name'] = long_name
    return ds

def check(field, max, min):
    '''Check the validity of the input data '''
    if np.nanmax(field) > max or np.nanmin(field) < min:
        print('\n\nWARNING! Please check the data, its seems they are out of a reasonable range %s MAX: %.2f MIN: %.2f \n' % (str.capitalize(field.name), np.nanmax(field), np.nanmin(field)))

def wrf_rh(T2, Q2, PSFC):
    pq0 = 379.90516
    a2 = 17.2693882
    a3 = 273.16
    a4 = 35.86
    rh = Q2 * 100 / ( (pq0 / PSFC) * np.exp(a2 * (T2 - a3) / (T2 - a4)) )
    
    rh[rh>100.0] = 100.0
    rh[rh<0.0] = 0.0
    return rh

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Create 2D input file from WRF file.')
    parser.add_argument('-i', '-wrf_file', dest='wrf_file', help='WRF file')
    parser.add_argument('-o', '-cosipy_file', dest='cosipy_file', help='Name of the resulting COSIPY file')
    parser.add_argument('-b', '-start_date', dest='start_date', const=None, help='Start date')
    parser.add_argument('-e', '-end_date', dest='end_date', const=None, help='End date')

    args = parser.parse_args()
    create_input(args.wrf_file, args.cosipy_file, args.start_date, args.end_date) 
