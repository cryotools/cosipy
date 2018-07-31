import xarray as xr
import pandas as pd
import numpy as np

def insert_var(ds, var, name, units, FillValue, missing_value, long_name):
    ds[name] = xr.DataArray(var, coords=[('time', df.index)])
    ds[name].attrs['units'] = units
    ds[name].attrs['long_name'] = long_name

    return ds


def insert_var_2D(ds, data, lat, lon, name, units, FillValue, missing_value, long_name):
    ds[name] = (('lat','lon'), data)
    ds[name].attrs['units'] = units
    ds[name].attrs['long_name'] = long_name

    return ds

def write_static_data():
    # Static data
    ds_dem = xr.open_dataset('./orig/dem.nc')
    ds_slo = xr.open_dataset('./orig/slope.nc')
    ds_asp = xr.open_dataset('./orig/aspect.nc')
    ds_mask = xr.open_dataset('./orig/glaciergrid.nc')
    
    # Write static data
    ds = xr.Dataset()
    ds.coords['lat'] = ds_dem.lat 
    ds.coords['lon'] = ds_dem.lon 

    insert_var_2D(ds, ds_dem.Band1, ds.lat, ds.lon, 'HGT', 'm', '-9999', -9999, 'Elevation')
    insert_var_2D(ds, ds_slo.Band1, ds.lat, ds.lon, 'SLOPE', 'degree', '-9999', -9999, 'Slope angle')
    insert_var_2D(ds, ds_asp.Band1, ds.lat, ds.lon, 'ASPECT', 'degree', '-9999', -9999, 'Aspect')
    insert_var_2D(ds, ds_mask.Band1, ds.lat, ds.lon, 'MASK', '-', '-9999', -9999, 'Glacier mask')
    ds.to_netcdf('static.nc')



write_static_data()


## Account for the missing values in G
#for i in range(len(df['G'])):
#    if (np.isnan(df['G'][i])):
#        df['G'][i] = df['G'][i-1]
#
#insert_var_2D(ds, df['T']+273.15, lat, lon, 'T2', 'degC', '-9999', -9999, 'Temperature at 2 m')
#insert_var_2D(ds, df['RH'], lat, lon, 'RH2', '%', '-9999', -9999, 'Relative humidity at 2 m')
#insert_var_2D(ds, df['G'], lat, lon,  'G', 'W m^-2', '-9999', -9999, 'Shortwave radiation')
#insert_var_2D(ds, df['U'], lat, lon, 'U2', 'm s^-1','-9999', -9999, 'Wind velocity at 2 m')
#insert_var_2D(ds, df['P']/1000.0, lat, lon,  'RRR', 'm', '-9999', -9999, 'Precipitation')
#
#if ('N' in df):
#    print('Cloud cover data (N) exists ')
#    insert_var_2D(ds, df['N'], lat, lon,  'N', '-', '-9999', -9999, 'Cloud cover')
#else:
#    print('Cloud cover data (N) is missing ')
#    print('The value is set to a constant value of 0.5 ')
#    insert_var_2D(ds, 0.5*np.ones_like(df['T']), lat, lon,  'N', '-', '-9999', -9999, 'Cloud cover')
#
#
#if ('PRES' in df):
#    print('Pressure data (PRES) exists ')
#    insert_var_2D(ds, df['PRES'], lat, lon,  'PRES', 'hPa', '-9999', -9999, 'Pressure')
#else:
#    print('Pressure data (PRES) is missing ')
#    print('The value is set to a constant value of 1013.0 hPa ')
#    insert_var_2D(ds, 1013.0*np.ones_like(df['T']), lat, lon,  'PRES', 'hPa', '-9999', -9999, 'Pressure')
#
#print(ds)
#ds.to_netcdf('data_amalia_2D.nc')

