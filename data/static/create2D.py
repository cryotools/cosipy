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


