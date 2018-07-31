import xarray as xr
import pandas as pd
import numpy as npz

df = pd.read_csv('./centinella_master.csv',delimiter=';',index_col=0, parse_dates=['Date'], na_values=' ')

print(df.head())

def insert_var(ds, var, name, units, FillValue, missing_value, long_name):
    ds[name] = xr.DataArray(var, coords=[('time', df.index)])
    ds[name].attrs['units'] = units
    ds[name].attrs['long_name'] = long_name

    return ds

ds = xr.Dataset()
ds.coords['lat'] =  -50.21
ds.coords['lon'] =  -72.23
ds.coords['time'] = df.index

# Account for the missing values in G
for i in range(len(df['G'])):
    if (np.isnan(df['G'][i])):
        df['G'][i] = df['G'][i-1]
        print(df['G'][i])

insert_var(ds, df['T']+273.15, 'T2', 'degC', '-9999', -9999, 'Temperature at 2 m')
insert_var(ds, df['RH'], 'RH2', '%', '-9999', -9999, 'Relative humidity at 2 m')
insert_var(ds, df['G'], 'G', 'W m^-2', '-9999', -9999, 'Shortwave radiation')
insert_var(ds, df['U'], 'U2', 'm s^-1','-9999', -9999, 'Wind velocity at 2 m')
insert_var(ds, df['P'], 'RRR', 'm', '-9999', -9999, 'Precipitation')

print(ds)
ds.to_netcdf('data_amalia.nc')

