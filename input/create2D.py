import xarray as xr
import pandas as pd
import numpy as np

1
import numpy as np

2


def mmm(input):
    print("max: ", np.nanmax(input))
    print("min: ", np.nanmin(input))
    print("mean: ", np.nanmean(input))
    print("sum: ", np.nansum(input),)
    print("")

df = pd.read_csv('./centinella_master.csv',delimiter=';',index_col=0, parse_dates=['Date'], na_values=' ')

def insert_var(ds, var, name, units, FillValue, missing_value, long_name):
    ds[name] = xr.DataArray(var, coords=[('time', df.index)])
    ds[name].attrs['units'] = units
    ds[name].attrs['long_name'] = long_name

    return ds


def insert_var_2D(ds, var, lat, lon, name, units, FillValue, missing_value, long_name):
    x, y, z = np.meshgrid(lon,lat,var)
    r = (0.2*np.random.randn(z.shape[0],z.shape[1],z.shape[2]))
    print(r)
    ds[name] = (('lat','lon','time'), z+r)
    ds[name].attrs['units'] = units
    ds[name].attrs['long_name'] = long_name

    return ds

lat = np.arange(-45,-48,-1.0)
lon = np.arange(-80,-78,1.0)
A = np.meshgrid(lon,lat)
ds = xr.Dataset()
ds.coords['lat'] =  np.arange(-45.0, -48.0, -1.0)
ds.coords['lon'] =  np.arange(-80.0, -78.0, 1.0)
ds.coords['time'] = df.index.values

# Account for the missing values in G
for i in range(len(df['G'])):
    if (np.isnan(df['G'][i])):
        df['G'][i] = df['G'][i-1]

insert_var_2D(ds, df['T']+273.15, lat, lon, 'T2', 'degC', '-9999', -9999, 'Temperature at 2 m')
insert_var_2D(ds, df['RH'], lat, lon, 'RH2', '%', '-9999', -9999, 'Relative humidity at 2 m')
insert_var_2D(ds, df['G'], lat, lon,  'G', 'W m^-2', '-9999', -9999, 'Shortwave radiation')
insert_var_2D(ds, df['U'], lat, lon, 'U2', 'm s^-1','-9999', -9999, 'Wind velocity at 2 m')
insert_var_2D(ds, df['P']/1000.0, lat, lon,  'RRR', 'm', '-9999', -9999, 'Precipitation')

if ('N' in df):
    print('Cloud cover data (N) exists ')
    insert_var_2D(ds, df['N'], lat, lon,  'N', '-', '-9999', -9999, 'Cloud cover')
else:
    print('Cloud cover data (N) is missing ')
    print('The value is set to a constant value of 0.5 ')
    insert_var_2D(ds, 0.5*np.ones_like(df['T']), lat, lon,  'N', '-', '-9999', -9999, 'Cloud cover')


if ('PRES' in df):
    print('Pressure data (PRES) exists ')
    insert_var_2D(ds, df['PRES'], lat, lon,  'PRES', 'hPa', '-9999', -9999, 'Pressure')
else:
    print('Pressure data (PRES) is missing ')
    print('The value is set to a constant value of 1013.0 hPa ')
    insert_var_2D(ds, 1013.0*np.ones_like(df['T']), lat, lon,  'PRES', 'hPa', '-9999', -9999, 'Pressure')

print(ds.T2.long_name)
mmm(ds.T2.values)

print(ds.RH2.long_name)
mmm(ds.RH2.values)

print(ds.G.long_name)
mmm(ds.G.values)

print(ds.U2.long_name)
mmm(ds.U2.values)

print(ds.RRR.long_name)
mmm(ds.RRR.values)

print(ds.N.long_name)
mmm(ds.N.values)

print(ds.PRES.long_name)
mmm(ds.PRES.values)
print(ds)
ds.to_netcdf('data_amalia_2D.nc')