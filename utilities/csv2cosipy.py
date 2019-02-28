"""
 This file reads the input data (model forcing) and write the output to netcdf file
"""
import sys
import xarray as xr
import pandas as pd
import numpy as np
import time
import dateutil

#np.warnings.filterwarnings('ignore')

sys.path.append('../')

from utilities.csv2cosipyConfig import *
from modules.radCor import correctRadiation

import argparse

def create_input(cs_file, cosipy_file, static_file, start_date, end_date):
    """ This function creates an input dataset from the Hintereisferner CR3000 Logger Dataset 
        Here you need to define how to interpolate the data.
    """

    print('-------------------------------------------')
    print('Create input \n')
    print('Read input file %s' % (cs_file))

    # Read data
    date_parser = lambda x: dateutil.parser.parse(x, ignoretz=True)
    df = pd.read_csv(cs_file,
       delimiter=',', index_col=['TIMESTAMP'],
        parse_dates=['TIMESTAMP'], na_values='NAN',date_parser=date_parser)

    df[T2_var] = df[T2_var].apply(pd.to_numeric, errors='coerce')
    df[RH2_var] = df[RH2_var].apply(pd.to_numeric, errors='coerce')
    df[U2_var] = df[U2_var].apply(pd.to_numeric, errors='coerce')
    df[G_var] = df[G_var].apply(pd.to_numeric, errors='coerce')
    df[PRES_var] = df[PRES_var].apply(pd.to_numeric, errors='coerce')
    df[RRR_var] = df[RRR_var].apply(pd.to_numeric, errors='coerce')
    df[N_var] = df[N_var].apply(pd.to_numeric, errors='coerce')

    # Select time slice
    df = df.loc[start_date:end_date]

    if(SNOWFALL_var in df):
        # Make hourly data
        df = df.resample('H').agg({T2_var:'mean', RH2_var:'mean',U2_var:'mean',
                               RRR_var:'sum',G_var:'mean',PRES_var:'mean',N_var:'mean', SNOWFALL_var:'sum'})

    else:
        df = df.resample('H').agg({T2_var: 'mean', RH2_var: 'mean', U2_var: 'mean',
                               RRR_var: 'sum', G_var: 'mean', PRES_var: 'mean', N_var:'mean'})

    # Load static data
    print('Read static file %s \n' % (static_file))
    ds = xr.open_dataset(static_file)
    ds.coords['time'] = df.index.values

    # Variable names in csv file
    T2 = df[T2_var]         # Temperature
    RH2 = df[RH2_var]       # Relative humdity
    U2 = df[U2_var]         # Wind velocity
    G = df[G_var]           # Incoming shortwave radiation
    PRES = df[PRES_var]     # Pressure
    RRR = df[RRR_var]       # Precipitation
    N = df[N_var]           # Cloud cover fraction

    # Create data arrays for the 2D fields
    T_interp = np.zeros([len(ds.time), len(ds.lat), len(ds.lon)])
    RH_interp = np.zeros([len(ds.time), len(ds.lat), len(ds.lon)])
    RRR_interp = np.zeros([len(ds.time), len(ds.lat), len(ds.lon)])
    U_interp = np.zeros([len(ds.time), len(ds.lat), len(ds.lon)])
    G_interp = np.zeros([len(ds.time), len(ds.lat), len(ds.lon)])
    P_interp = np.zeros([len(ds.time), len(ds.lat), len(ds.lon)])
    N_interp = np.zeros([len(ds.time), len(ds.lat), len(ds.lon)])

    if(SNOWFALL_var in df):
        SNOWFALL = df[SNOWFALL_var]      # Incoming longwave radiation
        SNOWFALL_interp = np.zeros([len(ds.time), len(ds.lat), len(ds.lon)])

    print('Interpolate CR file to grid')
    # Interpolate data (T, RH, RRR, U)  to grid using lapse rates
    for t in range(len(ds.time)):
        T_interp[t,:,:] = (T2[t]) + (ds.HGT.values-stationAlt)*lapse_T
        RH_interp[t,:,:] = RH2[t] + (ds.HGT.values-stationAlt)*lapse_RH
        RRR_interp[t,:,:] = RRR[t]
        U_interp[t,:,:] = U2[t]
        N_interp[t, :, :] = N[t]

        # Interpolate pressure using the barometric equation
        SLP = PRES[t]/np.power((1-(0.0065*stationAlt)/(288.15)), 5.255)
        P_interp[t,:,:] = SLP * np.power((1-(0.0065*ds.HGT.values)/(288.15)), 5.255)

        if (SNOWFALL_var in df):
            SNOWFALL_interp[t, :, :] = SNOWFALL[t]

    print("T_inter after", np.min(T_interp))
    # Change aspect to south==0, east==negative, west==positive
    ds['ASPECT'] = np.mod(ds['ASPECT']+180.0, 360.0)
    mask = ds['ASPECT'].where(ds['ASPECT']<=180.0)
    aspect = ds['ASPECT'].values
    aspect[aspect<180] = aspect[aspect<180]*-1.0
    aspect[aspect>=180] = 360.0 - aspect[aspect>=180]
    ds['ASPECT'] = (('lat','lon'),aspect)
    print(np.count_nonzero(~np.isnan(ds['MASK'].values)))

    # Auxiliary variables
    mask = ds.MASK.values
    slope = ds.SLOPE.values
    aspect = ds.ASPECT.values
    lats = ds.lat.values
    lons = ds.lon.values
    sw = G.values

    if radiationModule:
        print('Run the radiation module')
    else:
        print('No radiation module used')

    for t in range(len(ds.time)):
        doy = df.index[t].dayofyear
        hour = df.index[t].hour
        for i in range(len(ds.lat)):
            for j in range(len(ds.lon)):
                if (mask[i,j]==1):
                    if radiationModule:
                        G_interp[t,i,j] = np.maximum(0.0, correctRadiation(lats[i],lons[j], timezone_lon, doy, hour, slope[i,j], aspect[i,j], sw[t], zeni_thld))
                    else:
                        G_interp[t,i,j] = sw[t]

    # Check temperature, rh2, pressure,
    RH_interp[RH_interp > 100.0] = 100.0
    RH_interp[RH_interp < 0.0] = 0.1

    # Add arrays to dataset and write file
    add_variable_2D(ds, T_interp, 'T2', 'K', 'Temperature at 2 m')
    add_variable_2D(ds, RH_interp, 'RH2', '%', 'Relative humidity at 2 m')
    add_variable_2D(ds, RRR_interp, 'RRR', 'mm', 'Total precipitation (liquid+solid)')
    add_variable_2D(ds, U_interp, 'U2', 'm s\u207b\xb9', 'Wind velocity at 2 m')
    add_variable_2D(ds, G_interp, 'G', 'W m\u207b\xb2', 'Incoming shortwave radiation')
    add_variable_2D(ds, P_interp, 'PRES', 'hPa', 'Atmospheric Pressure')
    add_variable_2D(ds, N_interp, 'N', '%', 'Cloud cover fraction')

    if(SNOWFALL_var in df):
        add_variable_2D(ds, SNOWFALL_interp, 'SNOWFALL', 'm', 'Snowfall')

    ds.to_netcdf(cosipy_file)

    print('Input file created \n')
    print('-------------------------------------------')

    print(ds)

    check(ds.T2,316.16,223.16)
    check(ds.RH2,100.0,0.0)
    check(ds.U2, 50.0, 0.0)
    check(ds.RRR,20.0,0.0)
    check(ds.G,1600.0,0.0)
    check(ds.PRES,1080.0,200.0)
    check(ds.N,1.0,0.0)

    if(SNOWFALL_var in df):
        check(ds.SNOWFALL,0.1,0.0)

def add_variable_2D(ds, var, name, units, long_name):
    """ This function adds missing variables to the DATA class """
    ds[name] = (('time','lat','lon'), var)
    ds[name].attrs['units'] = units
    ds[name].attrs['long_name'] = long_name
    return ds

def check(field, max, min):
    '''Check the validity of the input data '''
    if np.nanmax(field) > max or np.nanmin(field) < min:
        print('\n\nWARNING! Please check the data, its seems they are out of a reasonalbe range %s MAX: %.2f MIN: %.2f \n' % (str.capitalize(field.name), np.nanmax(field), np.nanmin(field)))

    if np.isnan((np.min(field.values))):
        print('ERROR this does not work! %s VALUE: %.2f \n' % (str.capitalize(field.name), np.min(field.values)))

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Create 2D input file from CR-Logger file.')
    parser.add_argument('-c', '-cs_file', dest='cs_file', help='Campbell Scientific logger file (see readme for file convention)')
    parser.add_argument('-o', '-cosipy_file', dest='cosipy_file', help='Name of the resulting COSIPY file')
    parser.add_argument('-s', '-static_file', dest='static_file', help='Static file containing DEM, Slope etc.')
    parser.add_argument('-b', '-start_date', dest='start_date', help='Start date')
    parser.add_argument('-e', '-end_date', dest='end_date', help='End date')

    args = parser.parse_args()
    create_input(args.cs_file, args.cosipy_file, args.static_file, args.start_date, args.end_date) 
