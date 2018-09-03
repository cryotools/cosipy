"""
 This file reads the input data (model forcing) and write the output to netcdf file
"""
import sys
import xarray as xr
import pandas as pd
import numpy as np
import time

sys.path.append('../')

from cs2cosipyConfig import * 
from config import *
from modules.radCor import correctRadiation

import argparse

def create_input(cs_file, cosipy_file, static_file, start_date, end_date):
    """ This function creates an input dataset from the Hintereisferner CR3000 Logger Dataset 
        Here you need to define how to interpolate the data.
    """

    print('-------------------------------------------')
    print('Create input \n')
    print('Read input file %s' % (cs_file))

    # Read Im Hinteren Eis Logger data
    df = pd.read_csv(cs_file,
       delimiter=',', index_col=['TIMESTAMP'],
        parse_dates=['TIMESTAMP'], na_values='NAN', skiprows=[0, 2, 3])
    
    df[T_var] = df[T_var].apply(pd.to_numeric, errors='coerce')    
    df[RH_var] = df[RH_var].apply(pd.to_numeric, errors='coerce')    
    df[U_var] = df[U_var].apply(pd.to_numeric, errors='coerce')    
    df[RRR_var] = df[RRR_var].apply(pd.to_numeric, errors='coerce')    
    df[G_var] = df[G_var].apply(pd.to_numeric, errors='coerce')    
    
    if(P_var not in df):
        df[P_var] = 660.00
    
    # Select time slice
    df = df.loc[start_date:end_date]

    if(LW_var in df):
        # Make hourly data
        df = df.resample('H').agg({T_var:'mean', RH_var:'mean',U_var:'mean',
                               RRR_var:'sum',G_var:'mean',LW_var:'mean',P_var:'mean'})
    else:
        df = df.resample('H').agg({T_var:'mean', RH_var:'mean',U_var:'mean',
                               RRR_var:'sum',G_var:'mean',P_var:'mean'})

    # Load static data
    print('Read static file %s \n' % (static_file))
    ds = xr.open_dataset(static_file)
    ds.coords['time'] = df.index.values

    # Variable names in CR Logger file 
    T2 = df[T_var]        # Temperature 
    RH2 = df[RH_var]     # Relative humdity
    U2 = df[U_var]       # Wind velocity
    RRR = df[RRR_var]    # Precipitation
    G = df[G_var]        # Incoming shortwave radiation
    P = df[P_var]        # Pressure
    
    # Create data arrays for the 2D fields
    T_interp = np.zeros([len(ds.lat), len(ds.lon), len(ds.time)])
    RH_interp = np.zeros([len(ds.lat), len(ds.lon), len(ds.time)])
    RRR_interp = np.zeros([len(ds.lat), len(ds.lon), len(ds.time)])
    U_interp = np.zeros([len(ds.lat), len(ds.lon), len(ds.time)])
    G_interp = np.zeros([len(ds.lat), len(ds.lon), len(ds.time)])
    N_interp = np.zeros([len(ds.lat), len(ds.lon), len(ds.time)])
    P_interp = np.zeros([len(ds.lat), len(ds.lon), len(ds.time)])
    
    if(LW_var in df):
        LW = df[LW_var]      # Incoming longwave radiation
        LW_interp = np.zeros([len(ds.lat), len(ds.lon), len(ds.time)])
 
    print('Interpolate CR file to grid')
    # Interpolate data (T, RH, RRR, U)  to grid using lapse rates
    for t in range(len(ds.time)):
        T_interp[:,:,t] = (T2[t]+273.16) + (ds.HGT.values-stationAlt)*lapse_T 
        RH_interp[:,:,t] = RH2[t] + (ds.HGT.values-stationAlt)*lapse_RH 
        RRR_interp[:,:,t] = RRR[t]  
        U_interp[:,:,t] = U2[t] 
       
        # Interpolate pressure using the barometric equation 
        SLP = P[t]/np.power((1-(0.0065*stationAlt)/(288.15)), 5.255)
        P_interp[:,:,t] = SLP * np.power((1-(0.0065*ds.HGT.values)/(288.15)), 5.255)
         
        if(LW_var in df):
            LW_interp[:,:,t] = LW[t] 

    # Change aspect to south==0, east==negative, west==positive
    ds['ASPECT'] = np.mod(ds['ASPECT']+180.0, 360.0)
    print(ds['ASPECT'])
    mask = ds['ASPECT'].where(ds['ASPECT']<=180.0) 
    aspect = ds['ASPECT'].values
    aspect[aspect<180] = aspect[aspect<180]*-1.0
    aspect[aspect>=180] = 360.0 - aspect[aspect>=180]
    ds['ASPECT'] = (('lat','lon'),aspect)
    print(ds['ASPECT'])

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
                        G_interp[i,j,t] = np.maximum(0.0, correctRadiation(lats[i],lons[j], timezone_lon, doy, hour, slope[i,j], aspect[i,j], sw[t], zeni_thld))
                    else:
                        G_interp[i,j,t] = sw[t]

    # Add arrays to dataset and write file
    add_variable_2D(ds, T_interp, 'T2', 'K', 'Temperature at 2 m')
    add_variable_2D(ds, RH_interp, 'RH2', '%', 'Relative humidity at 2 m')
    add_variable_2D(ds, RRR_interp, 'RRR', 'mm', 'Total precipitation (liquid+solid)')
    add_variable_2D(ds, U_interp, 'U2', 'm/s', 'Wind velocity at 2 m')
    add_variable_2D(ds, G_interp, 'G', 'W m^-2', 'Incoming shortwave radiation')
    add_variable_2D(ds, P_interp, 'PRES', 'hPa', 'Atmospheric Pressure')
    add_variable_2D(ds, N_interp, 'N', '%', 'Cloud cover fraction')
        
    if(LW_var in df):
        add_variable_2D(ds, LW_interp, 'LWin', 'W m^-2', 'Incoming longwave radiation')
    print(ds)
    ds.to_netcdf(cosipy_file)
    
    print('Input file created \n')
    print('-------------------------------------------')

    print(ds)


def add_variable_2D(ds, var, name, units, long_name):
    """ This function adds missing variables to the DATA class """
    ds[name] = (('lat','lon','time'), var)
    ds[name].attrs['units'] = units
    ds[name].attrs['long_name'] = long_name
    return ds


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Create 2D input file from CR-Logger file.')
    parser.add_argument('-c', '-cs_file', dest='cs_file', help='Campbell Scientific logger file (see readme for file convention)')
    parser.add_argument('-o', '-cosipy_file', dest='cosipy_file', help='Name of the resulting COSIPY file')
    parser.add_argument('-s', '-static_file', dest='static_file', help='Static file containing DEM, Slope etc.')
    parser.add_argument('-b', '-start_date', dest='start_date', help='Start date')
    parser.add_argument('-e', '-end_date', dest='end_date', help='End date')

    args = parser.parse_args()
    create_input(args.cs_file, args.cosipy_file, args.static_file, args.start_date, args.end_date) 
