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

sys.path.append('../../')

from utilities.csv2cosipy.csv2cosipyConfig import *
from modules.radCor import correctRadiation

import argparse

def create_input(cs_file, cosipy_file, static_file, start_date, end_date):
    """ This function creates an input dataset from an offered csv file with input point data
        Here you need to define how to interpolate the data.
    """

    print('-------------------------------------------')
    print('Create input \n')
    print('Read input file %s' % (cs_file))

    #-----------------------------------
    # Read data
    #-----------------------------------
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

    #-----------------------------------
    # Select time slice
    #-----------------------------------
    df = df.loc[start_date:end_date]

    #-----------------------------------
    # Average to hourly data 
    #-----------------------------------
    if(SNOWFALL_var in df):
        # Make hourly data
        df = df.resample('H').agg({T2_var:'mean', RH2_var:'mean',U2_var:'mean',
                               RRR_var:'sum',G_var:'mean',PRES_var:'mean',N_var:'mean', SNOWFALL_var:'sum'})

    else:
        df = df.resample('H').agg({T2_var: 'mean', RH2_var: 'mean', U2_var: 'mean',
                               RRR_var: 'sum', G_var: 'mean', PRES_var: 'mean', N_var:'mean'})

    #-----------------------------------
    # Load static data
    #-----------------------------------
    print('Read static file %s \n' % (static_file))
    ds = xr.open_dataset(static_file)
    ds.coords['time'] = df.index.values

    #-----------------------------------
    # Variable names in csv file
    #-----------------------------------
    T2 = df[T2_var]         # Temperature
    RH2 = df[RH2_var]       # Relative humdity
    U2 = df[U2_var]         # Wind velocity
    G = df[G_var]           # Incoming shortwave radiation
    PRES = df[PRES_var]     # Pressure
    RRR = df[RRR_var]       # Precipitation
    N = df[N_var]           # Cloud cover fraction

    #-----------------------------------
    # Create numpy arrays for the 2D fields
    #-----------------------------------
    T_interp = np.zeros([len(ds.time), len(ds.south_north), len(ds.west_east)])
    RH_interp = np.zeros([len(ds.time), len(ds.south_north), len(ds.west_east)])
    RRR_interp = np.zeros([len(ds.time), len(ds.south_north), len(ds.west_east)])
    U_interp = np.zeros([len(ds.time), len(ds.south_north), len(ds.west_east)])
    G_interp = np.zeros([len(ds.time), len(ds.south_north), len(ds.west_east)])
    P_interp = np.zeros([len(ds.time), len(ds.south_north), len(ds.west_east)])
    N_interp = np.zeros([len(ds.time), len(ds.south_north), len(ds.west_east)])

    if(SNOWFALL_var in df):
        SNOWFALL = df[SNOWFALL_var]      # Incoming longwave radiation
        SNOWFALL_interp = np.zeros([len(ds.time), len(ds.south_north), len(ds.west_east)])

    #-----------------------------------
    # Interpolate point data to grid 
    #-----------------------------------
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

    # Change aspect to south==0, east==negative, west==positive
    ds['ASPECT'] = np.mod(ds['ASPECT']+180.0, 360.0)
    mask = ds['ASPECT'].where(ds['ASPECT']<=180.0)
    aspect = ds['ASPECT'].values
    aspect[aspect<180] = aspect[aspect<180]*-1.0
    aspect[aspect>=180] = 360.0 - aspect[aspect>=180]
    ds['ASPECT'] = (('south_north','west_east'),aspect)
    print(('Number of glacier cells: %i') % (np.count_nonzero(~np.isnan(ds['MASK'].values))))

    # Auxiliary variables
    mask = ds.MASK.values
    slope = ds.SLOPE.values
    aspect = ds.ASPECT.values
    south_norths = ds.south_north.values
    west_easts = ds.west_east.values
    sw = G.values

    #-----------------------------------
    # Run radiation module 
    #-----------------------------------
    if radiationModule:
        print('Run the radiation module')
    else:
        print('No radiation module used')

    for t in range(len(ds.time)):
        doy = df.index[t].dayofyear
        hour = df.index[t].hour
        for i in range(len(ds.south_north)):
            for j in range(len(ds.west_east)):
                if (mask[i,j]==1):
                    if radiationModule:
                        G_interp[t,i,j] = np.maximum(0.0, correctRadiation(south_norths[i],west_easts[j], timezone_lon, doy, hour, slope[i,j], aspect[i,j], sw[t], zeni_thld))
                    else:
                        G_interp[t,i,j] = sw[t]

    #-----------------------------------
    # Check bounds for relative humidity 
    #-----------------------------------
    RH_interp[RH_interp > 100.0] = 100.0
    RH_interp[RH_interp < 0.0] = 0.1


    #-----------------------------------
    # Create dataset
    #-----------------------------------
    # First create 2D lat/lon fields
    x, y = np.meshgrid(ds.west_east, ds.south_north)

    dso = xr.Dataset()
    dso.coords['time'] = (('time'), ds.time.values)

    #-----------------------------------
    # Add variables to file 
    #-----------------------------------
    dso.coords['lat'] = (('south_north','west_east'), y)
    dso.coords['lon'] = (('south_north','west_east'), x)
    add_variable_along_latlon(dso, ds.HGT.values, 'HGT', 'm', 'Elevation')
    add_variable_along_latlon(dso, ds.ASPECT.values, 'ASPECT', 'degrees', 'Aspect of slope')
    add_variable_along_latlon(dso, ds.SLOPE.values, 'SLOPE', 'degrees', 'Terrain slope')
    add_variable_along_latlon(dso, ds.MASK.values, 'MASK', 'boolean', 'Glacier mask')
    add_variable_along_timelatlon(dso, T_interp, 'T2', 'K', 'Temperature at 2 m')
    add_variable_along_timelatlon(dso, RH_interp, 'RH2', '%', 'Relative humidity at 2 m')
    add_variable_along_timelatlon(dso, RRR_interp, 'RRR', 'mm', 'Total precipitation (liquid+solid)')
    add_variable_along_timelatlon(dso, U_interp, 'U2', 'm s\u207b\xb9', 'Wind velocity at 2 m')
    add_variable_along_timelatlon(dso, G_interp, 'G', 'W m\u207b\xb2', 'Incoming shortwave radiation')
    add_variable_along_timelatlon(dso, P_interp, 'PRES', 'hPa', 'Atmospheric Pressure')
    add_variable_along_timelatlon(dso, N_interp, 'N', '%', 'Cloud cover fraction')
    
    if(SNOWFALL_var in df):
        add_variable_along_timelatlon(ds, SNOWFALL_interp, 'SNOWFALL', 'm', 'Snowfall')

    #-----------------------------------
    # Write file to disc 
    #-----------------------------------
    dso.to_netcdf(cosipy_file)


    print('Input file created \n')
    print('-------------------------------------------')

   
    #-----------------------------------
    # Do some checks
    #-----------------------------------
    check(dso.T2,316.16,223.16)
    check(dso.RH2,100.0,0.0)
    check(dso.U2, 50.0, 0.0)
    check(dso.RRR,20.0,0.0)
    check(dso.G,1600.0,0.0)
    check(dso.PRES,1080.0,200.0)
    check(dso.N,1.0,0.0)

    if(SNOWFALL_var in df):
        check(ds.SNOWFALL,0.1,0.0)

def add_variable_along_timelatlon(ds, var, name, units, long_name):
    """ This function adds missing variables to the DATA class """
    ds[name] = (('time','south_north','west_east'), var)
    ds[name].attrs['units'] = units
    ds[name].attrs['long_name'] = long_name
    return ds

def add_variable_along_latlon(ds, var, name, units, long_name):
    """ This function self.adds missing variables to the self.DATA class """
    ds[name] = (('south_north','west_east'), var)
    ds[name].attrs['units'] = units
    ds[name].attrs['long_name'] = long_name
    ds[name].encoding['_FillValue'] = -9999
    return ds

def check(field, max, min):
    '''Check the validity of the input data '''
    if np.nanmax(field) > max or np.nanmin(field) < min:
        print('\n\nWARNING! Please check the data, its seems they are out of a reasonalbe range %s MAX: %.2f MIN: %.2f \n' % (str.capitalize(field.name), np.nanmax(field), np.nanmin(field)))

    if np.isnan((np.min(field.values))):
        print('ERROR this does not work! %s VALUE: %.2f \n' % (str.capitalize(field.name), np.min(field.values)))

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Create 2D input file from csv file.')
    parser.add_argument('-c', '-csv_file', dest='csv_file', help='Csv file(see readme for file convention)')
    parser.add_argument('-o', '-cosipy_file', dest='cosipy_file', help='Name of the resulting COSIPY file')
    parser.add_argument('-s', '-static_file', dest='static_file', help='Static file containing DEM, Slope etc.')
    parser.add_argument('-b', '-start_date', dest='start_date', help='Start date')
    parser.add_argument('-e', '-end_date', dest='end_date', help='End date')

    args = parser.parse_args()
    create_input(args.csv_file, args.cosipy_file, args.static_file, args.start_date, args.end_date) 
