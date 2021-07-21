
"""
 This file reads the input data (model forcing) and write the output to netcdf file.  There is the create_1D_input
 (point model) and create_2D_input (distirbuted simulations) function. In case of the 1D input, the function works
 without a static file, in that file the static variables are created. For both cases, lapse rates can be determined
 in the aws2cosipyConfig.py file.
 This file has been extended to include dynamic lapse rates using data from two weather stations for T2 and RH.
 For U2 the mean is calculated or a constant value has to be defined in the asw2cosipConfig_2aws.py file.
 It is possible to weight T2, function has to be changed in this file (aws2cosipy_2aws.py). 
 It is possible to add a constant value to T2 in aws2cosipyConfig_2aws.py.
 It is possible to add precipitation (in % of total precipitation) in aws2cosipyConfig_2aws.py.
 Use "valley" as initial weather station.
 To executed the script with two meteo-files:
 python aws2cosipy.py /
 -cv ../../data/input/meteo-data-weatherstation-valley.csv /
 -cm ../../data/input/meteo-data-weatherstation-mountain.csv /
 -o ../../data/input/output-file.nc /
 -s ../../data/static/static-file.nc /
 -b yyyymmdd -e yyyymmdd
"""
import sys
import xarray as xr
import pandas as pd
import numpy as np
import time
import dateutil
from itertools import product

#np.warnings.filterwarnings('ignore')

sys.path.append('../../')

from utilities.aws2cosipy.aws2cosipyConfig_2aws import *
from cosipy.modules.radCor import correctRadiation

import argparse

def create_1D_input(cs_file_valley, cs_file_mountain, cosipy_file, static_file, start_date, end_date):
    """ This function creates an input dataset from an offered csv file with input point data
        Here you need to define how to interpolate the data.

        Please note, there should be only one header line in the file.

        Latest update: 
            Tobias Sauter 07.07.2019
	        Anselm 04.07.2020
	    	Christine Seupel 21.06.2021
		   Change this function to use data of 2 weather stations:
                   aws_valley and aws_mountain
    """

    print('-------------------------------------------')
    print('Create input \n')
    print('Read input file valley  %s' % (cs_file_valley))
    print('Read input file mountain %s' % (cs_file_mountain))


    #-----------------------------------
    # Read data
    #-----------------------------------
    date_parser = lambda x: dateutil.parser.parse(x, ignoretz=True)
    df_v = pd.read_csv(cs_file_valley,
       delimiter=',', index_col=['TIMESTAMP'],
        parse_dates=['TIMESTAMP'], na_values='NAN',date_parser=date_parser)
    df_m = pd.read_csv(cs_file_mountain,
       delimiter=',', index_col=['TIMESTAMP'],
        parse_dates=['TIMESTAMP'], na_values='NAN',date_parser=date_parser)

    print(df_v[pd.isnull(df_v).any(axis=1)])
    df_v = df_v.fillna(method='ffill')
    print(df_v[pd.isnull(df_v).any(axis=1)])
    
    print(df_m[pd.isnull(df_m).any(axis=1)])
    df_m = df_m.fillna(method='ffill')
    print(df_m[pd.isnull(df_m).any(axis=1)])

    if (LWin_var not in df_v):
        df_v[LWin_var] = np.nan
    if (N_var not in df_v):
        df_v[N_var] = np.nan
    if (SNOWFALL_var not in df_v):
        df_v[SNOWFALL_var] = np.nan
        
    if (LWin_var not in df_m):
        df_m[LWin_var] = np.nan
    if (N_var not in df_m):
        df_m[N_var] = np.nan
    if (SNOWFALL_var not in df_m):
        df_m[SNOWFALL_var] = np.nan
 
    # Improved function to sum dataframe columns which contain nan's
    def nansumwrapper(a, **kwargs):
        if np.isnan(a).all():
            return np.nan
        else:
            return np.nansum(a, **kwargs)

    col_list = [T2_var,RH2_var,U2_var,G_var,RRR_var,PRES_var,LWin_var,N_var,SNOWFALL_var]
    df_v = df_v[col_list]
    df_m = df_m[col_list]
    
    #df_v = df_v.resample('1H').agg({T2_var:np.mean, RH2_var:np.mean, U2_var:np.mean, G_var:np.mean, PRES_var:np.mean, RRR_var:nansumwrapper, LWin_var:np.mean, N_var:np.mean, SNOWFALL_var:nansumwrapper})
    df_v = df_v.dropna(axis=1,how='all')
    print(df_v.head())
    
    #df_m = df_m.resample('1H').agg({T2_var:np.mean, RH2_var:np.mean, U2_var:np.mean, G_var:np.mean, PRES_var:np.mean, RRR_var:nansumwrapper, LWin_var:np.mean, N_var:np.mean, SNOWFALL_var:nansumwrapper})
    df_m = df_m.dropna(axis=1,how='all')
    print(df_m.head())

    #-----------------------------------
    # Select time slice
    #-----------------------------------
    if ((start_date != None) & (end_date !=None)): 
        df_v = df_v.loc[start_date:end_date]
        
    if ((start_date != None) & (end_date !=None)): 
        df_m = df_m.loc[start_date:end_date]

    #-----------------------------------
    # Load static data
    #-----------------------------------

    if WRF:
        ds = xr.Dataset()
        lon, lat = np.meshgrid(plon, plat)
        ds.coords['lat'] = (('south_north', 'west_east'), lon)
        ds.coords['lon'] = (('south_north', 'west_east'), lat)

    else:
        if (static_file):
            print('Read static file %s \n' % (static_file))
            ds = xr.open_dataset(static_file)
            ds = ds.sel(lat=plat,lon=plon,method='nearest')
            ds.coords['lon'] = np.array([ds.lon.values])
            ds.coords['lat'] = np.array([ds.lat.values])
            print('ds: ', ds)
        else:
            ds = xr.Dataset()
            ds.coords['lon'] = np.array([plon])

            ds.coords['lat'] = np.array([plat])

        ds.lon.attrs['standard_name'] = 'lon'
        ds.lon.attrs['long_name'] = 'longitude'
        ds.lon.attrs['units'] = 'degrees_east'

        ds.coords['lat'] = np.array([plat])
        ds.lat.attrs['standard_name'] = 'lat'
        ds.lat.attrs['long_name'] = 'latitude'
        ds.lat.attrs['units'] = 'degrees_north'

    ds.coords['time'] = (('time'), df_v.index.values)

    #-----------------------------------
    # Order variables valley
    #-----------------------------------
    
    if (in_K):
        df_v[T2_var] = df_v[T2_var].apply(pd.to_numeric, errors='coerce')
    else:
        df_v[T2_var] = df_v[T2_var].values + 273.16
        df_v[T2_var] = df_v[T2_var].apply(pd.to_numeric, errors='coerce')

    if np.nanmax(df_v[T2_var]) > 373.16:
        print('Maximum temperature is: %s K please check the input temperature' % (np.nanmax(T2)))
        sys.exit()
    elif np.nanmin(df_v[T2_var]) < 173.16:
        print('Minimum temperature is: %s K please check the input temperature' % (np.nanmin(T2)))
        sys.exit()
    
    df_v[RH2_var] = df_v[RH2_var].apply(pd.to_numeric, errors='coerce')
    df_v[U2_var] = df_v[U2_var].apply(pd.to_numeric, errors='coerce')
    df_v[G_var] = df_v[G_var].apply(pd.to_numeric, errors='coerce')
    df_v[PRES_var] = df_v[PRES_var].apply(pd.to_numeric, errors='coerce')
    
    if (RRR_var in df_v):
        df_v[RRR_var] = df_v[RRR_var].apply(pd.to_numeric, errors='coerce')

    if (PRES_var not in df_v):
        df_v[PRES_var] = 660.00

    if (LWin_var not in df_v and N_var not in df_v):
        print("ERROR no longwave incoming or cloud cover data")
        sys.exit()

    elif (LWin_var in df_v):
        df_v[LWin_var] = df_v[LWin_var].apply(pd.to_numeric, errors='coerce')

    elif (N_var in df_v):
        df_v[N_var] = df_v[N_var].apply(pd.to_numeric, errors='coerce')

    if (SNOWFALL_var in df_v):
        df_v[SNOWFALL_var] = df_v[SNOWFALL_var].apply(pd.to_numeric, errors='coerce')
     
    #-----------------------------------
    # Order variables mountain
    #-----------------------------------
    
    if (in_K):
        df_m[T2_var] = df_m[T2_var].apply(pd.to_numeric, errors='coerce')
    else:
        df_m[T2_var] = df_m[T2_var].values + 273.16
        df_m[T2_var] = df_m[T2_var].apply(pd.to_numeric, errors='coerce')

    if np.nanmax(df_m[T2_var]) > 373.16:
        print('Maximum temperature is: %s K please check the input temperature' % (np.nanmax(T2)))
        sys.exit()
    elif np.nanmin(df_m[T2_var]) < 173.16:
        print('Minimum temperature is: %s K please check the input temperature' % (np.nanmin(T2)))
        sys.exit()
    
    df_m[RH2_var] = df_m[RH2_var].apply(pd.to_numeric, errors='coerce')
    df_m[U2_var] = df_m[U2_var].apply(pd.to_numeric, errors='coerce')
    df_m[G_var] = df_m[G_var].apply(pd.to_numeric, errors='coerce')
    df_m[PRES_var] = df_m[PRES_var].apply(pd.to_numeric, errors='coerce')
    
    if (RRR_var in df_m):
        df_m[RRR_var] = df_m[RRR_var].apply(pd.to_numeric, errors='coerce')

    if (PRES_var not in df_m):
        df_m[PRES_var] = 660.00

    if (LWin_var not in df_m and N_var not in df_m):
        print("ERROR no longwave incoming or cloud cover data")
        sys.exit()

    elif (LWin_var in df_m):
        df_m[LWin_var] = df_m[LWin_var].apply(pd.to_numeric, errors='coerce')

    elif (N_var in df_m):
        df_m[N_var] = df_m[N_var].apply(pd.to_numeric, errors='coerce')

    if (SNOWFALL_var in df_m):
        df_m[SNOWFALL_var] = df_m[SNOWFALL_var].apply(pd.to_numeric, errors='coerce')

    #----------------------------------------------------------
    # define lapse rates for T2 and RH, weight T, cal U2_mean
    #----------------------------------------------------------
    
    diffAlt = stationAlt_m - stationAlt_v
    diffAlt_mean_m_v = stationAlt_v + diffAlt/2 
    
    lapse_T = (df_m[T2_var].values - df_v[T2_var].values) / diffAlt
    if (T2_weighted):    
        T2_mean_weighted = ((2*df_m[T2_var].values + df_v[T2_var].values) / 3)
    else:
        T2_mean_weighted = np.nan

    lapse_RH = (df_m[RH2_var].values - df_v[RH2_var].values) / diffAlt
    
    if (U2_constant):
        U2_mean_aws = np.full(len(df_v), U2_const)
    else:
        U2_mean_aws = abs((df_m[U2_var].values + df_v[U2_var].values) / 2)
    
    #------------------------------------
    # print which weather station should be used as initial value
    #------------------------------------
    
    if (intialValley):
        df = df_v
        stationAlt = stationAlt_v
        print('weather station used for calculation: %s' % (stationName_valley))
    else:
        df = df_m
        stationAlt = stationAlt_m
        print('weather station used for calculation: %s' % (stationName_mountain))

    #-----------------------------------
    # Get values from file
    #-----------------------------------
    
    if (T2_weighted):
        T2 = T2_mean_weighted + (hgt - diffAlt_mean_m_v) * lapse_T + T_const			# Temperature
    else:
        T2 = df[T2_var].values + (hgt - stationAlt) * lapse_T + T_const

    RH2 = df[RH2_var].values + (hgt - stationAlt) * lapse_RH					# Relative humidity
    U2 = U2_mean_aws										# Wind velocity
    G = df[G_var].values          # Incoming shortwave radiation
 
    SLP = df[PRES_var].values / np.power((1 - (0.0065 * stationAlt) / (288.15)), 5.255)
    PRES = SLP * np.power((1 - (0.0065 * hgt)/(288.15)), 5.22)				# Pressure

    if (RRR_var in df):
        RRR = df[RRR_var].values       # Precipitation
        RRR_additional = np.zeros(len(df[RRR_var]))

        # add additional precipitation
        RRR_additional = ((RRR/100)* RRR_additional_in_percentage)
        RRR = RRR + RRR_additional
        RRR = np.where(RRR == 0.0, 0, np.maximum(RRR + (hgt - stationAlt_RRR_mean) * lapse_RRR, 0)) 

    if(SNOWFALL_var in df):
        SNOWFALL = np.maximum(df[SNOWFALL_var].values + (hgt-stationAlt) * lapse_SNOWFALL, 0)   # SNOWFALL

    if(LWin_var in df):
        LW = df[LWin_var].values                                                                # Incoming longwave radiation

    if(N_var in df):
        N = df[N_var].values                                                                    # Cloud cover fraction

    # Change aspect to south==0, east==negative, west==positive
    if (static_file):
        aspect = ds['ASPECT'].values - 180.0
        ds['ASPECT'] = aspect
        
        # Auxiliary variables
        mask = ds.MASK.values
        slope = ds.SLOPE.values
        aspect = ds.ASPECT.values
        mask_artificial_snow = ds.MASK_ARTIFICIAL_SNOW.values
    else:
        # Auxiliary variables
        mask = 1 
        slope = 0
        aspect = 0 
        mask_artificial_snow = 0

    #-----------------------------------
    # Check bounds for relative humidity 
    #-----------------------------------
    RH2[RH2 > 100.0] = 100.0
    RH2[RH2 < 0.0] = 0.1

    #-----------------------------------
    # Add variables to file 
    #-----------------------------------
    add_variable_along_point(ds, hgt, 'HGT', 'm', 'Elevation')
    add_variable_along_point(ds, aspect, 'ASPECT', 'degrees', 'Aspect of slope')
    add_variable_along_point(ds, slope, 'SLOPE', 'degrees', 'Terrain slope')
    add_variable_along_point(ds, mask, 'MASK', 'boolean', 'Glacier mask')
    add_variable_along_point(ds, mask_artificial_snow, 'MASK_ARTIFICIAL_SNOW', 'boolean', 'Artificial snow on glacier')
    add_variable_along_timelatlon_point(ds, T2, 'T2', 'K', 'Temperature at 2 m')
    add_variable_along_timelatlon_point(ds, RH2, 'RH2', '%', 'Relative humidity at 2 m')
    add_variable_along_timelatlon_point(ds, U2, 'U2', 'm s\u207b\xb9', 'Wind velocity at 2 m')
    add_variable_along_timelatlon_point(ds, G, 'G', 'W m\u207b\xb2', 'Incoming shortwave radiation')
    add_variable_along_timelatlon_point(ds, PRES, 'PRES', 'hPa', 'Atmospheric Pressure')
    
    if (RRR_var in df):
        add_variable_along_timelatlon_point(ds, RRR, 'RRR', 'mm', 'Total precipitation (liquid+solid)')
    
    if(SNOWFALL_var in df):
        add_variable_along_timelatlon_point(ds, SNOWFALL, 'SNOWFALL', 'm', 'Snowfall')

    if(LWin_var in df):
        add_variable_along_timelatlon_point(ds, LW, 'LWin', 'W m\u207b\xb2', 'Incoming longwave radiation')

    if(N_var in df):
        add_variable_along_timelatlon_point(ds, N, 'N', '%', 'Cloud cover fraction')

    #-----------------------------------
    # Write file to disc 
    #-----------------------------------
    check_for_nan_point(ds)
    ds.to_netcdf(cosipy_file)

    print('Input file created \n')
    print('-------------------------------------------')

    #-----------------------------------
    # Do some checks
    #-----------------------------------
    check(ds.T2,316.16,223.16)
    check(ds.RH2,100.0,0.0)
    check(ds.U2, 50.0, 0.0)
    check(ds.G,1600.0,0.0)
    check(ds.PRES,1080.0,200.0)

    if(RRR_var in df):
        check(ds.RRR,20.0,0.0)

    if (SNOWFALL_var in df):
        check(ds.SNOWFALL, 0.05, 0.0)

    if (LWin_var in df):
        check(ds.LWin, 400, 0.0)

    if (N_var in df):
        check(ds.N, 1.0, 0.0)

def create_2D_input(cs_file_valley, cs_file_mountain, cosipy_file, static_file, start_date, end_date, x0=None, x1=None, y0=None, y1=None):
    """ This function creates an input dataset from an offered csv file with input point data
        Here you need to define how to interpolate the data.

        Please note, there should be only one header line in the file.

        Latest update: 
        Tobias Sauter 07.07.2019
	Anselm 01.07.2020

	Christine Seupel 21.06.2021
	   Change this function to use data of 2 weather stations:
           aws_valley and aws_mountain
	"""

    print('-------------------------------------------')
    print('Create input \n')
    print('Read input file aws valley %s' % (cs_file_valley))
    print('Read input file aws mountain %s' % (cs_file_mountain))

    #-----------------------------------
    # Read data
    #-----------------------------------
    date_parser = lambda x: dateutil.parser.parse(x, ignoretz=True)
    df_v = pd.read_csv(cs_file_valley,
        delimiter=delimiter, index_col=['TIMESTAMP'], parse_dates=['TIMESTAMP'],
        na_values='NAN',date_parser=date_parser)
    df_m = pd.read_csv(cs_file_mountain,
        delimiter=delimiter, index_col=['TIMESTAMP'], parse_dates=['TIMESTAMP'],
        na_values='NAN',date_parser=date_parser)

    #-----------------------------------
    # Select time slice
    #-----------------------------------
    if ((start_date != None) & (end_date !=None)): 
        df_v = df_v.loc[start_date:end_date]
        df_m = df_m.loc[start_date:end_date]
        
    #-----------------------------------
    # Aggregate data to selected value
    #-----------------------------------
    if aggregate:
        if ((N_var in df_v) and (RRR_var in df_v) and (LWin_var in df_v) and (SNOWFALL_var in df_v)):
            df_v = df_v.resample(aggregation_step).agg({PRES_var:'mean', T2_var:'mean', RH2_var:'mean', G_var:'mean', U2_var:'mean', N_var:'mean',
                RRR_var:'sum', LWin_var:'mean', SNOWFALL_var:'sum'})

        elif ((N_var in df_v) and (RRR_var in df_v) and (LWin_var in df_v)):
            df_v = df_v.resample(aggregation_step).agg({PRES_var:'mean', T2_var:'mean', RH2_var:'mean', G_var:'mean', U2_var:'mean', N_var:'mean',
                RRR_var:'sum', LWin_var:'mean'})

        elif ((N_var in df_v) and (RRR_var in df_v) and (SNOWFALL_var in df_v)):
            df_v = df_v.resample(aggregation_step).agg({PRES_var:'mean', T2_var:'mean', RH2_var:'mean', G_var:'mean', U2_var:'mean', N_var:'mean',
                RRR_var:'sum', SNOWFALL_var:'sum'})

        elif ((N_var in df_v) and (LWin_var in df_v) and (SNOWFALL_var in df_v)):
            df_v = df_v.resample(aggregation_step).agg({PRES_var:'mean', T2_var:'mean', RH2_var:'mean', G_var:'mean', U2_var:'mean', N_var:'mean',
                LWin_var:'mean', SNOWFALL_var:'sum'})

        elif ((RRR_var in df_v) and (LWin_var in df_v) and (SNOWFALL_var in df_v)):
            df_v = df_v.resample(aggregation_step).agg({PRES_var:'mean', T2_var:'mean', RH2_var:'mean', G_var:'mean', U2_var:'mean', RRR_var:'sum',
                LWin_var:'mean', SNOWFALL_var:'sum'})

        elif ((N_var in df_v) and (RRR_var in df_v)):
            df_v = df_v.resample(aggregation_step).agg({PRES_var:'mean', T2_var:'mean', RH2_var:'mean', G_var:'mean', U2_var:'mean', N_var:'mean',
                RRR_var:'sum', LWin_var:'mean', SNOWFALL_var:'sum'})

        elif ((N_var in df_v) and (SNOWFALL_var in df_v)):
            df_v = df_v.resample(aggregation_step).agg({PRES_var:'mean', T2_var:'mean', RH2_var:'mean', G_var:'mean', U2_var:'mean', N_var:'mean',
                RRR_var:'sum', LWin_var:'mean', SNOWFALL_var:'sum'})

        elif ((RRR_var in df_v) and (LWin_var in df_v)):
            df_v = df_v.resample(aggregation_step).agg({PRES_var:'mean', T2_var:'mean', RH2_var:'mean', G_var:'mean', U2_var:'mean', N_var:'mean',
                RRR_var:'sum', LWin_var:'mean', SNOWFALL_var:'sum'})

        elif ((LWin_var in df_v) and (SNOWFALL_var in df_v)):
            df_v = df_v.resample(aggregation_step).agg({PRES_var:'mean', T2_var:'mean', RH2_var:'mean', G_var:'mean', U2_var:'mean', N_var:'mean',
                RRR_var:'sum', LWin_var:'mean', SNOWFALL_var:'sum'})
            
            
        if ((N_var in df_m) and (RRR_var in df_m) and (LWin_var in df_m) and (SNOWFALL_var in df_m)):
            df_m = df_m.resample(aggregation_step).agg({PRES_var:'mean', T2_var:'mean', RH2_var:'mean', G_var:'mean', U2_var:'mean', N_var:'mean',
                RRR_var:'sum', LWin_var:'mean', SNOWFALL_var:'sum'})

        elif ((N_var in df_m) and (RRR_var in df_m) and (LWin_var in df_m)):
            df_m = df_m.resample(aggregation_step).agg({PRES_var:'mean', T2_var:'mean', RH2_var:'mean', G_var:'mean', U2_var:'mean', N_var:'mean',
                RRR_var:'sum', LWin_var:'mean'})

        elif ((N_var in df_m) and (RRR_var in df_m) and (SNOWFALL_var in df_m)):
            df_m = df_m.resample(aggregation_step).agg({PRES_var:'mean', T2_var:'mean', RH2_var:'mean', G_var:'mean', U2_var:'mean', N_var:'mean',
                RRR_var:'sum', SNOWFALL_var:'sum'})

        elif ((N_var in df_m) and (LWin_var in df_m) and (SNOWFALL_var in df_m)):
            df_m = df_m.resample(aggregation_step).agg({PRES_var:'mean', T2_var:'mean', RH2_var:'mean', G_var:'mean', U2_var:'mean', N_var:'mean',
                LWin_var:'mean', SNOWFALL_var:'sum'})

        elif ((RRR_var in df_m) and (LWin_var in df_m) and (SNOWFALL_var in df_m)):
            df_m = df_m.resample(aggregation_step).agg({PRES_var:'mean', T2_var:'mean', RH2_var:'mean', G_var:'mean', U2_var:'mean', RRR_var:'sum',
                LWin_var:'mean', SNOWFALL_var:'sum'})

        elif ((N_var in df_m) and (RRR_var in df_m)):
            df_m = df_m.resample(aggregation_step).agg({PRES_var:'mean', T2_var:'mean', RH2_var:'mean', G_var:'mean', U2_var:'mean', N_var:'mean',
                RRR_var:'sum', LWin_var:'mean', SNOWFALL_var:'sum'})

        elif ((N_var in df_m) and (SNOWFALL_var in df_m)):
            df_m = df_m.resample(aggregation_step).agg({PRES_var:'mean', T2_var:'mean', RH2_var:'mean', G_var:'mean', U2_var:'mean', N_var:'mean',
                RRR_var:'sum', LWin_var:'mean', SNOWFALL_var:'sum'})

        elif ((RRR_var in df_m) and (LWin_var in df_m)):
            df_m = df_m.resample(aggregation_step).agg({PRES_var:'mean', T2_var:'mean', RH2_var:'mean', G_var:'mean', U2_var:'mean', N_var:'mean',
                RRR_var:'sum', LWin_var:'mean', SNOWFALL_var:'sum'})

        elif ((LWin_var in df_m) and (SNOWFALL_var in df_m)):
            df_m = df_m.resample(aggregation_step).agg({PRES_var:'mean', T2_var:'mean', RH2_var:'mean', G_var:'mean', U2_var:'mean', N_var:'mean',
                RRR_var:'sum', LWin_var:'mean', SNOWFALL_var:'sum'})
        

    #-----------------------------------
    # Load static data
    #-----------------------------------
    print('Read static file %s \n' % (static_file))
    ds = xr.open_dataset(static_file)

    #-----------------------------------
    # Create subset
    #-----------------------------------
    ds = ds.sel(lat=slice(y0,y1), lon=slice(x0,x1))

    if WRF:
        dso = xr.Dataset()
        x, y = np.meshgrid(ds.lon, ds.lat)
        dso.coords['time'] = (('time'), df.index.values)
        dso.coords['lat'] = (('south_north','west_east'), y)
        dso.coords['lon'] = (('south_north','west_east'), x)

    else:
        dso = ds    
        dso.coords['time'] = df_v.index.values
        dso.coords['time'] = df_m.index.values
        
    #-----------------------------------
    # Order variables aws_valley
    #-----------------------------------

    if (in_K):
        df_v[T2_var] = df_v[T2_var].apply(pd.to_numeric, errors='coerce')
    else:
        df_v[T2_var] = df_v[T2_var].values + 273.16
        df_v[T2_var] = df_v[T2_var].apply(pd.to_numeric, errors='coerce')

    if np.nanmax(df_v[T2_var]) > 373.16:
        print('Maximum temperature is: %s K please check the input temperature' % (np.nanmax(T2)))
        sys.exit()
    elif np.nanmin(df_v[T2_var]) < 173.16:
        print('Minimum temperature is: %s K please check the input temperature' % (np.nanmin(T2)))
        sys.exit()

    df_v[RH2_var] = df_v[RH2_var].apply(pd.to_numeric, errors='coerce')
    df_v[U2_var] = df_v[U2_var].apply(pd.to_numeric, errors='coerce')
    df_v[G_var] = df_v[G_var].apply(pd.to_numeric, errors='coerce')
    df_v[PRES_var] = df_v[PRES_var].apply(pd.to_numeric, errors='coerce')

    if (PRES_var not in df_v):
        df_v[PRES_var] = 660.00

    if (RRR_var in df_v):
        df_v[RRR_var] = df_v[RRR_var].apply(pd.to_numeric, errors='coerce')

    if (LWin_var not in df_v and N_var not in df_v):
        print("ERROR no longwave incoming or cloud cover data")
        sys.exit()

    elif (LWin_var in df_v):
        df_v[LWin_var] = df_v[LWin_var].apply(pd.to_numeric, errors='coerce')
        print("LWin in data")

    elif (N_var in df_v):
        df_v[N_var] = df_v[N_var].apply(pd.to_numeric, errors='coerce')

    if (SNOWFALL_var in df_v):
        df_v[SNOWFALL_var] = df_v[SNOWFALL_var].apply(pd.to_numeric, errors='coerce')

    #-----------------------------------
    # Order variables aws_mountain
    #-----------------------------------

    if (in_K):
        df_m[T2_var] = df_m[T2_var].apply(pd.to_numeric, errors='coerce')
    else:
        df_m[T2_var] = df_m[T2_var].values + 273.16
        df_m[T2_var] = df_m[T2_var].apply(pd.to_numeric, errors='coerce')

    if np.nanmax(df_m[T2_var]) > 373.16:
        print('Maximum temperature is: %s K please check the input temperature' % (np.nanmax(T2)))
        sys.exit()
    elif np.nanmin(df_m[T2_var]) < 173.16:
        print('Minimum temperature is: %s K please check the input temperature' % (np.nanmin(T2)))
        sys.exit()

    df_m[RH2_var] = df_m[RH2_var].apply(pd.to_numeric, errors='coerce')
    df_m[U2_var] = df_m[U2_var].apply(pd.to_numeric, errors='coerce')
    df_m[G_var] = df_m[G_var].apply(pd.to_numeric, errors='coerce')
    df_m[PRES_var] = df_m[PRES_var].apply(pd.to_numeric, errors='coerce')

    if (PRES_var not in df_m):
        df_m[PRES_var] = 660.00

    if (RRR_var in df_m):
        df_m[RRR_var] = df_m[RRR_var].apply(pd.to_numeric, errors='coerce')

    if (LWin_var not in df_m and N_var not in df_m):
        print("ERROR no longwave incoming or cloud cover data")
        sys.exit()

    elif (LWin_var in df_m):
        df_m[LWin_var] = df_m[LWin_var].apply(pd.to_numeric, errors='coerce')
        print("LWin in data")

    elif (N_var in df_m):
        df_m[N_var] = df_m[N_var].apply(pd.to_numeric, errors='coerce')

    if (SNOWFALL_var in df_m):
        df_m[SNOWFALL_var] = df_m[SNOWFALL_var].apply(pd.to_numeric, errors='coerce')
    
    #---------------------------------------------------------
    # define lapse rates for T2 and RH, weight T, cal U2_mean
    #---------------------------------------------------------
    
    diffAlt = stationAlt_m - stationAlt_v
    diffAlt_mean_m_v = stationAlt_v + diffAlt/2 
    
    lapse_T = (df_m[T2_var] - df_v[T2_var]) / diffAlt
    if (T2_weighted):    
        T2_mean_weighted = ((2*df_m[T2_var].values + df_v[T2_var].values) / 3)
    else:
        T2_mean_weighted = np.nan    

    lapse_RH = (df_m[RH2_var] - df_v[RH2_var]) / diffAlt

    if (U2_constant):
        U2_mean_aws = np.full(len(df_v), U2_const)
    else:
        U2_mean_aws = abs((df_m[U2_var].values + df_v[U2_var].values) / 2)
    
    #------------------------------------
    # print which weather station should be used as initial value
    #------------------------------------
    
    if (intialValley):
        df = df_v
        stationAlt = stationAlt_v
        print('weather station used for calculation: %s' % (stationName_valley))
    else:
        df = df_m
        stationAlt = stationAlt_m
        print('weather station used for calculation: %s' % (stationName_mountain))


    #-----------------------------------
    # Get values from file
    #-----------------------------------
    T2 = df[T2_var]         # Temperature
    RH2 = df[RH2_var]       # Relative humidity
    U2 = df[U2_var]         # Wind velocity
    G = df[G_var]           # Incoming shortwave radiation
    PRES = df[PRES_var]     # Pressure

    #-----------------------------------
    # Create numpy arrays for the 2D fields
    #-----------------------------------
    T_interp = np.zeros([len(dso.time), len(ds.lat), len(ds.lon)])
    RH_interp = np.zeros([len(dso.time), len(ds.lat), len(ds.lon)])
    U_interp = np.zeros([len(dso.time), len(ds.lat), len(ds.lon)])
    G_interp = np.full([len(dso.time), len(ds.lat), len(ds.lon)], np.nan)
    P_interp = np.zeros([len(dso.time), len(ds.lat), len(ds.lon)])

    if (RRR_var in df):
        RRR = df[RRR_var]       # Precipitation
        RRR_additional = np.zeros(len(df[RRR_var]))
	
        # add additional precipitation
        RRR_additional = ((RRR/100)* RRR_additional_in_percentage)
        RRR = RRR + RRR_additional
        RRR_interp = np.zeros([len(dso.time), len(ds.lat), len(ds.lon)])

    if(SNOWFALL_var in df):
        SNOWFALL = df[SNOWFALL_var]      # Incoming longwave radiation
        SNOWFALL_interp = np.zeros([len(dso.time), len(ds.lat), len(ds.lon)])

    if(LWin_var in df):
        LW = df[LWin_var]      # Incoming longwave radiation
        LW_interp = np.zeros([len(dso.time), len(ds.lat), len(ds.lon)])

    if(N_var in df):
        N = df[N_var]        # Cloud cover fraction
        N_interp = np.zeros([len(dso.time), len(ds.lat), len(ds.lon)])

    #-----------------------------------
    # Interpolate point data to grid 
    #-----------------------------------
    print('Interpolate CR file to grid')
   
    # Interpolate data (T, RH, RRR, U)  to grid using lapse rates
    for t in range(len(dso.time)):
        
        if (T2_weighted):
            T_interp[t,:,:] = (T2_mean_weighted[t])+(ds.HGT.values-diffAlt_mean_m_v)*lapse_T[t] + T_const
        else:
            T_interp[t,:,:] = (T2[t]) + (ds.HGT.values-stationAlt)*lapse_T[t]+T_const

        RH_interp[t,:,:] = RH2[t] + (ds.HGT.values-stationAlt)*lapse_RH[t]
        U_interp[t,:,:] = U2_mean_aws[t]

        # Interpolate pressure using the barometric equation
        SLP = PRES[t]/np.power((1-(0.0065*stationAlt)/(288.15)), 5.255)
        P_interp[t,:,:] = SLP * np.power((1-(0.0065*ds.HGT.values)/(288.15)), 5.255)

        if (RRR_var in df):
            if RRR[t] == 0:
                RRR_interp[t,:,:] = 0.0
            else:
                RRR_interp[t,:,:] = np.maximum(RRR[t] + (ds.HGT.values-stationAlt_RRR_mean)*lapse_RRR, 0.0)

        if (SNOWFALL_var in df):
            SNOWFALL_interp[t, :, :] = SNOWFALL[t] + (ds.HGT.values-stationAlt)*lapse_SNOWFALL

        if(LWin_var in df):
            LW_interp[t,:,:] = LW[t]
        
        if(N_var in df):
            N_interp[t,:,:] = N[t]

    # Change aspect to south==0, east==negative, west==positive
    aspect = ds['ASPECT'].values - 180.0
    ds['ASPECT'] = (('lat','lon'),aspect)

    print(('Number of glacier cells: %i') % (np.count_nonzero(~np.isnan(ds['MASK'].values))))
    print(('Number of glacier cells: %i') % (np.nansum(ds['MASK'].values)))

    # Auxiliary variables
    mask = ds.MASK.values
    slope = ds.SLOPE.values
    aspect = ds.ASPECT.values
    lats = ds.lat.values
    lons = ds.lon.values
    sw = G.values
    mask_artificial_snow = ds.MASK_ARTIFICIAL_SNOW.values

    #-----------------------------------
    # Run radiation module 
    #-----------------------------------
    if radiationModule:
        print('Run the radiation module')
    else:
        print('No radiation module used')

    for t in range(len(dso.time)):
        doy = df.index[t].dayofyear
        hour = df.index[t].hour
        for i in range(len(ds.lat)):
            for j in range(len(ds.lon)):
                if ((mask[i, j] == 1) | (mask_artificial_snow[i, j] == 1)):
                    if radiationModule:
                        G_interp[t,i,j] = np.maximum(0.0, correctRadiation(lats[i],lons[j], timezone_lon, doy, hour, slope[i,j], aspect[i,j], sw[t], zeni_thld))
                    else:
                        G_interp[t,i,j] = sw[t]

    #-----------------------------------
    # Check bounds for relative humidity 
    #-----------------------------------
    RH_interp[RH_interp > 100.0] = 100.0
    RH_interp[RH_interp < 0.0] = 0.1

    #-----------------------------------
    # Add variables to file 
    #-----------------------------------
    add_variable_along_latlon(dso, ds.HGT.values, 'HGT', 'm', 'Elevation')
    add_variable_along_latlon(dso, ds.ASPECT.values, 'ASPECT', 'degrees', 'Aspect of slope')
    add_variable_along_latlon(dso, ds.SLOPE.values, 'SLOPE', 'degrees', 'Terrain slope')
    add_variable_along_latlon(dso, ds.MASK.values, 'MASK', 'boolean', 'Glacier mask')
    add_variable_along_latlon(dso, ds.MASK_ARTIFICIAL_SNOW.values, 'MASK_ARTIFICIAL_SNOW', 'boolean', 'Artificial snow on glacier')
    add_variable_along_timelatlon(dso, T_interp, 'T2', 'K', 'Temperature at 2 m')
    add_variable_along_timelatlon(dso, RH_interp, 'RH2', '%', 'Relative humidity at 2 m')
    add_variable_along_timelatlon(dso, U_interp, 'U2', 'm s\u207b\xb9', 'Wind velocity at 2 m')
    add_variable_along_timelatlon(dso, G_interp, 'G', 'W m\u207b\xb2', 'Incoming shortwave radiation')
    add_variable_along_timelatlon(dso, P_interp, 'PRES', 'hPa', 'Atmospheric Pressure')

    if (RRR_var in df):
        add_variable_along_timelatlon(dso, RRR_interp, 'RRR', 'mm', 'Total precipitation (liquid+solid)')

    if(SNOWFALL_var in df):
        add_variable_along_timelatlon(dso, SNOWFALL_interp, 'SNOWFALL', 'm', 'Snowfall')

    if(LWin_var in df):
        add_variable_along_timelatlon(dso, LW_interp, 'LWin', 'W m\u207b\xb2', 'Incoming longwave radiation')

    if(N_var in df):
        add_variable_along_timelatlon(dso, N_interp, 'N', '%', 'Cloud cover fraction')

    encoding = dict()
    # for var in IO.get_result().data_vars:
    #     dataMin = IO.get_result()[var].min(skipna=True).values
    #     dataMax = IO.get_result()[var].max(skipna=True).values
    #
    #     dtype = 'int16'
    #     FillValue = -9999
    #     scale_factor, add_offset = compute_scale_and_offset(dataMin, dataMax, 16)
    #     encoding[var] = dict(zlib=True, complevel=2, dtype=dtype, scale_factor=scale_factor, add_offset=add_offset, _FillValue=FillValue)

    #-----------------------------------
    # Write file to disc 
    #-----------------------------------
    #dso.to_netcdf(cosipy_file, encoding=encoding)
    check_for_nan(dso)
    dso.to_netcdf(cosipy_file)

    print('Input file created \n')
    print('-------------------------------------------')

    #-----------------------------------
    # Do some checks
    #-----------------------------------
    check(dso.T2,316.16,223.16)
    check(dso.RH2,100.0,0.0)
    check(dso.U2, 50.0, 0.0)
    check(dso.G,1600.0,0.0)
    check(dso.PRES,1080.0,200.0)

    if(RRR_var in df):
        check(dso.RRR,25.0,0.0)

    if (SNOWFALL_var in df):
        check(dso.SNOWFALL, 0.05, 0.0)

    if (LWin_var in df):
        check(dso.LWin, 400, 0.0)

    if (N_var in df):
        check(dso.N, 1.0, 0.0)

def add_variable_along_timelatlon(ds, var, name, units, long_name):
    """ This function adds missing variables to the DATA class """
    if WRF:
         ds[name] = (('time','south_north','west_east'), var)	
    else:
        ds[name] = (('time','lat','lon'), var)
    ds[name].attrs['units'] = units
    ds[name].attrs['long_name'] = long_name
    return ds

def add_variable_along_latlon(ds, var, name, units, long_name):
    """ This function self.adds missing variables to the self.DATA class """
    if WRF: 
        ds[name] = (('south_north','west_east'), var)
    else:
        ds[name] = (('lat','lon'), var)
    ds[name].attrs['units'] = units
    ds[name].attrs['long_name'] = long_name
    ds[name].encoding['_FillValue'] = -9999
    return ds

def add_variable_along_timelatlon_point(ds, var, name, units, long_name):
    """ This function adds missing variables to the DATA class """
    ds[name] = (('time','lat','lon'), np.reshape(var,(len(var),1,1)))
    ds[name].attrs['units'] = units
    ds[name].attrs['long_name'] = long_name
    return ds

def add_variable_along_point(ds, var, name, units, long_name):
    """ This function adds missing variables to the DATA class """
    ds[name] = (('lat','lon'), np.reshape(var,(1,1)))
    ds[name].attrs['units'] = units
    ds[name].attrs['long_name'] = long_name
    return ds

def check(field, max, min):
    '''Check the validity of the input data '''
    if np.nanmax(field) > max or np.nanmin(field) < min:
        print('\n\nWARNING! Please check the data, its seems they are out of a reasonable range %s MAX: %.2f MIN: %.2f \n' % (str.capitalize(field.name), np.nanmax(field), np.nanmin(field)))

def nan_in_dataset(dataset, x, y):
    for variable in dataset.data_vars:
        if variable != "MASK_ARTIFICIAL_SNOW":
            if np.isnan(dataset[variable].isel(lat=y, lon=x)).any():
                return True

def check_for_nan(dataset: xr.Dataset) -> None:
    if WRF is True:
        for y,x in product(range(dataset.dims['south_north']),range(dataset.dims['west_east'])):
            if dataset.MASK.sel(south_north=y, west_east=x) == 1:
                if nan_in_dataset(dataset=dataset, x=x , y=y):
                    print('ERROR!!!!!!!!!!! There are NaNs in the dataset')
                    sys.exit()
    else:
        for y, x in product(range(dataset.dims['lat']), range(dataset.dims['lon'])):
            if dataset.MASK.isel(lat=y, lon=x) == 1:
                if nan_in_dataset(dataset=dataset, x=x, y=y):
                    print('ERROR!!!!!!!!!!! There are NaNs in the dataset')
                    sys.exit()

def check_for_nan_point(ds):
    if np.isnan(ds.to_array()).any():
        print('ERROR!!!!!!!!!!! There are NaNs in the dataset')
        sys.exit()

def compute_scale_and_offset(min, max, n):
    # stretch/compress data to the available packed range
    scale_factor = (max - min) / (2 ** n - 1)
    # translate the range to be symmetric about zero
    add_offset = min + 2 ** (n - 1) * scale_factor
    return (scale_factor, add_offset)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Create 2D input file from csv file.')
    parser.add_argument('-cv', '-csv_file_valley', dest='csv_file_valley', help='Csv file valley weather station (see readme for file convention)')
    parser.add_argument('-cm', '-csv_file_mountain', dest='csv_file_mountain', help='Csv file mountain weather station (see readme for file convention)')
    parser.add_argument('-o', '-cosipy_file', dest='cosipy_file', help='Name of the resulting COSIPY file')
    parser.add_argument('-s', '-static_file', dest='static_file', help='Static file containing DEM, Slope etc.')
    parser.add_argument('-b', '-start_date', dest='start_date', help='Start date')
    parser.add_argument('-e', '-end_date', dest='end_date', help='End date')
    parser.add_argument('-xl', '-xl', dest='xl', type=float, const=None, help='left longitude value of the subset')
    parser.add_argument('-xr', '-xr', dest='xr', type=float, const=None, help='right longitude value of the subset')
    parser.add_argument('-yl', '-yl', dest='yl', type=float, const=None, help='lower latitude value of the subset')
    parser.add_argument('-yu', '-yu', dest='yu', type=float, const=None, help='upper latitude value of the subset')

    args = parser.parse_args()
    if point_model:
        create_1D_input(args.csv_file_valley, args.csv_file_mountain, args.cosipy_file, args.static_file, args.start_date, args.end_date) 
    else:
        create_2D_input(args.csv_file_valley, args.csv_file_mountain, args.cosipy_file, args.static_file, args.start_date, args.end_date, args.xl, args.xr, args.yl, args.yu) 
