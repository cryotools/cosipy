"""
This file reads the input data (model forcing) and writes the output to
netcdf file. There are the `create_1D_input` (point model) and
`create_2D_input` (distributed simulations) function.

The 1D input function works without a static file, in which the static
variables are created.

For both cases, lapse rates can be determined in a .toml configuration
file. See the sample `utilities_config.toml` for more information.

To run from source:
``python -m cosipy.utilities.aws2cosipy.aws2cosipy``

Otherwise, use the entry point:
``cosipy-aws2cosipy``
"""
import argparse
import sys
from itertools import product

import dateutil
import netCDF4 as nc
import numpy as np
import pandas as pd
import xarray as xr

import cosipy.modules.radCor as mod_radCor
from cosipy.utilities.config_utils import UtilitiesConfig

_args = None
_cfg = None


def create_1D_input(cs_file, cosipy_file, static_file, start_date, end_date):
    """ This function creates an input dataset from an offered csv file with input point data
        Here you need to define how to interpolate the data.

        Please note, there should be only one header line in the file.

        Latest update: 
            Tobias Sauter 07.07.2019
	        Anselm 04.07.2020
    """

    print('-------------------------------------------')
    print('Create input \n')
    print('Read input file %s' % (cs_file))

    #-----------------------------------
    # Read data
    #-----------------------------------
    date_parser = lambda x: dateutil.parser.parse(x, ignoretz=True)
    df = pd.read_csv(cs_file,
       delimiter=_cfg.coords["delimiter"], index_col=['TIMESTAMP'],
        parse_dates=['TIMESTAMP'], na_values='NAN',date_parser=date_parser)

    print(df[pd.isnull(df).any(axis=1)])
    df = df.fillna(method='ffill')
    print(df[pd.isnull(df).any(axis=1)])

    if (_cfg.names["LWin_var"] not in df):
        df[_cfg.names["LWin_var"]] = np.nan
    if (_cfg.names["N_var"] not in df):
        df[_cfg.names["N_var"]] = np.nan
    if (_cfg.names["SNOWFALL_var"] not in df):
        df[_cfg.names["SNOWFALL_var"]] = np.nan
 
    # Improved function to sum dataframe columns which contain nan's
    def nansumwrapper(a, **kw_args):
        if np.isnan(a).all():
            return np.nan
        else:
            return np.nansum(a, **kw_args)

    col_list = [_cfg.names["T2_var"],_cfg.names["RH2_var"],_cfg.names["U2_var"],_cfg.names["G_var"],_cfg.names["RRR_var"],_cfg.names["PRES_var"],_cfg.names["LWin_var"],_cfg.names["N_var"],_cfg.names["SNOWFALL_var"]]
    df = df[col_list]
    
    df = df.resample('1H').agg({_cfg.names["T2_var"]:np.mean, _cfg.names["RH2_var"]:np.mean, _cfg.names["U2_var"]:np.mean, _cfg.names["G_var"]:np.mean, _cfg.names["PRES_var"]:np.mean, _cfg.names["RRR_var"]:nansumwrapper, _cfg.names["LWin_var"]:np.mean, _cfg.names["N_var"]:np.mean, _cfg.names["SNOWFALL_var"]:nansumwrapper})
    df = df.dropna(axis=1,how='all')
    print(df.head())

    #-----------------------------------
    # Select time slice
    #-----------------------------------
    if ((start_date != None) & (end_date !=None)): 
        df = df.loc[start_date:end_date]

    #-----------------------------------
    # Load static data
    #-----------------------------------

    if _cfg.coords["WRF"]:
        ds = xr.Dataset()
        lon, lat = np.meshgrid(_cfg.points["plon"], _cfg.points["plat"])
        ds.coords['lat'] = (('south_north', 'west_east'), lon)
        ds.coords['lon'] = (('south_north', 'west_east'), lat)

    else:
        if (static_file):
            print('Read static file %s \n' % (static_file))
            ds = xr.open_dataset(static_file)
            ds = ds.sel(lat=_cfg.points["plat"],lon=_cfg.points["plon"],method='nearest')
            ds.coords['lon'] = np.array([ds.lon.values])
            ds.coords['lat'] = np.array([ds.lat.values])

        else:
            ds = xr.Dataset()
            ds.coords['lon'] = np.array([_cfg.points["plon"]])
            ds.coords['lat'] = np.array([_cfg.points["plat"]])

        ds.lon.attrs['standard_name'] = 'lon'
        ds.lon.attrs['long_name'] = 'longitude'
        ds.lon.attrs['units'] = 'degrees_east'

        ds.coords['lat'] = np.array([_cfg.points["plat"]])
        ds.lat.attrs['standard_name'] = 'lat'
        ds.lat.attrs['long_name'] = 'latitude'
        ds.lat.attrs['units'] = 'degrees_north'

    ds.coords['time'] = (('time'), df.index.values)

    #-----------------------------------
    # Order variables
    #-----------------------------------
    df[_cfg.names["T2_var"]] = df[_cfg.names["T2_var"]].apply(pd.to_numeric, errors='coerce')
    df[_cfg.names["RH2_var"]] = df[_cfg.names["RH2_var"]].apply(pd.to_numeric, errors='coerce')
    df[_cfg.names["U2_var"]] = df[_cfg.names["U2_var"]].apply(pd.to_numeric, errors='coerce')
    df[_cfg.names["G_var"]] = df[_cfg.names["G_var"]].apply(pd.to_numeric, errors='coerce')
    df[_cfg.names["PRES_var"]] = df[_cfg.names["PRES_var"]].apply(pd.to_numeric, errors='coerce')
    
    if (_cfg.names["RRR_var"] in df):
        df[_cfg.names["RRR_var"]] = df[_cfg.names["RRR_var"]].apply(pd.to_numeric, errors='coerce')

    if (_cfg.names["PRES_var"] not in df):
        df[_cfg.names["PRES_var"]] = 660.00

    if (_cfg.names["LWin_var"] not in df and _cfg.names["N_var"] not in df):
        print("ERROR no longwave incoming or cloud cover data")
        sys.exit()

    elif (_cfg.names["LWin_var"] in df):
        df[_cfg.names["LWin_var"]] = df[_cfg.names["LWin_var"]].apply(pd.to_numeric, errors='coerce')

    elif (_cfg.names["N_var"] in df):
        df[_cfg.names["N_var"]] = df[_cfg.names["N_var"]].apply(pd.to_numeric, errors='coerce')

    if (_cfg.names["SNOWFALL_var"] in df):
        df[_cfg.names["SNOWFALL_var"]] = df[_cfg.names["SNOWFALL_var"]].apply(pd.to_numeric, errors='coerce')

    #-----------------------------------
    # Get values from file
    #-----------------------------------
    if (_cfg.names["in_K"]):
        T2 = df[_cfg.names["T2_var"]].values + (_cfg.points["hgt"] - _cfg.station["stationAlt"]) * _cfg.lapse["lapse_T"]                                   # Temperature
    else:
        T2 = df[_cfg.names["T2_var"]].values + 273.16 + (_cfg.points["hgt"] - _cfg.station["stationAlt"]) * _cfg.lapse["lapse_T"]

    if np.nanmax(T2) > 373.16:
        print('Maximum temperature is: %s K please check the input temperature' % (np.nanmax(T2)))
        sys.exit()
    elif np.nanmin(T2) < 173.16:
        print('Minimum temperature is: %s K please check the input temperature' % (np.nanmin(T2)))
        sys.exit()

    RH2 = df[_cfg.names["RH2_var"]].values + (_cfg.points["hgt"] - _cfg.station["stationAlt"]) * _cfg.lapse["lapse_RH"]  # Relative humidity
    U2 = df[_cfg.names["U2_var"]].values  # Wind velocity
    G = df[_cfg.names["G_var"]].values  # Incoming shortwave radiation

    SLP = df[_cfg.names["PRES_var"]].values / np.power((1 - (0.0065 * _cfg.station["stationAlt"]) / (288.15)), 5.255)
    PRES = SLP * np.power((1 - (0.0065 * _cfg.points["hgt"])/(288.15)), 5.22)  # Pressure


    if (_cfg.names["RRR_var"] in df):
        RRR = np.maximum(df[_cfg.names["RRR_var"]].values + (_cfg.points["hgt"] - _cfg.station["stationAlt"]) * _cfg.lapse["lapse_RRR"], 0)  # Precipitation

    if(_cfg.names["SNOWFALL_var"] in df):
        SNOWFALL = np.maximum(df[_cfg.names["SNOWFALL_var"]].values + (_cfg.points["hgt"]-_cfg.station["stationAlt"]) * _cfg.lapse["lapse_SNOWFALL"], 0)  # SNOWFALL

    if(_cfg.names["LWin_var"] in df):
        LW = df[_cfg.names["LWin_var"]].values  # Incoming longwave radiation

    if(_cfg.names["N_var"] in df):
        N = df[_cfg.names["N_var"]].values  # Cloud cover fraction

    # Change aspect to south==0, east==negative, west==positive
    if (static_file):
        aspect = ds['ASPECT'].values - 180.0
        ds['ASPECT'] = aspect
        
        # Auxiliary variables
        mask = ds.MASK.values
        slope = ds.SLOPE.values
        aspect = ds.ASPECT.values
    else:
        # Auxiliary variables
        mask = 1 
        slope = 0
        aspect = 0

    #-----------------------------------
    # Check bounds for relative humidity 
    #-----------------------------------
    RH2[RH2 > 100.0] = 100.0
    RH2[RH2 < 0.0] = 0.1

    #-----------------------------------
    # Add variables to file 
    #-----------------------------------
    add_variable_along_point(ds, _cfg.points["hgt"], 'HGT', 'm', 'Elevation')
    add_variable_along_point(ds, aspect, 'ASPECT', 'degrees', 'Aspect of slope')
    add_variable_along_point(ds, slope, 'SLOPE', 'degrees', 'Terrain slope')
    add_variable_along_point(ds, mask, 'MASK', 'boolean', 'Glacier mask')
    add_variable_along_timelatlon_point(ds, T2, 'T2', 'K', 'Temperature at 2 m')
    add_variable_along_timelatlon_point(ds, RH2, 'RH2', '%', 'Relative humidity at 2 m')
    add_variable_along_timelatlon_point(ds, U2, 'U2', 'm s\u207b\xb9', 'Wind velocity at 2 m')
    add_variable_along_timelatlon_point(ds, G, 'G', 'W m\u207b\xb2', 'Incoming shortwave radiation')
    add_variable_along_timelatlon_point(ds, PRES, 'PRES', 'hPa', 'Atmospheric Pressure')
    
    if (_cfg.names["RRR_var"] in df):
        add_variable_along_timelatlon_point(ds, RRR, 'RRR', 'mm', 'Total precipitation (liquid+solid)')
    
    if(_cfg.names["SNOWFALL_var"] in df):
        add_variable_along_timelatlon_point(ds, SNOWFALL, 'SNOWFALL', 'm', 'Snowfall')

    if(_cfg.names["LWin_var"] in df):
        add_variable_along_timelatlon_point(ds, LW, 'LWin', 'W m\u207b\xb2', 'Incoming longwave radiation')

    if(_cfg.names["N_var"] in df):
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

    if(_cfg.names["RRR_var"] in df):
        check(ds.RRR,20.0,0.0)

    if (_cfg.names["SNOWFALL_var"] in df):
        check(ds.SNOWFALL, 0.05, 0.0)

    if (_cfg.names["LWin_var"] in df):
        check(ds.LWin, 400, 0.0)

    if (_cfg.names["N_var"] in df):
        check(ds.N, 1.0, 0.0)

def create_2D_input(cs_file, cosipy_file, static_file, start_date, end_date, x0=None, x1=None, y0=None, y1=None):
    """ This function creates an input dataset from an offered csv file with input point data
        Here you need to define how to interpolate the data.

        Please note, there should be only one header line in the file.

        Latest update: 
        Tobias Sauter 07.07.2019
	Anselm 01.07.2020
	Franziska Temme 03.08.2021
	"""

    print('-------------------------------------------')
    print('Create input \n')
    print('Read input file %s' % (cs_file))

    #-----------------------------------
    # Read data
    #-----------------------------------
    date_parser = lambda x: dateutil.parser.parse(x, ignoretz=True)
    df = pd.read_csv(cs_file, 
        delimiter=_cfg.coords["delimiter"], index_col=['TIMESTAMP'], parse_dates=['TIMESTAMP'],
        na_values='NAN',date_parser=date_parser)

    #-----------------------------------
    # Select time slice
    #-----------------------------------
    if ((start_date != None) & (end_date !=None)): 
        df = df.loc[start_date:end_date]

    #-----------------------------------
    # Aggregate data to selected value
    #-----------------------------------
    if _cfg.coords["aggregate"]:
        if ((_cfg.names["N_var"] in df) and (_cfg.names["RRR_var"] in df) and (_cfg.names["LWin_var"] in df) and (_cfg.names["SNOWFALL_var"] in df)):
            df = df.resample(_cfg.coords["aggregation_step"]).agg({_cfg.names["PRES_var"]:'mean', _cfg.names["T2_var"]:'mean', _cfg.names["RH2_var"]:'mean', _cfg.names["G_var"]:'mean', _cfg.names["U2_var"]:'mean', _cfg.names["N_var"]:'mean',
                _cfg.names["RRR_var"]:'sum', _cfg.names["LWin_var"]:'mean', _cfg.names["SNOWFALL_var"]:'sum'})

        elif ((_cfg.names["N_var"] in df) and (_cfg.names["RRR_var"] in df) and (_cfg.names["LWin_var"] in df)):
            df = df.resample(_cfg.coords["aggregation_step"]).agg({_cfg.names["PRES_var"]:'mean', _cfg.names["T2_var"]:'mean', _cfg.names["RH2_var"]:'mean', _cfg.names["G_var"]:'mean', _cfg.names["U2_var"]:'mean', _cfg.names["N_var"]:'mean',
                _cfg.names["RRR_var"]:'sum', _cfg.names["LWin_var"]:'mean'})

        elif ((_cfg.names["N_var"] in df) and (_cfg.names["RRR_var"] in df) and (_cfg.names["SNOWFALL_var"] in df)):
            df = df.resample(_cfg.coords["aggregation_step"]).agg({_cfg.names["PRES_var"]:'mean', _cfg.names["T2_var"]:'mean', _cfg.names["RH2_var"]:'mean', _cfg.names["G_var"]:'mean', _cfg.names["U2_var"]:'mean', _cfg.names["N_var"]:'mean',
                _cfg.names["RRR_var"]:'sum', _cfg.names["SNOWFALL_var"]:'sum'})

        elif ((_cfg.names["N_var"] in df) and (_cfg.names["LWin_var"] in df) and (_cfg.names["SNOWFALL_var"] in df)):
            df = df.resample(_cfg.coords["aggregation_step"]).agg({_cfg.names["PRES_var"]:'mean', _cfg.names["T2_var"]:'mean', _cfg.names["RH2_var"]:'mean', _cfg.names["G_var"]:'mean', _cfg.names["U2_var"]:'mean', _cfg.names["N_var"]:'mean',
                _cfg.names["LWin_var"]:'mean', _cfg.names["SNOWFALL_var"]:'sum'})

        elif ((_cfg.names["RRR_var"] in df) and (_cfg.names["LWin_var"] in df) and (_cfg.names["SNOWFALL_var"] in df)):
            df = df.resample(_cfg.coords["aggregation_step"]).agg({_cfg.names["PRES_var"]:'mean', _cfg.names["T2_var"]:'mean', _cfg.names["RH2_var"]:'mean', _cfg.names["G_var"]:'mean', _cfg.names["U2_var"]:'mean', _cfg.names["RRR_var"]:'sum',
                _cfg.names["LWin_var"]:'mean', _cfg.names["SNOWFALL_var"]:'sum'})

        elif ((_cfg.names["N_var"] in df) and (_cfg.names["RRR_var"] in df)):
            df = df.resample(_cfg.coords["aggregation_step"]).agg({_cfg.names["PRES_var"]:'mean', _cfg.names["T2_var"]:'mean', _cfg.names["RH2_var"]:'mean', _cfg.names["G_var"]:'mean', _cfg.names["U2_var"]:'mean', _cfg.names["N_var"]:'mean',
                _cfg.names["RRR_var"]:'sum'})

        elif ((_cfg.names["N_var"] in df) and (_cfg.names["SNOWFALL_var"] in df)):
            df = df.resample(_cfg.coords["aggregation_step"]).agg({_cfg.names["PRES_var"]:'mean', _cfg.names["T2_var"]:'mean', _cfg.names["RH2_var"]:'mean', _cfg.names["G_var"]:'mean', _cfg.names["U2_var"]:'mean', _cfg.names["N_var"]:'mean',
                _cfg.names["SNOWFALL_var"]:'sum'})

        elif ((_cfg.names["RRR_var"] in df) and (_cfg.names["LWin_var"] in df)):
            df = df.resample(_cfg.coords["aggregation_step"]).agg({_cfg.names["PRES_var"]:'mean', _cfg.names["T2_var"]:'mean', _cfg.names["RH2_var"]:'mean', _cfg.names["G_var"]:'mean', _cfg.names["U2_var"]:'mean', _cfg.names["RRR_var"]:'sum',
                _cfg.names["LWin_var"]:'mean'})

        elif ((_cfg.names["LWin_var"] in df) and (_cfg.names["SNOWFALL_var"] in df)):
            df = df.resample(_cfg.coords["aggregation_step"]).agg({_cfg.names["PRES_var"]:'mean', _cfg.names["T2_var"]:'mean', _cfg.names["RH2_var"]:'mean', _cfg.names["G_var"]:'mean', _cfg.names["U2_var"]:'mean', _cfg.names["LWin_var"]:'mean',
                _cfg.names["SNOWFALL_var"]:'sum'})

    #-----------------------------------
    # Load static data
    #-----------------------------------
    print('Read static file %s \n' % (static_file))
    ds = xr.open_dataset(static_file)

    #-----------------------------------
    # Create subset
    #-----------------------------------
    ds = ds.sel(lat=slice(y0,y1), lon=slice(x0,x1))

    if _cfg.coords["WRF"]:
        dso = xr.Dataset()
        x, y = np.meshgrid(ds.lon, ds.lat)
        dso.coords['time'] = (('time'), df.index.values)
        dso.coords['lat'] = (('south_north','west_east'), y)
        dso.coords['lon'] = (('south_north','west_east'), x)

    else:
        dso = ds    
        dso.coords['time'] = df.index.values

    #-----------------------------------
    # Order variables
    #-----------------------------------
    df[_cfg.names["T2_var"]] = df[_cfg.names["T2_var"]].apply(pd.to_numeric, errors='coerce')
    df[_cfg.names["RH2_var"]] = df[_cfg.names["RH2_var"]].apply(pd.to_numeric, errors='coerce')
    df[_cfg.names["U2_var"]] = df[_cfg.names["U2_var"]].apply(pd.to_numeric, errors='coerce')
    df[_cfg.names["G_var"]] = df[_cfg.names["G_var"]].apply(pd.to_numeric, errors='coerce')
    df[_cfg.names["PRES_var"]] = df[_cfg.names["PRES_var"]].apply(pd.to_numeric, errors='coerce')

    #if (_cfg.names["PRES_var"] not in df):
    #    df[_cfg.names["PRES_var"]] = 660.00

    if (_cfg.names["RRR_var"] in df):
        df[_cfg.names["RRR_var"]] = df[_cfg.names["RRR_var"]].apply(pd.to_numeric, errors='coerce')

    if (_cfg.names["LWin_var"] not in df and _cfg.names["N_var"] not in df):
        print("ERROR no longwave incoming or cloud cover data")
        sys.exit()

    elif (_cfg.names["LWin_var"] in df):
        df[_cfg.names["LWin_var"]] = df[_cfg.names["LWin_var"]].apply(pd.to_numeric, errors='coerce')
        print("LWin in data")

    elif (_cfg.names["N_var"] in df):
        df[_cfg.names["N_var"]] = df[_cfg.names["N_var"]].apply(pd.to_numeric, errors='coerce')

    if (_cfg.names["SNOWFALL_var"] in df):
        df[_cfg.names["SNOWFALL_var"]] = df[_cfg.names["SNOWFALL_var"]].apply(pd.to_numeric, errors='coerce')

    #-----------------------------------
    # Get values from file
    #-----------------------------------
    RH2 = df[_cfg.names["RH2_var"]]       # Relative humidity
    U2 = df[_cfg.names["U2_var"]]         # Wind velocity
    G = df[_cfg.names["G_var"]]           # Incoming shortwave radiation
    PRES = df[_cfg.names["PRES_var"]]     # Pressure

    if (_cfg.names["in_K"]):
        T2 = df[_cfg.names["T2_var"]].values         # Temperature
    else:
        T2 = df[_cfg.names["T2_var"]].values + 273.16      

    if np.nanmax(T2) > 373.16:
        print('Maximum temperature is: %s K please check the input temperature' % (np.nanmax(T2)))
        sys.exit()
    elif np.nanmin(T2) < 173.16:
        print('Minimum temperature is: %s K please check the input temperature' % (np.nanmin(T2)))
        sys.exit()

    #-----------------------------------
    # Create numpy arrays for the 2D fields
    #-----------------------------------
    T_interp = np.zeros([len(dso.time), len(ds.lat), len(ds.lon)])
    RH_interp = np.zeros([len(dso.time), len(ds.lat), len(ds.lon)])
    U_interp = np.zeros([len(dso.time), len(ds.lat), len(ds.lon)])
    G_interp = np.full([len(dso.time), len(ds.lat), len(ds.lon)], np.nan)
    P_interp = np.zeros([len(dso.time), len(ds.lat), len(ds.lon)])

    if (_cfg.names["RRR_var"] in df):
        RRR = df[_cfg.names["RRR_var"]]       # Precipitation
        RRR_interp = np.zeros([len(dso.time), len(ds.lat), len(ds.lon)])

    if(_cfg.names["SNOWFALL_var"] in df):
        SNOWFALL = df[_cfg.names["SNOWFALL_var"]]      # Incoming longwave radiation
        SNOWFALL_interp = np.zeros([len(dso.time), len(ds.lat), len(ds.lon)])

    if(_cfg.names["LWin_var"] in df):
        LW = df[_cfg.names["LWin_var"]]      # Incoming longwave radiation
        LW_interp = np.zeros([len(dso.time), len(ds.lat), len(ds.lon)])

    if(_cfg.names["N_var"] in df):
        N = df[_cfg.names["N_var"]]        # Cloud cover fraction
        N_interp = np.zeros([len(dso.time), len(ds.lat), len(ds.lon)])

    #-----------------------------------
    # Interpolate point data to grid 
    #-----------------------------------
    print('Interpolate CR file to grid')
   
    # Interpolate data (T, RH, RRR, U)  to grid using lapse rates
    for t in range(len(dso.time)):
        T_interp[t,:,:] = (T2[t]) + (ds.HGT.values-_cfg.station["stationAlt"])*_cfg.lapse["lapse_T"]
        RH_interp[t,:,:] = RH2[t] + (ds.HGT.values-_cfg.station["stationAlt"])*_cfg.lapse["lapse_RH"]
        U_interp[t,:,:] = U2[t]

        # Interpolate pressure using the barometric equation
        SLP = PRES[t]/np.power((1-(0.0065*_cfg.station["stationAlt"])/(288.15)), 5.255)
        P_interp[t,:,:] = SLP * np.power((1-(0.0065*ds.HGT.values)/(288.15)), 5.255)

        if (_cfg.names["RRR_var"] in df):
            RRR_interp[t,:,:] = np.maximum(RRR[t] + (ds.HGT.values-_cfg.station["stationAlt"])*_cfg.lapse["lapse_RRR"], 0.0)
        
        if (_cfg.names["SNOWFALL_var"] in df):
            SNOWFALL_interp[t, :, :] = SNOWFALL[t] + (ds.HGT.values-_cfg.station["stationAlt"])*_cfg.lapse["lapse_SNOWFALL"]

        if(_cfg.names["LWin_var"] in df):
            LW_interp[t,:,:] = LW[t]
        
        if(_cfg.names["N_var"] in df):
            N_interp[t,:,:] = N[t]


    print(('Number of glacier cells: %i') % (np.count_nonzero(~np.isnan(ds['MASK'].values))))
    print(('Number of glacier cells: %i') % (np.nansum(ds['MASK'].values)))

    # Auxiliary variables
    mask = ds.MASK.values
    heights = ds.HGT.values
    slope = ds.SLOPE.values
    aspect = ds.ASPECT.values
    lats = ds.lat.values
    lons = ds.lon.values
    sw = G.values

    #-----------------------------------
    # Run radiation module 
    #-----------------------------------
    if _cfg.radiation["radiationModule"] == 'Wohlfahrt2016' or _cfg.radiation["radiationModule"] == 'none':
        print('Run the Radiation Module Wohlfahrt2016 or no Radiation Module.')

        # Change aspect to south==0, east==negative, west==positive
        aspect = ds['ASPECT'].values - 180.0
        ds['ASPECT'] = (('lat', 'lon'), aspect)

        for t in range(len(dso.time)):
            doy = df.index[t].dayofyear
            hour = df.index[t].hour
            for i in range(len(ds.lat)):
                for j in range(len(ds.lon)):
                    if (mask[i, j] == 1):
                        if _cfg.radiation["radiationModule"] == 'Wohlfahrt2016':
                            G_interp[t, i, j] = np.maximum(0.0, mod_radCor.correctRadiation(lats[i], lons[j], _cfg.radiation["timezone_lon"], doy, hour, slope[i, j], aspect[i, j], sw[t], _cfg.radiation["zeni_thld"]))
                        else:
                            G_interp[t, i, j] = sw[t]

    elif _cfg.radiation["radiationModule"] == 'Moelg2009':
        print('Run the Radiation Module Moelg2009')

        # Calculate solar Parameters
        solPars, timeCorr = mod_radCor.solpars(_cfg.station["stationLat"])

        if _cfg.radiation["LUT"]:
            print('Read in look-up-tables')
            ds_LUT = xr.open_dataset('../../data/static/LUT_Rad.nc')
            shad1yr = ds_LUT.SHADING.values
            svf = ds_LUT.SVF.values

        else:
            print('Build look-up-tables')

            # Sky view factor
            svf = mod_radCor.LUTsvf(np.flipud(heights), np.flipud(mask), np.flipud(slope), np.flipud(aspect), lats[::-1], lons)
            print('Look-up-table sky view factor done')

            # Topographic shading
            shad1yr = mod_radCor.LUTshad(solPars, timeCorr, _cfg.station["stationLat"], np.flipud(heights), np.flipud(mask), lats[::-1], lons, _cfg.radiation["dtstep"], _cfg.radiation["tcart"])
            print('Look-up-table topographic shading done')

            # Save look-up tables
            Nt = int(366 * (3600 / _cfg.radiation["dtstep"]) * 24)  # number of time steps
            Ny = len(lats)  # number of latitudes
            Nx = len(lons)  # number of longitudes

            f = nc.Dataset('../../data/static/LUT_Rad.nc', 'w')
            f.createDimension('time', Nt)
            f.createDimension('lat', Ny)
            f.createDimension('lon', Nx)

            LATS = f.createVariable('lat', 'f4', ('lat',))
            LATS.units = 'degree'
            LONS = f.createVariable('lon', 'f4', ('lon',))
            LONS.units = 'degree'

            LATS[:] = lats
            LONS[:] = lons

            shad = f.createVariable('SHADING', float, ('time', 'lat', 'lon'))
            shad.long_name = 'Topographic shading'
            shad[:] = shad1yr

            SVF = f.createVariable('SVF', float, ('lat', 'lon'))
            SVF.long_name = 'sky view factor'
            SVF[:] = svf

            f.close()

        # In both cases, run the radiation model
        for t in range(len(dso.time)):
            doy = df.index[t].dayofyear
            hour = df.index[t].hour
            G_interp[t, :, :] = mod_radCor.calcRad(solPars, timeCorr, doy, hour, _cfg.station["stationLat"], T_interp[t, ::-1, :], P_interp[t, ::-1, :], RH_interp[t, ::-1, :], N_interp[t, ::-1, :], np.flipud(heights), np.flipud(mask), np.flipud(slope), np.flipud(aspect), shad1yr, svf, _cfg.radiation["dtstep"], _cfg.radiation["tcart"])

        # Change aspect to south == 0, east == negative, west == positive
        aspect2 = ds['ASPECT'].values - 180.0
        ds['ASPECT'] = (('lat', 'lon'), aspect2)

    else:
        print('Error! Radiation module not available.\nAvailable options are: Wohlfahrt2016, Moelg2009, none.')
        sys.exit()

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
    add_variable_along_timelatlon(dso, T_interp, 'T2', 'K', 'Temperature at 2 m')
    add_variable_along_timelatlon(dso, RH_interp, 'RH2', '%', 'Relative humidity at 2 m')
    add_variable_along_timelatlon(dso, U_interp, 'U2', 'm s\u207b\xb9', 'Wind velocity at 2 m')
    add_variable_along_timelatlon(dso, G_interp, 'G', 'W m\u207b\xb2', 'Incoming shortwave radiation')
    add_variable_along_timelatlon(dso, P_interp, 'PRES', 'hPa', 'Atmospheric Pressure')
    
    if (_cfg.names["RRR_var"] in df):
        add_variable_along_timelatlon(dso, RRR_interp, 'RRR', 'mm', 'Total precipitation (liquid+solid)')
    
    if (_cfg.names["SNOWFALL_var"] in df):
        add_variable_along_timelatlon(dso, SNOWFALL_interp, 'SNOWFALL', 'm', 'Snowfall')

    if (_cfg.names["LWin_var"] in df):
        add_variable_along_timelatlon(dso, LW_interp, 'LWin', 'W m\u207b\xb2', 'Incoming longwave radiation')
    if (_cfg.names["N_var"] in df):
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

    if(_cfg.names["RRR_var"] in df):
        check(dso.RRR,25.0,0.0)

    if (_cfg.names["SNOWFALL_var"] in df):
        check(dso.SNOWFALL, 0.05, 0.0)

    if (_cfg.names["LWin_var"] in df):
        check(dso.LWin, 400, 0.0)

    if (_cfg.names["N_var"] in df):
        check(dso.N, 1.0, 0.0)

def add_variable_along_timelatlon(ds, var, name, units, long_name):
    """ This function adds missing variables to the DATA class """
    if _cfg.coords["WRF"]:
         ds[name] = (('time','south_north','west_east'), var)	
    else:
        ds[name] = (('time','lat','lon'), var)
    ds[name].attrs['units'] = units
    ds[name].attrs['long_name'] = long_name
    return ds

def add_variable_along_latlon(ds, var, name, units, long_name):
    """ This function self.adds missing variables to the self.DATA class """
    if _cfg.coords["WRF"]: 
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

def check_for_nan(ds):
    if _cfg.coords["WRF"] is True:
        for y,x in product(range(ds.dims['south_north']),range(ds.dims['west_east'])):
            mask = ds.MASK.sel(south_north=y, west_east=x)
            if mask==1:
                if np.isnan(ds.sel(south_north=y, west_east=x).to_array()).any():
                    print('ERROR!!!!!!!!!!! There are NaNs in the dataset')
                    sys.exit()
    else:
        for y,x in product(range(ds.dims['lat']),range(ds.dims['lon'])):
            mask = ds.MASK.isel(lat=y, lon=x)
            if mask==1:
                if np.isnan(ds.isel(lat=y, lon=x).to_array()).any():
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


def get_user_arguments(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """Get user arguments for converting AWS data.
    
    Args:
        parser: An initialised argument parser.
    
    Returns:
        User arguments for conversion.
    """
    parser.description = "Create 2D input file from csv file."
    parser.prog = __package__

    # Required arguments
    parser.add_argument('-c', '--csv_file', dest='csv_file', type=str, metavar='<path>', required=True,
                        help='Path to .csv file with meteorological data')
    parser.add_argument('-o', '--cosipy_file', dest='cosipy_file', type=str, metavar='<path>', required=True,
                        help='Path to the resulting COSIPY netCDF file')
    parser.add_argument('-s', '--static_file', type=str, dest='static_file', help='Path to static file with DEM, slope etc.')

    # Optional arguments
    parser.add_argument('-b', '--start_date', type=str, metavar='<yyyy-mm-dd>', dest='start_date', help='Start date')
    parser.add_argument('-e', '--end_date', type=str, metavar='<yyyy-mm-dd>', dest='end_date', help='End date')
    parser.add_argument('--xl', dest='xl', type=float, metavar='<float>', const=None, help='left longitude value of the subset')
    parser.add_argument('--xr', dest='xr', type=float, metavar='<float>', const=None, help='right longitude value of the subset')
    parser.add_argument('--yl', dest='yl', type=float, metavar='<float>', const=None, help='lower latitude value of the subset')
    parser.add_argument('--yu', dest='yu', type=float, metavar='<float>', const=None, help='upper latitude value of the subset')
    arguments = parser.parse_args()

    return arguments


def load_config(module_name: str) -> tuple:
    """Load configuration for module.
    
    Args:
        module_name: name of this module.

    Returns:
        User arguments and configuration parameters.
    """
    params = UtilitiesConfig()
    arguments = get_user_arguments(params.parser)
    params.load(arguments.utilities_path)
    params = params.get_config_expansion(name=module_name)

    return arguments, params


def main():
    global _args  # Yes, it's bad practice
    global _cfg
    _args, _cfg = load_config(module_name="aws2cosipy")

    if _cfg.points["point_model"]:
        create_1D_input(_args.csv_file, _args.cosipy_file, _args.static_file, _args.start_date, _args.end_date) 
    else:
        create_2D_input(_args.csv_file, _args.cosipy_file, _args.static_file, _args.start_date, _args.end_date, _args.xl, _args.xr, _args.yl, _args.yu) 


if __name__ == "__main__":
    main()
