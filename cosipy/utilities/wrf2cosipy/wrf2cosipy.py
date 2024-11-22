"""
Create 2D input from WRF data.

Reads the input data (model forcing) and writes the output to
a netcdf file. Edit the configuration by supplying a valid .toml file.
See the sample ``utilities_config.toml`` for more information.

Usage:

From source:
``python -m cosipy.utilities.wrf2cosipy.wrf2cosipy -i <path> -o <path> -u [<path] [-b <date>] [-e <date>]``

Entry point:
``cosipy-wrf2cosipy -i <path> -o <path> -u [<path] [-b <date>] [-e <date>]``

Options and arguments:

Required arguments:
    -i, --input <path>      Path to WRF file.
    -o, --output <path>     Path to the resulting COSIPY file.

Optional arguments:
    -u, --u <path>                  Relative path to utilities'
                                        configuration file.
    -b, --start_date <yyyymmdd>     Start date.
    -e, --end_date <yyyymmdd>       End date. 
"""

import argparse

import numpy as np
import pandas as pd
import xarray as xr
import xrspatial as xrs

from cosipy.constants import Constants
from cosipy.utilities.config_utils import UtilitiesConfig

_args = None
_cfg = None


def create_input(wrf_file, cosipy_file, start_date, end_date):
    """Create an input dataset from WRF data."""

    print('-------------------------------------------')
    print(f"Create input\nRead input file {wrf_file}")

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
    z  = _cfg.constants["hu"]     # Height of measurement
    z0 = 0.0040 # Roughness length for momentum
    umag = np.sqrt(ds.V10.values**2+ds.U10.values**2)   # Mean wind velocity
    U2 = umag * (np.log(2 / z0) / np.log(10 / z0))
    dso = add_variable_along_timelatlon(dso, U2, 'U2', 'm s^-1', 'Wind velocity at 2 m')

    # Add glacier mask to file (derived from the land use category)
    mask = ds.LU_INDEX[0].values
    mask[mask!=_cfg.constants["lu_class"]] = 0
    mask[mask==_cfg.constants["lu_class"]] = 1
    dso = add_variable_along_latlon(dso, mask, 'MASK', '-', 'Glacier mask')
    
    # Derive precipitation from accumulated values
    rrr = np.full_like(ds.T2, 0.)
    for t in np.arange(1,len(dso.time)):
        rrr[t,:,:] = (ds.RAINNC[t,:,:] + ds.RAINC[t,:,:])  - (ds.RAINNC[t-1,:,:] + ds.RAINC[t-1,:,:])
    dso = add_variable_along_timelatlon(dso, rrr, 'RRR', 'mm', 'Total precipitation')
 
    # Calc fresh snow density
    if (Constants.densification_method!='constant'):
        density_fresh_snow = np.maximum(109.0+6.0*(ds.T2.values-273.16)+26.0*np.sqrt(U2), 50.0)
    else:
        density_fresh_snow = Constants.constant_density

    # Derive snowfall from accumulated values
    snowf = np.full_like(ds.T2, 0.)
    for t in np.arange(1,len(dso.time)):
        snowf[t,:,:] = ds.SNOWNC[t,:,:]-ds.SNOWNC[t-1,:,:]
    snowf = (snowf/1000.0)*(Constants.water_density/density_fresh_snow)
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
    print(dso.dims['south_north']) 
  
    # Do some checks 
    check(dso.T2,316.16,223.16)
    check(dso.RH2,100.0,0.0)
    check(dso.SNOWFALL,1.0,0.0)
    check(dso.U2,50.0,0.0)
    check(dso.G,1600.0,0.0)
    check(dso.PRES,1080.0,200.0)
    check(dso.LWin,500.0,100.0)


def add_variable_along_latlon(ds, var, name, units, long_name):
    """Add spatial data to a dataset."""
    ds[name] = (('south_north','west_east'), var)
    ds[name].attrs['units'] = units
    ds[name].attrs['long_name'] = long_name
    ds[name].encoding['_FillValue'] = -9999
    return ds


def add_variable_along_timelatlon(ds, var, name, units, long_name):
    """Add spatiotemporal data to a dataset."""
    ds[name] = (('time','south_north','west_east'), var)
    ds[name].attrs['units'] = units
    ds[name].attrs['long_name'] = long_name
    return ds


def check(field, max_bound, min_bound):
    """Check the validity of the input data."""
    if np.nanmax(field) > max_bound or np.nanmin(field) < min_bound:
        msg = f"{str.capitalize(field.name)} MAX: {np.nanmax(field):.2f} MIN: {np.nanmin(field):.2f}"
        print(
            f"\n\nWARNING! Please check the data, it seems they are out of a reasonable range {msg}"
        )


def wrf_rh(T2, Q2, PSFC):
    """Get the relative humidity."""
    pq0 = 379.90516
    a2 = 17.2693882
    a3 = 273.16
    a4 = 35.86
    rh = Q2 * 100 / ( (pq0 / PSFC) * np.exp(a2 * (T2 - a3) / (T2 - a4)) )
    
    rh[rh>100.0] = 100.0
    rh[rh<0.0] = 0.0
    return rh


def get_user_arguments(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """Get user arguments for converting AWS data.

    Args:
        parser: An initialised argument parser.

    Returns:
        User arguments for conversion.
    """
    parser.description = "Create 2D input file from WRF file."
    parser.prog = __package__

    # Required arguments
    parser.add_argument(
        "-i",
        "--input",
        dest="wrf_file",
        type=str,
        metavar="<path>",
        required=True,
        help="Path to WRF file",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="cosipy_file",
        type=str,
        metavar="<path>",
        required=True,
        help="Path to the resulting COSIPY file",
    )

    # Optional arguments
    parser.add_argument(
        "-b", "--start_date", dest="start_date", const=None, help="Start date"
    )
    parser.add_argument(
        "-e", "--end_date", dest="end_date", const=None, help="End date"
    )
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
    global _args
    global _cfg
    _args, _cfg = load_config(module_name="wrf2cosipy")

    create_input(_args.wrf_file, _args.cosipy_file, _args.start_date, _args.end_date)


if __name__ == "__main__":
    main()