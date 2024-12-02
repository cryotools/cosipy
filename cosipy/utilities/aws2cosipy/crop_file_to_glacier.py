import sys
import xarray as xr
import numpy as np
from itertools import product

#np.warnings.filterwarnings('ignore')

from cosipy.utilities.config_utils import UtilitiesConfig
import argparse

_args = None
_cfg = None

def crop_file_to_glacier(ds, WRF, check_vars):
       
    dic_attrs= {
        "HGT": ("HGT", "m", "Elevation"),
        "MASK": ("MASK", "boolean", "Glacier mask"),
        "SLOPE": ("SLOPE", "degrees", "Terrain slope"),
        "ASPECT": ("ASPECT", "degrees", "Aspect of slope"),
        "T2": ("T2", "K", "Air temperature at 2 m"),
        "RH2": ("RH2", "%", "Relative humidity at 2 m"),
        "U2": ("U2", "m s\u207b\xb9", "Wind velocity at 2 m"),
        "PRES": ("PRES", "hPa", "Atmospheric pressure"),
        "G": ("G", "W m\u207b\xb2", "Incoming shortwave radiation"),
        "RRR": ("RRR", "mm", "Total precipitation"),
        "SNOWFALL": ("SNOWFALL", "m", "Snowfall"),
        "N": ("N", "-", "Cloud fraction"),
        "LWin": ("LWin", "W m\u207b\xb2", "Incoming longwave radiation"),
        'N_Points': ('N_Points', 'count','Number of Points in each bin'),
        'sw_dir_cor': ('sw_dir_cor', '-', 'correction factor for direct downward shortwave radiation'),
        'slope': ('slope','degrees', 'Horayzon Slope'),
        'aspect': ('aspect','degrees','Horayzon Aspect measured clockwise from the North'),
        'surf_enl_fac': ('surf_enl_fac','-','Surface enlargement factor'),
        'elevation': ('elevation','m','Orthometric Height')}


    dso = ds 

    print('Create cropped file.')
    dso_mod = xr.Dataset()
    for var in list(dso.variables):
        print(var)
        arr = bbox_2d_array(dso.MASK.values, dso[var].values, var)
        if var in ['lat','latitude','south_north', 'lon','longitude', 'west_east', 'time','Time']:
            if var in ['lat','latitude','south_north']:
                dso_mod.coords[var] = arr
                dso_mod[var].attrs['standard_name'] = var
                dso_mod[var].attrs['long_name'] = 'latitude'
                dso_mod[var].attrs['units'] = 'degrees_north'
            elif var in ['lon', 'longitude', 'west_east']:
                dso_mod.coords[var] = arr
                dso_mod[var].attrs['standard_name'] = var
                dso_mod[var].attrs['long_name'] = 'longitude'
                dso_mod[var].attrs['units'] = 'degrees_east'
            else:
                dso_mod.coords['time'] = arr
        elif var in ['HGT','ASPECT','SLOPE','MASK','N_Points','surf_enl_fac','slope','aspect','elevation']:
            add_variable_along_latlon(dso_mod, arr, dic_attrs[var][0], dic_attrs[var][1], dic_attrs[var][2], WRF)
        else:
            add_variable_along_timelatlon(dso_mod, arr, dic_attrs[var][0], dic_attrs[var][1], dic_attrs[var][2], WRF)
    
    #----------------------
    # Do some checks
    #----------------------
    print("Performing checks.")
    check_for_nan(dso_mod, WRF)
    
    # Check data
    if check_vars is True:
        check_data(dataset=dso_mod)

    return dso_mod




### Functions ###
#Note this function has issues if the glacier extent is near a border already:
#In that case the index+1 may result in index error
#This should however never be the case as this script is only applied if we take a larger extent around the glacier for e.g., 
#Mölgs radiation scheme

def bbox_2d_array(mask, arr, varname):
    if arr.ndim == 1:
        if varname in ['time','Time']:
            i_min = 0
            i_max = None
        elif varname in ['lat','latitude','south_north']:
            ix = np.where(np.any(mask == 1, axis=1))[0]
            i_min, i_max = ix[[0, -1]]
            i_min = i_min #lower bounds included
            i_max = i_max +1
        elif varname in ['lon','longitude','west_east']:
            ix = np.where(np.any(mask == 1, axis=0))[0]
            i_min, i_max = ix[[0, -1]]
            i_min = i_min #lower bounds included
            i_max = i_max +1
        bbox = arr[i_min:i_max]
    elif arr.ndim == 2:
        ix_c = np.where(np.any(mask == 1, axis=0))[0]
        ix_r = np.where(np.any(mask == 1, axis=1))[0]
        c_min, c_max = ix_c[[0, -1]]
        r_min, r_max = ix_r[[0, -1]]
    
        #Draw box with one non-value border
        #Now we got bounding box -> just add +1 at maxima and voila
        bbox = arr[r_min:r_max+1,c_min:c_max+1]
    elif arr.ndim == 3:
        ix_c = np.where(np.any(mask == 1, axis=0))[0]
        ix_r = np.where(np.any(mask == 1, axis=1))[0]
        c_min, c_max = ix_c[[0, -1]]
        r_min, r_max = ix_r[[0, -1]]
        bbox = arr[:, r_min:r_max+1,c_min:c_max+1]
    return bbox 


def add_variable_along_timelatlon(ds, var, name, units, long_name, WRF):
    """Add spatiotemporal data to a dataset."""
    if WRF:
        ds[name] = (("time", "south_north", "west_east"), var)
    else:
        ds[name] = (("time", "lat", "lon"), var)
    ds[name].attrs["units"] = units
    ds[name].attrs["long_name"] = long_name
    return ds

def add_variable_along_latlon(ds, var, name, units, long_name, WRF):
    """Add spatial data to a dataset."""
    if WRF:
        ds[name] = (("south_north", "west_east"), var)
    else:
        ds[name] = (("lat", "lon"), var)
    ds[name].attrs["units"] = units
    ds[name].attrs["long_name"] = long_name
    ds[name].encoding["_FillValue"] = -9999
    return ds

def check_data(dataset: xr.Dataset):
    """Check data is within physically reasonable bounds."""
    T2_var = _cfg.names['T2_var']
    PRES_var = _cfg.names['PRES_var']
    RH2_var = _cfg.names['RH2_var']
    G_var = _cfg.names['G_var']
    RRR_var = _cfg.names['RRR_var']
    U2_var = _cfg.names['U2_var']
    LWin_var = _cfg.names['LWin_var']
    SNOWFALL_var = _cfg.names['SNOWFALL_var']
    N_var = _cfg.names['N_var']
    
    var_list = list(dataset.variables)
    
    if T2_var in var_list:
        check(dataset[T2_var], 316.16, 223.16)
    if RH2_var in var_list:
        check(dataset[RH2_var], 100.0, 0.0)
    if U2_var in var_list:
        check(dataset[U2_var], 50.0, 0.0)
    if G_var in var_list:
        check(dataset[G_var], 1600.0, 0.0)
    if PRES_var in var_list:
        check(dataset[PRES_var], 1080.0, 200.0)
    if RRR_var in var_list:
        check(dataset[RRR_var], 25.0, 0.0)
    if SNOWFALL_var in var_list:
        check(dataset[SNOWFALL_var], 0.05, 0.0)
    if LWin_var in var_list:
        check(dataset[LWin_var], 400, 0.0)
    if N_var in var_list:
        check(dataset[N_var], 1.0, 0.0)


def check(field, max_bound, min_bound):
    """Check the validity of the input data."""

    if np.nanmax(field) > max_bound or np.nanmin(field) < min_bound:
        msg = f"{str.capitalize(field.name)} MAX: {np.nanmax(field):.2f} MIN: {np.nanmin(field):.2f}"
        print(
            f"\n\nWARNING! Please check the data, it seems they are out of a reasonable range {msg}"
        )

def raise_nan_error():
    """Raise error if NaNs are in the dataset.

    Raises:
        ValueError: There are NaNs in the dataset.
    """
    raise ValueError("ERROR! There are NaNs in the dataset.")
     
def check_for_nan(ds, WRF):
    if WRF is True:
        for y, x in product(
            range(ds.dims["south_north"]), range(ds.dims["west_east"])
        ):
            mask = ds.MASK.sel(south_north=y, west_east=x)
            if mask == 1:
                if np.isnan(
                    ds.sel(south_north=y, west_east=x).to_array()
                ).any():
                    raise_nan_error()
    else:
        for y, x in product(range(ds.dims["lat"]), range(ds.dims["lon"])):
            mask = ds.MASK.isel(lat=y, lon=x)
            if mask == 1:
                if np.isnan(ds.isel(lat=y, lon=x).to_array()).any():
                    raise_nan_error()

def empty_user_arguments(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """Get user arguments for converting AWS data.
    
    Args:
        parser: An initialised argument parser.
    
    Returns:
        User arguments for conversion.
    """
    parser.description = "Create static file."
    parser.prog = __package__

    arguments = parser.parse_args()

    return arguments

def get_user_arguments(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """Get user arguments for converting AWS data.

    Args:
        parser: An initialised argument parser.

    Returns:
        User arguments for conversion.
    """
    parser.description = "Create netCDF input file from a .csv file."
    parser.prog = __package__

    # Required arguments
    parser.add_argument(
        "-i",
        "--input",
        dest="input_file",
        type=str,
        metavar="<path>",
        required=True,
        help="Path to .nc file that needs to be cropped",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="cosipy_file",
        type=str,
        metavar="<path>",
        required=True,
        help="Path to the resulting COSIPY netCDF file",
    )

    arguments = parser.parse_args()

    return arguments

def load_config(module_name: str) -> tuple:
    """Load configuration for module.

    Args:
        module_name: Name of this module.

    Returns:
        User arguments and configuration parameters.
    """
    params = UtilitiesConfig()
    arguments = get_user_arguments(params.parser)
    params.load(arguments.utilities_path)
    params = params.get_config_expansion(name=module_name)

    return arguments, params

def load_config_empty(module_name: str) -> tuple:
    """Load configuration for module.

    Args:
        module_name: Name of this module.

    Returns:
        User arguments and configuration parameters.
    """
    params = UtilitiesConfig()
    arguments = empty_user_arguments(params.parser)
    params.load(arguments.utilities_path)
    params = params.get_config_expansion(name=module_name)

    return arguments, params


def main():
    global _args  # Yes, it's bad practice
    global _cfg
    _args, _cfg = load_config(module_name="aws2cosipy")
        
    print('Read input file %s \n' % (_args.input_file))
    ds = xr.open_dataset(_args.input_file)
    dso_mod = crop_file_to_glacier(ds, WRF=_cfg.coords["WRF"], check_vars=True)
    ## write out to file ##
    print("Writing cropped cosipy file.")
    dso_mod.to_netcdf(_args.cosipy_file)

if __name__ == "__main__":
    main()


