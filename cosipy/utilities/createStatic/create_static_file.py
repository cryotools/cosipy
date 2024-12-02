"""
Reads the DEM of the study site and the shapefile and creates a
corresponding static file, ``static.nc``.

Edit the configuration by supplying a valid .toml file. See the sample
``utilities_config.toml`` for more information.

Usage:

From source:
``python -m cosipy.utilities.createStatic.create_static_file [-u <path>]``

Entry point:
``cosipy-create-static [-u <path>]``

Optional arguments:
    -u, --u <path>  Relative path to utilities' configuration file.
"""
import argparse
import os
from itertools import product

import numpy as np
import richdem as rd
import xarray as xr
import fiona
from horayzon.domain import curved_grid

from cosipy.utilities.config_utils import UtilitiesConfig
from cosipy.utilities.aws2cosipy.crop_file_to_glacier import crop_file_to_glacier

_args = None
_cfg = None

def check_folder_path(path: str) -> str:
    """Check the folder path includes a forward slash."""
    if not path.endswith("/"):
        path = f"{path}/"

    return path


def check_for_nan(ds,var=None):
    for y,x in product(range(ds.dims['lat']),range(ds.dims['lon'])):
        mask = ds.MASK.isel(lat=y, lon=x)
        if mask==1:
            if var is None:
                if np.isnan(ds.isel(lat=y, lon=x).to_array()).any():
                    raise ValueError("ERROR! There are NaNs in the static fields")
            else:
                if np.isnan(ds[var].isel(lat=y, lon=x)).any():
                    raise ValueError("ERROR! There are NaNs in the static fields")


def insert_var(ds, var, name, units, long_name):
    """Insert variables in dataset"""
    ds[name] = (('lat','lon'), var)
    ds[name].attrs['units'] = units
    ds[name].attrs['long_name'] = long_name
    ds[name].attrs['_FillValue'] = -9999


def get_user_arguments(parser: argparse.ArgumentParser) -> argparse.Namespace:
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

#domain creation assumes WGS84 is valid
def domain_creation(shp_path, dist_search, ellps="WGS84"):
    print("Using automatic domain creation.")
    #Get bound of glacier shapefile
    shp = fiona.open(shp_path)
    domain = {"lon_min": shp.bounds[0], "lon_max": shp.bounds[2],
              "lat_min": shp.bounds[1], "lat_max": shp.bounds[3]}

    #Additionally there is the option of a planar grid which requires importing the function
    domain_outer = curved_grid(domain, dist_search=dist_search, ellps=ellps)
    ### Get lat/lon corners ###
    #Note setup is created based on Northern Hemisphere glaciers
    longitude_upper_left = str(domain_outer['lon_min'])
    latitude_upper_left = str(domain_outer['lat_max'])
    longitude_lower_right = str(domain_outer['lon_max'])
    latitude_lower_right = str(domain_outer['lat_min'])

    return (longitude_upper_left, latitude_upper_left, longitude_lower_right, latitude_lower_right)


def main():
    _, _cfg = load_config(module_name="create_static")

    static_folder = _cfg.paths["static_folder"]
    tile = _cfg.coords["tile"]
    aggregate = _cfg.coords["aggregate"]

    # input digital elevation model (DEM)
    dem_path_tif = f"{static_folder}{_cfg.paths['dem_path']}"
    # input shape of glacier or study area, e.g. from the Randolph glacier inventory
    shape_path = f"{static_folder}{_cfg.paths['shape_path']}"
    # path where the static.nc file is saved
    output_path = f"{static_folder}{_cfg.paths['output_file']}"
    output_path_crop = f"{static_folder}{_cfg.paths['output_file_crop']}"

    automatic_domain = _cfg.coords["automatic_domain"]
    dist_search = _cfg.coords["dist_search"]
    if automatic_domain:
        longitude_upper_left, latitude_upper_left,\
            longitude_lower_right, latitude_lower_right = domain_creation(shape_path,
                                                                          dist_search=dist_search,
                                                                          ellps="WGS84")

    else:
        ### to shrink the DEM use the following lat/lon corners
        print("Please be aware that you switched off the automatic domain creation which requires by-hand adjustment of the borders.")
        
        # to shrink the DEM use the following lat/lon corners
        longitude_upper_left = str(_cfg.coords["longitude_upper_left"])
        latitude_upper_left = str(_cfg.coords["latitude_upper_left"])
        longitude_lower_right = str(_cfg.coords["longitude_lower_right"])
        latitude_lower_right = str(_cfg.coords["latitude_lower_right"])

    # to aggregate the DEM to a coarser spatial resolution
    aggregate_degree = str(_cfg.coords["aggregate_degree"])

    # intermediate files, will be removed afterwards
    dem_path_tif_temp = f"{static_folder}DEM_temp.tif"
    dem_path_tif_temp2 = f"{static_folder}DEM_temp2.tif"
    dem_path = f"{static_folder}dem.nc"
    aspect_path = f"{static_folder}aspect.nc"
    mask_path = f"{static_folder}mask.nc"
    slope_path = f"{static_folder}slope.nc"

    if tile:
        os.system(
            f"gdal_translate -projwin {longitude_upper_left} "
            + f"{latitude_upper_left} {longitude_lower_right} "
            + f"{latitude_lower_right} {dem_path_tif} {dem_path_tif_temp}"
        )
        dem_path_tif = dem_path_tif_temp

    if aggregate:
        os.system(
            f"gdalwarp -tr {aggregate_degree} {aggregate_degree} -r average "
            + f"{dem_path_tif} {dem_path_tif_temp2}"
        )
        dem_path_tif = dem_path_tif_temp2

    # convert DEM from tif to NetCDF
    os.system(f"gdal_translate -of NETCDF {dem_path_tif} {dem_path}")

    # calculate slope as NetCDF from DEM
    os.system(f"gdaldem slope -of NETCDF {dem_path} {slope_path} -s 111120")

    # calculate aspect from DEM
    #rd might throw an error when the no_data value is not directly specified
    try:
        aspect = np.flipud(rd.TerrainAttribute(rd.LoadGDAL(dem_path_tif), attrib = 'aspect'))
    except:
        aspect = np.flipud(rd.TerrainAttribute(rd.LoadGDAL(dem_path_tif, no_data=-9999), attrib = 'aspect'))

    # calculate mask as NetCDF with DEM and shapefile
    os.system(
        f"gdalwarp -of NETCDF --config GDALWARP_IGNORE_BAD_CUTLINE YES "
        + f"-cutline {shape_path} {dem_path_tif} {mask_path}"
    )

    # open intermediate netcdf files
    dem = xr.open_dataset(dem_path)
    mask = xr.open_dataset(mask_path)
    slope = xr.open_dataset(slope_path)

    # set NaNs in mask to -9999 and elevation within the shape to 1
    mask=mask.Band1.values
    mask[np.isnan(mask)]=-9999
    mask[mask>0]=1
    print(mask)

    ## create output dataset
    ds = xr.Dataset()
    ds.coords['lon'] = dem.lon.values
    ds.lon.attrs['standard_name'] = 'lon'
    ds.lon.attrs['long_name'] = 'longitude'
    ds.lon.attrs['units'] = 'degrees_east'

    ds.coords['lat'] = dem.lat.values
    ds.lat.attrs['standard_name'] = 'lat'
    ds.lat.attrs['long_name'] = 'latitude'
    ds.lat.attrs['units'] = 'degrees_north'

    # insert needed static variables
    insert_var(ds, dem.Band1.values,'HGT','meters','meter above sea level')
    insert_var(ds, aspect,'ASPECT','degrees','Aspect of slope')
    insert_var(ds, slope.Band1.values,'SLOPE','degrees','Terrain slope')
    insert_var(ds, mask,'MASK','boolean','Glacier mask')

    os.system(
        f"rm {dem_path} {mask_path} {slope_path} "
        + f"{dem_path_tif_temp} {dem_path_tif_temp2}"
    )

    """Save combined static file, delete intermediate files, print
    number of glacier grid points."""
    check_for_nan(ds)
    
    # crop file to glacier if desired - but do not set to NaN using xr.where
    crop_file = _cfg.coords['crop_file']
    if crop_file:
        ds_crop = crop_file_to_glacier(ds, WRF=False, check_vars=False)
        if aggregate: #save cropped file only
            ds_crop.to_netcdf(output_path)
        else:
            #uncropped file is needed for e.g., HORAYZON calculation (surrounding terrain)
            #cropped file is needed as static data for simulation
            ds.to_netcdf(output_path)
            ds_crop.to_netcdf(output_path_crop)
        ds.close()
        ds_crop.close()
    else:
        ds.to_netcdf(output_path)
        ds.close()    
    
    print("Study area consists of ", np.nansum(mask[mask==1]), " glacier points")
    print("Done")

if __name__ == "__main__":
    main()
##old version has NaNs, new version only 0 and 1s