"""
 This file reads the DEM of the study site and the shapefile and creates the needed static.nc
"""
import os
import sys
from itertools import product

import numpy as np
import richdem as rd
import xarray as xr

from cosipy.utilities.config import UtilitiesConfig


def check_folder_path(path: str) -> str:
    """Check folder path includes a forward slash."""
    if not path.endswith("/"):
        path = f"{path}/"

    return path


def check_for_nan(ds,var=None):
    for y,x in product(range(ds.dims['lat']),range(ds.dims['lon'])):
        mask = ds.MASK.isel(lat=y, lon=x)
        if mask==1:
            if var is None:
                if np.isnan(ds.isel(lat=y, lon=x).to_array()).any():
                    print('ERROR!!!!!!!!!!! There are NaNs in the static fields')
                    sys.exit()
            else:
                if np.isnan(ds[var].isel(lat=y, lon=x)).any():
                    print('ERROR!!!!!!!!!!! There are NaNs in the static fields')
                    sys.exit()


def insert_var(ds, var, name, units, long_name):
    """Insert variables in dataset"""
    ds[name] = (('lat','lon'), var)
    ds[name].attrs['units'] = units
    ds[name].attrs['long_name'] = long_name
    ds[name].attrs['_FillValue'] = -9999


def main():
    cfg = UtilitiesConfig()
    cfg = cfg.create_static
    static_folder = cfg.paths["static_folder"]

    tile = cfg.coords["tile"]
    aggregate = cfg.coords["aggregate"]

    ### input digital elevation model (DEM)
    dem_path_tif = f"{static_folder}{cfg.paths['dem_path']}"
    ### input shape of glacier or study area, e.g. from the Randolph glacier inventory
    shape_path = f"{static_folder}{cfg.paths['shape_path']}"
    ### path were the static.nc file is saved
    output_path = f"{static_folder}{cfg.paths['output_file']}"

    ### to shrink the DEM use the following lat/lon corners
    longitude_upper_left = str(cfg.coords["longitude_upper_left"])
    latitude_upper_left = str(cfg.coords["latitude_upper_left"])
    longitude_lower_right = str(cfg.coords["longitude_lower_right"])
    latitude_lower_right = str(cfg.coords["latitude_lower_right"])

    ### to aggregate the DEM to a coarser spatial resolution
    aggregate_degree = str(cfg.coords["aggregate_degree"])

    ### intermediate files, will be removed afterwards
    dem_path_tif_temp = static_folder + 'DEM_temp.tif'
    dem_path_tif_temp2 = static_folder + 'DEM_temp2.tif'
    dem_path = static_folder + 'dem.nc'
    aspect_path = static_folder + 'aspect.nc'
    mask_path = static_folder + 'mask.nc'
    slope_path = static_folder + 'slope.nc'

    if tile:
        os.system('gdal_translate -projwin ' + longitude_upper_left + ' ' + latitude_upper_left + ' ' +
              longitude_lower_right + ' ' + latitude_lower_right + ' ' + dem_path_tif + ' ' + dem_path_tif_temp)
        dem_path_tif = dem_path_tif_temp

    if aggregate:
        os.system('gdalwarp -tr ' + aggregate_degree + ' ' + aggregate_degree + ' -r average ' + dem_path_tif + ' ' + dem_path_tif_temp2)
        dem_path_tif = dem_path_tif_temp2

    ### convert DEM from tif to NetCDF
    os.system('gdal_translate -of NETCDF ' + dem_path_tif  + ' ' + dem_path)

    ### calculate slope as NetCDF from DEM
    os.system('gdaldem slope -of NETCDF ' + dem_path + ' ' + slope_path + ' -s 111120')

    ### calculate aspect from DEM
    aspect = np.flipud(rd.TerrainAttribute(rd.LoadGDAL(dem_path_tif), attrib = 'aspect'))

    ### calculate mask as NetCDF with DEM and shapefile
    os.system('gdalwarp -of NETCDF  --config GDALWARP_IGNORE_BAD_CUTLINE YES -cutline ' + shape_path + ' ' + dem_path_tif  + ' ' + mask_path)

    ### open intermediate netcdf files
    dem = xr.open_dataset(dem_path)
    mask = xr.open_dataset(mask_path)
    slope = xr.open_dataset(slope_path)

    ### set NaNs in mask to -9999 and elevation within the shape to 1
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

    ### insert needed static variables
    insert_var(ds, dem.Band1.values,'HGT','meters','meter above sea level')
    insert_var(ds, aspect,'ASPECT','degrees','Aspect of slope')
    insert_var(ds, slope.Band1.values,'SLOPE','degrees','Terrain slope')
    insert_var(ds, mask,'MASK','boolean','Glacier mask')

    os.system('rm '+ dem_path + ' ' + mask_path + ' ' + slope_path + ' ' + dem_path_tif_temp + ' '+ dem_path_tif_temp2)

    ### save combined static file, delete intermediate files and print number of glacier grid points

    check_for_nan(ds)
    ds.to_netcdf(output_path)
    print("Study area consists of ", np.nansum(mask[mask==1]), " glacier points")
    print("Done")

if __name__ == "__main__":
    main()