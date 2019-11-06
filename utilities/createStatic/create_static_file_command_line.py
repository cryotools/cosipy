"""
 This file reads the DEM of the study site and the shapefile and creates the needed static.nc
"""

import xarray as xr
import numpy as np
import netCDF4
import os

static_folder = '../../data/static/'

### input digital elevation model (DEM)
dem_path_tif = static_folder + 'DEM/Peru/ASTGTM2_S09W078_dem_clip.tif'
### input shape of glacier or study area, e.g. from the Randolph glacier inventory
shape_path = static_folder + 'Shapefiles/Peru/artesonraju_glacier_sentinel_f.shp'
### path were the static.nc file is saved
output_path = static_folder + 'Peru_static.nc'

### intermediate files, will be removed afterwars
dem_path_tif_temp = static_folder + 'DEM/DEM_temp.tif'
dem_path_tif_temp2 = static_folder + 'DEM/DEM_temp2.tif'
dem_path = static_folder + 'dem.nc'
aspect_path = static_folder + 'aspect.nc'
mask_path = static_folder + 'mask.nc'
slope_path = static_folder + 'slope.nc'

aggregate_degree = '0.00045475436839202433'
os.system('gdalwarp -tr ' + aggregate_degree + ' ' + aggregate_degree + ' -r average ' + dem_path_tif + ' ' + dem_path_tif_temp)
dem_path_tif = dem_path_tif_temp

### convert DEM from tif to NetCDF
os.system('gdal_translate -of NETCDF ' + dem_path_tif  + ' ' + dem_path)
### calcualte slope as NetCDF from DEM
os.system('gdaldem slope -of NETCDF ' + dem_path + ' ' + slope_path + ' -s 111120')
### calcualte aspect as NetCDF from DEM
os.system('gdaldem aspect -of NETCDF ' + dem_path + ' ' + aspect_path)
### calculate mask as NetCDF with DEM and shapefile
os.system('gdalwarp -of NETCDF  --config GDALWARP_IGNORE_BAD_CUTLINE YES -cutline ' + shape_path + ' ' + dem_path_tif  + ' ' + mask_path)

### open intermediate netcdf files
dem = xr.open_dataset(dem_path)
aspect = xr.open_dataset(aspect_path)
mask = xr.open_dataset(mask_path)
slope = xr.open_dataset(slope_path)

### set NaNs in mask to -9999 and elevation within the shape to 1
mask=mask.Band1.values
mask[np.isnan(mask)]=-9999
mask[mask>0]=1
print(mask)

### create output dataset
ds = xr.Dataset()
ds.coords['south_north'] = dem.lat.values
ds.south_north.attrs['standard_name'] = 'south_north'
ds.south_north.attrs['long_name'] = 'longitude'
ds.south_north.attrs['units'] = 'degrees_east'
ds.coords['west_east'] = dem.lon.values
ds.west_east.attrs['standard_name'] = 'west_east'
ds.west_east.attrs['long_name'] = 'latitude'
ds.west_east.attrs['units'] = 'degrees_north'

### function to insert variarbles to dataset
def insert_var(ds, var, name, units, long_name):
    ds[name] = (('south_north','west_east'), var)
    ds[name].attrs['units'] = units
    ds[name].attrs['long_name'] = long_name
    ds[name].attrs['_FillValue'] = -9999

### insert needed static variables
insert_var(ds, dem.Band1.values,'HGT','meters','meter above sea level')
insert_var(ds, aspect.Band1.values,'ASPECT','degress','Aspect of slope')
insert_var(ds, slope.Band1.values,'SLOPE','degress','Terrain slope')
insert_var(ds, mask,'MASK','boolean','Glacier mask')

### save combined static file, delete intermediate files and print number of glacier grid points
ds.to_netcdf(output_path)
os.system('rm '+ dem_path + ' ' + aspect_path + ' ' + mask_path + ' ' + slope_path + ' ' + dem_path_tif_temp + ' '+ dem_path_tif_temp2)
print("Study area consists of ", np.nansum(mask[mask==1]), " glacier points")
