"""
 This file reads the DEM of the study site and the shapefile and creates the needed static.nc
"""
import sys
import os
import xarray as xr
import numpy as np
from itertools import product
import richdem as rd

static_folder = '../../data/static/'

tile = True
aggregate = True

### input digital elevation model (DEM)
dem_path_tif = static_folder + 'DEM/morteratsch_dhm25_WGS84.tif'
### input shape of glacier or study area, e.g. from the Randolph glacier inventory
shape_path = static_folder + 'Shapefiles/Morteratsch_final.shp'
shape_path_artificial_snow = static_folder + 'Shapefiles/Artificial_Snow.shp'
### path were the static.nc file is saved
output_path = static_folder + 'mort_static_100m_artificial_snow.nc'

### to shrink the DEM use the following lat/lon corners
longitude_upper_left = '9.88'
latitude_upper_left = '46.44'
longitude_lower_right = '10.01'
latitude_lower_right = '46.33'

### to aggregate the DEM to a coarser spatial resolution
#aggregate_degree = '0.003'
aggregate_degree = '0.0009259259259'

### intermediate files, will be removed afterwards
dem_path_tif_temp = static_folder + 'DEM_temp.tif'
dem_path_tif_temp2 = static_folder + 'DEM_temp2.tif'
dem_path = static_folder + 'dem.nc'
aspect_path = static_folder + 'aspect.nc'
mask_path = static_folder + 'mask.nc'
mask_path_artificial_snow = static_folder + 'mask_artificial_snow.nc'
slope_path = static_folder + 'slope.nc'

### If you do not want to shrink the DEM, comment out the following to three lines
if tile:
    os.system('gdal_translate -projwin ' + longitude_upper_left + ' ' + latitude_upper_left + ' ' +
          longitude_lower_right + ' ' + latitude_lower_right + ' ' + dem_path_tif + ' ' + dem_path_tif_temp)
    dem_path_tif = dem_path_tif_temp

### If you do not want to aggregate DEM, comment out the following to two lines
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

### calculate mask artificial snow as NetCDF with DEM and shapefile
os.system('gdalwarp -of NETCDF  --config GDALWARP_IGNORE_BAD_CUTLINE YES -cutline ' + shape_path_artificial_snow + ' ' + dem_path_tif  + ' ' + mask_path_artificial_snow)

### open intermediate netcdf files
dem = xr.open_dataset(dem_path)
mask = xr.open_dataset(mask_path)
mask_artificial_snow = xr.open_dataset(mask_path_artificial_snow)
slope = xr.open_dataset(slope_path)

### set NaNs in mask to -9999 and elevation within the shape to 1
mask=mask.Band1.values
mask[np.isnan(mask)]=-9999
mask[mask>0]=1
print(mask)

mask_artificial_snow = mask_artificial_snow.Band1.values
mask_artificial_snow[np.isnan(mask_artificial_snow)]=-9999
mask_artificial_snow[mask_artificial_snow>0]=1
print(mask_artificial_snow)

# test if all grid points from artificial snow are located within the glacier area.
if(1 in mask_artificial_snow[np.where(np.equal(mask, mask_artificial_snow) == False)]):
        print('\n Warning: Not all grid points of artificial snow are within the glacier geometry! \n')

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

### function to insert variables to dataset
def insert_var(ds, var, name, units, long_name):
    ds[name] = (('lat','lon'), var)
    ds[name].attrs['units'] = units
    ds[name].attrs['long_name'] = long_name
    ds[name].attrs['_FillValue'] = -9999

### insert needed static variables
insert_var(ds, dem.Band1.values,'HGT','meters','meter above sea level')
insert_var(ds, aspect,'ASPECT','degrees','Aspect of slope')
insert_var(ds, slope.Band1.values,'SLOPE','degrees','Terrain slope')
insert_var(ds, mask,'MASK','boolean','Glacier mask')
insert_var(ds, mask_artificial_snow,'MASK_ARTIFICIAL_SNOW','boolean','Artificial Snow on glacier mask')

os.system('rm '+ dem_path + ' ' + mask_path + ' ' + slope_path + ' ' + dem_path_tif_temp + ' '+ dem_path_tif_temp2 + ' ' + mask_path_artificial_snow)

### save combined static file, delete intermediate files and print number of glacier grid points
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
check_for_nan(ds)
ds.to_netcdf(output_path)
print("Study area consists of ", np.nansum(mask[mask==1]), " glacier points")
print("Area with artificial snow consists of ", np.nansum(mask_artificial_snow[mask_artificial_snow==1]), "glacier points")
print("Done")
