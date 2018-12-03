import xarray as xr
import shapefile
import richdem as rd
import matplotlib.pyplot as plt
import numpy as np

dem_path = 'DEM_Hintereisferner.nc'
dem_path_tif = 'DEM_Hintereisferner.tif'
shape_path = 'HEF.shp'
output_path = 'static_py.nc'
sf = shapefile.Reader(shape_path)
#sf = salem.read_shapefile(shape_path)

dem = xr.open_dataset(dem_path)
#dem = salem.xr.open_dataset(dem_path)
dem_tif = rd.LoadGDAL(dem_path_tif)

ds = xr.Dataset()
ds.attrs['Conventions']='CF-1.4'
ds.attrs['crs']='+proj=longlat +datum=WGS84 +no_defs +ellps=WGS84 +towgs84=0,0,0'
ds.attrs['crs_format']='PROJ.4'

ds.coords['longitude'] = dem.lon.values
ds.longitude.attrs['standard_name'] = 'longitude'
ds.longitude.attrs['long_name'] = 'longitude'
ds.longitude.attrs['units'] = 'degrees_east'
ds.longitude.attrs['axis'] = 'X'

ds.coords['latitude'] = dem.lat.values
ds.latitude.attrs['standard_name'] = 'latitude'
ds.latitude.attrs['long_name'] = 'latitude'
ds.latitude.attrs['units'] = 'degrees_north'
ds.latitude.attrs['axis'] = 'Y'

slope = rd.TerrainAttribute(dem_tif, attrib='slope_percentage',zscale=111120)
rd.rdShow(slope, axes=False, cmap='magma', figsize=(8, 5.5))
plt.show()
aspect = rd.TerrainAttribute(dem_tif, attrib='aspect')
rd.rdShow(aspect, axes=False, cmap='jet', figsize=(8, 5.5))
plt.show()

def insert_var(ds, var, name, units, missing_value, long_name):
    ds[name] = (('latitude','longitude'), var)
    ds[name].attrs['units'] = units
    ds[name].attrs['long_name'] = long_name
    ds[name].attrs['missing_value'] = missing_value

insert_var(ds, dem.Band1.values,'HGT','meters',np.nan,'meter above sea level')
insert_var(ds, slope,'SLOPE','degress',np.nan,'Terrain slope')
insert_var(ds, aspect,'ASPECT','degress',np.nan,'Aspect of slope')

ds.to_netcdf(output_path)
