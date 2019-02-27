import xarray as xr
import richdem as rd
import matplotlib.pyplot as plt
import netCDF4
import fiona
import rasterio.mask
import rasterio

dem_path = 'DEM_Hintereisferner.nc'
dem_path_tif = 'DEM_Hintereisferner.tif'
shape_path = 'Hintereisferner_RGI_6.shp'
output_path = 'static_py.nc'

print(dem_path)

dem = xr.open_dataset(dem_path)
dem_tif = rd.LoadGDAL(dem_path_tif)
dem_tif2 = rasterio.open(dem_path_tif)

with fiona.open(shape_path, "r") as shapefile:
    features = [feature["geometry"] for feature in shapefile]

with rasterio.open(dem_path_tif) as src:
    out_image, out_transform = rasterio.mask.mask(src, features, crop=False, nodata=-9999)
    out_meta = src.meta.copy()

slope=xr.DataArray(rd.TerrainAttribute(dem_tif, attrib='slope_degrees'))
#rd.rdShow(slope, axes=False, cmap='magma', figsize=(8, 5.5))
#plt.show()

aspect = xr.DataArray(rd.TerrainAttribute(dem_tif, attrib='aspect'))
#rd.rdShow(aspect, axes=False, cmap='jet', figsize=(8, 5.5))
#plt.show()

mask = out_image[0,:,:]
mask[mask>0]=1

ds = xr.Dataset()

ds.coords['longitude'] = dem.lon.values
ds.longitude.attrs['standard_name'] = 'longitude'
ds.longitude.attrs['long_name'] = 'longitude'
ds.longitude.attrs['units'] = 'degrees_east'

ds.coords['latitude'] = dem.lat.values
ds.latitude.attrs['standard_name'] = 'latitude'
ds.latitude.attrs['long_name'] = 'latitude'
ds.latitude.attrs['units'] = 'degrees_north'


def insert_var(ds, var, name, units, long_name):
    ds[name] = (('latitude','longitude'), var)
    ds[name].attrs['units'] = units
    ds[name].attrs['long_name'] = long_name
    ds[name].attrs['_FillValue'] = -9999
    ds[name].attrs['Longitude'] = -9999

insert_var(ds, dem.Band1.values,'HGT','meters','meter above sea level')
insert_var(ds, slope.values,'SLOPE','degress','Terrain slope')
insert_var(ds, aspect.values,'ASPECT','degress','Aspect of slope')
insert_var(ds, mask,'MASK','boolean','Glacier mask')

ds.to_netcdf(output_path)

#ds.attrs['Conventions']='CF-1.4'
#ds.attrs['crs']='+proj=longlat +datum=WGS84 +no_defs +ellps=WGS84 +towgs84=0,0,0'
#ds.attrs['crs_format']='PROJ.4'
#ds.longitude.attrs['axis'] = 'X'
#ds.latitude.attrs['axis'] = 'Y'
