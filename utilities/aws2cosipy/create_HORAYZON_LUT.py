# Description: Compute gridded correction factor for downward direct shortwave
#              radiation from given DEM data (~30 m) and
#              mask all non-glacier grid cells according to the glacier outline.
#              Consider Earth's surface curvature.
#
# Important note: An Earthdata account is required and 'wget' has to be set
#                 (https://disc.gsfc.nasa.gov/data-access) to download NASADEM
#                 data successfully.
#
# Source of applied DEM data: https://lpdaac.usgs.gov/products/nasadem_hgtv001/
#
# Copyright (c) 2022 ETH Zurich, Christian R. Steger
# MIT License

##WORK IN PROGRESS!##

# Load modules
import os
import numpy as np
import subprocess
from netCDF4 import Dataset, date2num
import zipfile
from skyfield.api import load, wgs84
import time
import fiona
from rasterio.features import rasterize
from rasterio.transform import Affine
from shapely.geometry import shape
import datetime
import datetime as dt
import horayzon as hray
import xarray as xr
import sys
import xesmf as xe
import pandas as pd

sys.path.append("../../")
from utilities.aws2cosipy.crop_file_to_glacier import crop_file_to_glacier
from utilities.aws2cosipy.aws2cosipyConfig import WRF

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------

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

# set paths
regrid = True #regrid to coarser resolution
ellps = "WGS84"  # Earth's surface approximation (sphere, GRS80 or WGS84)
path_out = "../../data/static/HEF/"
file_sw_dir_cor = "LUT_HORAYZON_sw_dir_cor.nc"

static_file = "../../data/static/HEF/HEF_static_raw.nc" #path to high resolution dataset
coarse_static_file = "../../data/static/HEF/HEF_static_agg.nc" #Load coarse grid

# -----------------------------------------------------------------------------
# Prepare data and initialise Terrain class
# -----------------------------------------------------------------------------

# Check if output directory exists
if not os.path.isdir(path_out):
    os.makedirs(path_out, exist_ok=True)

# Load high resolution static data
ds = xr.open_dataset(static_file)
elevation = ds["HGT"].values
lon = ds["lon"].values
lat = ds["lat"].values

# Compute indices of inner domain -> needs to encompass everything in range for aggregation
slice_in = (slice(1,lat.shape[0]-1, None), slice(1, lon.shape[0]-1))

offset_0 = slice_in[0].start
offset_1 = slice_in[1].start
print("Inner domain size: " + str(elevation[slice_in].shape))

#orthometric height (-> height above mean sea level)
elevation_ortho = np.ascontiguousarray(elevation[slice_in])

# Compute ellipsoidal heights
elevation += hray.geoid.undulation(lon, lat, geoid="EGM96")  # [m]

# Compute glacier mask
mask_glacier = ds["MASK"].values
#set NaNs to zero, relict from create static file
mask_glacier[np.isnan(mask_glacier)] = 0
mask_glacier = mask_glacier.astype(bool)
mask_glacier = mask_glacier[slice_in]

#mask with buffer for aggregation to lower spatial resolutions
#set +- 11 grid cells to "glacier" to allow ensure regridding
ilist = []
jlist = []
for i in np.arange(0,mask_glacier.shape[0]):
    for j in np.arange(0,mask_glacier.shape[1]):
        if mask_glacier[i,j] == True:
            print("Grid cell is glacier.")
            ilist.append(i)
            jlist.append(j)
#create buffer around glacier
ix_latmin = np.min(ilist)
ix_latmax = np.max(ilist)
ix_lonmin = np.min(jlist)
ix_lonmax = np.max(jlist)

#Watch out that the large domain incorporates the buffer
slice_buffer = (slice(ix_latmin-11,ix_latmax+11), slice(ix_lonmin-11, ix_lonmax+11))
mask_glacier[slice_buffer] = True

# Compute ECEF coordinates
x_ecef, y_ecef, z_ecef = hray.transform.lonlat2ecef(*np.meshgrid(lon, lat),
                                                    elevation, ellps=ellps)
dem_dim_0, dem_dim_1 = elevation.shape

# Compute ENU coordinates
trans_ecef2enu = hray.transform.TransformerEcef2enu(
    lon_or=lon[int(len(lon) / 2)], lat_or=lat[int(len(lat) / 2)], ellps=ellps)
x_enu, y_enu, z_enu = hray.transform.ecef2enu(x_ecef, y_ecef, z_ecef,
                                              trans_ecef2enu)

# Compute unit vectors (up and north) in ENU coordinates for inner domain
vec_norm_ecef = hray.direction.surf_norm(*np.meshgrid(lon[slice_in[1]],
                                                      lat[slice_in[0]]))
vec_north_ecef = hray.direction.north_dir(x_ecef[slice_in], y_ecef[slice_in],
                                          z_ecef[slice_in], vec_norm_ecef,
                                          ellps=ellps)
del x_ecef, y_ecef, z_ecef
vec_norm_enu = hray.transform.ecef2enu_vector(vec_norm_ecef, trans_ecef2enu)
vec_north_enu = hray.transform.ecef2enu_vector(vec_north_ecef, trans_ecef2enu)
del vec_norm_ecef, vec_north_ecef

# Merge vertex coordinates and pad geometry buffer
# holds all the data
vert_grid = hray.auxiliary.rearrange_pad_buffer(x_enu, y_enu, z_enu)

# Compute rotation matrix (global ENU -> local ENU)

rot_mat_glob2loc = hray.transform.rotation_matrix_glob2loc(vec_north_enu,
                                                           vec_norm_enu)

del vec_north_enu

# Compute slope (in global ENU coordinates!)
slice_in_a1 = (slice(slice_in[0].start - 1, slice_in[0].stop + 1),
               slice(slice_in[1].start - 1, slice_in[1].stop + 1))

## Slope vs plain method -> for comparison later
vec_tilt_enu = \
    np.ascontiguousarray(hray.topo_param.slope_vector_meth(
        x_enu[slice_in_a1], y_enu[slice_in_a1], z_enu[slice_in_a1],
        rot_mat=rot_mat_glob2loc, output_rot=False)[1:-1, 1:-1])

# Compute surface enlargement factor
surf_enl_fac = 1.0 / (vec_norm_enu * vec_tilt_enu).sum(axis=2)
print("Surface enlargement factor (min/max): %.3f" % surf_enl_fac.min()
      + ", %.3f" % surf_enl_fac.max())

# Initialise terrain
mask = np.ones(vec_tilt_enu.shape[:2], dtype=np.uint8)
mask[~mask_glacier] = 0  # mask non-glacier grid cells

terrain = hray.shadow.Terrain()
dim_in_0, dim_in_1 = vec_tilt_enu.shape[0], vec_tilt_enu.shape[1]
terrain.initialise(vert_grid, dem_dim_0, dem_dim_1,
                   offset_0, offset_1, vec_tilt_enu, vec_norm_enu,
                   surf_enl_fac, mask=mask, elevation=elevation_ortho,
                   refrac_cor=False)
# -> neglect atmospheric refraction -> effect is weak due to high
#    surface elevation and thus low atmospheric surface pressure

# Load Skyfield data
load.directory = path_out
planets = load("de421.bsp")
sun = planets["sun"]
earth = planets["earth"]
loc_or = earth + wgs84.latlon(trans_ecef2enu.lat_or, trans_ecef2enu.lon_or)
# -> position lies on the surface of the ellipsoid by default

# -----------------------------------------------------------------------------
# Compute Slope and Aspect
# -----------------------------------------------------------------------------
# Compute slope (in local ENU coordinates!)
vec_tilt_enu_loc = \
    np.ascontiguousarray(hray.topo_param.slope_vector_meth(
        x_enu, y_enu, z_enu,
        rot_mat=rot_mat_glob2loc, output_rot=True)[1:-1, 1:-1])

# Compute slope angle and aspect (in local ENU coordinates)
slope = np.arccos(vec_tilt_enu_loc[:, :, 2].clip(max=1.0))
#beware of aspect orientation -> N = 0 in HORAYZON, adjust here
aspect = np.pi / 2.0 - np.arctan2(vec_tilt_enu_loc[:, :, 1],
                                  vec_tilt_enu_loc[:, :, 0])
aspect[aspect < 0.0] += np.pi * 2.0  # [0.0, 2.0 * np.pi]

#Create output file for HRZ
static_ds = xr.Dataset()
static_ds.coords['lat'] = lat[slice_buffer[0]]
static_ds.coords['lon'] = lon[slice_buffer[1]]
add_variable_along_latlon(static_ds, elevation_ortho[slice_buffer], "elevation", "m", "Orthometric Height")
add_variable_along_latlon(static_ds, np.rad2deg(slope)[slice_buffer], "slope", "degree", "Slope")
add_variable_along_latlon(static_ds, np.rad2deg(aspect)[slice_buffer], "aspect", "m", "Aspect measured clockwise from North")
add_variable_along_latlon(static_ds, surf_enl_fac[slice_buffer], "surf_enl_fac", "-", "Surface enlargement factor")

# -----------------------------------------------------------------------------
# Compute correction factor for direct downward shortwave radiation
# -----------------------------------------------------------------------------

# Create time axis
# time in UTC, set timeframe here
time_dt_beg = dt.datetime(2020, 1, 1, 0, 00, tzinfo=dt.timezone.utc)
time_dt_end = dt.datetime(2021, 1, 1, 0, 00, tzinfo=dt.timezone.utc)
dt_step = dt.timedelta(hours=1)
num_ts = int((time_dt_end - time_dt_beg) / dt_step)
ta = [time_dt_beg + dt_step * i for i in range(num_ts)]

# Add sw dir correction and regrid
comp_time_shadow = []
sw_dir_cor = np.zeros(vec_tilt_enu.shape[:2], dtype=np.float32)

##Load coarse grid
ds_coarse = xr.open_dataset(coarse_static_file)
ds_coarse['mask'] = ds_coarse['MASK'] #prepare for masked regridding

### Build regridder ###
#Create sample dataset to use regridding for
#Create data for first timestep.
ts = load.timescale()
t = ts.from_datetime(ta[0])
astrometric = loc_or.at(t).observe(sun)
alt, az, d = astrometric.apparent().altaz()
x_sun = d.m * np.cos(alt.radians) * np.sin(az.radians)
y_sun = d.m * np.cos(alt.radians) * np.cos(az.radians)
z_sun = d.m * np.sin(alt.radians)
sun_position = np.array([x_sun, y_sun, z_sun], dtype=np.float32)

terrain.sw_dir_cor(sun_position, sw_dir_cor)

result = xr.Dataset()
result.coords['time'] = [pd.to_datetime(ta[0])]
#ix_latmin-11:ix_latmax+11,ix_lonmin-11:ix_lonmax+11
result.coords['lat'] = lat[slice_buffer[0]]
result.coords['lon'] = lon[slice_buffer[1]]
sw_holder = np.zeros(shape=[1, lat[slice_buffer[0]].shape[0], lon[slice_buffer[1]].shape[0]])
sw_holder[0,:,:] = sw_dir_cor[slice_buffer]
mask_crop = mask[slice_buffer]
add_variable_along_timelatlon(result, sw_holder, "sw_dir_cor", "-", "correction factor for direct downward shortwave radiation")
add_variable_along_latlon(result, mask_crop, "mask", "-", "Boolean Glacier Mask")

#build regridder
if regrid == True:
    regrid_mask = xe.Regridder(result, ds_coarse, method="conservative_normed")

result.close()
del result
#Iterate over timesteps

datasets = []
for i in range(len(ta)): #loop over timesteps

    t_beg = time.time()

    ts = load.timescale()
    t = ts.from_datetime(ta[i])
    astrometric = loc_or.at(t).observe(sun)
    alt, az, d = astrometric.apparent().altaz()
    x_sun = d.m * np.cos(alt.radians) * np.sin(az.radians)
    y_sun = d.m * np.cos(alt.radians) * np.cos(az.radians)
    z_sun = d.m * np.sin(alt.radians)
    sun_position = np.array([x_sun, y_sun, z_sun], dtype=np.float32)

    terrain.sw_dir_cor(sun_position, sw_dir_cor)

    comp_time_shadow.append((time.time() - t_beg))
    
    result = xr.Dataset()
    result.coords['time'] = [pd.to_datetime(ta[i])]
    #ix_latmin-11:ix_latmax+11,ix_lonmin-11:ix_lonmax+11
    result.coords['lat'] = lat[slice_buffer[0]]
    result.coords['lon'] = lon[slice_buffer[1]]
    sw_holder = np.zeros(shape=[1, lat[slice_buffer[0]].shape[0], lon[slice_buffer[1]].shape[0]])
    sw_holder[0,:,:] = sw_dir_cor[slice_buffer]
    mask_crop = mask[slice_buffer]
    add_variable_along_timelatlon(result, sw_holder, "sw_dir_cor", "-", "correction factor for direct downward shortwave radiation")
    add_variable_along_latlon(result, mask_crop, "mask", "-", "Boolean Glacier Mask")
    
    now = time.time()
    if regrid == True:  
        datasets.append(regrid_mask(result))
        #print("regridding took:", time.time()-now)
    else:
        datasets.append(result)
    #Close and delete files to free memory
    result.close()
    del result

#afterwards regrid aspect and slope as they do not require time!
#still test if result is different when shape is not the same
#how to solve issue of time?

#Merge single timestep files
now = time.time()
ds_sw_cor = xr.concat(datasets, dim='time')
ds_sw_cor['time'] = pd.to_datetime(ds_sw_cor['time'].values)
if regrid == True:
    ds_sw_cor['MASK'] = ds_coarse['MASK'] #replace with original mask
else:
    ds_sw_cor['MASK'] = ds['MASK']
ds_sw_cor = ds_sw_cor[['sw_dir_cor','MASK']]
print("concat took:", time.time()-now)

time_tot = np.array(comp_time_shadow).sum()
print("Elapsed time (total / per time step): " + "%.2f" % time_tot
      + " , %.2f" % (time_tot / len(ta)) + " s")

#regrid static ds and merge with sw_dir_cor
if regrid == True:
    regrid_no_mask = xe.Regridder(static_ds, ds_coarse[["HGT"]], method="conservative_normed")
    regrid = regrid_no_mask(static_ds, ds_coarse[["HGT"]])
    combined = xr.merge([ds_sw_cor, regrid])
else:
    combined = xr.merge([ds_sw_cor, static_ds])

#BBox script to crop to minimal extent!
cropped_combined = crop_file_to_glacier(combined)
cropped_combined.to_netcdf(path_out+file_sw_dir_cor)
