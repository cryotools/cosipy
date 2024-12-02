"""
Creates the ray-tracing based correction factors for incoming shortwave radiation. 
Processing routine is based on the HORAYZON package (https://github.com/ChristianSteger/HORAYZON).
Please refer to the publication by Steger et al., (2022): https://gmd.copernicus.org/articles/15/6817/2022/.

This script requires the installation of HORAYZON and xesmf (https://xesmf.readthedocs.io/en/stable/).

Prerequisite is a static file that incorporates surrounding terrain, created through the COSIPY utilities.
If regridding is desired, correction factors should be calculated at high resolution first and then regridded to lower resolution
instead of calculation correction factors on low resolution static files directly.

Usage:

python -m cosipy.utilities.aws2cosipy.createHORAYZONLUT -s <input static data> -o <output name> [-c <coarse static data>]\
    [-r <regrid>] [-e <elevation profile>] [-es <elevation bin size>] [-d <elevation band static output] 
    
Options and arguments:

Required arguments:
    -s, --static <path>                         Path to the static file that should be used for the calculation.
    -o, --output <path>                         Path to the resulting netCDF file.

All other arguments are supplied with a default behaviour of -r False, -e False, -c None, -es 30, -d None
Elevation bands are only calculated when regridding is inactive which will be enforced if -e True.
Optional arguments:
    -c, --coarse-static <path>                  Path to the coarse static file to which the results should be regridded. Only used when -r True 
    -r, --regridding <str>                      String conveted to boolean information on whether to regrid or not
    -e, --elevation_prof <str>                  String converted to boolean information on whether to compute 1D elevation bands or not
    -es, --elevation_size <int>                 Elevation bin size, used when -e True
    -d, --elev_data <path>                       Only used when -e True: Path where elevation bands static file should be written to.
"""

import argparse
import numpy as np
from skyfield.api import load, wgs84
import time
import datetime as dt
import horayzon as hray
import xarray as xr
import xesmf as xe
import pandas as pd

from cosipy.utilities.config_utils import UtilitiesConfig

_args = None
_cfg = None

ellps = "WGS84"  # Earth's surface approximation (sphere, GRS80 or WGS84)

# ----------------------------
# Some helper functions
# ----------------------------

#https://stackoverflow.com/a/43357954
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def add_variable_along_timelatlon(ds, var, name, units, long_name):
    """ This function adds missing variables to the DATA class """
    ds[name] = (('time','lat','lon'), var)
    ds[name].attrs['units'] = units
    ds[name].attrs['long_name'] = long_name
    return ds

def add_variable_along_latlon(ds, var, name, units, long_name):
    """ This function self.adds missing variables to the self.DATA class """
    ds[name] = (('lat','lon'), var)
    ds[name].attrs['units'] = units
    ds[name].attrs['long_name'] = long_name
    ds[name].encoding['_FillValue'] = -9999
    return ds

### function to assign attributes to variable
def assign_attrs(ds, name, units, long_name):
    ds[name].attrs['units'] = units
    ds[name].attrs['long_name'] = long_name
    ds[name].attrs['_FillValue'] = -9999

#function for mean of circular values strongly inspired by http://webspace.ship.edu/pgmarr/Geo441/Lectures/Lec%201
def aspect_means(x):
    mean_sine = np.nanmean(np.sin(np.radians(x)))
    mean_cosine = np.nanmean(np.cos(np.radians(x)))
    r = np.sqrt(mean_cosine**2 + mean_sine**2)
    cos_mean = mean_cosine/r
    sin_mean = mean_sine/r
    mean_angle = np.arctan2(sin_mean, cos_mean)
    return np.degrees(mean_angle) 


def calculate_1d_elevationband(xds, elevation_var, mask_var, var_of_interest, elev_bandsize, slice_idx=None):
    
    ## first mask vals
    xds = xds.where(xds[mask_var] == 1, drop=True)
    
    #test groupby bins
    full_elev_range = xds[elevation_var].values[xds[mask_var] == 1]
    bins = np.arange(np.nanmin(full_elev_range), np.nanmax(full_elev_range)+elev_bandsize, elev_bandsize)
    labels = bins[:-1] + elev_bandsize/2
    
    if var_of_interest in ["lat","lon"]:
        values = []
        for i in (bins):
            sub = xds.where((xds[mask_var] == 1) & (xds[elevation_var] >= i) & (xds[elevation_var] < i + elev_bandsize), drop=True)
            values.append(np.nanmean(sub[var_of_interest].values))
    elif var_of_interest == "ASPECT":
        elvs = xds[elevation_var].values.flatten()[xds[mask_var].values.flatten() == 1]
        aspects = xds[var_of_interest].values.flatten()[xds[mask_var].values.flatten() == 1]
        values = []
        for i in (bins):
            values.append(aspect_means(aspects[np.logical_and(elvs >= i, elvs < i+elev_bandsize)]))
    elif var_of_interest == mask_var:
        if slice_idx is None:
            values = xds[var_of_interest].groupby_bins(xds[elevation_var], bins, labels=labels, include_lowest=True).sum(skipna=True, min_count=1)
        else:
            values = xds[var_of_interest][slice_idx].groupby_bins(xds[elevation_var][slice_idx], bins, labels=labels, include_lowest=True).sum(skipna=True, min_count=1)
        ## below calculation doesnt work
    #elif var_of_interest in ["aspect","ASPECT"]:
    #    if slice_idx is None:          
    #        values = xds[var_of_interest].groupby_bins(xds[elevation_var], bins, labels=labels, include_lowest=True).map(aspect_means)
    #    else:
    #        values = xds[var_of_interest][slice_idx].groupby_bins(xds[elevation_var][slice_idx], bins, labels=labels, include_lowest=True).map(aspect_means)
    else:
        if slice_idx is None:
            values = xds[var_of_interest].groupby_bins(xds[elevation_var], bins, labels=labels, include_lowest=True).mean(skipna=True)
            
        else:
            values = xds[var_of_interest][slice_idx].groupby_bins(xds[elevation_var][slice_idx], bins, labels=labels, include_lowest=True).mean(skipna=True)
    
    return values    

def construct_1d_dataset(df):
    elev_ds = df.to_xarray()
    elev_ds.lon.attrs['standard_name'] = 'lon'
    elev_ds.lon.attrs['long_name'] = 'longitude'
    elev_ds.lon.attrs['units'] = 'Average Lon of elevation bands'
    
    elev_ds.lat.attrs['standard_name'] = 'lat'
    elev_ds.lat.attrs['long_name'] = 'latitude'
    elev_ds.lat.attrs['units'] = 'Average Lat of elevation bands'
    assign_attrs(elev_ds, 'HGT','meters','Mean of elevation range per bin as meter above sea level')
    assign_attrs(elev_ds, 'ASPECT','degrees','Mean Aspect of slope')
    assign_attrs(elev_ds, 'SLOPE','degrees','Mean Terrain slope')
    assign_attrs(elev_ds, 'MASK','boolean','Glacier mask')
    assign_attrs(elev_ds, 'N_Points','count','Number of Points in each bin')
    assign_attrs(elev_ds, 'sw_dir_cor','-','Average shortwave radiation correction factor per elevation band')
    
    return elev_ds

def compute_and_slice(latitudes, longitudes, mask_obj):
    # Compute indices of inner domain -> needs to encompass everything in range for aggregation
    slice_in = (slice(1,latitudes.shape[0]-1, None), slice(1, longitudes.shape[0]-1))

    # Compute glacier mask
    mask_glacier_original = mask_obj
    #set NaNs to zero, relict from create static file
    mask_glacier_original[np.isnan(mask_glacier_original)] = 0
    mask_glacier = mask_glacier_original.astype(bool)
    mask_glacier = mask_glacier[slice_in] #-1 -1 verywhere

    #mask with buffer for aggregation to lower spatial resolutions
    #set +- 11 grid cells to "glacier" to allow ensure regridding
    ilist = []
    jlist = []
    ## Note that this list is not based on the original shape, see slice_in above
    for i in np.arange(0,mask_glacier.shape[0]):
        for j in np.arange(0,mask_glacier.shape[1]):
            if mask_glacier[i,j] == True:
                #print("Grid cell is glacier.")
                ilist.append(i)
                jlist.append(j)
    #create buffer around glacier
    ix_latmin = np.min(ilist)
    ix_latmax = np.max(ilist)
    ix_lonmin = np.min(jlist)
    ix_lonmax = np.max(jlist)

    #Watch out that the large domain incorporates the buffer - here selected 11 grid cells
    slice_buffer = (slice(ix_latmin-11,ix_latmax+11), slice(ix_lonmin-11, ix_lonmax+11))
    mask_glacier[slice_buffer] = True
    
    return (slice_in, slice_buffer, mask_glacier, mask_glacier_original)

def compute_coords(lat, lon, elevation, slice_in):
        # Compute ECEF coordinates
    x_ecef, y_ecef, z_ecef = hray.transform.lonlat2ecef(*np.meshgrid(lon, lat),
                                                        elevation, ellps=ellps)

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
        
    return (vert_grid, vec_norm_enu, vec_tilt_enu, trans_ecef2enu,
            x_enu, y_enu, z_enu, rot_mat_glob2loc)

def merge_timestep_files(datasets, regrid, ds_coarse, static_ds,
                         elevation_profile, mask_glacier_original, slice_in, slice_buffer,
                         ):
    #Merge single timestep files
    now = time.time()
    ds_sw_cor = xr.concat(datasets, dim='time')
    ds_sw_cor['time'] = pd.to_datetime(ds_sw_cor['time'].values)
    if regrid == True:
        ds_sw_cor['MASK'] = ds_coarse["MASK"] #replace with original mask, should have same dimensions
    else:
        #dont have same dimensions
        if elevation_profile == True:
            ds_sw_cor['HGT'] = ds_sw_cor['HGT'].isel(time=0)
            ds_sw_cor['ASPECT'] = ds_sw_cor['ASPECT'].isel(time=0)
            ds_sw_cor['SLOPE'] = ds_sw_cor['SLOPE'].isel(time=0)
            ds_sw_cor['MASK'] = ds_sw_cor['MASK'].isel(time=0)
            ds_sw_cor['N_Points'] = ds_sw_cor['N_Points'].isel(time=0)
        else:
            mask_holder = mask_glacier_original[slice_in]
            add_variable_along_latlon(ds_sw_cor,mask_holder[slice_buffer], "MASK", "-", "Actual Glacier Mask" )


    ds_sw_cor['MASK'] = (('lat','lon'),np.where(ds_sw_cor['MASK'] == 1, ds_sw_cor['MASK'], np.nan))
    if elevation_profile == False:
        ds_sw_cor = ds_sw_cor[['sw_dir_cor','MASK']]
    print("concat took:", time.time()-now)

    #regrid static ds and merge with sw_dir_cor
    if regrid == True:
        regrid_no_mask = xe.Regridder(static_ds, ds_coarse[["HGT"]], method="conservative_normed")
        regrid = regrid_no_mask(static_ds, ds_coarse[["HGT"]])
        combined = xr.merge([ds_sw_cor, regrid])
    else:
        if elevation_profile == True:
            combined = ds_sw_cor.copy()
        if elevation_profile == False:
            combined = xr.merge([ds_sw_cor, static_ds])
            
    return combined


def run_horayzon_scheme(static_file, file_sw_dir_cor, coarse_static_file=None,
                        regrid=False, elevation_profile=False,
                        elev_bandsize=10, elev_stat_file=None):
    # -----------------------------------------------------------------------------
    # Prepare data and initialise Terrain class
    # -----------------------------------------------------------------------------
    if elevation_profile == True:
        print("Routine check. Regrid Option is set to: ", regrid)
        print("Setting regrid to False.")
        print("Elevation band size is set to: ", elev_bandsize, "m")
        regrid = False



    # Load high resolution static data
    ds = xr.open_dataset(static_file)
    elevation = ds["HGT"].values.copy() #else values get overwritten by later line
    elevation_original = ds["HGT"].values.copy()
    lon = ds["lon"].values
    lat = ds["lat"].values

    slice_in, slice_buffer, mask_glacier, mask_glacier_original = compute_and_slice(lat, lon, ds["MASK"].values)
    
    print("Inner domain size: " + str(elevation[slice_in].shape))
    
    #orthometric height (-> height above mean sea level)
    elevation_ortho = np.ascontiguousarray(elevation[slice_in])
    
    # Compute ellipsoidal heights
    elevation += hray.geoid.undulation(lon, lat, geoid="EGM96")  # [m]
    
    offset_0 = slice_in[0].start
    offset_1 = slice_in[1].start

    dem_dim_0, dem_dim_1 = elevation.shape

    vert_grid, vec_norm_enu, vec_tilt_enu, trans_ecef2enu,\
        x_enu, y_enu, z_enu, rot_mat_glob2loc = compute_coords(lat, lon, elevation, slice_in)

    # Compute surface enlargement factor
    surf_enl_fac = 1.0 / (vec_norm_enu * vec_tilt_enu).sum(axis=2)
    print("Surface enlargement factor (min/max): %.3f" % surf_enl_fac.min()
        + ", %.3f" % surf_enl_fac.max())

    # Initialise terrain
    mask = np.ones(vec_tilt_enu.shape[:2], dtype=np.uint8)
    mask[~mask_glacier] = 0  # mask non-glacier grid cells

    terrain = hray.shadow.Terrain()
    #dim_in_0, dim_in_1 = vec_tilt_enu.shape[0], vec_tilt_enu.shape[1]
    terrain.initialise(vert_grid, dem_dim_0, dem_dim_1,
                    offset_0, offset_1, vec_tilt_enu, vec_norm_enu,
                    surf_enl_fac, mask=mask, elevation=elevation_ortho,
                    refrac_cor=False)
    # -> neglect atmospheric refraction -> effect is weak due to high
    #    surface elevation and thus low atmospheric surface pressure

    # Load Skyfield data
    load.directory = _cfg.paths['static_folder']
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
    if coarse_static_file != None:
        ds_coarse = xr.open_dataset(coarse_static_file)
        ds_coarse['mask'] = ds_coarse['MASK'] #prepare for masked regridding
    else:
        #assign empty var
        ds_coarse = None

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

    ## Construct regridder outside of loop - create empty place holder netcdf
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

    ## Note: this script is slower than the original, because we are creating a dataset
    ## for each individual timeframe. This could be optimised when there is no regridding required.
    ## To allow for a smooth run of both, we ignore this time constraint, as the script is still comparatively fast

    datasets = []
    for i in range(len(ta)): #loop over timesteps

        t_beg = time.time()

        ts = load.timescale()
        t = ts.from_datetime(ta[i])
        print(i)
        astrometric = loc_or.at(t).observe(sun)
        alt, az, d = astrometric.apparent().altaz()
        x_sun = d.m * np.cos(alt.radians) * np.sin(az.radians)
        y_sun = d.m * np.cos(alt.radians) * np.cos(az.radians)
        z_sun = d.m * np.sin(alt.radians)
        sun_position = np.array([x_sun, y_sun, z_sun], dtype=np.float32)

        terrain.sw_dir_cor(sun_position, sw_dir_cor)

        comp_time_shadow.append((time.time() - t_beg))
        
        ## first create distributed 2d xarray
        result = xr.Dataset()
        result.coords['time'] = [pd.to_datetime(ta[i])]
        #ix_latmin-11:ix_latmax+11,ix_lonmin-11:ix_lonmax+11
        result.coords['lat'] = lat[slice_buffer[0]]
        result.coords['lon'] = lon[slice_buffer[1]]
        sw_holder = np.zeros(shape=[1, lat[slice_buffer[0]].shape[0], lon[slice_buffer[1]].shape[0]])
        sw_holder[0,:,:] = sw_dir_cor[slice_buffer]
        ## this sets the whole small domain to mask == 1 - might have issues in regridding (considers values outside actual mask) but nvm
        mask_crop = mask[slice_buffer]
        add_variable_along_timelatlon(result, sw_holder, "sw_dir_cor", "-", "correction factor for direct downward shortwave radiation")
        # XESMF regridding requires fieldname "mask" to notice it
        add_variable_along_latlon(result, mask_crop, "mask", "-", "Boolean Glacier Mask")
        
        if elevation_profile == True:
            elev_holder = elevation_original[slice_in] #could also use elevation_ortho here
            mask_holder = mask_glacier_original[slice_in]
            add_variable_along_latlon(result,elev_holder[slice_buffer], "HGT", "m asl", "Surface elevation" )
            ## load actual mask
            add_variable_along_latlon(result,mask_holder[slice_buffer], "mask_real", "-", "Actual Glacier Mask" )
            
            full_elev_range = result["HGT"].values[result["mask_real"] == 1]
            bins = np.arange(np.nanmin(full_elev_range), np.nanmax(full_elev_range)+elev_bandsize, elev_bandsize)
            labels = bins[:-1] + elev_bandsize/2
            
            placeholder = {}
            for var in ["SLOPE","ASPECT","lat","lon"]:
                placeholder[var] = calculate_1d_elevationband(ds, "HGT", "MASK", var, elev_bandsize)
            
            for var in ["sw_dir_cor","mask_real"]:
                placeholder[var] = calculate_1d_elevationband(result, "HGT", "mask_real", var, elev_bandsize)
            
            ## construct the dataframe and xarray dataset
            #This is the crudest and most simplest try and here I want to avoid having a 26x26 grid filled with NaNs due to computational time
            mask_elev = np.ones_like(placeholder['lat'][:-1])
            ## Suggest all points on glacier
            df = pd.DataFrame({'lat':placeholder['lat'][:-1],
                            'lon': np.mean(placeholder['lon'][:-1]), #just assign the same value for now for simplicity
                            'time': pd.to_datetime(ta[i]), 
                            'HGT': labels,
                            'ASPECT': placeholder['ASPECT'][:-1],        
                            'SLOPE': placeholder['SLOPE'].data,
                            'MASK': mask_elev,
                            'N_Points': placeholder["mask_real"].data,
                            'sw_dir_cor': placeholder["sw_dir_cor"][0,:].data})
            
            #drop the timezone argument from pandas datetime object to ensure fluent conversion into xarray
            df['time'] = df['time'].dt.tz_localize(None)
            ##sort values by index vars, just in case
            df.sort_values(by=["time","lat","lon"], inplace=True)
            ## if elevation bins are too small, we will not have a unique index .. manual adjust that
            # the latitude information is not really useful anymore if we use HORAYZON, so adjust it
            try:
                df['lat'] = df['lat'] + df['HGT']*1e-9
                df.set_index(['time','lat','lon'], inplace=True)
                print(df)
                elev_ds = construct_1d_dataset(df)
            except:
                df['lat'] = df['lat'] + df['HGT']*1e-8
                df.set_index(['time','lat','lon'], inplace=True)
                print(df)
                elev_ds = construct_1d_dataset(df)
        
        if regrid == True:  
            datasets.append(regrid_mask(result))
            #print("regridding took:", time.time()-now)
        else:
            if elevation_profile == True:
                datasets.append(elev_ds)
                elev_ds.close()
                del elev_ds
                del df
            else:
                datasets.append(result)
        #Close and delete files to free memory
        result.close()
        del result

    time_tot = np.array(comp_time_shadow).sum()
    print("Elapsed time (total / per time step): " + "%.2f" % time_tot
        + " , %.2f" % (time_tot / len(ta)) + " s")
    
    combined = merge_timestep_files(datasets=datasets, regrid=regrid, ds_coarse=ds_coarse,
                                    static_ds=static_ds,elevation_profile=elevation_profile,
                                    mask_glacier_original=mask_glacier_original,
                                    slice_in=slice_in, 
                                    slice_buffer=slice_buffer
                                    )

    #BBox script to crop to minimal extent!
    if elevation_profile == True:
        combined.to_netcdf(file_sw_dir_cor)
        combined[['HGT','ASPECT','SLOPE','MASK','N_Points']].to_netcdf(elev_stat_file)
    else:
        cropped_combined = combined.where(combined.MASK == 1, drop=True)
        cropped_combined.to_netcdf(file_sw_dir_cor)

#### !! BEWARE: elevation is not the same when using regridding. Files are the same when using 1D approach.


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
        "-s",
        "--static",
        dest="static_file",
        type=str,
        metavar="<path>",
        required=True,
        help="Path to .nc file with static data",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="file_sw_dir_cor",
        type=str,
        metavar="<path>",
        required=True,
        help="Path to the resulting netCDF file",
    )
    parser.add_argument(
        "-c",
        "--coarse-static",
        dest="coarse_static_file",
        type=str,
        #const=None,
        default=None,
        metavar="<path>",
        required=False,
        help="Path to coarse static file",
    )
    parser.add_argument(
        "-r",
        "--regridding",
        type=str2bool,
        #const=False,
        default=False,
        dest="regrid",
        required=False,
        help="Boolean whether to regrid or not.",
    )
    parser.add_argument(
        "-e",
        "--elevation_prof",
        type=str2bool,
        #const=False,
        default=False,
        dest="elevation_profile",
        required=False,
        help="Boolean whether to calculate 1D elevation bands",
    )
    parser.add_argument(
        "-es",
        "--elevation_size",
        type=int,
        dest="elev_bandsize",
        #const=None,
        default=30,
        required=False,
        help="Integer on the size of the elevation bands in meters",
    )
    parser.add_argument(
        "-d",
        "--elev_data",
        dest="elev_stat_file",
        type=str,
        #const=None,
        default=None,
        metavar="<path>",
        required=False,
        help="Left longitude value of the subset",
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


def main():
    global _args  # Yes, it's bad practice
    global _cfg
    _args, _cfg = load_config(module_name="create_static")
    _args, _cfg = load_config(module_name="create_static")

    run_horayzon_scheme(
        _args.static_file,
        _args.file_sw_dir_cor,
        _args.coarse_static_file,
        _args.regrid,
        _args.elevation_profile,
        _args.elev_bandsize,
        _args.elev_stat_file
    )


if __name__ == "__main__":
    main()
