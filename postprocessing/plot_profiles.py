import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import os
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import argparse

def plot_profile(filename, pdate, lat, lon):
    """ This creates a simple plot showing the 2D fields"""
    DATA = xr.open_dataset(filename)

    (c_y, c_x) = naive_fast(DATA.lat.values, DATA.lon.values, lat, lon)
    DATA = DATA.sel(time=pdate,west_east=c_x,south_north=c_y)
    
    plt.figure(figsize=(20, 12))
 
    depth = np.append(0,np.cumsum(DATA.LAYER_HEIGHT.values))
    rho = np.append(DATA.LAYER_RHO[0],DATA.LAYER_RHO.values)
    plt.step(rho,depth)
    ax = plt.gca()
    ax.invert_yaxis()
    plt.show()

def naive_fast(latvar,lonvar,lat0,lon0):
    # Read latitude and longitude from file into numpy arrays
    latvals = latvar[:]
    lonvals = lonvar[:]
    dist_sq = (latvals-lat0)**2 + (lonvals-lon0)**2
    minindex_flattened = dist_sq.argmin()  # 1D index of min element
    iy_min,ix_min = np.unravel_index(minindex_flattened, latvals.shape)
    return iy_min,ix_min
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Quick plot of the results file.')
    parser.add_argument('-f', '-file', dest='file', help='File name')
    parser.add_argument('-d', '-date', dest='pdate', help='Which date to plot')
    parser.add_argument('-v', '-var', dest='var', help='Which variable')
    parser.add_argument('-n', '-lat', dest='lat', help='Latitude', type=float)
    parser.add_argument('-m', '-lon', dest='lon', help='Longitude', type=float)

    args = parser.parse_args()
    
    if (args.lat is not None) & (args.lon is not None) & (args.pdate is not None):
        print('Profile')
        plot_profile(args.file, args.pdate, args.lat, args.lon) 
    
