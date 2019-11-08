import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import os
import numpy as np
import xarray as xr
import argparse
from scipy.interpolate import griddata
from scipy import interpolate
import matplotlib.dates as mdates
import pandas as pd

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


def plot_profile_1D_timeseries(filename, var, domainy):
    
    # Get dataset
    ds = xr.open_dataset(filename)

    # Get first layer height
    fl = ds.attrs['First_layer_height_log_profile']

    # Get data
    if var=='T':
        V = ds.LAYER_T[:,0,0,:].values
        lb = (np.floor(np.nanmin(V)/10.0))*10
        levels = np.arange(lb, 273.16, 2)
        barLabel = 'Temperature [K]'
    if var=='RHO':
        V = ds.LAYER_RHO[:,0,0,:].values
        #cmap = lt.cm.bone
        levels = np.arange(0,550,50)
        barLabel = 'Density [kg m^-3]'
    if var=='IF':
        V = ds.LAYER_ICE_FRACTION[:,0,0,:].values
        #cmap = lt.cm.bone
        levels = np.arange(0,1,0.1)
        barLabel = 'Ice fraction [-]'
    if var=='LWC':
        V = ds.LAYER_LWC[:,0,0,:].values
        #cmap = lt.cm.bone
        levels = np.arange(0,1,0.1)
        barLabel = 'Liquid Water Content [-]'
    if var=='POR':
        V = ds.LAYER_POROSITY[:,0,0,:].values
        #cmap = lt.cm.bone
        levels = np.arange(0,1,0.1)
        barLabel = 'Air Porosity [-]'
        
        
    D = ds.LAYER_HEIGHT[:,0,0,:].values.cumsum(axis=1)
    
    # Get dimensions
    time = np.arange(ds.dims['time'])
    depth = ds.TOTALHEIGHT[:,0,0].values
    
    # Calc plotting domain height
    Dn = (np.int(np.floor(ds.TOTALHEIGHT.max()))+2)
    
    # Create new grid
    xi = time
    yi = np.arange(domainy, Dn, fl)
    X, Y = np.meshgrid(xi,yi)
    data = np.full_like(X, np.nan, dtype=np.double)
    
    # Re-calc depth data top=zero
    D = (-(D.transpose()-depth).transpose())

    for i in xi:
        # Get non-nan values
        Dsubset = D[i,~np.isnan(D[i,:])]
        Tsubset = V[i,~np.isnan(D[i,:])]
        idx = np.where(yi<=depth[i])
    
        f = interpolate.interp1d(Dsubset, Tsubset, fill_value="extrapolate")
        data[idx,i] = f(yi[idx])

    fig, ax = plt.subplots(figsize=(20, 10))
    CS = ax.contourf(X,Y,data,levels=levels,extend='max')
    
    N = pd.date_range(ds.time[0].values, ds.time[-1].values, freq='m')
    M = pd.date_range(ds.time[0].values, ds.time[-1].values, freq='H')
    
    labIdx = []
    label = []
    for q in N:
        o = np.where(M==q)
        labIdx.append(o[0][0])
        label.append(q.strftime('%Y-%m-%d'))
    
    plt.xticks(labIdx, label,rotation=45,fontsize=16,weight='normal')
    plt.ylabel('Depth [m]',fontsize=16, weight='normal')
    plt.xlabel('Date',fontsize=16,weight='normal')
    plt.title(var+'-Profile',fontsize=16, weight='normal')
    
    # Make a colorbar for the ContourSet returned by the contourf call.
    cbar = plt.colorbar(CS)
    cbar.ax.set_ylabel(barLabel,fontsize=16,fontname = "Helvetica", weight='normal')
    cbar.ax.set_yticks(barLabel,fontname = "Helvetica", weight='normal')

    plt.show()


def plot_profile_1D(filename, pdate, d):
    """ This creates a simple plot showing the 2D fields"""

    DATA = xr.open_dataset(filename)
    print(DATA)
    DATA = DATA.sel(time=pdate)
    plt.figure(figsize=(5, 5))
    depth = np.append(0,np.cumsum(DATA.LAYER_HEIGHT.values))
    rho = np.append(DATA.LAYER_RHO[:,:,0],DATA.LAYER_RHO.values)
    t = np.append(DATA.LAYER_T[:,:,0],DATA.LAYER_T.values)
    
    idx, val = find_nearest(depth,d)
    print('nearest depth: ', val)
    print('density: ',rho[idx])
    print('temperature: ',t[idx])

    plt.step(rho,depth)
    ax1 = plt.gca()
    ax1.invert_yaxis()
    ax1.set_ylabel('Depth [m]')
    ax1.tick_params(axis='x', labelcolor='blue')
    ax1.set_xlabel('Density [kg m^-3]', color='blue')
    ax2 = ax1.twiny()
    ax2.plot(t,depth, color='red')
    ax2.set_xlabel('Temperature [K]', color='red')
    ax2.tick_params(axis='x', labelcolor='red')
    plt.show()


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.nanargmin(np.abs(array - value)))
    return (idx,array[idx])

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
    parser.add_argument('-e', '-depth', dest='d', help='depth', type=float)

    args = parser.parse_args()
    
    if (args.lat is None) & (args.lon is None) & (args.pdate is None):
        plot_profile_1D_timeseries(args.file, args.var, args.d)

    if (args.lat is None) & (args.lon is None) & (args.pdate is not None):
        plot_profile_1D(args.file, args.pdate, args.d) 
