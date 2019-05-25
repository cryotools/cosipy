import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import os
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import argparse

def plotMesh(filename, pdate, var=None):
    """ This creates a simple plot showing the 2D fields"""
    DATA = xr.open_dataset(filename)
    
    plt.figure(figsize=(20, 12))
    print(DATA)

    if var is None:
        ax = plt.subplot(4,3,1,projection=ccrs.PlateCarree())
        DATA.H.sel(time=pdate).plot.pcolormesh('lon', 'lat', ax=ax);
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=1, color='gray', alpha=0.1, linestyle='--')
        gl.xlabels_top = None
        gl.ylabels_right = False
        
        ax = plt.subplot(4,3,2,projection=ccrs.PlateCarree())
        DATA.LE.sel(time=pdate).plot.pcolormesh('lon', 'lat', ax=ax);
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=1, color='gray', alpha=0.1, linestyle='--')
        gl.xlabels_top = None
        gl.ylabels_right = False
        
        ax = plt.subplot(4,3,3,projection=ccrs.PlateCarree())
        DATA.B.sel(time=pdate).plot.pcolormesh('lon', 'lat', ax=ax);
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=1, color='gray', alpha=0.1, linestyle='--')
        gl.xlabels_top = None
        gl.ylabels_right = False
        
        ax = plt.subplot(4,3,4,projection=ccrs.PlateCarree())
        DATA.SNOWHEIGHT.sel(time=pdate).plot.pcolormesh('lon', 'lat', ax=ax);
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=1, color='gray', alpha=0.1, linestyle='--')
        gl.xlabels_top = None
        gl.ylabels_right = False
        
        ax = plt.subplot(4,3,5,projection=ccrs.PlateCarree())
        DATA.surfM.sel(time=pdate).plot.pcolormesh('lon', 'lat', ax=ax);
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=1, color='gray', alpha=0.1, linestyle='--')
        gl.xlabels_top = None
        gl.ylabels_right = False

        ax = plt.subplot(4,3,6,projection=ccrs.PlateCarree())
        DATA.TS.sel(time=pdate).plot.pcolormesh('lon', 'lat', ax=ax);
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=1, color='gray', alpha=0.1, linestyle='--')
        gl.xlabels_top = None
        gl.ylabels_right = False
        
        ax = plt.subplot(4,3,7,projection=ccrs.PlateCarree())
        DATA.G.sel(time=pdate).plot.pcolormesh('lon', 'lat', ax=ax);
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=1, color='gray', alpha=0.1, linestyle='--')
        gl.xlabels_top = None
        gl.ylabels_right = False
        
        ax = plt.subplot(4,3,8,projection=ccrs.PlateCarree())
        DATA.Q.sel(time=pdate).plot.pcolormesh('lon', 'lat', ax=ax);
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=1, color='gray', alpha=0.1, linestyle='--')
        gl.xlabels_top = None
        gl.ylabels_right = False
        
        ax = plt.subplot(4,3,9,projection=ccrs.PlateCarree())
        DATA.LWout.sel(time=pdate).plot.pcolormesh('lon', 'lat', ax=ax);
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=1, color='gray', alpha=0.1, linestyle='--')
        gl.xlabels_top = None
        gl.ylabels_right = False
        
        ax = plt.subplot(4,3,10,projection=ccrs.PlateCarree())
        DATA.surfMB.sel(time=pdate).plot.pcolormesh('lon', 'lat', ax=ax);
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=1, color='gray', alpha=0.1, linestyle='--')
        gl.xlabels_top = None
        gl.ylabels_right = False
        
        ax = plt.subplot(4,3,11,projection=ccrs.PlateCarree())
        DATA.MB.sel(time=pdate).plot.pcolormesh('lon', 'lat', ax=ax);
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=1, color='gray', alpha=0.1, linestyle='--')
        gl.xlabels_top = None
        gl.ylabels_right = False
        
        ax = plt.subplot(4,3,12,projection=ccrs.PlateCarree())
        DATA.REFREEZE.sel(time=pdate).plot.pcolormesh('lon', 'lat', ax=ax);
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=1, color='gray', alpha=0.1, linestyle='--')
        gl.xlabels_top = None
        gl.ylabels_right = False

    
    ax.coastlines()
    plt.show()

def plotContour(filename, pdate, var=None):
    """ This creates a simple plot showing the 2D fields"""
    DATA = xr.open_dataset(filename)
    
    plt.figure(figsize=(20, 12))
    print(DATA)

    if var is None:
        ax = plt.subplot(4,3,1,projection=ccrs.PlateCarree())
        DATA.H.sel(time=pdate).plot.contourf('lon', 'lat', ax=ax);
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=1, color='gray', alpha=0.1, linestyle='--')
        gl.xlabels_top = None
        gl.ylabels_right = False
        
        ax = plt.subplot(4,3,2,projection=ccrs.PlateCarree())
        DATA.LE.sel(time=pdate).plot.contourf('lon', 'lat', ax=ax);
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=1, color='gray', alpha=0.1, linestyle='--')
        gl.xlabels_top = None
        gl.ylabels_right = False
        
        ax = plt.subplot(4,3,3,projection=ccrs.PlateCarree())
        DATA.B.sel(time=pdate).plot.contourf('lon', 'lat', ax=ax);
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=1, color='gray', alpha=0.1, linestyle='--')
        gl.xlabels_top = None
        gl.ylabels_right = False
        
        ax = plt.subplot(4,3,4,projection=ccrs.PlateCarree())
        DATA.SNOWHEIGHT.sel(time=pdate).plot.contourf('lon', 'lat', ax=ax);
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=1, color='gray', alpha=0.1, linestyle='--')
        gl.xlabels_top = None
        gl.ylabels_right = False
        
        ax = plt.subplot(4,3,5,projection=ccrs.PlateCarree())
        DATA.surfM.sel(time=pdate).plot.contourf('lon', 'lat', ax=ax);
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=1, color='gray', alpha=0.1, linestyle='--')
        gl.xlabels_top = None
        gl.ylabels_right = False

        ax = plt.subplot(4,3,6,projection=ccrs.PlateCarree())
        DATA.TS.sel(time=pdate).plot.contourf('lon', 'lat', ax=ax);
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=1, color='gray', alpha=0.1, linestyle='--')
        gl.xlabels_top = None
        gl.ylabels_right = False
        
        ax = plt.subplot(4,3,7,projection=ccrs.PlateCarree())
        DATA.G.sel(time=pdate).plot.contourf('lon', 'lat', ax=ax);
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=1, color='gray', alpha=0.1, linestyle='--')
        gl.xlabels_top = None
        gl.ylabels_right = False
        
        ax = plt.subplot(4,3,8,projection=ccrs.PlateCarree())
        DATA.LWin.sel(time=pdate).plot.contourf('lon', 'lat', ax=ax);
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=1, color='gray', alpha=0.1, linestyle='--')
        gl.xlabels_top = None
        gl.ylabels_right = False
        
        ax = plt.subplot(4,3,9,projection=ccrs.PlateCarree())
        DATA.LWout.sel(time=pdate).plot.contourf('lon', 'lat', ax=ax);
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=1, color='gray', alpha=0.1, linestyle='--')
        gl.xlabels_top = None
        gl.ylabels_right = False
        
        ax = plt.subplot(4,3,10,projection=ccrs.PlateCarree())
        DATA.MB.sel(time=pdate).plot.contourf('lon', 'lat', ax=ax);
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=1, color='gray', alpha=0.1, linestyle='--')
        gl.xlabels_top = None
        gl.ylabels_right = False
        
        ax = plt.subplot(4,3,11,projection=ccrs.PlateCarree())
        DATA.surfMB.sel(time=pdate).plot.contourf('lon', 'lat', ax=ax);
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=1, color='gray', alpha=0.1, linestyle='--')
        gl.xlabels_top = None
        gl.ylabels_right = False
        
        ax = plt.subplot(4,3,12,projection=ccrs.PlateCarree())
        DATA.Q.sel(time=pdate).plot.contourf('lon', 'lat', ax=ax);
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=1, color='gray', alpha=0.1, linestyle='--')
        gl.xlabels_top = None
        gl.ylabels_right = False


        #DATA[var].sel(time=pdate).plot.contourf('lon', 'lat', ax=ax);
        
    
    ax.coastlines()
    plt.show()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Quick plot of the results file.')
    parser.add_argument('-f', '-file', dest='file', help='File name')
    parser.add_argument('-d', '-date', dest='pdate', help='Which date to plot')
    parser.add_argument('-v', '-var', dest='var', help='Which variable')
    parser.add_argument('-t', '-type', dest='type', help='(1) contour, (2) mesh', type=int)

    args = parser.parse_args()
    print(args.type)
    if args.type==1:
        print('Contour')
        plotContour(args.file, args.pdate, args.var) 
    else:
        print('Mesh')
        plotMesh(args.file, args.pdate, args.var) 
    
