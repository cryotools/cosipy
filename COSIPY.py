#!/usr/bin/env python

"""
    This is the main code file of the 'COupled Snowpack and Ice surface energy
    and MAss balance glacier model in Python' (COSIPY). The model was initially written by
    Tobias Sauter. The version is constantly under development by a core developer team.
    
    Core developer team:

    Tobias Sauter
    Anselm Arndt
    David Loibl
    Bjoern Sass

    You are allowed to use and modify this code in a noncommercial manner and by
    appropriately citing the above mentioned developers.

    The code is available on github. https://github.com/cryotools/cosipy

    For more information read the README and see https://cryo-tools.org/

    The model is written in Python 3.6.3 and is tested on Anaconda3-4.4.7 64-bit.

    Correspondence: tobias.sauter@fau.de

"""
import os
from datetime import datetime
from itertools import product

from config import *
from cpkernel.cosipy_core import cosipy_core
from cpkernel.io import *

from distributed import Client, LocalCluster
from dask import compute, delayed
import dask as da
from dask.diagnostics import ProgressBar
from dask.distributed import progress

def main():

    #------------------------------------------
    # Create input and output dataset
    #------------------------------------------
    
    # Check, if restart
    if (restart==True):
        print('--------------------------------------------------------------')
        print('\t RESTART FROM PREVIOUS STATE')
        print('--------------------------------------------------------------')
        
        # Load the restart file
        if os.path.isfile(restart_netcdf):
            GRID_RESTART = xr.open_dataset(restart_netcdf)
            latest_time = GRID_RESTART.time     # Get time of the last calculation
            DATA = read_data(latest_time)       # Read data from the last date to the end of the data file
        else:
            print('\n No restart file available!')  # if there is a problem kill the program
            sys.exit(1)
    else:
        DATA = read_data()  # If no restart, read data according to the dates defined in the config.py
    
    # Initialize the result dataset
    RESULT = init_result_dataset(DATA)



    #-----------------------------------------------
    # Create a client for distributed calculations
    #-----------------------------------------------
    cluster = LocalCluster(scheduler_port=8786, n_workers=8, silence_logs=False)
    client = Client(cluster, processes=False)
    print('--------------------------------------------------------------')
    print('\t Starting clients ... \n')
    print(client)
    print('--------------------------------------------------------------')

    # Measure time
    start_time = datetime.now()


    #----------------------------------------------
    # Calculation - Multithreading using all cores  
    #----------------------------------------------
    fut = []
    nfut = 0
    for i,j in product(DATA.lat, DATA.lon):
        mask = DATA.MASK.sel(lat=i, lon=j)
       
        # Provide restart grid if necessary
        if ((mask==1) & (restart==False)):
            nfut = nfut+1
            fut.append(client.submit(cosipy_core, DATA.sel(lat=i, lon=j)))
        elif ((mask==1) & (restart==True)):
            nfut = nfut+1
            fut.append(client.submit(cosipy_core, DATA.sel(lat=i,lon=j), GRID_RESTART.sel(lat=i,lon=j)))
  

    # Finally, do the calculations and print the progress
    progress(fut)
    results = client.gather(fut)

    #----------------------------------------------
    # Save data  
    #----------------------------------------------
    
    # First close restart file
    if (restart==True):
        GRID_RESTART.close()
    
    # Aggregate and save result and restart file
    RESULT = write_results(RESULT, results, output_netcdf)
    write_restart(RESULT, restart_netcdf)
 
    # Stop time measurement
    duration_run = datetime.now() - start_time
    print("\n \n Total run duration in seconds %4.2f \n\n" % (duration_run.total_seconds()))


    print('--------------------------------------------------------------')
    print('\t SIMULATION WAS SUCCESSFUL')
    print('--------------------------------------------------------------')



def write_results(RESULT, results, output_netcdf):
    """ This function aggregates the point result 
    
    results         ::  List with the result from COSIPI
    """
     
    # Assign point results to the aggregated dataset
    for i in np.arange(len(results)):
        RESULT.SNOWHEIGHT.loc[dict(lon=results[i].lon.values, lat=results[i].lat.values)] = results[i].SNOWHEIGHT
        RESULT.EVAPORATION.loc[dict(lon=results[i].lon.values, lat=results[i].lat.values)] = results[i].EVAPORATION
        RESULT.SUBLIMATION.loc[dict(lon=results[i].lon.values, lat=results[i].lat.values)] = results[i].SUBLIMATION
        RESULT.MELT.loc[dict(lon=results[i].lon.values, lat=results[i].lat.values)] = results[i].MELT
        RESULT.LWin.loc[dict(lon=results[i].lon.values, lat=results[i].lat.values)] = results[i].LWin
        RESULT.LWout.loc[dict(lon=results[i].lon.values, lat=results[i].lat.values)] = results[i].LWout
        RESULT.H.loc[dict(lon=results[i].lon.values, lat=results[i].lat.values)] = results[i].H
        RESULT.LE.loc[dict(lon=results[i].lon.values, lat=results[i].lat.values)] = results[i].LE
        RESULT.B.loc[dict(lon=results[i].lon.values, lat=results[i].lat.values)] = results[i].B
        RESULT.TS.loc[dict(lon=results[i].lon.values, lat=results[i].lat.values)] = results[i].TS
        RESULT.RH2.loc[dict(lon=results[i].lon.values, lat=results[i].lat.values)] = results[i].RH2
        RESULT.T2.loc[dict(lon=results[i].lon.values, lat=results[i].lat.values)] = results[i].T2
        RESULT.G.loc[dict(lon=results[i].lon.values, lat=results[i].lat.values)] = results[i].G
        RESULT.U2.loc[dict(lon=results[i].lon.values, lat=results[i].lat.values)] = results[i].U2
        RESULT.N.loc[dict(lon=results[i].lon.values, lat=results[i].lat.values)] = results[i].N
        RESULT.Z0.loc[dict(lon=results[i].lon.values, lat=results[i].lat.values)] = results[i].Z0
        RESULT.ALBEDO.loc[dict(lon=results[i].lon.values, lat=results[i].lat.values)] = results[i].ALBEDO
        RESULT.RUNOFF.loc[dict(lon=results[i].lon.values, lat=results[i].lat.values)] = results[i].RUNOFF
        
        RESULT.NLAYERS.loc[dict(lon=results[i].lon.values, lat=results[i].lat.values)] = results[i].NLAYERS
        RESULT.LAYER_HEIGHT.loc[dict(lon=results[i].lon.values, lat=results[i].lat.values, layer=np.arange(max_layers))] = results[i].LAYER_HEIGHT
        RESULT.LAYER_RHO.loc[dict(lon=results[i].lon.values, lat=results[i].lat.values, layer=np.arange(max_layers))] = results[i].LAYER_RHO
        RESULT.LAYER_T.loc[dict(lon=results[i].lon.values, lat=results[i].lat.values, layer=np.arange(max_layers))] = results[i].LAYER_T
        RESULT.LAYER_LWC.loc[dict(lon=results[i].lon.values, lat=results[i].lat.values, layer=np.arange(max_layers))] = results[i].LAYER_LWC
        RESULT.LAYER_CC.loc[dict(lon=results[i].lon.values, lat=results[i].lat.values, layer=np.arange(max_layers))] = results[i].LAYER_CC
        RESULT.LAYER_POROSITY.loc[dict(lon=results[i].lon.values, lat=results[i].lat.values, layer=np.arange(max_layers))] = results[i].LAYER_POROSITY
        RESULT.LAYER_VOL.loc[dict(lon=results[i].lon.values, lat=results[i].lat.values, layer=np.arange(max_layers))] = results[i].LAYER_VOL
        RESULT.LAYER_REFREEZE.loc[dict(lon=results[i].lon.values, lat=results[i].lat.values, layer=np.arange(max_layers))] = results[i].LAYER_REFREEZE

    # Write results to netcdf
#    encoding = {'lat': {'zlib': False, '_FillValue': False},
#                'lon': {'zlib': False, '_FillValue': False},
#            }

    comp = dict(zlib=True, complevel=9)
    encoding = {var: comp for var in RESULT.data_vars}

    print('\n')
    print('--------------------------------------------------------------')
    print('\t Writing result file \n')
    
    RESULT.to_netcdf(output_netcdf, encoding=encoding)
    return RESULT


def write_restart(RESULT, restart_netcdf):
    """ Writes the restart file 
    
    RESULT      :: RESULT dataset
    
    """

    # Create restart file (last time step)
    print('\t Writing restart file')
    print('--------------------------------------------------------------')

    RESTART = RESULT.isel(time=-1)
    RESTART.to_netcdf(restart_netcdf)



''' MODEL EXECUTION '''
if __name__ == "__main__":
    main()
