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
from dask.distributed import progress, wait, as_completed
import dask

def main():

    #------------------------------------------
    # Create input and output dataset
    #------------------------------------------ 
    IO = IOClass()
    DATA = IO.create_data_file() 
    RESULT = IO.create_result_file() 
    RESTART = IO.create_restart_file() 


    #-----------------------------------------------
    # Create a client for distributed calculations
    #-----------------------------------------------
    cluster = LocalCluster(scheduler_port=8786, silence_logs=True)
    #dask.config.set(scheduler='single-threaded')
    client = Client(cluster, processes=False)

    print('--------------------------------------------------------------')
    print('\t Starting clients ... \n')
    print(client)
    print('-------------------------------------------------------------- \n')


    #----------------------------------------------
    # Calculation - Multithreading using all cores  
    #----------------------------------------------
    
    # Auxiliary variables for futures
    futures = []
    nfutures = 0

    # Measure time
    start_time = datetime.now()

    # Go over the whole grid
    for i,j in product(DATA.lat, DATA.lon):
        mask = DATA.MASK.sel(lat=i, lon=j)
       
        # Provide restart grid if necessary
        if ((mask==1) & (restart==False)):
            nfutures = nfutures+1
            futures.append(client.submit(cosipy_core, DATA.sel(lat=i, lon=j)))
        elif ((mask==1) & (restart==True)):
            nfutures = nfutures+1
            futures.append(client.submit(cosipy_core, DATA.sel(lat=i,lon=j), IO.create_grid_restart().sel(lat=i,lon=j)))
  
    # Finally, do the calculations and print the progress
    progress(futures)

    if (restart==True):
        IO.get_grid_restart().close()

    for future in as_completed(futures):   
        results = future.result()
        result_data = results[0]
        restart_data = results[1]
        IO.write_results_future(result_data)
        IO.write_restart_future(restart_data)

    comp = dict(zlib=True, complevel=9)
    encoding = {var: comp for var in IO.get_result().data_vars}
    IO.get_result().to_netcdf(output_netcdf, encoding=encoding)
  
    comp = dict(zlib=True, complevel=9)
    encoding = {var: comp for var in IO.get_restart().data_vars}
    IO.get_restart().to_netcdf(restart_netcdf, encoding=encoding)
 
    # Stop time measurement
    duration_run = datetime.now() - start_time

    # Print out some information
    print("\n \n Total run duration in seconds %4.2f \n\n" % (duration_run.total_seconds()))
    print('--------------------------------------------------------------')
    print('\t SIMULATION WAS SUCCESSFUL')
    print('--------------------------------------------------------------')


''' MODEL EXECUTION '''
if __name__ == "__main__":
    main()
