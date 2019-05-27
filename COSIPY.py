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
import itertools

import logging
import yaml

from config import *
from slurm_config import *
from cpkernel.cosipy_core import * 
from cpkernel.io import *

from distributed import Client, LocalCluster
from dask import compute, delayed
import dask as da
from dask.diagnostics import ProgressBar
from dask.distributed import progress, wait, as_completed
import dask
from tornado import gen
from dask_jobqueue import SLURMCluster

import cProfile

def main():

    start_logging()
    
    #------------------------------------------
    # Create input and output dataset
    #------------------------------------------ 
    IO = IOClass()
    DATA = IO.create_data_file() 
    
    # Create global result and restart datasets
    RESULT = IO.create_result_file() 
    RESTART = IO.create_restart_file() 

    #----------------------------------------------
    # Calculation - Multithreading using all cores  
    #----------------------------------------------
    
    # Auxiliary variables for futures
    futures = []
    nfutures = 0

    # Measure time
    start_time = datetime.now()

    #-----------------------------------------------
    # Create a client for distributed calculations
    #-----------------------------------------------
    if (slurm_use):
  
        with SLURMCluster(scheduler_port=port, cores=cores, processes=processes, memory=str(memory_per_process * processes) + 'GB', name=name, job_extra=extra_slurm_parameters, local_directory='dask-worker-space') as cluster:
            cluster.scale(processes * nodes)   
            print(cluster.job_script())
            print("You are using SLURM!\n")
            print(cluster)
            run_cosipy(cluster, IO, DATA, RESULT, RESTART, futures)

    else:
        with LocalCluster(scheduler_port=local_port, n_workers=workers, threads_per_worker=1, silence_logs=True) as cluster:
            print(cluster)
            run_cosipy(cluster, IO, DATA, RESULT, RESTART, futures)

    print('\n')
    print('--------------------------------------------------------------')
    print('Write results to netcdf')
    print('-------------------------------------------------------------- \n')
    start_writing = datetime.now()

    #-----------------------------------------------
    # Write results and restart files
    #-----------------------------------------------
    timestamp = pd.to_datetime(str(IO.get_restart().time.values)).strftime('%Y-%m-%dT%H-%M-%S')
    comp = dict(zlib=True, complevel=9)
    
    encoding = {var: comp for var in IO.get_result().data_vars}
    IO.get_result().to_netcdf(os.path.join(data_path,'output',output_netcdf), encoding=encoding, mode = 'w')

    encoding = {var: comp for var in IO.get_restart().data_vars}
    IO.get_restart().to_netcdf(os.path.join(data_path,'restart','restart_'+timestamp+'.nc'), encoding=encoding)
    
    #-----------------------------------------------
    # Stop time measurement
    #-----------------------------------------------
    duration_run = datetime.now() - start_time
    duration_run_writing = datetime.now() - start_writing

    #-----------------------------------------------
    # Print out some information
    #-----------------------------------------------
    print("\n \n Total run duration in seconds %4.2f \n" % (duration_run.total_seconds()))
    print("\n \n Needed time for writing restart and output in seconds %4.2f \n" % (duration_run_writing.total_seconds()))
    
    if duration_run.total_seconds() >= 60 and duration_run.total_seconds() < 3600:
        print("Total run duration in minutes %4.2f \n\n" %(duration_run.total_seconds() / 60))
    if duration_run.total_seconds() >= 3600:
        print("Total run duration in hours %4.2f \n\n" %(duration_run.total_seconds() / 3600))

    print('--------------------------------------------------------------')
    print('\t SIMULATION WAS SUCCESSFUL')
    print('--------------------------------------------------------------')


def run_cosipy(cluster, IO, DATA, RESULT, RESTART, futures):

    with Client(cluster,processes=False) as client:
        print('--------------------------------------------------------------')
        print('\t Starting clients and submit jobs ... \n')
        print('-------------------------------------------------------------- \n')

        print(cluster)
        print(client)

        # Get dimensions of the whole domain
        ny = DATA.dims['south_north']
        nx = DATA.dims['west_east']

        cp = cProfile.Profile()

        # Get some information about the cluster/nodes
        total_grid_points = DATA.dims['south_north']*DATA.dims['west_east']
        total_cores = processes*nodes
        points_per_core = total_grid_points // total_cores
        print(total_grid_points, total_cores, points_per_core)

        # Distribute data and model to workers
        for y,x in product(range(DATA.dims['south_north']),range(DATA.dims['west_east'])):
            mask = DATA.MASK.sel(south_north=y, west_east=x)
            # Provide restart grid if necessary
            if ((mask==1) & (restart==False)):
                futures.append(client.submit(cosipy_core, DATA.sel(south_north=y, west_east=x), y, x))
            elif ((mask==1) & (restart==True)):
                futures.append(client.submit(cosipy_core, DATA.sel(south_north=y, west_east=x), y, x, GRID_RESTART=IO.create_grid_restart().sel(south_north=y, west_east=x)))


        # Finally, do the calculations and print the progress
        progress(futures)

        #---------------------------------------
        # Guarantee that restart file is closed
        #---------------------------------------
        if (restart==True):
            IO.get_grid_restart().close()
      
        # Create numpy arrays which aggregates all local results
        IO.create_global_result_arrays()

        #---------------------------------------
        # Assign local results to global 
        #---------------------------------------
        start_res = datetime.now()
        for future in as_completed(futures):

                # Get the results from the workers
#                res = future.result() 
                indY,indX,local_restart,RAIN,SNOWFALL,LWin,LWout,H,LE,B,MB,surfMB,Q,SNOWHEIGHT,TOTALHEIGHT,TS,ALBEDO,NLAYERS, \
                                ME,intMB,EVAPORATION,SUBLIMATION,CONDENSATION,DEPOSITION,REFREEZE,subM,Z0,surfM, \
                                LAYER_HEIGHT,LAYER_RHO,LAYER_T,LAYER_LWC,LAYER_CC,LAYER_POROSITY,LAYER_LW,LAYER_ICE_FRACTION, \
                                LAYER_IRREDUCIBLE_WATER,LAYER_REFREEZE = future.result()
                
                IO.copy_local_to_global(indY,indX,RAIN,SNOWFALL,LWin,LWout,H,LE,B,MB,surfMB,Q,SNOWHEIGHT,TOTALHEIGHT,TS,ALBEDO,NLAYERS, \
                                ME,intMB,EVAPORATION,SUBLIMATION,CONDENSATION,DEPOSITION,REFREEZE,subM,Z0,surfM,LAYER_HEIGHT,LAYER_RHO, \
                                LAYER_T,LAYER_LWC,LAYER_CC,LAYER_POROSITY,LAYER_LW,LAYER_ICE_FRACTION,LAYER_IRREDUCIBLE_WATER,LAYER_REFREEZE)

                # res[0]::y  res[1]::x  res[2]::restart  res[3]::local_io
                #y = res[0]
                #x = res[1]
                #local_restart = res[2]
                #local_io = res[3]

                # Copy the local results from worker to the global result array
                #IO.copy_local_to_global(local_io, y, x)
                
                # Write results to file
                IO.write_results_to_file()

                # Write the restart file
                IO.write_restart_future(local_restart,y,x)

        # Measure time
        end_res = datetime.now()-start_res 
        print("\n \n Needed time for save results to xarray in seconds %4.2f \n" % (end_res.total_seconds()))
        



def start_logging():
    ''' Start the python logging'''

    if os.path.exists('./cosipy.yaml'):
        with open('./cosipy.yaml', 'rt') as f:
            config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    else:
       logging.basicConfig(level=logging.INFO)

    logger = logging.getLogger(__name__)
    logger.info('COSIPY simulation started')    



@gen.coroutine
def close_everything(scheduler):
    yield scheduler.retire_workers(workers=scheduler.workers, close_workers=True)
    yield scheduler.close()



''' MODEL EXECUTION '''
if __name__ == "__main__":
    main()
