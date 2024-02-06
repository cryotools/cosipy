#!/usr/bin/env python

"""
    This is the main code file of the 'COupled Snowpack and Ice surface energy
    and MAss balance glacier model in Python' (COSIPY). The model was initially written by
    Tobias Sauter. The version is constantly under development by a core developer team.
    
    Core developer team:

    Tobias Sauter
    Anselm Arndt

    You are allowed to use and modify this code in a noncommercial manner and by
    appropriately citing the above mentioned developers.

    The code is available on github. https://github.com/cryotools/cosipy

    For more information read the README and see https://cryo-tools.org/

    The model is written in Python 3.6.3 and is tested on Anaconda3-4.4.7 64-bit.

    Correspondence: tobias.sauter@fau.de

"""
import cProfile
import logging
import os
import sys
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd
import scipy
import yaml
# from dask import compute, delayed
# import dask as da
# from dask.diagnostics import ProgressBar
from dask.distributed import as_completed, progress
from dask_jobqueue import SLURMCluster
from distributed import Client, LocalCluster
# import dask
from tornado import gen

from config import *
from cosipy.cpkernel.cosipy_core import cosipy_core
from cosipy.cpkernel.io import IOClass
from slurm_config import *

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

    # Measure time
    start_time = datetime.now()

    #-----------------------------------------------
    # Create a client for distributed calculations
    #-----------------------------------------------
    if (slurm_use):

        with SLURMCluster(job_name=name, cores=cores, processes=cores, memory=memory, 
			 account=account, job_extra_directives=slurm_parameters,
                         local_directory='logs/dask-worker-space') as cluster:
            cluster.scale(nodes*cores)   
            print(cluster.job_script())
            print("You are using SLURM!\n")
            print(cluster)
            run_cosipy(cluster, IO, DATA, RESULT, RESTART, futures)

    else:
        with LocalCluster(scheduler_port=local_port, n_workers=workers, local_directory='logs/dask-worker-space', threads_per_worker=1, silence_logs=True) as cluster:
            print(cluster)
            run_cosipy(cluster, IO, DATA, RESULT, RESTART, futures)

    print('\n')
    print('--------------------------------------------------------------')
    print('Write results ...')
    print('-------------------------------------------------------------- \n')
    start_writing = datetime.now()

    #-----------------------------------------------
    # Write results and restart files
    #-----------------------------------------------
    timestamp = pd.to_datetime(str(IO.get_restart().time.values)).strftime('%Y-%m-%dT%H-%M')
   
    encoding = dict()
    for var in IO.get_result().data_vars:
        dataMin = IO.get_result()[var].min(skipna=True).values
        dataMax = IO.get_result()[var].max(skipna=True).values
        dtype = 'int16'
        FillValue = -9999 
        scale_factor, add_offset = compute_scale_and_offset(dataMin, dataMax, 16)
        #encoding[var] = dict(zlib=True, complevel=compression_level, dtype=dtype, scale_factor=scale_factor, add_offset=add_offset, _FillValue=FillValue)
        encoding[var] = dict(zlib=True, complevel=compression_level)
  
    IO.get_result().to_netcdf(os.path.join(data_path,'output',output_netcdf), encoding=encoding, mode = 'w')

    encoding = dict()
    for var in IO.get_restart().data_vars:
        dataMin = IO.get_restart()[var].min(skipna=True).values
        dataMax = IO.get_restart()[var].max(skipna=True).values
        dtype = 'int16'
        FillValue = -9999 
        scale_factor, add_offset = compute_scale_and_offset(dataMin, dataMax, 16)
        #encoding[var] = dict(zlib=True, complevel=compression_level, dtype=dtype, scale_factor=scale_factor, add_offset=add_offset, _FillValue=FillValue)
        encoding[var] = dict(zlib=True, complevel=compression_level)
    
    IO.get_restart().to_netcdf(os.path.join(data_path,'restart','restart_'+timestamp+'.nc'), encoding=encoding)
    
    #-----------------------------------------------
    # Stop time measurement
    #-----------------------------------------------
    duration_run = datetime.now() - start_time
    duration_run_writing = datetime.now() - start_writing

    #-----------------------------------------------
    # Print out some information
    #-----------------------------------------------
    print("\t Time required tor write restart and output files: %4g minutes %2g seconds \n" % (duration_run_writing.total_seconds()//60.0,duration_run_writing.total_seconds()%60.0))
    print("\t Total run duration: %4g minutes %2g seconds \n" % (duration_run.total_seconds()//60.0,duration_run.total_seconds()%60.0))
    print('--------------------------------------------------------------')
    print('\t SIMULATION WAS SUCCESSFUL')
    print('--------------------------------------------------------------')


def run_cosipy(cluster, IO, DATA, RESULT, RESTART, futures):

    with Client(cluster) as client:
        print('--------------------------------------------------------------')
        print('\t Starting clients and submit jobs ... \n')
        print('-------------------------------------------------------------- \n')

        print(cluster)
        print(client)

        # Get dimensions of the whole domain
        ny = DATA.sizes[northing]
        nx = DATA.sizes[easting]

        cp = cProfile.Profile()

        # Get some information about the cluster/nodes
        total_grid_points = DATA.sizes[northing]*DATA.sizes[easting]
        if slurm_use is True:
            total_cores = cores*nodes
            points_per_core = total_grid_points // total_cores
            print(total_grid_points, total_cores, points_per_core)

        # Check if evaluation is selected:
        if stake_evaluation is True:
            # Read stake data (data must be given as cumulative changes)
            df_stakes_loc = pd.read_csv(stakes_loc_file, delimiter='\t', na_values='-9999')
            df_stakes_data = pd.read_csv(stakes_data_file, delimiter='\t', index_col='TIMESTAMP', na_values='-9999')
            df_stakes_data.index = pd.to_datetime(df_stakes_data.index)

            # Uncomment, if stake data is given as changes between measurements
            # df_stakes_data = df_stakes_data.cumsum(axis=0)

            # Init dataframes to store evaluation statistics
            df_stat = pd.DataFrame()
            df_val = df_stakes_data.copy()

            # reshape and stack coordinates
            if WRF:
                coords = np.column_stack((DATA.lat.values.ravel(), DATA.lon.values.ravel()))
            else:
                # in case lat/lon are 1D coordinates
                lons, lats = np.meshgrid(DATA.lon,DATA.lat)
                coords = np.column_stack((lats.ravel(),lons.ravel()))

            # construct KD-tree, in order to get closes grid cell
            ground_pixel_tree = scipy.spatial.cKDTree(transform_coordinates(coords))

            # Check for stake data
            stakes_list = []
            for index, row in df_stakes_loc.iterrows():
                index = ground_pixel_tree.query(transform_coordinates((row['lat'], row['lon'])))
                if WRF:
                    index = np.unravel_index(index[1], DATA.lat.shape)
                else:
                    index = np.unravel_index(index[1], lats.shape)
                stakes_list.append((index[0][0], index[1][0], row['id']))

        else:
            stakes_loc = None
            df_stakes_data = None


        # Distribute data and model to workers
        start_res = datetime.now()
        for y,x in product(range(DATA.sizes[northing]),range(DATA.sizes[easting])):
            if stake_evaluation is True:
                stake_names = []
                # Check if the grid cell contain stakes and store the stake names in a list
                for idx, (stake_loc_y, stake_loc_x, stake_name) in enumerate(stakes_list):
                    if ((y == stake_loc_y) & (x == stake_loc_x)):
                        stake_names.append(stake_name)
            else:
                stake_names = None
                
            if WRF is True:
                mask = DATA.MASK.sel(south_north=y, west_east=x)
                # Provide restart grid if necessary
                if ((mask==1) & (not restart)):
                    if np.isnan(DATA.sel(south_north=y, west_east=x).to_array()).any():
                        print('ERROR!!!!!!!!!!! There are NaNs in the dataset')
                        sys.exit()
                    futures.append(client.submit(cosipy_core, DATA.sel(south_north=y, west_east=x), y, x, stake_names=stake_names, stake_data=df_stakes_data))
                elif ((mask==1) & (restart)):
                    if np.isnan(DATA.sel(south_north=y, west_east=x).to_array()).any():
                        print('ERROR!!!!!!!!!!! There are NaNs in the dataset')
                        sys.exit()
                    futures.append(client.submit(cosipy_core, DATA.sel(south_north=y, west_east=x), y, x, 
                                             GRID_RESTART=IO.create_grid_restart().sel(south_north=y, west_east=x), 
                                             stake_names=stake_names, stake_data=df_stakes_data))
            else:
                mask = DATA.MASK.isel(lat=y, lon=x)
                # Provide restart grid if necessary
                if ((mask==1) & (not restart)):
                    if np.isnan(DATA.isel(lat=y,lon=x).to_array()).any():
                        print('ERROR!!!!!!!!!!! There are NaNs in the dataset')
                        sys.exit()
                    futures.append(client.submit(cosipy_core, DATA.isel(lat=y, lon=x), y, x, stake_names=stake_names, stake_data=df_stakes_data))
                elif ((mask==1) & (restart)):
                    if np.isnan(DATA.isel(lat=y,lon=x).to_array()).any():
                        print('ERROR!!!!!!!!!!! There are NaNs in the dataset')
                        sys.exit()
                    futures.append(client.submit(cosipy_core, DATA.isel(lat=y, lon=x), y, x, 
                                             GRID_RESTART=IO.create_grid_restart().isel(lat=y, lon=x), 
                                             stake_names=stake_names, stake_data=df_stakes_data))
        # Finally, do the calculations and print the progress
        progress(futures)

        #---------------------------------------
        # Guarantee that restart file is closed
        #---------------------------------------
        if (restart==True):
            IO.get_grid_restart().close()
      
        # Create numpy arrays which aggregates all local results
        IO.create_global_result_arrays()

        # Create numpy arrays which aggregates all local results
        IO.create_global_restart_arrays()

        #---------------------------------------
        # Assign local results to global 
        #---------------------------------------
        for future in as_completed(futures):

                # Get the results from the workers
                indY,indX,local_restart,RAIN,SNOWFALL,LWin,LWout,H,LE,B,QRR,MB,surfMB,Q,SNOWHEIGHT,TOTALHEIGHT,TS,ALBEDO,NLAYERS, \
                                ME,intMB,EVAPORATION,SUBLIMATION,CONDENSATION,DEPOSITION,REFREEZE,subM,Z0,surfM,MOL, \
                                LAYER_HEIGHT,LAYER_RHO,LAYER_T,LAYER_LWC,LAYER_CC,LAYER_POROSITY,LAYER_ICE_FRACTION, \
                                LAYER_IRREDUCIBLE_WATER,LAYER_REFREEZE,stake_names,stat,df_eval = future.result()
               
                IO.copy_local_to_global(indY,indX,RAIN,SNOWFALL,LWin,LWout,H,LE,B,QRR,MB,surfMB,Q,SNOWHEIGHT,TOTALHEIGHT,TS,ALBEDO,NLAYERS, \
                                ME,intMB,EVAPORATION,SUBLIMATION,CONDENSATION,DEPOSITION,REFREEZE,subM,Z0,surfM,MOL,LAYER_HEIGHT,LAYER_RHO, \
                                LAYER_T,LAYER_LWC,LAYER_CC,LAYER_POROSITY,LAYER_ICE_FRACTION,LAYER_IRREDUCIBLE_WATER,LAYER_REFREEZE)

                IO.copy_local_restart_to_global(indY,indX,local_restart)

                # Write results to file
                IO.write_results_to_file()
                
                # Write restart data to file
                IO.write_restart_to_file()

                if stake_evaluation is True:
                    # Store evaluation of stake measurements to dataframe
                    stat = stat.rename('rmse')
                    df_stat = pd.concat([df_stat, stat])

                    for i in stake_names:
                        if (obs_type == 'mb'):
                            df_val[i] = df_eval.mb
                        if (obs_type == 'snowheight'):
                            df_val[i] = df_eval.snowheight

        # Measure time
        end_res = datetime.now()-start_res 
        print("\t Time required to do calculations: %4g minutes %2g seconds \n" % (end_res.total_seconds()//60.0,end_res.total_seconds()%60.0))
      
        if stake_evaluation is True:
            # Save the statistics and the mass balance simulations at the stakes to files
            df_stat.to_csv(os.path.join(data_path,'output','stake_statistics.csv'),sep='\t', float_format='%.2f')
            df_val.to_csv(os.path.join(data_path,'output','stake_simulations.csv'),sep='\t', float_format='%.2f')



def start_logging():
    """Start the python logging"""

    if os.path.exists('./cosipy.yaml'):
        with open('./cosipy.yaml', 'rt') as f:
            config = yaml.load(f.read(),Loader=yaml.SafeLoader)
        logging.config.dictConfig(config)
    else:
       logging.basicConfig(level=logging.INFO)

    logger = logging.getLogger(__name__)
    logger.info('COSIPY simulation started')    


def transform_coordinates(coords):
    """ Transform coordinates from geodetic to cartesian
    an array of tuples)
    """
    # WGS 84 reference coordinate system parameters
    A = 6378.137 # major axis [km]   
    E2 = 6.69437999014e-3 # eccentricity squared    
    
    coords = np.asarray(coords).astype(float)
                                                      
    # is coords a tuple? Convert it to an one-element array of tuples
    if coords.ndim == 1:
        coords = np.array([coords])
    
    # convert to radiants
    lat_rad = np.radians(coords[:,0])
    lon_rad = np.radians(coords[:,1]) 
    
    # convert to cartesian coordinates
    r_n = A / (np.sqrt(1 - E2 * (np.sin(lat_rad) ** 2)))
    x = r_n * np.cos(lat_rad) * np.cos(lon_rad)
    y = r_n * np.cos(lat_rad) * np.sin(lon_rad)
    z = r_n * (1 - E2) * np.sin(lat_rad)
    
    return np.column_stack((x, y, z))


def compute_scale_and_offset(min, max, n):
    # stretch/compress data to the available packed range
    scale_factor = (max - min) / (2 ** n - 1)
    # translate the range to be symmetric about zero
    add_offset = min + 2 ** (n - 1) * scale_factor
    return (scale_factor, add_offset)


@gen.coroutine
def close_everything(scheduler):
    yield scheduler.retire_workers(workers=scheduler.workers, close_workers=True)
    yield scheduler.close()



""" MODEL EXECUTION """
if __name__ == "__main__":
    main()
