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


def main():

    start_logging()
    
    #------------------------------------------
    # Create input and output dataset
    #------------------------------------------ 
    IO = IOClass()
    DATA = IO.create_data_file() 
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
  
        with SLURMCluster(scheduler_port=port, cores=cores, processes=processes, memory=memory, shebang=shebang, name=name, queue=queue, job_extra=slurm_parameters) as cluster:
            cluster.scale(processes*nodes)   
            print(cluster.job_script())
            print("You are using SLURM!\n")
            print(cluster)
            run_cosipy(cluster, IO, DATA, RESULT, RESTART, futures)

    else:
        with LocalCluster(scheduler_port=local_port, n_workers=1, threads_per_worker=20, silence_logs=True) as cluster:
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
        print(" Total run duration in minutes %4.2f \n\n" %(duration_run.total_seconds() / 60))
    if duration_run.total_seconds() >= 3600:
        print(" Total run duration in hours %4.2f \n\n" %(duration_run.total_seconds() / 3600))

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


        #---------------------------------------
        # Create the result arrays
        #---------------------------------------
        RAIN = np.full((DATA.dims['time'],ny,nx), np.nan)
        SNOWFALL = np.full((DATA.dims['time'],ny,nx), np.nan)
        LWin = np.full((DATA.dims['time'],ny,nx), np.nan)
        LWout = np.full((DATA.dims['time'],ny,nx), np.nan)
        H = np.full((DATA.dims['time'],ny,nx), np.nan)
        LE = np.full((DATA.dims['time'],ny,nx), np.nan)
        B = np.full((DATA.dims['time'],ny,nx), np.nan)
        MB = np.full((DATA.dims['time'],ny,nx), np.nan)
        surfMB = np.full((DATA.dims['time'],ny,nx), np.nan)
        Q = np.full((DATA.dims['time'],ny,nx), np.nan)
        SNOWHEIGHT = np.full((DATA.dims['time'],ny,nx), np.nan)
        TOTALHEIGHT = np.full((DATA.dims['time'],ny,nx), np.nan)
        TS = np.full((DATA.dims['time'],ny,nx), np.nan)
        ALBEDO = np.full((DATA.dims['time'],ny,nx), np.nan)
        NLAYERS = np.full((DATA.dims['time'],ny,nx), np.nan)
        ME = np.full((DATA.dims['time'],ny,nx), np.nan)
        intMB = np.full((DATA.dims['time'],ny,nx), np.nan)
        EVAPORATION = np.full((DATA.dims['time'],ny,nx), np.nan)
        SUBLIMATION = np.full((DATA.dims['time'],ny,nx), np.nan)
        CONDENSATION = np.full((DATA.dims['time'],ny,nx), np.nan)
        DEPOSITION = np.full((DATA.dims['time'],ny,nx), np.nan)
        REFREEZE = np.full((DATA.dims['time'],ny,nx), np.nan)
        subM = np.full((DATA.dims['time'],ny,nx), np.nan)
        Z0 = np.full((DATA.dims['time'],ny,nx), np.nan)
        surfM= np.full((DATA.dims['time'],ny,nx), np.nan)

        if full_field:
            LAYER_HEIGHT = np.full((DATA.dims['time'],ny,nx,max_layers), np.nan)
            LAYER_RHO = np.full((DATA.dims['time'],ny,nx,max_layers), np.nan)
            LAYER_T = np.full((DATA.dims['time'],ny,nx,max_layers), np.nan)
            LAYER_LWC = np.full((DATA.dims['time'],ny,nx,max_layers), np.nan)
            LAYER_CC = np.full((DATA.dims['time'],ny,nx,max_layers), np.nan)
            LAYER_POROSITY = np.full((DATA.dims['time'],ny,nx,max_layers), np.nan)
            LAYER_LW = np.full((DATA.dims['time'],ny,nx,max_layers), np.nan)
            LAYER_ICE_FRACTION = np.full((DATA.dims['time'],ny,nx,max_layers), np.nan)
            LAYER_IRREDUCIBLE_WATER = np.full((DATA.dims['time'],ny,nx,max_layers), np.nan)
            LAYER_REFREEZE = np.full((DATA.dims['time'],ny,nx,max_layers), np.nan)


        #---------------------------------------
        # Guarantee that restart file is closed
        #---------------------------------------
        if (restart==True):
            IO.get_grid_restart().close()
       
        #---------------------------------------
        # Assign local results to global 
        #---------------------------------------
        start_res = datetime.now()
        for future in as_completed(futures):
                res = future.result() 
                RAIN[:,res[0],res[1]] = res[3]
                SNOWFALL[:,res[0],res[1]] = res[4]
                LWin[:,res[0],res[1]] = res[5]
                LWout[:,res[0],res[1]] = res[6]
                H[:,res[0],res[1]] = res[7]
                LE[:,res[0],res[1]] = res[8]
                B[:,res[0],res[1]] = res[9]
                MB[:,res[0],res[1]] = res[10]
                surfMB[:,res[0],res[1]] = res[11]
                Q[:,res[0],res[1]] = res[12]
                SNOWHEIGHT[:,res[0],res[1]] = res[13]
                TOTALHEIGHT[:,res[0],res[1]] = res[14]
                TS[:,res[0],res[1]] = res[15]
                ALBEDO[:,res[0],res[1]] = res[16]
                NLAYERS[:,res[0],res[1]] = res[17]
                ME[:,res[0],res[1]] = res[18]
                intMB[:,res[0],res[1]] = res[19]
                EVAPORATION[:,res[0],res[1]] = res[20]
                SUBLIMATION[:,res[0],res[1]] = res[21]
                CONDENSATION[:,res[0],res[1]] = res[22]
                DEPOSITION[:,res[0],res[1]] = res[23]
                REFREEZE[:,res[0],res[1]] = res[24]
                subM[:,res[0],res[1]] = res[25]
                Z0[:,res[0],res[1]] = res[26]
                surfM[:,res[0],res[1]] = res[27]

                if full_field:
                    LAYER_HEIGHT[:,res[0],res[1],:] = res[28]
                    LAYER_RHO[:,res[0],res[1],:] = res[29]
                    LAYER_T[:,res[0],res[1],:] = res[30]
                    LAYER_LWC[:,res[0],res[1],:] = res[31]
                    LAYER_CC[:,res[0],res[1],:] = res[32]
                    LAYER_POROSITY[:,res[0],res[1],:] = res[33]
                    LAYER_LW[:,res[0],res[1],:] = res[34]
                    LAYER_ICE_FRACTION[:,res[0],res[1],:] = res[35]
                    LAYER_IRREDUCIBLE_WATER[:,res[0],res[1],:] = res[36]
                    LAYER_REFREEZE[:,res[0],res[1],:] = res[37]

                # Write the restart file
                IO.write_restart_future(res[2],res[0],res[1])

        #---------------------------------------
        # Auxiliary function 
        #---------------------------------------
        def add_variable_along_latlontime(ds, var, name, units, long_name):
            """ This function self.adds missing variables to the self.DATA class """
            ds[name] = (('time','south_north','west_east'), var)
            ds[name].attrs['units'] = units
            ds[name].attrs['long_name'] = long_name
            ds[name].encoding['_FillValue'] = -9999
            return ds

        def add_variable_along_latlonlayertime(ds, var, name, units, long_name):
            """ This function self.adds missing variables to the self.DATA class """
            ds[name] = (('time','south_north','west_east','layer'), var)
            ds[name].attrs['units'] = units
            ds[name].attrs['long_name'] = long_name
            ds[name].encoding['_FillValue'] = -9999
            return ds

        print('Adding results to NetCDF file')

        #---------------------------------------
        # Add variables to output file 
        #---------------------------------------
        add_variable_along_latlontime(RESULT, RAIN, 'RAIN', 'mm', 'Liquid precipitation') 
        add_variable_along_latlontime(RESULT, SNOWFALL, 'SNOWFALL', 'm', 'Snowfall') 
        add_variable_along_latlontime(RESULT, LWin, 'LWin', 'W m\u207b\xb2', 'Incoming longwave radiation') 
        add_variable_along_latlontime(RESULT, LWout, 'LWout', 'W m\u207b\xb2', 'Outgoing longwave radiation') 
        add_variable_along_latlontime(RESULT, H, 'H', 'W m\u207b\xb2', 'Sensible heat flux') 
        add_variable_along_latlontime(RESULT, LE, 'LE', 'W m\u207b\xb2', 'Latent heat flux') 
        add_variable_along_latlontime(RESULT, B, 'B', 'W m\u207b\xb2', 'Ground heat flux') 
        add_variable_along_latlontime(RESULT, surfMB, 'surfMB', 'm w.e.', 'Surface mass balance') 
        add_variable_along_latlontime(RESULT, Q, 'Q', 'm w.e.', 'Runoff') 
        add_variable_along_latlontime(RESULT, SNOWHEIGHT, 'SNOWHEIGHT', 'm', 'Snowheight') 
        add_variable_along_latlontime(RESULT, TOTALHEIGHT, 'TOTALHEIGHT', 'm', 'Total domain height') 
        add_variable_along_latlontime(RESULT, TS, 'TS', 'K', 'Surface temperature') 
        add_variable_along_latlontime(RESULT, ALBEDO, 'ALBEDO', '-', 'Albedo') 
        add_variable_along_latlontime(RESULT, NLAYERS, 'NLAYERS', '-', 'Number of layers') 
        add_variable_along_latlontime(RESULT, ME, 'ME', 'W m\u207b\xb2', 'Available melt energy') 
        add_variable_along_latlontime(RESULT, intMB, 'intMB', 'm w.e.', 'Internal mass balance') 
        add_variable_along_latlontime(RESULT, EVAPORATION, 'EVAPORATION', 'm w.e.', 'Evaporation') 
        add_variable_along_latlontime(RESULT, SUBLIMATION, 'SUBLIMATION', 'm w.e.', 'Sublimation') 
        add_variable_along_latlontime(RESULT, CONDENSATION, 'CONDENSATION', 'm w.e.', 'Condensation') 
        add_variable_along_latlontime(RESULT, DEPOSITION, 'DEPOSITION', 'm w.e.', 'Deposition') 
        add_variable_along_latlontime(RESULT, REFREEZE, 'REFREEZE', 'm w.e.', 'Refreezing') 
        add_variable_along_latlontime(RESULT, subM, 'subM', 'm w.e.', 'Subsurface melt') 
        add_variable_along_latlontime(RESULT, Z0, 'Z0', 'm', 'Roughness length') 
        add_variable_along_latlontime(RESULT, surfM, 'surfM', 'm w.e.', 'Surface melt') 
       
        if full_field:
            add_variable_along_latlonlayertime(RESULT, LAYER_HEIGHT, 'LAYER_HEIGHT', 'm', 'Layer height') 
            add_variable_along_latlonlayertime(RESULT, LAYER_RHO, 'LAYER_RHO', 'kg m^-3', 'Layer density') 
            add_variable_along_latlonlayertime(RESULT, LAYER_T, 'LAYER_T', 'K', 'Layer temperature') 
            add_variable_along_latlonlayertime(RESULT, LAYER_LWC, 'LAYER_LWC', 'kg m^-2', 'Liquid water content') 
            add_variable_along_latlonlayertime(RESULT, LAYER_CC, 'LAYER_CC', 'J m^-2', 'Cold content') 
            add_variable_along_latlonlayertime(RESULT, LAYER_POROSITY, 'LAYER_POROSITY', '-', 'Porosity') 
            add_variable_along_latlonlayertime(RESULT, LAYER_LW, 'LAYER_LW', 'm w.e.', 'Liquid water') 
            add_variable_along_latlonlayertime(RESULT, LAYER_ICE_FRACTION, 'LAYER_ICE_FRACTION', '-', 'Ice fraction') 
            add_variable_along_latlonlayertime(RESULT, LAYER_IRREDUCIBLE_WATER, 'LAYER_IRREDUCIBLE_WATER', '-', 'Irreducible water') 
            add_variable_along_latlonlayertime(RESULT, LAYER_REFREEZE, 'LAYER_REFREEZE', 'm w.e.', 'Refreezing') 



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
