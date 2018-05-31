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
from datetime import datetime
from itertools import product

from config import *
from cpkernel.cosipy_core import cosipy_core
from cpkernel.io import *

from distributed import Client, LocalCluster
from dask import compute, delayed
import dask as da

def main():

    # Measure time
    start_time = datetime.now()

    # Create input and output dataset
    DATA = read_data()
    RESULT = init_result_dataset(DATA)
    
    # Create a client for distributed calculations
    cluster = LocalCluster(scheduler_port=8786, n_workers=8, silence_logs=False)
    client = Client(cluster, processes=False)
    print(client)

    # Multithreading using all cores  
    fut = [client.submit(cosipy_core,DATA.sel(lat=i, lon=j)) for i,j in product(DATA.lat, DATA.lon)]
    #cosipy_core(DATA.sel(lat=-45,lon=-80))
   
    try:
        # Gather the results
        results = client.gather(fut)
    except:
        pass

    # Assign the results
    for i in np.arange(len(results)):
        RESULT.SNOWHEIGHT.loc[dict(lon=results[i].lon.values, lat=results[i].lat.values)] = results[i].SNOWHEIGHT
        RESULT.EVAPORATION.loc[dict(lon=results[i].lon.values, lat=results[i].lat.values)] = results[i].EVAPORATION
        RESULT.SUBLIMATION.loc[dict(lon=results[i].lon.values, lat=results[i].lat.values)] = results[i].SUBLIMATION
        RESULT.MELT.loc[dict(lon=results[i].lon.values, lat=results[i].lat.values)] = results[i].MELT
        RESULT.H.loc[dict(lon=results[i].lon.values, lat=results[i].lat.values)] = results[i].H
        RESULT.LE.loc[dict(lon=results[i].lon.values, lat=results[i].lat.values)] = results[i].LE
        RESULT.B.loc[dict(lon=results[i].lon.values, lat=results[i].lat.values)] = results[i].B
        RESULT.TS.loc[dict(lon=results[i].lon.values, lat=results[i].lat.values)] = results[i].TS

    RESULT.to_netcdf(output_netcdf)
    print(RESULT)

    # Stop time measurement
    duration_run = datetime.now() - start_time
    print("run duration in seconds ", duration_run.total_seconds())



''' MODEL EXECUTION '''
if __name__ == "__main__":
    main()
