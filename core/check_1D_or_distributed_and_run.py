import numpy as np
from datetime import datetime
import itertools

from core.core_1D import core_1D
from core.io import *

from dask.distributed import Client, LocalCluster, progress 
from dask.distributed import Worker

def check_1D_or_distributed_and_run():

    start_time = datetime.now()

    ### read input data
    DATA = preprocessing()

    if DATA.T2.ndim == 1:
        
        print('------------------------------')
        print("1D run")
        print('------------------------------')

        ### run model in 1D version
        core_1D(DATA)


    elif DATA.T2.ndim == 3:


        cluster = LocalCluster(n_workers=1,scheduler_port=8786,threads_per_worker=1)
        client = Client('127.0.0.1:8786')
        print(client)

        #fut = [client.submit(core_1D, DATA.sel(lat=i, lon=j)) for i,j in itertools.product(DATA.lat, DATA.lon)]
        fut = client.submit(core_1D, DATA.sel(lat=-45, lon=-77)) 
        progress(fut) 
        result = client.gather(fut)
        #print(result[0])

    duration_run = datetime.now() - start_time
    print("run duration in seconds ", duration_run.total_seconds())
