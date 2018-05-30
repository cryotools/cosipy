import numpy as np
from datetime import datetime

from core.core_1D import core_1D
from core.io import *

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

         ### rund model in distributed version (multiple 1D versions)
         print('------------------------------')
         print("distributed run")
         print('------------------------------')

         for i in DATA.lat:
             for j in DATA.lon:

                print(DATA.sel(lat=i, lon=j))
                ### run 1D core version
                core_1D(DATA.sel(lat=i, lon=j))

    else:
         print("input not suitable")

    duration_run = datetime.now() - start_time
    print("run duration in seconds ", duration_run.total_seconds())
