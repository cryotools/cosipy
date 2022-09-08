import numpy as np
from config import eval_method, obs_type
from cosipy.utils.options import read_opt


def evaluate(stake_names, stake_data, df_, opt_dict=None):
    """ This methods evaluates the simulation with the stake measurements
        stake_name  ::  """
            
    # Read and set options
    read_opt(opt_dict, globals())
    if eval_method == 'rmse':
        stat = rmse(stake_names, stake_data, df_)
    else:
        stat = None
       
    return stat


def rmse(stake_names, stake_data, df_):
    if (obs_type=='mb'):
        rmse = ((stake_data[stake_names].subtract(df_['mb'],axis=0))**2).mean()**.5
    if (obs_type=='snowheight'):
        rmse = ((stake_data[stake_names].subtract(df_['snowheight'],axis=0))**2).mean()**.5
    return rmse
