import numpy as np
from config import eval_method, obs_type


def evaluate(stake_data, df_):
    """ This methods evaluates the simulation with the stake measurements
        stake_name  ::  """

    if eval_method == 'rmse':
        stat = rmse(stake_data, df_)
    else:
        stat = None
       
    return stat


def rmse(stake_data, df_):
    # types = ['mb', 'snowheight', 'volume', 'surfTemp', 'bulkTemp']
    # rmse = []
    # print(stake_data.head())
    # print(df_.head())
    for type in obs_type:
        # rmse.append(((stake_data[stake_data[type].notnull()].subtract(df_[type],axis=0))**2).mean()**.5)
        rmse = ((stake_data[type].subtract(df_[type],axis=0))**2).mean()**.5
    print("RMSE", rmse)
    return rmse
    # if ('mb' in obs_type):
    #     rmse = ((stake_data[stake_names].subtract(df_['mb'],axis=0))**2).mean()**.5
    # if ('snowheight' in obs_type):
    #     rmse = ((stake_data[stake_names].subtract(df_['snowheight'],axis=0))**2).mean()**.5
    # if ('volume' in obs_type):
    #     rmse = ((stake_data[stake_names].subtract(df_['volume'],axis=0))**2).mean()**.5
    # if ('surfTemp' in obs_type):
    #     rmse = ((stake_data[stake_names].subtract(df_['surfTemp'],axis=0))**2).mean()**.5
    # if ('bulkTemp' in obs_type):
    #     rmse = ((stake_data[stake_names].subtract(df_['bulkTemp'],axis=0))**2).mean()**.5
