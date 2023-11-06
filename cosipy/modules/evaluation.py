from config import eval_method, obs_type


def evaluate(stake_names, stake_data, df_):
    """ This methods evaluates the simulation with the stake measurements
        stake_name  ::  """

    if eval_method == 'rmse':
        stat = rmse(stake_names, stake_data, df_)
    else:
        stat = None
       
    return stat


def rmse(stake_names, stake_data, df_):
    if (obs_type=='mb'):
        rmse = ((stake_data[stake_names].subtract(df_['mb'],axis=0))**2).mean()**.5
    elif (obs_type=='snowheight'):
        rmse = ((stake_data[stake_names].subtract(df_['snowheight'],axis=0))**2).mean()**.5
    else:
        msg = f'RMSE not implemented for obs_type="{obs_type}" in config.py.'
        raise NotImplementedError(msg)
    return rmse
