from cosipy.config import Config


def evaluate(stake_names, stake_data, df_):
    """Evaluate the simulation using stake measurements.

    Args:
        stake_names (list): Stake IDs.
        stake_data (pd.Dataframe): Stake measurements.
        df\_ (pd.Dataframe): Simulated mass balance and snow height.
    
    Returns:
        Statistical evaluation.
    """

    if Config.eval_method == 'rmse':
        stat = rmse(stake_names, stake_data, df_)
    else:
        stat = None
       
    return stat


def rmse(stake_names, stake_data, df_):
    if (Config.obs_type=='mb'):
        rmse = ((stake_data[stake_names].subtract(df_['mb'],axis=0))**2).mean()**.5
    elif (Config.obs_type=='snowheight'):
        rmse = ((stake_data[stake_names].subtract(df_['snowheight'],axis=0))**2).mean()**.5
    else:
        msg = f'RMSE not implemented for obs_type="{Config.obs_type}" in config.py.'
        raise NotImplementedError(msg)
    return rmse
