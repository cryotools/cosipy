from cosipy.config import Config


def evaluate(stake_names, stake_data, df):
    """Evaluate the simulation using stake measurements.

    Implemented stake evaluation methods:

        - **rmse**: RMSE of simulated mass balance.

    Args:
        stake_names (list): Stake IDs.
        stake_data (pd.Dataframe): Stake measurements.
        df (pd.Dataframe): Simulated mass balance and snow height.

    Returns:
        float or None: Statistical evaluation.
    """

    if Config.eval_method == "rmse":
        stat = rmse(stake_names, stake_data, df)
    else:
        stat = None

    return stat


def rmse(stake_names: list, stake_data, df) -> float:
    """Get RMSE of simulated stake measurements.

    Args:
        stake_names: Stake IDs.
        stake_data (pd.Dataframe): Stake measurements.
        df (pd.Dataframe): Simulated mass balance and snow height.

    Returns:
        RMSE of simulated measurements.
    """
    if Config.obs_type not in ["mb", "snowheight"]:
        msg = f'RMSE not implemented for obs_type="{Config.obs_type}" in config.toml.'
        raise NotImplementedError(msg)
    else:
        rmse = (
            (stake_data[stake_names].subtract(df[Config.obs_type], axis=0))
            ** 2
        ).mean() ** 0.5

    return rmse
