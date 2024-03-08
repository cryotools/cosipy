"""
Hook configuration files for COSIPY.
"""

import sys

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # backwards compatibility


class TomlLoader(object):
    """Load and parse configuration files."""

    @staticmethod
    def get_raw_toml(file_path: str = "./config.toml") -> dict:
        """Open and load .toml configuration file.

        Args:
            file_path: Relative path to .toml configuration file.

        Returns:
            Loaded .toml data.
        """
        with open(file_path, "rb") as f:
            raw_config = tomllib.load(f)

        return raw_config

    @staticmethod
    def set_config_values(config_table: dict):
        """Overwrite attributes with configuration data.

        Args:
            config_table: Loaded .toml data.
        """
        for _, table in config_table.items():
            for k, v in table.items():
                setattr(Config, k, v)


class Config(TomlLoader):
    """Model configuration.

    Loads, parses, and sets model configuration for COSIPY from a valid
    .toml file.

    .. note::
        Attributes must be declared to avoid import errors, but are
        overwritten once the configuration file is read.

    """

    time_start = "2009-01-01T06:00"
    time_end = "2009-01-10T00:00"
    data_path = "./data/"
    input_netcdf = "Zhadang/Zhadang_ERA5_2009.nc"
    output_prefix = "Zhadang_ERA5"
    restart = False
    stake_evaluation = False
    stakes_loc_file = "./data/input/HEF/loc_stakes.csv"
    stakes_data_file = "./data/input/HEF/data_stakes_hef.csv"
    eval_method = "rmse"
    obs_type = "snowheight"
    WRF = False
    northing = "lat"
    easting = "lon"
    WRF_X_CSPY = False
    compression_level = 2
    slurm_use = False
    workers = None
    local_port = 8786
    full_field = False
    force_use_TP = False
    force_use_N = False
    tile = False
    xstart = 20
    xend = 40
    ystart = 20
    yend = 40

    def __init__(self, path: str = "./config.toml"):
        raw_toml = self.get_raw_toml(path)
        parsed_toml = self.set_correct_config(raw_toml)
        self.set_config_values(parsed_toml)

    @staticmethod
    def set_correct_config(config_table: dict) -> dict:
        """Adjust invalid or mutually exclusive configuration values.

        Args:
            config_table: Loaded .toml data.

        Returns:
            Adjusted .toml data.
        """
        # WRF Compatibility
        if config_table["WRF"]["WRF"]:
            config_table["WRF"]["northing"] = "south_north"
            config_table["WRF"]["easting"] = "west_east"
        if config_table["WRF"]["WRF_X_CSPY"]:
            config_table["FULL_FIELDS"]["full_field"] = True
        # TOML doesn't support null values
        if config_table["PARALLELIZATION"]["workers"] == 0:
            config_table["PARALLELIZATION"]["workers"] = None

        return config_table


class SlurmConfig(TomlLoader):
    """SLURM configuration.

    Loads, parses, and sets Slurm configuration for COSIPY from a valid
    .toml file.

    .. note::
        Attributes must be declared to avoid import errors, but are
        overwritten once the configuration file is read.

    Attributes:
        account (str): Slurm account/group, equivalent to Slurm
            parameter `--account`.
        name (str): Equivalent to Slurm parameter `--job-name`.
        project (str): Project name.
        queue (str): Queue name.
        port (int): Network port number.
        cores (int): One grid point per core, do not change.
        nodes (str): Grid points submitted in one sbatch script.
        processes (int): Number of processes.
        memory (str): Memory per process.
        shebang (str): Shebang string.
        slurm_parameters (List[str]): Additional Slurm parameters.

    """

    account = ""
    name = ""
    project = ""
    queue = ""
    port = 8786
    cores = 1
    processes = 20
    memory = ""
    nodes = 1
    shebang = ""
    slurm_parameters = []

    def __init__(self, path: str = "./slurm_config.toml"):
        raw_toml = self.get_raw_toml(path)
        parsed_toml = self.set_correct_config(raw_toml)
        self.set_config_values(parsed_toml)

    @staticmethod
    def set_correct_config(config_table: dict) -> dict:
        """Adjust invalid or mutually exclusive configuration values.

        Args:
            config_table: Loaded .toml data.

        Returns:
            Adjusted .toml data.
        """
        if config_table["OVERRIDES"]["memory_per_process"]:
            memory = (
                config_table["OVERRIDES"]["memory_per_process"]
                * config_table["MEMORY"]["cores"]
            )
            config_table["MEMORY"]["memory"] = f"{memory}GB"

        return config_table


def main():
    pass


if __name__ == "__main__":
    main()
