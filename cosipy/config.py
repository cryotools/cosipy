"""
Hook configuration files for COSIPY.
"""

import argparse
import sys
from importlib.metadata import entry_points
from pathlib import Path
from typing import Annotated, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.types import StringConstraints
from typing_extensions import Self

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # backwards compatibility

# FIXME: Will this work for all occasions or do we need to use frame?
cwd = Path.cwd()
default_path = cwd / "config.toml"
default_slurm_path = cwd / "slurm_config.toml"
default_constants_path = cwd / "constants.toml"
default_utilities_path = cwd / "utilities_config.toml"


def set_parser() -> argparse.ArgumentParser:
    """Set argument parser for COSIPY."""
    tagline = (
        "Coupled snowpack and ice surface energy and mass balance model in Python."
    )
    parser = argparse.ArgumentParser(prog="COSIPY", description=tagline)

    # Optional arguments
    parser.add_argument(
        "-c",
        "--config",
        default=default_path,
        dest="config_path",
        type=Path,
        metavar="<path>",
        required=False,
        help="relative path to configuration file",
    )

    parser.add_argument(
        "-x",
        "--constants",
        default=default_constants_path,
        dest="constants_path",
        type=Path,
        metavar="<path>",
        required=False,
        help="relative path to constants file",
    )

    parser.add_argument(
        "-s",
        "--slurm",
        default=default_slurm_path,
        dest="slurm_path",
        type=Path,
        metavar="<path>",
        required=False,
        help="relative path to Slurm configuration file",
    )

    return parser


def get_user_arguments() -> argparse.Namespace:
    """Parse user arguments when run as main.

    Optional switches:
        -h, --help              Show this help message and exit.

    Optional arguments:
        -c, --config <str>      Relative path to configuration file.
        -x, --constants <str>   Relative path to constants file.
        -s, --slurm <str>       Relative path to slurm configuration
                                file.

    Returns:
        Namespace of user arguments.

    Raises:
        TypeError: Illegal arguments.
    """

    parser = set_parser()

    arguments, unknown = parser.parse_known_args()

    # Conflicts with Slurm
    # if "pytest" not in sys.modules and unknown:
    #     illegal_args = ", ".join(unknown)
    #     raise TypeError(f"Illegal arguments: {illegal_args}")

    return arguments


DatetimeStr = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True, pattern=r"\d{4}-[01]\d-[0-3]\dT[0-2]\d:[0-5]\d"
    ),
]


class CosipyConfigModel(BaseModel):
    """COSIPY configuration model."""

    model_config = ConfigDict(from_attributes=True)
    time_start: DatetimeStr = Field(
        description="Start time of the simulation in ISO format"
    )
    time_end: DatetimeStr = Field(description="End time of the simulation in ISO format")
    data_path: Path = Field(description="Path to the data directory")
    input_netcdf: Path = Field(description="Input NetCDF file path")
    output_prefix: str = Field(description="Prefix for output files")
    restart: bool = Field(description="Restart flag")
    stake_evaluation: bool = Field(description="Flag for stake data evaluation")
    stakes_loc_file: Path = Field(description="Path to stake location file")
    stakes_data_file: Path = Field(description="Path to stake data file")
    eval_method: Literal["rmse"] = Field(
        "rmse", description="Evaluation method for simulations"
    )
    obs_type: Literal["mb", "snowheight"] = Field(description="Type of stake data used")
    WRF: bool = Field(description="Flag for WRF input")
    WRF_X_CSPY: bool = Field(description="Interactive simulation with WRF flag")
    northing: str = Field(description="Name of northing dimension")
    easting: str = Field(description="Name of easting dimension")
    compression_level: int = Field(
        ge=0, le=9, description="Output NetCDF compression level"
    )
    slurm_use: bool = Field(description="Use SLURM flag")
    workers: Optional[int] = Field(
        default=None,
        ge=0,
        description="""
        Setting is only used is slurm_use is False.
        Number of workers (cores), with 0 all available cores are used.
        """,
    )
    local_port: int = Field(default=8786, gt=0, description="Port for local cluster")
    full_field: bool = Field(description="Flag for writing full fields to file")
    force_use_TP: bool = Field(..., description="Total precipitation flag")
    force_use_N: bool = Field(..., description="Cloud cover fraction flag")
    tile: bool = Field(description="Flag for tiling")
    xstart: int = Field(ge=0, description="Start x index")
    xend: int = Field(ge=0, description="End x index")
    ystart: int = Field(ge=0, description="Start y index")
    yend: int = Field(ge=0, description="End y index")
    output_atm: str = Field(description="Atmospheric output variables")
    output_internal: str = Field(description="Internal output variables")
    output_full: str = Field(description="Full output variables")

    @model_validator(mode="after")
    def validate_output_variables(self) -> Self:
        if self.WRF:
            self.northing = "south_north"
            self.easting = "west_east"
        if self.WRF_X_CSPY:
            self.full_field = True
        if self.workers == 0:
            self.workers = None
        return self


def get_help():
    """Print help for commands."""
    parser = set_parser()
    parser.print_help()
    sys.exit(1)


def get_entry_points(package_name: str = "cosipy"):
    """Get package entry points.

    Returns:
        Generator: All of the package's available entry points.
    """

    if sys.version_info >= (3, 10):
        entries = entry_points(group="console_scripts")
    else:
        entries = entry_points()["console_scripts"]
    entrypoints = (
        ep
        for ep in entries
        if ep.name.startswith(package_name.upper())
        or ep.name.startswith(package_name.lower())
        or package_name.lower() in ep.name
    )

    return entrypoints


def print_entry_points(package_name: str = "cosipy"):
    """Print available entry points and their associated function."""
    entrypoints = get_entry_points(package_name=package_name)
    for ep in entrypoints:
        print(f"{ep.name}:\t{ep.value}")


class TomlLoader(object):
    """Load and parse configuration files."""

    @staticmethod
    def get_raw_toml(file_path: Path = default_path) -> dict:
        """Open and load .toml configuration file.

        Args:
            file_path: Relative path to .toml configuration file.

        Returns:
            Loaded .toml data.
        """
        with file_path.open("rb") as f:
            return tomllib.load(f)

    @staticmethod
    def flatten(config_table: dict[str, dict]) -> dict:
        """Overwrite attributes with configuration data.

        Args:
            config_table: Loaded .toml data.
        """
        flat_dict = {}
        for table in config_table.values():
            flat_dict = {**flat_dict, **table}
        return flat_dict


class Config(TomlLoader):
    """Model configuration.

    Loads, parses, and sets model configuration for COSIPY from a valid
    .toml file.
    """

    def __init__(self, path: Path = default_path) -> None:
        raw_toml = self.get_raw_toml(path)
        self.raw_toml = self.flatten(raw_toml)

    def validate(self) -> CosipyConfigModel:
        """Validate configuration using Pydantic class.

        Returns:
            CosipyConfigModel: Validated configuration.
        """
        return CosipyConfigModel(**self.raw_toml)


ShebangStr = Annotated[str, StringConstraints(strip_whitespace=True, pattern=r"^#!")]


class SlurmConfigModel(BaseModel):
    """Slurm configuration model."""

    account: str = Field(description="Slurm account/group")
    name: str = Field(description="Equivalent to Slurm parameter `--job-name`")
    queue: str = Field(description="Queue name")
    slurm_parameters: list[str] = Field(description="Additional Slurm parameters")
    shebang: ShebangStr = Field(description="Shebang string")
    local_directory: Path = Field(description="Local directory")
    port: int = Field(description="Network port number")
    cores: int = Field(description="One grid point per core")
    nodes: int = Field(description="Grid points submitted in one sbatch script")
    processes: int = Field(description="Number of processes")
    memory: str = Field(description="Memory per process")
    memory_per_process: Optional[int] = Field(gt=0, description="Memory per process")

    @model_validator(mode="after")
    def validate_output_variables(self):
        if self.memory_per_process:
            memory = self.memory_per_process * self.cores
            self.memory = f"{memory}GB"

        return self


class SlurmConfig(TomlLoader):
    """Slurm configuration.

    Loads, parses, and sets Slurm configuration for COSIPY from a valid
    .toml file.

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

    def __init__(self, path: Path = default_slurm_path) -> None:
        raw_toml = self.get_raw_toml(path)
        self.raw_toml = self.flatten(raw_toml)

    def validate(self) -> SlurmConfigModel:
        """Validate configuration using Pydantic class.

        Returns:
            CosipyConfigModel: Validated configuration.
        """
        return SlurmConfigModel(**self.raw_toml)


def main() -> tuple[CosipyConfigModel, Optional[SlurmConfigModel]]:
    args = get_user_arguments()
    cfg = Config(args.config_path).validate()
    slurm_cfg = SlurmConfig(args.slurm_path).validate() if cfg.slurm_use else None
    return cfg, slurm_cfg


if __name__ == "__main__":
    main()
else:
    main_config, slurm_config = main()
