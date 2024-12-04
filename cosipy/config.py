"""Hook configuration files for COSIPY."""

import argparse
import os
import sys
from importlib.metadata import entry_points
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # backwards compatibility


def get_cosipy_path_from_env(name: str = "COSIPY_DIR") -> Path:
    """Get path to COSIPY directory.

    When using WRFxCSPY, the coupler will default to searching for
    config files in the current working directory, which may contain the
    COSIPY source code. This function instead loads an environment
    variable.

    Args:
        name: Name of environment variable pointing to the COSIPY
            directory.

    Returns:
        Path to the COSIPY directory.

    Raises:
        NotADirectoryError: Invalid path.
    """
    cosipy_path = Path(os.environ.get(name, Path.cwd()))
    if not cosipy_path.is_dir():
        raise NotADirectoryError(f"Invalid path at: {cosipy_path}")

    return cosipy_path


cosipy_path = get_cosipy_path_from_env()
default_config_path = cosipy_path / "config.toml"
default_slurm_path = cosipy_path / "slurm_config.toml"
default_constants_path = cosipy_path / "constants.toml"
default_utilities_path = cosipy_path / "utilities_config.toml"


def set_parser() -> argparse.ArgumentParser:
    """Set argument parser for COSIPY."""
    tagline = "Coupled snowpack and ice surface energy and mass balance model in Python."
    parser = argparse.ArgumentParser(prog="COSIPY", description=tagline)

    # Optional arguments
    parser.add_argument(
        "-c",
        "--config",
        default=default_config_path,
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
    def get_raw_toml(file_path: Path = default_config_path) -> dict:
        """Open and load .toml configuration file.

        Args:
            file_path: Relative path to .toml configuration file.

        Returns:
            Loaded .toml data.
        """
        with file_path.open("rb") as f:
            raw_config = tomllib.load(f)

        return raw_config

    @classmethod
    def set_config_values(cls, config_table: dict):
        """Overwrite attributes with configuration data.

        Args:
            config_table: Loaded .toml data.
        """
        for _, table in config_table.items():
            for k, v in table.items():
                setattr(cls, k, v)


class Config(TomlLoader):
    """Model configuration.

    Loads, parses, and sets model configuration for COSIPY from a valid
    .toml file.
    """

    def __init__(self):
        self.args = get_user_arguments()
        self.load(self.args.config_path)

    @classmethod
    def load(cls, path: Path = default_config_path):
        raw_toml = cls.get_raw_toml(path)
        parsed_toml = cls.set_correct_config(raw_toml)
        cls.set_config_values(parsed_toml)

    @classmethod
    def set_correct_config(cls, config_table: dict) -> dict:
        """Adjust invalid or mutually exclusive configuration values.

        Args:
            config_table: Loaded .toml data.

        Returns:
            Adjusted .toml data.
        """
        # WRF Compatibility
        if config_table["DIMENSIONS"]["WRF"]:
            config_table["DIMENSIONS"]["northing"] = "south_north"
            config_table["DIMENSIONS"]["easting"] = "west_east"
        if config_table["DIMENSIONS"]["WRF_X_CSPY"]:
            config_table["FULL_FIELDS"]["full_field"] = True
        # TOML doesn't support null values
        if config_table["PARALLELIZATION"]["workers"] == 0:
            config_table["PARALLELIZATION"]["workers"] = None

        return config_table


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

    def __init__(self):
        self.args = get_user_arguments()
        self.load(self.args.slurm_path)

    @classmethod
    def load(cls, path: Path = default_slurm_path):
        raw_toml = cls.get_raw_toml(path)
        parsed_toml = cls.set_correct_config(raw_toml)
        cls.set_config_values(parsed_toml)

    @classmethod
    def set_correct_config(cls, config_table: dict) -> dict:
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
    cfg = Config()
    if cfg.slurm_use:
        SlurmConfig()


if __name__ == "__main__":
    main()
else:
    main()
