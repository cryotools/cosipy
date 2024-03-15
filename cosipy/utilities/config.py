"""
Hook configuration files for COSIPY utilities.
"""

import argparse
import sys
from collections import namedtuple

from cosipy.config import TomlLoader, get_user_arguments

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # backwards compatibility


def get_user_arguments() -> argparse.Namespace:
    """Parse user arguments when run as main.

    Optional switches:
        -h, --help              Show this help message and exit.

    Optional arguments:
        -u, --utilities <str>   Relative path to utilities'
                                configuration file.

    Returns:
        Namespace of user arguments.
    """

    tagline = "COSIPY Utilities."
    parser = argparse.ArgumentParser(prog="cosipy", description=tagline)

    # Optional arguments
    parser.add_argument(
        "-u",
        "--utilities",
        default="./utilities_config.toml",
        dest="utilities_path",
        type=str,
        metavar="<path>",
        required=False,
        help="relative path to utilities' configuration file",
    )

    arguments = parser.parse_args()

    return arguments


class UtilitiesConfig(TomlLoader):
    """Configuration for utilities.

    Loads, parses, and sets configuration for COSIPY utilities from a
    valid .toml file.

    Attributes:
        aws2cosipy: Configuration parameters for `aws2cosipy.py`.
        create_static: Configuration parameters for `create_static.py`.
        wrf2cosipy: Configuration parameters for `wrf2cosipy.py`.
    """

    def __init__(self):
        args = get_user_arguments()
        self.load(args.utilities_path)

    @classmethod
    def load(cls, path: str = "./utilities_config.toml"):
        raw_toml = cls.get_raw_toml(path)
        cls.set_config_values(raw_toml)

    @staticmethod
    def set_config_values(config_table: dict):
        """Overwrite attributes with configuration data.

        Args:
            config_table: Loaded .toml data.
        """
        for header, table in config_table.items():
            data = namedtuple(header, table.keys())
            params = data(**table)
            setattr(UtilitiesConfig, header, params)


def main():
    args = get_user_arguments()
    UtilitiesConfig.load(args.utilities_path)


if __name__ == "__main__":
    main()
else:
    args = get_user_arguments()
    UtilitiesConfig.load(args.utilities_path)
