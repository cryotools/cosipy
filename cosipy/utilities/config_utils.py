"""
Hook configuration files for COSIPY utilities.
"""

import argparse
import sys
from collections import namedtuple

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # backwards compatibility


class TomlLoader(object):
    """Load and parse configuration files."""

    @staticmethod
    def get_raw_toml(file_path: str = "./utilities_config.toml") -> dict:
        """Open and load .toml configuration file.

        Args:
            file_path: Relative path to .toml configuration file.

        Returns:
            Loaded .toml data.
        """
        with open(file_path, "rb") as f:
            raw_config = tomllib.load(f)

        return raw_config


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
        self.parser = self.set_arg_parser()

    @classmethod
    def load(cls, path: str = "./utilities_config.toml"):
        raw_toml = cls.get_raw_toml(path)
        cls.set_config_values(raw_toml)

    def get_config_expansion(self, name: str):
        return getattr(self, name)

    @classmethod
    def set_config_values(cls, config_table: dict):
        """Overwrite attributes with configuration data.

        Args:
            config_table: Loaded .toml data.
        """
        for header, table in config_table.items():
            data = namedtuple(header, table.keys())
            params = data(**table)
            setattr(cls, header, params)

    @classmethod
    def set_arg_parser(cls) -> argparse.ArgumentParser:
        tagline = "COSIPY Utilities."
        parser = argparse.ArgumentParser(description=tagline)
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

        return parser


def main():
    UtilitiesConfig()


if __name__ == "__main__":
    main()
else:
    main()
