import sys

from cosipy.config import Config, TomlLoader, get_user_arguments

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # backwards compatibility


class Constants(TomlLoader):
    """Constants configuration.

    Loads, parses, and sets constants for COSIPY from a valid .toml
    file.
    """

    def __init__(self):
        self.args = get_user_arguments()
        self.load(self.args.constants_path)

    @classmethod
    def load(cls, path: str = "./constants.toml"):
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
        # WRF_X_CSPY: for efficiency and consistency
        if Config.WRF_X_CSPY:
            config_table["PARAMETERIZATIONS"]["albedo_method"] = "Oerlemans98"
            config_table["PARAMETERIZATIONS"]["stability_correction"] = "MO"
            # config_table["PARAMETERIZATIONS"][
            #     "sfc_temperature_method"
            # ] = "Secant"

        return config_table


def main():
    Constants()


if __name__ == "__main__":
    main()
else:
    main()
