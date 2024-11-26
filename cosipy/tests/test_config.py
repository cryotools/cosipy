import argparse
from pathlib import Path

import pytest

import cosipy.config
from cosipy.config import Config, CosipyConfigModel, main_config

file_path = Path("config.toml")

@pytest.fixture()
def test_cfg():
    return Config(file_path).validate()

class TestConfigParser:
    """Test argument parsing for config hook."""

    def test_set_parser(self):
        compare_parser = cosipy.config.set_parser()
        assert isinstance(compare_parser, argparse.ArgumentParser)

        actions = []
        for action in compare_parser._actions:
            longest = ""
            for option in action.option_strings:
                option = option.lstrip("-")
                if len(option) > len(longest):
                    longest = option
            actions.append(longest)

        for name in ["help", "config", "constants", "slurm"]:
            assert name in actions


class TestConfig:
    """Test rtoml support."""

    def test_init_config(self, test_cfg):
        assert isinstance(test_cfg, CosipyConfigModel)
        assert test_cfg == main_config

    def test_get_raw_toml(self):
        variable_names = [
            "SIMULATION_PERIOD",
            "FILENAMES",
            "RESTART",
            "STAKE_DATA",
            "DIMENSIONS",
            "COMPRESSION",
            "PARALLELIZATION",
            "FULL_FIELDS",
            "FORCINGS",
            "SUBSET",
        ]

        compare_toml = Config(file_path).get_raw_toml(file_path)
        assert isinstance(compare_toml, dict)
        for name in variable_names:
            assert name in compare_toml

    def test_load_config(self, test_cfg):
        variable_names = [
            "time_start",
            "time_end",
            "data_path",
            "input_netcdf",
            "output_prefix",
            "restart",
            "stake_evaluation",
            "stakes_loc_file",
            "stakes_data_file",
            "eval_method",
            "obs_type",
            "WRF",
            "WRF_X_CSPY",
            "northing",
            "easting",
            "compression_level",
            "slurm_use",
            "workers",
            "full_field",
            "force_use_TP",
            "force_use_N",
            "tile",
            "xstart",
            "xend",
            "ystart",
            "yend",
        ]

        for name in variable_names:
            assert hasattr(test_cfg, name)
