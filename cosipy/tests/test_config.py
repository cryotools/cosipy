import argparse
import os
import pathlib
from unittest.mock import patch

import pytest

import cosipy.config
from cosipy.config import Config


class TestConfigParser:
    """Test argument parsing for config hook."""

    file_path = "./config.toml"

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

    @pytest.mark.dependency(name="TestConfigParser::test_set_parser")
    @pytest.mark.parametrize("arg_type", (str, pathlib.Path))
    def test_user_arguments(self, arg_type):
        test_parser = cosipy.config.set_parser()
        test_path = "./some/path/config.toml"
        assert isinstance(test_path, str)

        test_args = [
            "--config",
            test_path,
            "--constants",
            test_path,
            "--slurm",
            test_path,
        ]
        arguments, unknown = test_parser.parse_known_args(test_args)
        assert isinstance(arguments, argparse.Namespace)

        for user_path in [
            arguments.config_path,
            arguments.constants_path,
            arguments.slurm_path,
        ]:
            assert isinstance(user_path, pathlib.Path)
            assert user_path == pathlib.Path(test_path)

    @patch.dict(os.environ, {"COSIPY_DIR": "./path/to/wrong/cosipy/"})
    def test_check_directory_exists(self):
        """Raise error if directory not found."""
        wrong_path = "./path/to/wrong/cosipy/"
        error_message = f"Invalid path at: {pathlib.Path(wrong_path)}"
        with pytest.raises(NotADirectoryError, match=error_message):
            cosipy.config.get_cosipy_path_from_env(name="COSIPY_DIR")

    @pytest.mark.parametrize(
        "arg_env", ((True, "COSIPY_DIR"), (True, "XFAIL"), (False, ""))
    )
    @patch.dict(os.environ, {"COSIPY_DIR": "./path/to/cosipy/"})
    def test_get_cosipy_path(
        self, arg_env, conftest_mock_check_directory_exists
    ):
        _ = conftest_mock_check_directory_exists
        test_name = arg_env[1]
        compare_path = cosipy.config.get_cosipy_path_from_env(name=test_name)

        if arg_env[1] == "COSIPY_DIR":
            assert compare_path == pathlib.Path("./path/to/cosipy/")
        else:
            assert compare_path == pathlib.Path.cwd()


class TestConfig:
    """Test rtoml support."""

    file_path = "config.toml"

    def test_init_config(self):
        test_cfg = Config()
        assert isinstance(test_cfg, Config)

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

        test_cfg = Config()
        compare_toml = test_cfg.get_raw_toml()
        assert isinstance(compare_toml, dict)
        for name in variable_names:
            assert name in compare_toml.keys()

    def test_load_config(self):
        test_cfg = Config()
        test_cfg.load()

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
            compare_attr = getattr(test_cfg, name)
            assert (
                isinstance(compare_attr, (int, str, bool))
                or compare_attr is None
            )
