import pytest

from cosipy.utilities.config_utils import UtilitiesConfig


class TestConfigUtilities:
    """Test rtoml support."""

    file_path = "utilities_config.toml"

    def test_get_config(self):
        test_cfg = UtilitiesConfig()
        assert isinstance(test_cfg, UtilitiesConfig)

    @pytest.mark.parametrize(
        "arg_table", ["aws2cosipy", "create_static", "wrf2cosipy"]
    )
    def test_load_config(self, arg_table):
        test_cfg = UtilitiesConfig()
        test_cfg.load()
        assert hasattr(test_cfg, arg_table)
        test_table = getattr(test_cfg, arg_table)
        assert isinstance(test_table, tuple)

    def test_get_config_aws2cosipy(self):
        test_cfg = UtilitiesConfig()
        test_table = test_cfg.aws2cosipy
        table_keys = [
            "names",
            "coords",
            "radiation",
            "points",
            "station",
            "lapse",
        ]
        for key in table_keys:
            assert hasattr(test_table, key)
        assert isinstance(test_table.station, dict)
        assert test_table.station["stationName"] == "Zhadang"

    def test_get_config_create_static(self):
        test_cfg = UtilitiesConfig()
        test_table = test_cfg.create_static
        table_keys = ["paths", "coords"]
        for key in table_keys:
            assert hasattr(test_table, key)
        assert isinstance(test_table.coords, dict)
        assert test_table.paths["static_folder"] == "./data/static/"

    def test_get_config_wrf2cosipy(self):
        test_cfg = UtilitiesConfig()
        test_table = test_cfg.wrf2cosipy
        table_keys = ["constants"]
        for key in table_keys:
            assert hasattr(test_table, key)
        assert isinstance(test_table.constants, dict)
        assert test_table.constants["lu_class"] == 24
