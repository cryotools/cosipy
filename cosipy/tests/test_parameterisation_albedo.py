import numpy as np
import pytest

import cosipy.modules.albedo as module_albedo
from COSIPY import start_logging
from cosipy.constants import Constants


class TestParamAlbedoUpdate:
    """Tests get/set methods for albedo properties."""

    def test_updateAlbedo(self, conftest_mock_grid):
        grid = conftest_mock_grid

        surface_albedo, snow_albedo = module_albedo.updateAlbedo(
            GRID=grid,
            surface_temperature=270.0,
            albedo_snow=Constants.albedo_fresh_snow,
        )
        assert isinstance(surface_albedo, float)
        assert (
            Constants.albedo_firn
            <= surface_albedo
            <= Constants.albedo_fresh_snow
        )
        assert isinstance(snow_albedo, float)

    def test_updateAlbedo_ice(
        self, conftest_mock_grid_ice, conftest_boilerplate
    ):
        grid_ice = conftest_mock_grid_ice
        albedo, snow_albedo = module_albedo.updateAlbedo(
            GRID=grid_ice,
            surface_temperature=270.0,
            albedo_snow=Constants.albedo_fresh_snow,
        )
        assert conftest_boilerplate.check_output(
            albedo, float, Constants.albedo_ice
        )
        assert conftest_boilerplate.check_output(
            snow_albedo, float, Constants.albedo_fresh_snow
        )


class TestParamAlbedoSelection:
    """Tests user selection of parametrisation method."""

    @pytest.mark.parametrize("arg_method", ["Oerlemans98", "Bougamont05"])
    def test_updateAlbedo_method(
        self, monkeypatch, conftest_mock_grid, conftest_boilerplate, arg_method
    ):
        """Set method from constants.py when calculating albedo."""

        grid = conftest_mock_grid

        conftest_boilerplate.patch_variable(
            monkeypatch, module_albedo, {"albedo_method": arg_method}
        )
        assert module_albedo.albedo_method == arg_method
        surface_albedo, snow_albedo = module_albedo.updateAlbedo(
            GRID=grid,
            surface_temperature=270.0,
            albedo_snow=Constants.albedo_fresh_snow,
        )
        assert isinstance(surface_albedo, float)
        assert isinstance(snow_albedo, float)

    @pytest.mark.parametrize("arg_method", ["Wrong Method", "", None])
    def test_updateAlbedo_method_error(
        self, monkeypatch, conftest_mock_grid, conftest_boilerplate, arg_method
    ):
        grid = conftest_mock_grid
        valid_methods = ["Oerlemans98", "Bougamont05"]

        conftest_boilerplate.patch_variable(
            monkeypatch, module_albedo, {"albedo_method": arg_method}
        )
        assert module_albedo.albedo_method == arg_method
        error_message = (
            f'Albedo method = "{module_albedo.albedo_method}"',
            f"is not allowed, must be one of",
            f'{", ".join(valid_methods)}',
        )

        with pytest.raises(ValueError, match=" ".join(error_message)):
            module_albedo.updateAlbedo(
                GRID=grid,
                surface_temperature=270.0,
                albedo_snow=Constants.albedo_fresh_snow,
            )


class TestParamAlbedoMethods:
    """Tests methods for parametrising albedo."""

    def test_method_Oerlemans(self, conftest_mock_grid):
        """Get surface albedo without accounting for snow depth."""

        grid = conftest_mock_grid
        albedo_limit = Constants.albedo_firn + (
            Constants.albedo_fresh_snow - Constants.albedo_firn
        ) * np.exp(0 / (Constants.albedo_mod_snow_aging * 24.0))

        compare_albedo = module_albedo.method_Oerlemans(grid)

        assert isinstance(compare_albedo, float)
        assert 0.0 <= compare_albedo <= 1.0
        assert compare_albedo <= albedo_limit
