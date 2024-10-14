import numpy as np
import pytest
import re

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

    @pytest.mark.parametrize("arg_height", [0.0, 0.05, 0.1, 0.5])
    @pytest.mark.parametrize(
        "arg_temperature",
        [Constants.temperature_bottom, Constants.zero_temperature, 280.0],
    )
    @pytest.mark.parametrize(
        "arg_time", [0, Constants.albedo_mod_snow_aging * 24]
    )
    def test_get_surface_properties(
        self,
        conftest_mock_grid,
        conftest_boilerplate,
        arg_height,
        arg_temperature,
        arg_time,
    ):
        """Get snow height, timestamp, and time since last snowfall."""

        grid = conftest_mock_grid
        surface = module_albedo.get_surface_properties(GRID=grid)

        assert isinstance(surface, tuple)
        assert len(surface) == 3
        assert all(isinstance(parameter, float) for parameter in surface)
        conftest_boilerplate.check_output(surface[0], float, 0.0)
        conftest_boilerplate.check_output(surface[1], float, 0.0)

        grid.add_fresh_snow(
            arg_height,
            Constants.constant_density,
            arg_temperature,
            0.0,
            0,
        )
        grid.set_fresh_snow_props_update_time(3600 * arg_time)
        fresh_surface = module_albedo.get_surface_properties(GRID=grid)

        assert isinstance(fresh_surface, tuple)
        assert all(isinstance(parameter, float) for parameter in fresh_surface)
        conftest_boilerplate.check_output(fresh_surface[0], float, arg_height)
        conftest_boilerplate.check_output(
            fresh_surface[1], float, arg_time * 3600
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
        error_message = " ".join(
            (
                f'Albedo method = "{module_albedo.albedo_method}"',
                f"is not allowed, must be one of",
                f'{", ".join(valid_methods)}',
            )
        )

        with pytest.raises(ValueError, match=re.escape(error_message)):
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

    @pytest.mark.parametrize("arg_hours", [0.0, 12.0, 25.0])
    def test_get_simple_albedo(self, arg_hours):
        """Get surface albedo without accounting for snow depth."""

        albedo_limit = Constants.albedo_firn + (
            Constants.albedo_fresh_snow - Constants.albedo_firn
        ) * np.exp((0) / (Constants.albedo_mod_snow_aging * 24.0))

        compare_albedo = module_albedo.get_simple_albedo(
            elapsed_time=arg_hours
        )

        assert isinstance(compare_albedo, float)
        assert 0.0 <= compare_albedo <= 1.0
        assert compare_albedo <= albedo_limit

    @pytest.mark.parametrize("arg_depth", [0.0, 0.1, 1.5])
    def test_get_albedo_with_decay(self, arg_depth):
        """Apply surface albedo decay due to the snow depth."""

        albedo_limit = Constants.albedo_firn + (
            Constants.albedo_fresh_snow - Constants.albedo_firn
        ) * np.exp((0) / (Constants.albedo_mod_snow_aging * 24.0))

        albedo_limit = Constants.albedo_firn + (
            Constants.albedo_ice - Constants.albedo_firn
        ) * np.exp(
            (-1.0 * arg_depth) / (Constants.albedo_mod_snow_depth / 100.0)
        )

        compare_albedo = module_albedo.get_albedo_with_decay(
            snow_albedo=Constants.albedo_firn, snow_height=arg_depth
        )

        assert isinstance(compare_albedo, float)
        assert 0.0 <= compare_albedo <= 1.0
        assert compare_albedo <= albedo_limit
