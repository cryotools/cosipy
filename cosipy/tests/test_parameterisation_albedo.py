from unittest.mock import patch

import numpy as np
import pytest

import constants
import cosipy.modules.albedo as test_albedo
from COSIPY import start_logging

class TestParamAlbedoProperties:
    """Tests get/set methods for albedo properties."""

    def test_updateAlbedo(self, conftest_mock_grid, conftest_mock_grid_ice):
        GRID = conftest_mock_grid
        GRID_ice = conftest_mock_grid_ice

        albedo = test_albedo.updateAlbedo(GRID)
        assert isinstance(albedo, float)
        assert (
            albedo >= constants.albedo_firn
            and albedo <= constants.albedo_fresh_snow
        )
        albedo = test_albedo.updateAlbedo(GRID_ice)
        assert np.isclose(albedo, constants.albedo_ice)

    @pytest.mark.parametrize("arg_method", ["Oerlemans98", "Lejeune13"])
    def test_updateAlbedo_method(self, conftest_mock_grid, arg_method):
        """Set method from constants.py when calculating albedo."""

        grid = conftest_mock_grid

        with patch("constants.albedo_method", arg_method):
            assert constants.albedo_method == arg_method
            albedo = test_albedo.updateAlbedo(grid)
            assert isinstance(albedo, float)

    @pytest.mark.parametrize("arg_method", ["Wrong Method", "", None])
    def test_updateAlbedo_method_error(self, conftest_mock_grid, arg_method):
        grid = conftest_mock_grid

        with patch("constants.albedo_method", arg_method):
            assert constants.albedo_method == arg_method
            error_message = f'Albedo method = "{constants.albedo_method}" is not allowed, must be one of {", ".join(["Oerlemans98", "Lejeune13"])}'

            with pytest.raises(ValueError, match=error_message):
                test_albedo.updateAlbedo(grid)

    @pytest.mark.parametrize("arg_height", [0.0, 0.05, 0.1, 0.5])
    @pytest.mark.parametrize(
        "arg_temperature",
        [constants.temperature_bottom, constants.zero_temperature, 280.0],
    )
    @pytest.mark.parametrize(
        "arg_time", [0, constants.albedo_mod_snow_aging * 24]
    )
    def test_get_surface_properties(
        self, conftest_mock_grid, arg_height, arg_temperature, arg_time
    ):
        """Get snow height, timestamp, and time since last snowfall."""

        grid = conftest_mock_grid
        surface = test_albedo.get_surface_properties(GRID=grid)

        assert isinstance(surface, tuple)
        assert len(surface) == 3
        assert all(isinstance(parameter, float) for parameter in surface)
        assert surface[0] == 0
        assert surface[1] == 0

        grid.add_fresh_snow(
            arg_height,
            constants.constant_density,
            arg_temperature,
            0.0,
        )
        grid.set_fresh_snow_props_update_time(3600 * arg_time)
        fresh_surface = test_albedo.get_surface_properties(GRID=grid)

        assert isinstance(fresh_surface, tuple)
        assert all(isinstance(parameter, float) for parameter in fresh_surface)
        assert np.isclose(fresh_surface[0], arg_height)
        assert fresh_surface[1] == arg_time * 3600


class TestParamAlbedoMethods:
    """Tests methods of albedo parametrisation."""

    @pytest.mark.parametrize("arg_hours", [0.0, 12.0, 25.0])
    def test_get_simple_albedo(self, arg_hours):
        """Get surface albedo without accounting for snow depth."""

        albedo_limit = constants.albedo_firn + (
            constants.albedo_fresh_snow - constants.albedo_firn
        ) * np.exp((0) / (constants.albedo_mod_snow_aging * 24.0))

        compare_albedo = test_albedo.get_simple_albedo(elapsed_time=arg_hours)

        assert isinstance(compare_albedo, float)
        assert 0.0 <= compare_albedo <= 1.0
        assert compare_albedo <= albedo_limit

    @pytest.mark.parametrize("arg_depth", [0.0, 0.05, 0.1, 1.0])
    def test_albedo_weighting_lejeune(self, arg_depth):
        """Get albedo weight."""

        weight = test_albedo.get_albedo_weight_lejeune(snow_depth=arg_depth)
        assert isinstance(weight, float)

        if arg_depth >= 0.1:
            assert weight == 1.0
        else:
            assert 0 <= weight < 1.0

    def test_method_lejeune(self, conftest_mock_grid):
        """Get albedo for snow-covered debris."""

        grid = conftest_mock_grid

        albedo = test_albedo.method_lejeune(GRID=grid)
        assert isinstance(albedo, float)
        assert 0.0 <= albedo <= 1.0
        assert np.isclose(albedo, constants.albedo_debris)

    def test_method_lejeune_ice(self, conftest_mock_grid_ice):
        """Get albedo for bare debris."""

        grid = conftest_mock_grid_ice

        albedo = test_albedo.method_lejeune(GRID=grid)
        assert isinstance(albedo, float)
        assert 0.0 <= albedo <= 1.0
        assert np.isclose(albedo, constants.albedo_debris)
