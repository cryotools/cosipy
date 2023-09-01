import numpy as np
import pytest

# import constants
from COSIPY import start_logging
from cosipy.cpkernel.grid import Grid


class TestGridUpdate:
    """Tests update methods for Grid objects."""

    def test_grid_update_functions(self, conftest_mock_grid):
        GRID = conftest_mock_grid
        GRID.set_node_liquid_water_content(0, 0.04)
        GRID.set_node_liquid_water_content(1, 0.03)
        GRID.set_node_liquid_water_content(2, 0.03)
        GRID.set_node_liquid_water_content(3, 0.02)
        GRID.set_node_liquid_water_content(4, 0.01)

        SWE_before = np.array(GRID.get_height()) / np.array(GRID.get_density())
        SWE_before_sum = np.nansum(SWE_before)

        GRID.update_grid()
        SWE_after = np.array(GRID.get_height()) / np.array(GRID.get_density())
        SWE_after_sum = np.nansum(SWE_after)

        GRID.adaptive_profile()
        SWE_after_adaptive = np.array(GRID.get_height()) / np.array(
            GRID.get_density()
        )
        SWE_after_adaptive_sum = np.nansum(SWE_after_adaptive)

        assert np.allclose(SWE_before_sum, SWE_after_sum, atol=1e-3)
        assert np.allclose(SWE_after_sum, SWE_after_adaptive_sum, atol=1e-3)

    @pytest.mark.parametrize("arg_height", [0.05, 0.1, 0.5])
    @pytest.mark.parametrize("arg_temperature", [273.16, 270.16, 280.0])
    def test_add_fresh_snow(
        self, conftest_mock_grid, arg_height, arg_temperature
    ):
        """Add fresh snow layer."""

        grid = conftest_mock_grid
        assert isinstance(grid, Grid)

        snow = grid.get_fresh_snow_props()
        assert isinstance(snow, tuple)
        assert all(isinstance(parameter, float) for parameter in snow)
        assert snow[0] == 0

        # grid.add_fresh_snow(height=arg_height, density=250, temperature=270.15, liquid_water_content=0.0)
        grid.add_fresh_snow(arg_height, 250.0, arg_temperature, 0.0)
        assert isinstance(grid, Grid)

        fresh_snow = grid.get_fresh_snow_props()
        assert isinstance(fresh_snow, tuple)
        assert all(isinstance(parameter, float) for parameter in fresh_snow)
        assert np.isclose(fresh_snow[0], arg_height)
        assert not np.isclose(fresh_snow[0], snow[0])
