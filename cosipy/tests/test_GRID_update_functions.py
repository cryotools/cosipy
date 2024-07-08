import numpy as np
import pytest

# import constants
from COSIPY import start_logging
from cosipy.cpkernel.node import Node


class TestGridUpdate:
    """Tests update methods for Grid objects."""

    def test_grid_set_node_height(
        self, conftest_mock_grid, conftest_boilerplate
    ):
        GRID = conftest_mock_grid
        test_height = GRID.get_node_height(0)
        GRID.set_node_height(0, test_height + 0.5)
        compare_height = GRID.get_node_height(0)
        conftest_boilerplate.check_output(
            compare_height, float, test_height + 0.5
        )

    def test_grid_set_node_temperature(
        self, conftest_mock_grid, conftest_boilerplate
    ):
        GRID = conftest_mock_grid
        test_temperature = GRID.get_node_temperature(0)
        GRID.set_node_temperature(0, test_temperature + 0.5)
        compare_temperature = GRID.get_node_temperature(0)
        conftest_boilerplate.check_output(
            compare_temperature, float, test_temperature + 0.5
        )

    def test_grid_set_node_ice_fraction(
        self, conftest_mock_grid, conftest_boilerplate
    ):
        GRID = conftest_mock_grid
        test_fraction = GRID.get_node_ice_fraction(0)
        GRID.set_node_ice_fraction(0, test_fraction + 0.5)
        compare_fraction = GRID.get_node_ice_fraction(0)
        conftest_boilerplate.check_output(
            compare_fraction, float, test_fraction + 0.5
        )

    def test_grid_set_node_liquid_water_content(
        self, conftest_mock_grid, conftest_boilerplate
    ):
        GRID = conftest_mock_grid
        test_lwc = GRID.get_node_liquid_water_content(0)
        GRID.set_node_liquid_water_content(0, test_lwc + 0.5)
        compare_lwc = GRID.get_node_liquid_water_content(0)
        conftest_boilerplate.check_output(compare_lwc, float, test_lwc + 0.5)

    def test_grid_set_node_refreeze(
        self, conftest_mock_grid, conftest_boilerplate
    ):
        GRID = conftest_mock_grid
        test_refreeze = GRID.get_node_liquid_water_content(0)
        GRID.set_node_liquid_water_content(0, test_refreeze + 0.5)
        compare_refreeze = GRID.get_node_liquid_water_content(0)
        conftest_boilerplate.check_output(
            compare_refreeze, float, test_refreeze + 0.5
        )

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


class TestGridInteractions:
    """Tests remeshing and interactions between layers."""

    @pytest.mark.parametrize("arg_height", [0.05, 0.1, 0.5])
    @pytest.mark.parametrize("arg_temperature", [273.16, 270.16, 280.0])
    @pytest.mark.parametrize("arg_lwc", [0.0, 0.5])
    def test_add_fresh_snow(
        self,
        conftest_mock_grid,
        conftest_boilerplate,
        arg_height,
        arg_temperature,
        arg_lwc,
    ):
        """Add fresh snow layer."""

        test_grid = conftest_mock_grid
        test_number_nodes = test_grid.number_nodes
        test_snow = test_grid.get_fresh_snow_props()
        assert isinstance(test_snow, tuple)
        assert all(isinstance(parameter, float) for parameter in test_snow)
        assert test_snow[0] == 0
        dt = 3600

        test_grid.add_fresh_snow(arg_height, 250.0, arg_temperature, arg_lwc, dt)
        assert test_grid.number_nodes == test_number_nodes + 1
        assert isinstance(test_grid.grid[0], Node)

        fresh_snow = test_grid.get_fresh_snow_props()
        assert isinstance(fresh_snow, tuple)
        assert all(isinstance(parameter, float) for parameter in fresh_snow)
        assert conftest_boilerplate.check_output(
            fresh_snow[0], float, arg_height
        )
        assert not np.isclose(fresh_snow[0], test_snow[0])

        compare_node = test_grid.grid[0]
        conftest_boilerplate.check_output(
            compare_node.height, float, arg_height
        )
        conftest_boilerplate.check_output(
            compare_node.temperature, float, arg_temperature
        )
        conftest_boilerplate.check_output(
            compare_node.liquid_water_content, float, arg_lwc
        )

    @pytest.mark.parametrize("arg_idx", [None, [-1], [0, 1, 2]])
    def test_grid_remove_node(
        self, conftest_mock_grid_values, conftest_mock_grid, arg_idx
    ):
        """Remove node from grid with or without indices."""

        data = conftest_mock_grid_values.copy()
        GRID = conftest_mock_grid
        if not arg_idx:
            indices = [0]
        else:
            indices = arg_idx
        assert isinstance(indices, list)

        GRID.remove_melt_weq(0.01)
        number_nodes_before = GRID.get_number_layers()
        GRID.remove_node(arg_idx)  # Remove node

        assert GRID.get_number_layers() == number_nodes_before - len(indices)
        assert np.isclose(  # matches new density
            np.nanmean(GRID.get_density()),
            np.nanmean(np.delete(data["layer_densities"], indices)),
        )


# class TestGridUpdate:
#     """Tests update methods for Grid objects."""

#     def test_grid_update_functions(self, conftest_mock_grid):
#         GRID = conftest_mock_grid
#         GRID.set_node_liquid_water_content(0, 0.04)
#         GRID.set_node_liquid_water_content(1, 0.03)
#         GRID.set_node_liquid_water_content(2, 0.03)
#         GRID.set_node_liquid_water_content(3, 0.02)
#         GRID.set_node_liquid_water_content(4, 0.01)

#         SWE_before = np.array(GRID.get_height()) / np.array(GRID.get_density())
#         SWE_before_sum = np.nansum(SWE_before)

#         GRID.update_grid()
#         SWE_after = np.array(GRID.get_height()) / np.array(GRID.get_density())
#         SWE_after_sum = np.nansum(SWE_after)

#         GRID.adaptive_profile()
#         SWE_after_adaptive = np.array(GRID.get_height()) / np.array(
#             GRID.get_density()
#         )
#         SWE_after_adaptive_sum = np.nansum(SWE_after_adaptive)

#         assert np.allclose(SWE_before_sum, SWE_after_sum, atol=1e-3)
#         assert np.allclose(SWE_after_sum, SWE_after_adaptive_sum, atol=1e-3)

#     @pytest.mark.parametrize("arg_height", [0.05, 0.1, 0.5])
#     @pytest.mark.parametrize("arg_temperature", [273.16, 270.16, 280.0])
#     def test_add_fresh_snow(
#         self, conftest_mock_grid, arg_height, arg_temperature
#     ):
#         """Add fresh snow layer."""

#         grid = conftest_mock_grid
#         assert isinstance(grid, Grid)

#         snow = grid.get_fresh_snow_props()
#         assert isinstance(snow, tuple)
#         assert all(isinstance(parameter, float) for parameter in snow)
#         assert snow[0] == 0

#         # grid.add_fresh_snow(height=arg_height, density=250, temperature=270.15, liquid_water_content=0.0)
#         grid.add_fresh_snow(arg_height, 250.0, arg_temperature, 0.0)
#         assert isinstance(grid, Grid)

#         fresh_snow = grid.get_fresh_snow_props()
#         assert isinstance(fresh_snow, tuple)
#         assert all(isinstance(parameter, float) for parameter in fresh_snow)
#         assert np.isclose(fresh_snow[0], arg_height)
#         assert not np.isclose(fresh_snow[0], snow[0])
