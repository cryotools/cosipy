import numpy as np
import pytest

# import constants
from COSIPY import start_logging


class TestGridGetter:
    """Tests get methods for Grid objects."""

    def test_grid_getter_functions(
        self, conftest_mock_grid_values, conftest_mock_grid
    ):
        data = conftest_mock_grid_values.copy()
        GRID = conftest_mock_grid

        # pytest documentation recommends np.allclose instead of pytest.approx
        assert np.allclose(GRID.get_height(), data["layer_heights"])
        assert np.allclose(
            GRID.get_density(), data["layer_densities"], atol=1e-3
        )
        assert np.allclose(GRID.get_temperature(), data["layer_temperatures"])
        assert np.allclose(
            GRID.get_liquid_water_content(), data["layer_liquid_water_content"]
        )
        assert np.allclose(GRID.get_snow_heights(), data["layer_heights"][0:3])
        assert np.allclose(GRID.get_ice_heights(), data["layer_heights"][3:5])
        assert np.allclose(GRID.get_node_height(0), data["layer_heights"][0])
        assert np.allclose(
            GRID.get_node_density(0), data["layer_densities"][0], atol=1e-3
        )
        assert np.allclose(
            GRID.get_node_temperature(0), data["layer_temperatures"][0]
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
