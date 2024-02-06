import numpy as np
import pytest
from numba import float64

import constants
from cosipy.cpkernel.grid import Grid


class TestGridSetup:
    """Tests initialisation methods for Grid objects."""

    def test_grid_init(self, conftest_boilerplate):
        data = {
            "layer_heights": [0.1, 0.2, 0.3],
            "layer_densities": [200.0, 210.0, 220.0],
            "layer_temperatures": [270.0, 260.0, 250.0],
            "layer_liquid_water_content": [0.0, 0.0, 0.0],
        }

        test_grid = Grid(
            layer_heights=float64(data["layer_heights"]),
            layer_densities=float64(data["layer_densities"]),
            layer_temperatures=float64(data["layer_temperatures"]),
            layer_liquid_water_content=float64(
                data["layer_liquid_water_content"]
            ),
        )
        assert isinstance(test_grid, Grid)
        conftest_boilerplate.check_output(
            test_grid.number_nodes, int, len(data["layer_heights"])
        )
        assert test_grid.layer_ice_fraction is None
        for i in range(test_grid.number_nodes):
            assert isinstance(test_grid.get_node_ice_fraction(i), float)

    def test_grid_init_grid(self, conftest_mock_grid):
        test_grid = conftest_mock_grid
        test_number_nodes = test_grid.number_nodes
        test_grid.init_grid()

        assert test_grid.number_nodes == test_number_nodes
        assert test_grid.layer_ice_fraction is None
        for i in range(test_grid.number_nodes):
            assert isinstance(test_grid.get_node_ice_fraction(i), float)


class TestGridGetter:
    """Tests get methods for Grid objects.

    ..
        Pytest documentation recommends `np.allclose` instead of
        `pytest.approx`.

    Attributes:
        data (dict[float64]): Dummy grid data.
    """

    data = {
        "layer_heights": float64([0.1, 0.2, 0.3, 0.5, 0.5]),
        "layer_densities": float64([250, 250, 250, 917, 917]),
        "layer_temperatures": float64([260, 270, 271, 271.5, 272]),
        "layer_liquid_water_content": float64([0.0, 0.0, 0.0, 0.0, 0.0]),
    }

    def create_grid(self):
        grid_object = Grid(
            layer_heights=self.data["layer_heights"],
            layer_densities=self.data["layer_densities"],
            layer_temperatures=self.data["layer_temperatures"],
            layer_liquid_water_content=self.data["layer_liquid_water_content"],
        )
        return grid_object

    @pytest.fixture(name="grid", autouse=False, scope="function")
    def fixture_grid(self):
        return self.create_grid()

    def get_ice_fraction(self, ice_fraction, density):
        if ice_fraction is None:
            a = (
                density
                - (1 - (density / constants.ice_density))
                * constants.air_density
            )
            ice_fraction = a / constants.ice_density
        else:
            ice_fraction = ice_fraction
        return ice_fraction

    @pytest.mark.parametrize("arg_ice_fraction", [0.1, None])
    def test_get_ice_fraction(self, arg_ice_fraction, conftest_boilerplate):
        test_density = 300
        if arg_ice_fraction is None:
            a = (
                test_density
                - (1 - (test_density / constants.ice_density))
                * constants.air_density
            )
            test_ice = a / constants.ice_density
        else:
            test_ice = arg_ice_fraction
        compare_ice = self.get_ice_fraction(arg_ice_fraction, test_density)
        assert conftest_boilerplate.check_output(compare_ice, float, test_ice)

    def test_create_grid(self):
        test_grid = self.create_grid()
        assert isinstance(test_grid, Grid)
        assert test_grid.number_nodes == len(self.data["layer_heights"])

    def test_grid_get_height(self, grid, conftest_boilerplate):
        assert np.allclose(grid.get_height(), self.data["layer_heights"])
        for i in range(grid.number_nodes):
            conftest_boilerplate.check_output(
                grid.get_node_height(i), float, self.data["layer_heights"][i]
            )

    def test_grid_get_temperature(self, grid, conftest_boilerplate):
        assert np.allclose(
            grid.get_temperature(), self.data["layer_temperatures"]
        )
        for i in range(grid.number_nodes):
            conftest_boilerplate.check_output(
                grid.get_node_temperature(i),
                float,
                self.data["layer_temperatures"][i],
            )

    def test_grid_get_liquid_water_content(self, grid, conftest_boilerplate):
        assert np.allclose(
            grid.get_liquid_water_content(),
            self.data["layer_liquid_water_content"],
        )
        for i in range(grid.number_nodes):
            conftest_boilerplate.check_output(
                grid.get_node_liquid_water_content(i),
                float,
                self.data["layer_liquid_water_content"][i],
            )

    def test_grid_get_density(self, grid, conftest_boilerplate):
        assert np.allclose(grid.get_density(), self.data["layer_densities"])
        for i in range(grid.number_nodes):
            conftest_boilerplate.check_output(
                grid.get_node_density(i),
                float,
                self.data["layer_densities"][i],
            )

    def test_grid_get_ice_fraction(self, grid, conftest_boilerplate):
        ice_fractions = [
            self.get_ice_fraction(None, density)
            for density in self.data["layer_densities"]
        ]
        assert np.allclose(grid.get_ice_fraction(), ice_fractions)

        for i in range(grid.number_nodes):
            conftest_boilerplate.check_output(
                grid.get_node_ice_fraction(i), float, ice_fractions[i]
            )

    def test_grid_get_refreeze(self, grid, conftest_boilerplate):
        refrozen = [0.0 for i in range(grid.number_nodes)]
        assert np.allclose(grid.get_refreeze(), refrozen)

        for i in range(grid.number_nodes):
            conftest_boilerplate.check_output(
                grid.get_node_refreeze(i), float, refrozen[i]
            )

    def test_grid_get_snow_ice_heights(
        self, conftest_mock_grid_values, conftest_mock_grid
    ):
        data = conftest_mock_grid_values.copy()
        GRID = conftest_mock_grid

        assert np.allclose(GRID.get_snow_heights(), data["layer_heights"][0:3])
        assert np.allclose(GRID.get_ice_heights(), data["layer_heights"][3:5])
        assert np.allclose(GRID.get_node_height(0), data["layer_heights"][0])

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
