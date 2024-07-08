import numba
import numpy as np
import pytest

from COSIPY import start_logging
from cosipy.cpkernel.grid import Grid
from cosipy.modules.refreezing import refreezing
from cosipy.constants import Constants


class TestParamRefreezing:
    """Tests methods for parametrising refreezing.

    Attributes:
        heights (np.ndarray[float]): Snowpack heights for each layer
            [:math:`m`].
        densities (np.ndarray[float]): Snow densities for each layer
            [:math:`kg~m^{-3}`].
        temperatures (np.ndarray[float]): Temperatures for each layer
            [:math:`K`].
        liquid_water_contents (np.ndarray[float]): Liquid water content
            for each layer [:math:`m~w.e.`].
    """

    # values are different to fixture
    heights = numba.float64([0.1, 0.2, 0.3, 0.5, 0.5])
    densities = numba.float64([250, 250, 250, 917, 917])
    temperatures = numba.float64([260, 270, 271, 271.5, 272])
    liquid_water_contents = numba.float64([0.01, 0.01, 0.01, 0.01, 0.01])

    def convert_mwe(self, mwe: float, depth: float) -> float:
        """Converts m w.e. to kgm^-3."""
        assert isinstance(mwe, float)
        water = mwe / depth

        return water

    @pytest.mark.parametrize("arg_mwe", [0.0, 0.5, 1.0])
    @pytest.mark.parametrize("arg_depth", [0.1, 1.0, 10.0])
    def test_percolation_convert_mwe(self, arg_mwe: float, arg_depth: float):
        compare_mwe = self.convert_mwe(mwe=arg_mwe, depth=arg_depth)
        assert isinstance(compare_mwe, float)
        if arg_depth < 1 and arg_mwe > 0:
            assert compare_mwe > arg_mwe
        else:
            assert compare_mwe <= arg_mwe

    def get_delta_theta_w(self, grid_obj):
        delta_temperature = (
            np.array(grid_obj.get_temperature()) - Constants.zero_temperature
        )

        delta_theta_w = (
            Constants.spec_heat_ice
            * Constants.ice_density
            * np.array(grid_obj.get_ice_fraction())
            * delta_temperature
        ) / (Constants.lat_heat_melting * Constants.water_density)

        return delta_theta_w

    def test_get_delta_theta_w(self, conftest_mock_grid):
        test_grid = conftest_mock_grid

        compare_theta_w = self.get_delta_theta_w(grid_obj=test_grid)

        assert (compare_theta_w < 0.0).all()

    def get_delta_theta_i(self, theta_w):

        delta_theta_i = -(
            Constants.water_density * theta_w / Constants.ice_density
        )
        return delta_theta_i

    def test_get_delta_theta_i(self, conftest_mock_grid):
        test_grid = conftest_mock_grid
        test_theta_w = self.get_delta_theta_w(grid_obj=test_grid)
        compare_theta_i = self.get_delta_theta_i(theta_w=test_theta_w)
        assert (compare_theta_i > 0.0).all()

    def test_refreezing(self):
        GRID = Grid(
            layer_heights=self.heights,
            layer_densities=self.densities,
            layer_temperatures=self.temperatures,
            layer_liquid_water_content=self.liquid_water_contents,
        )
        heights = np.array(GRID.get_height())

        initial_ice_fraction = np.array(GRID.get_ice_fraction())
        initial_lwc = np.array(GRID.get_liquid_water_content())
        initial_mass = np.nansum(
            initial_lwc * heights * Constants.water_density
        ) + np.nansum(initial_ice_fraction * heights * Constants.ice_density)

        water_refreezed = refreezing(GRID)
        assert isinstance(water_refreezed, float)
        assert water_refreezed >= 0.0
        final_ice_fraction = np.array(GRID.get_ice_fraction())
        final_lwc = np.array(GRID.get_liquid_water_content())
        assert np.nansum(final_ice_fraction) >= np.nansum(initial_ice_fraction)
        assert np.nansum(final_lwc) <= np.nansum(initial_lwc)
        final_mass = (
            np.nansum(
                np.array(GRID.get_liquid_water_content())
                * Constants.water_density
                * heights
            )
            # + water_refreezed
            + np.nansum(
                np.array(GRID.get_ice_fraction())
                * Constants.ice_density
                * heights
            )
        )

        delta_w = final_lwc - initial_lwc
        delta_i = final_ice_fraction - initial_ice_fraction
        delta_w_mass = -delta_w * Constants.water_density
        delta_i_mass = delta_i * Constants.ice_density

        delta_i_mwe = delta_i_mass / heights
        delta_w_mwe = delta_w_mass / heights
        assert np.allclose(delta_w_mass, delta_i_mass)
        assert np.allclose(delta_w_mwe, delta_i_mwe)
        assert np.isclose(initial_mass, final_mass)
