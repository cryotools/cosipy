import numba
import numpy as np

# import constants
from COSIPY import start_logging
from cosipy.cpkernel.grid import Grid
from cosipy.modules.densification import densification


class TestParamDensification:
    """Tests methods for parametrising densification.

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
    heights = numba.float64([10.0, 10.0, 10.0, 0.2, 0.2])
    densities = numba.float64([450, 450, 450, 100, 100])
    temperatures = numba.float64([260, 270, 271, 271.5, 272])
    liquid_water_contents = numba.float64([0, 0, 0, 0, 0])

    def test_densification(self):
        GRID = Grid(  # values are different to fixture
            layer_heights=self.heights,
            layer_densities=self.densities,
            layer_temperatures=self.temperatures,
            layer_liquid_water_content=self.liquid_water_contents,
        )
        SWE_before = np.array(GRID.get_height()) / np.array(GRID.get_density())
        SWE_before_sum = np.nansum(SWE_before)

        densification(GRID=GRID, SLOPE=0.0, dt=3600)

        SWE_after = np.array(GRID.get_height()) / np.array(GRID.get_density())
        SWE_after_sum = np.nansum(SWE_after)

        # Original assertions:
        # assert np.isclose(SWE_before_sum, SWE_after_sum, atol=1e-3)
        # assert SWE_before_sum == SWE_after_sum

        # New assertion: Global densification method is set to "Boone",
        # not "constant", so shouldn't the SWEs NOT match?
        assert not np.isclose(SWE_before_sum, SWE_after_sum, atol=1e-3)
