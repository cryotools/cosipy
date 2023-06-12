import numpy as np

# import constants
from COSIPY import start_logging
from cosipy.modules.percolation import percolation


class TestParamPercolation:
    """Tests percolation methods.

    Attributes:
        melt_water (float): Melt water at surface.
        timedelta (int): Integration time, dt [s].
    """

    melt_water = 1.0
    timedelta = 7200

    def test_percolation(self, conftest_mock_grid):
        GRID = conftest_mock_grid

        initial_water = self.melt_water + np.nansum(
            GRID.get_liquid_water_content()
        )
        runoff = percolation(GRID, self.melt_water, self.timedelta)
        total_water = runoff + np.nansum(GRID.get_liquid_water_content())

        # Bug? Total water is greater than the initially available water
        # assert np.isclose(self.melt_water, total_water)
        assert np.isclose(initial_water, total_water)
