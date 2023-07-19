import numpy as np

# import constants
from COSIPY import start_logging


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
