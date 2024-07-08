import numpy as np

from COSIPY import start_logging
from cosipy.constants import Constants
from cosipy.modules.roughness import updateRoughness


class TestParamRoughness:
    """Tests methods for parametrising roughness."""

    def test_updateRoughness(self, conftest_mock_grid, conftest_mock_grid_ice):
        GRID = conftest_mock_grid
        roughness = updateRoughness(GRID)
        assert (
            Constants.roughness_fresh_snow / 1000
            <= roughness
            <= Constants.roughness_firn / 1000
        )

        GRID_ice = conftest_mock_grid_ice
        ice_roughness = updateRoughness(GRID_ice)
        assert np.isclose(ice_roughness, Constants.roughness_ice / 1000)
