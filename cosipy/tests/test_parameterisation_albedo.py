import numpy as np

import constants
from COSIPY import start_logging
from cosipy.modules.albedo import updateAlbedo


class TestParamAlbedo:
    """Tests methods for parametrising albedo."""

    def test_updateAlbedo(self, conftest_mock_grid, conftest_mock_grid_ice):
        GRID = conftest_mock_grid
        GRID_ice = conftest_mock_grid_ice

        albedo = updateAlbedo(GRID)
        assert isinstance(albedo, float)
        assert (
            albedo >= constants.albedo_firn
            and albedo <= constants.albedo_fresh_snow
        )

        albedo = updateAlbedo(GRID_ice)
        assert np.isclose(albedo, constants.albedo_ice)
