import numpy as np
import pytest

from COSIPY import start_logging
from cosipy.constants import Constants
import cosipy.modules.roughness as module_roughness
from cosipy.modules.roughness import updateRoughness


class TestParamRoughness:
    """Tests methods for parametrising roughness."""

    @pytest.mark.parametrize("arg_method", ["Wrong Method", "", None])
    def test_updateRoughness_method_error(
        self, monkeypatch, conftest_mock_grid, conftest_boilerplate, arg_method
    ):
        grid = conftest_mock_grid
        valid_methods = ["Moelg12"]

        conftest_boilerplate.patch_variable(
            monkeypatch,
            module_roughness.Constants,
            {"roughness_method": arg_method},
        )
        assert module_roughness.Constants.roughness_method == arg_method
        error_message = (
            f'Roughness method = "{module_roughness.Constants.roughness_method}"',
            f"is not allowed, must be one of",
            f'{", ".join(valid_methods)}',
        )

        with pytest.raises(ValueError, match=" ".join(error_message)):
            module_roughness.updateRoughness(grid)

    def test_method_moelg(self, conftest_mock_grid, conftest_mock_grid_ice):
        test_grid = conftest_mock_grid
        compare_roughness = module_roughness.method_Moelg(test_grid)

        assert (
            Constants.roughness_fresh_snow / 1000
            <= compare_roughness
            <= Constants.roughness_firn / 1000
        )

        test_grid_ice = conftest_mock_grid_ice
        ice_roughness = module_roughness.method_Moelg(test_grid_ice)
        assert np.isclose(ice_roughness, Constants.roughness_ice / 1000)
