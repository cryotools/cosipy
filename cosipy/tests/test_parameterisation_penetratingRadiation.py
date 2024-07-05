import pytest
import constants
import cosipy.modules.penetratingRadiation as pRad


class TestParamRadiation:
    """Tests radiation methods."""

    melt_water = 1.0
    timedelta = 3600

    def test_penetrating_radiation_error(
        self,
        monkeypatch,
        conftest_mock_grid,
        conftest_boilerplate,
    ):
        test_grid = conftest_mock_grid
        allow_list = ["Bintanja95"]
        conftest_boilerplate.patch_variable(
            monkeypatch, pRad, {"penetrating_method": "WrongMethod"}
        )
        error_msg = (
            f'Penetrating method = "{pRad.penetrating_method}" ',
            f'is not allowed, must be one of {", ".join(allow_list)}',
        )
        with pytest.raises(ValueError, match="".join(error_msg)):
            pRad.penetrating_radiation(GRID=test_grid, SWnet=800.0, dt=3600)

    @pytest.mark.parametrize(
        "arg_density", [250.0, constants.snow_ice_threshold + 1]
    )
    def test_method_Bintanja(
        self,
        monkeypatch,
        conftest_mock_grid,
        conftest_boilerplate,
        arg_density,
    ):
        conftest_boilerplate.patch_variable(
            monkeypatch, pRad, {"penetrating_method": "Bintanja95"}
        )
        test_grid = conftest_mock_grid
        test_grid.add_fresh_snow(0.1, arg_density, 270.15, 0.0)
        test_swnet = 800.0
        if arg_density <= constants.snow_ice_threshold:
            test_si = test_swnet * 0.1
        else:
            test_si = test_swnet * 0.2

        melt_si = pRad.method_Bintanja(
            GRID=test_grid, SWnet=test_swnet, dt=constants.dt
        )
        assert isinstance(melt_si, tuple)
        compare_melt = melt_si[0]
        assert isinstance(compare_melt, float)
        compare_si = melt_si[1]
        conftest_boilerplate.check_output(compare_si, float, test_si)
