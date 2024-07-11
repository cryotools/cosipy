from COSIPY import start_logging
from cosipy.constants import Constants
from cosipy.modules.surfaceTemperature import update_surface_temperature


class TestParamSurfaceTemperature:
    """Tests methods for parametrising surface temperature."""

    def test_surface_Temperature_parameterisation(self, conftest_mock_grid):
        GRID = conftest_mock_grid

        (
            fun,
            surface_temperature,
            lw_radiation_in,
            lw_radiation_out,
            sensible_heat_flux,
            latent_heat_flux,
            ground_heat_flux,
            rain_heat_flux,
            rho,
            Lv,
            monin_obukhov_length,
            Cs_t,
            Cs_q,
            q0,
            q2,
        ) = update_surface_temperature(
            # Old args: GRID, 0.6, (0.24 / 1000), 275, 0.6, 789, 1000, 4.5, 0.0, 0.1
            GRID=GRID,
            dt=3600,
            z=2,
            z0=(0.24 / 1000),
            T2=275,
            rH2=50,
            p=1000,
            SWnet=789,
            u2=3.5,
            RAIN=0.1,
            SLOPE=0.0,
            LWin=None,
            N=0.5,  # otherwise tries to retrieve LWin from non-existent file
        )

        assert Constants.zero_temperature >= surface_temperature >= 220.0
        assert 400 >= lw_radiation_in >= 0
        assert 0 >= lw_radiation_out >= -400
        assert 250 >= sensible_heat_flux >= -250
        assert 200 >= latent_heat_flux >= -200
        assert 100 >= ground_heat_flux >= -100
