import numpy as np
import pytest

import cosipy.modules.surfaceTemperature as module_surface_temperature
from cosipy.constants import Constants


class TestParamSurfaceTemperature:
    """Tests methods for parametrising surface temperature."""

    dt = 3600
    z = 2
    z0 = 0.24 / 1000
    T2 = 275
    rH2 = 50
    p = 1000
    SWnet = 789
    u2 = 3.5
    RAIN = 0.1
    SLOPE = 0.0
    B_Ts = np.array([270.15, 268.15])
    LWin = None
    N = 0.5

    test_args = {
        "dt": dt,
        "z": z,
        "z0": z0,
        "T2": T2,
        "rH2": rH2,
        "p": p,
        "SWnet": SWnet,
        "u2": u2,
        "RAIN": RAIN,
        "SLOPE": SLOPE,
        "B_Ts": B_Ts,
        "LWin": LWin,
        "N": N,
    }

    def assert_minimisation_ranges(
        self, surface_temperature, lw_in, lw_out, shf, lhf, ghf
    ):
        assert Constants.zero_temperature >= surface_temperature >= 220.0
        assert 400 >= lw_in >= 0
        assert 0 >= lw_out >= -400
        assert 250 >= shf >= -250
        assert 200 >= lhf >= -200
        assert 100 >= ghf >= -100

        return True

    @pytest.mark.parametrize("arg_optim", ["L-BFGS-B", "SLSQP"])
    def test_minimize_surface_energy_balance(
        self, monkeypatch, conftest_mock_grid, conftest_boilerplate, arg_optim
    ):
        conftest_boilerplate.patch_variable(
            monkeypatch,
            Constants,
            {"sfc_temperature_method": arg_optim},
        )
        conftest_boilerplate.patch_variable(
            monkeypatch,
            module_surface_temperature.Constants,
            {"sfc_temperature_method": arg_optim},
        )

        test_grid = conftest_mock_grid
        bounds = (220.0, 330.0)
        (
            residual_function,
            residual,
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
        ) = module_surface_temperature.update_surface_temperature(
            GRID=test_grid,
            dt=self.dt,
            z=self.z,
            z0=self.z0,
            T2=self.T2,
            rH2=self.rH2,
            p=self.p,
            SWnet=self.SWnet,
            u2=self.u2,
            RAIN=self.RAIN,
            SLOPE=self.SLOPE,
            LWin=self.LWin,
            N=self.N,
        )

        assert isinstance(residual, np.ndarray)
        assert bounds[0] < residual < bounds[1]

        self.assert_minimisation_ranges(
            residual,
            lw_radiation_in,
            lw_radiation_out,
            sensible_heat_flux,
            latent_heat_flux,
            ground_heat_flux,
        )

    @pytest.mark.parametrize("arg_optim", ["Newton", "Secant"])
    def test_minimize_newton(
        self, monkeypatch, conftest_mock_grid, conftest_boilerplate, arg_optim
    ):
        conftest_boilerplate.patch_variable(
            monkeypatch,
            module_surface_temperature.Constants,
            {"sfc_temperature_method": arg_optim},
        )
        test_grid = conftest_mock_grid
        bounds = (220.0, 330.0)

        (
            residual_function,
            residual,
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
        ) = module_surface_temperature.update_surface_temperature(
            GRID=test_grid,
            dt=self.dt,
            z=self.z,
            z0=self.z0,
            T2=self.T2,
            rH2=self.rH2,
            p=self.p,
            SWnet=self.SWnet,
            u2=self.u2,
            RAIN=self.RAIN,
            SLOPE=self.SLOPE,
            LWin=self.LWin,
            N=self.N,
        )

        assert isinstance(residual, np.ndarray)
        assert bounds[0] < residual < bounds[1]
        self.assert_minimisation_ranges(
            residual,
            lw_radiation_in,
            lw_radiation_out,
            sensible_heat_flux,
            latent_heat_flux,
            ground_heat_flux,
        )

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
        ) = module_surface_temperature.update_surface_temperature(
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
