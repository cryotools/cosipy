import numpy as np


class TestParamHeatEquation:
    """Tests heat equation."""

    melt_water = 1.0
    timedelta = 3600

    def get_grid_spacing(self, layers, num_layers):
        spacing = np.divide(
            np.add(layers[0 : num_layers - 1], layers[1:num_layers]), 2.0
        )
        return spacing

    def test_get_grid_spacing(self, conftest_mock_grid):
        test_grid = conftest_mock_grid
        num_layers = test_grid.get_number_layers()
        test_layers = np.asarray(test_grid.get_height())

        # original function
        test_spacing = (test_layers[0 : num_layers - 1] / 2.0) + (
            test_layers[1:num_layers] / 2.0
        )
        compare_spacing = self.get_grid_spacing(
            layers=test_layers, num_layers=num_layers
        )
        assert isinstance(compare_spacing, np.ndarray)
        np.testing.assert_array_equal(test_spacing, compare_spacing)
        assert test_layers.shape == (num_layers,)

    def dt_stab_minimisation(
        self,
        hk: np.ndarray,
        hk1: np.ndarray,
        ku: np.ndarray,
        kl: np.ndarray,
        c_stab: float = 0.8,
    ) -> float:
        """Original minimisation function."""

        dt_stab = c_stab * (
            min([min(hk * hk / (2 * ku)), min(hk1 * hk1 / (2 * kl))])
        )

        return dt_stab

    def test_dt_stab_minimisation(self, conftest_boilerplate):
        shape = (30,)
        nl = shape[0]
        nl_1 = shape[0] - 1
        nl_2 = shape[0] - 2

        therm_diff = np.random.rand(*shape)
        therm_upper = (therm_diff[1:nl_1] + therm_diff[2:nl]) / 2.0
        therm_lower = (therm_diff[0:nl_2] + therm_diff[1:nl_1]) / 2.0

        heights = np.random.rand(*shape)
        spacing = self.get_grid_spacing(layers=heights, num_layers=shape[0])
        hk = spacing[0:nl_2]  # between z-1 and z
        hk1 = spacing[1:nl_1]  # between z and z+1

        c_stab = 0.8
        test_dt_stab = self.dt_stab_minimisation(
            hk=hk, hk1=hk1, ku=therm_upper, kl=therm_lower, c_stab=c_stab
        )
        assert isinstance(test_dt_stab, float)

        compare_dt_stab = c_stab * (
            min(min(hk**2 / therm_upper), min(hk1**2 / therm_lower)) / 2
        )
        conftest_boilerplate.check_output(compare_dt_stab, float, test_dt_stab)

    def get_t_new(
        self,
        temperatures,
        centre_pts,
        lower_pts,
        upper_pts,
        hk,
        hk1,
        Kl,
        Ku,
        stab_t=0.0,
        dt=3600,
    ):
        "Original iteration."
        c_stab = 0.8
        dt_stab = c_stab * (min(min(hk**2 / Ku), min(hk1**2 / lower_pts)) / 2)
        Tnew = temperatures.copy()
        while stab_t < dt:
            dt_use = min(dt_stab, dt - stab_t)
            stab_t = stab_t + dt_use
            Tnew[centre_pts] += (
                (
                    Kl
                    * dt_use
                    * (temperatures[lower_pts] - temperatures[centre_pts])
                    / (hk1)
                )
                - (
                    Ku
                    * dt_use
                    * (temperatures[centre_pts] - temperatures[upper_pts])
                    / (hk)
                )
            ) / (0.5 * (hk + hk1))
            temperatures = Tnew.copy()
        return temperatures

    def get_t_new_nocopy(
        self,
        temperatures,
        centre_pts,
        lower_pts,
        upper_pts,
        hk,
        hk1,
        Kl,
        Ku,
        stab_t=0.0,
        dt=3600,
    ):
        c_stab = 0.8
        dt_stab = c_stab * (min(min(hk**2 / Ku), min(hk1**2 / lower_pts)) / 2)
        while stab_t < dt:
            dt_use = min(dt_stab, dt - stab_t)
            stab_t = stab_t + dt_use
            temperatures[centre_pts] += (
                dt_use
                * 2
                * (
                    (
                        Kl
                        * (temperatures[lower_pts] - temperatures[centre_pts])
                        / hk1
                    )
                    - (
                        Ku
                        * (temperatures[centre_pts] - temperatures[upper_pts])
                        / hk
                    )
                )
                / (hk + hk1)
            )
        return temperatures

    def test_get_t_new(self):
        nl = 30
        shape = (nl,)
        centre_pts = np.arange(1, nl - 1)  # center points
        lower_pts = np.arange(2, nl)  # lower points
        upper_pts = np.arange(0, nl - 2)  # upper points

        temperatures = 273.15 - (np.random.rand(*shape) * 10)
        therm_diff = np.random.rand(*shape)
        therm_upper = (therm_diff[1 : nl - 1] + therm_diff[2:nl]) / 2.0
        therm_lower = (therm_diff[0 : nl - 2] + therm_diff[1 : nl - 1]) / 2.0

        heights = np.random.rand(*shape)
        spacing = self.get_grid_spacing(layers=heights, num_layers=shape[0])
        hk = spacing[0 : nl - 2]  # between z-1 and z
        hk1 = spacing[1 : nl - 1]  # between z and z+1
        dt = 200

        test_temperatures = temperatures.copy()
        test_temperatures = self.get_t_new(
            temperatures=test_temperatures,
            centre_pts=centre_pts,
            upper_pts=upper_pts,
            lower_pts=lower_pts,
            hk=hk,
            hk1=hk1,
            Kl=therm_lower,
            Ku=therm_upper,
            stab_t=0.0,
            dt=dt,
        )

        compare_temperatures = temperatures.copy()
        compare_temperatures = self.get_t_new_nocopy(
            temperatures=compare_temperatures,
            centre_pts=centre_pts,
            upper_pts=upper_pts,
            lower_pts=lower_pts,
            hk=hk,
            hk1=hk1,
            Kl=therm_lower,
            Ku=therm_upper,
            stab_t=0.0,
            dt=dt,
        )
        assert np.allclose(compare_temperatures, test_temperatures)
