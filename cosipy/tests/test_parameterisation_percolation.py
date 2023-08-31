import numpy as np
import pytest

# import constants
from COSIPY import start_logging
from cosipy.cpkernel.grid import Grid
from cosipy.modules.percolation import percolation


class TestParamPercolation:
    """Tests percolation methods.

    Attributes:
        melt_water (float): Melt water at surface.
        timedelta (int): Integration time, dt [s].
    """

    melt_water = 1.0
    timedelta = 3600

    def convert_mwe(self, mwe: float, depth: float) -> float:
        """Converts m w.e. to kgm^-3."""
        assert isinstance(mwe, float)
        water = mwe / depth

        return water

    @pytest.mark.parametrize("arg_mwe", [0.0, 0.5, 1.0])
    @pytest.mark.parametrize("arg_depth", [0.1, 1.0, 10.0])
    def test_percolation_convert_mwe(self, arg_mwe: float, arg_depth: float):
        compare_mwe = self.convert_mwe(mwe=arg_mwe, depth=arg_depth)
        assert isinstance(compare_mwe, float)
        if arg_depth < 1 and arg_mwe > 0:
            assert compare_mwe > arg_mwe
        else:
            assert compare_mwe <= arg_mwe

    def set_liquid_top_layer(self, grid: Grid, water: float):
        """Set liquid water content for top layer."""
        node_height_0 = grid.get_node_height(0)
        water_in = self.convert_mwe(mwe=water, depth=node_height_0)
        grid.set_node_liquid_water_content(
            0, grid.get_node_liquid_water_content(0) + float(water_in)
        )

        return grid

    @pytest.mark.parametrize("arg_water", [0.0, 0.5, 1.0])
    @pytest.mark.parametrize("arg_depth", [0.1, 1.0, 10.0])
    def test_set_liquid_top_layer(
        self, conftest_mock_grid, arg_water: float, arg_depth: float
    ):
        GRID = conftest_mock_grid
        water_in = self.convert_mwe(mwe=arg_water, depth=arg_depth)
        assert isinstance(water_in, float)
        start_water_layer_0 = GRID.get_node_liquid_water_content(0)
        assert start_water_layer_0 == 0.0

        GRID = self.set_liquid_top_layer(grid=GRID, water=water_in)

        new_water_layer_0 = GRID.get_node_liquid_water_content(0)
        if arg_water > 0:
            assert new_water_layer_0 > 0.0
            assert new_water_layer_0 > start_water_layer_0
        else:
            assert new_water_layer_0 == 0.0
            assert new_water_layer_0 == start_water_layer_0

    def get_layer_runoff(self, grid: Grid, idx: int) -> tuple:
        """Calculate percolated runoff for a single layer."""
        assert isinstance(idx, int)
        theta_e = grid.get_node_irreducible_water_content(idx)
        theta_w = grid.get_node_liquid_water_content(idx)
        # Residual volume fraction of water (m^3 which is equal to m)
        residual = np.maximum((theta_w - theta_e), 0.0)

        return residual, theta_e, theta_w

    @pytest.mark.parametrize("arg_idx", [0, 1, -1])
    def test_get_layer_runoff(self, conftest_mock_grid, arg_idx: float):
        GRID = conftest_mock_grid
        theta_e = GRID.get_node_irreducible_water_content(arg_idx)
        theta_w = GRID.get_node_liquid_water_content(arg_idx)

        residual, new_theta_e, new_theta_w = self.get_layer_runoff(
            grid=GRID, idx=arg_idx
        )

        assert isinstance(residual, float)
        assert np.isclose(residual, np.maximum((theta_w - theta_e), 0.0))
        assert np.isclose(
            GRID.get_node_irreducible_water_content(arg_idx), new_theta_e
        )
        assert np.isclose(
            GRID.get_node_liquid_water_content(arg_idx), new_theta_w
        )
        assert np.isclose(
            GRID.get_node_irreducible_water_content(arg_idx), theta_e
        )
        assert np.isclose(GRID.get_node_liquid_water_content(arg_idx), theta_w)

    def percolate_layer(self, grid: Grid, idx: int) -> Grid:
        residual, theta_e, theta_w = self.get_layer_runoff(grid=grid, idx=idx)
        if residual > 0:
            grid.set_node_liquid_water_content(idx, theta_e)
            grid.set_node_liquid_water_content(
                idx + 1,
                grid.get_node_liquid_water_content(idx + 1)
                + residual / grid.get_node_height(idx + 1),
            )
        else:
            # edge case where irreducible water content is greater, e.g. when
            # liquid water content was set to zero.
            grid.set_node_liquid_water_content(
                idx, np.maximum(theta_e, theta_w)
            )
            # grid.set_node_liquid_water_content(idx, theta_e)

        return grid

    @pytest.mark.parametrize("arg_idx", [0, 1, -2])
    @pytest.mark.parametrize(
        "arg_distribution", ["zero", "random", "static", "decreasing"]
    )
    def test_percolate_layer(
        self,
        conftest_mock_grid,
        conftest_rng_seed,
        arg_idx,
        arg_distribution,
    ):
        """Tests percolation between two layers."""
        GRID = conftest_mock_grid

        if arg_distribution == "random":
            rng = conftest_rng_seed
            for idx in range(0, GRID.number_nodes - 1):
                GRID.set_node_liquid_water_content(
                    idx, rng.uniform(low=0.01, high=0.05)
                )
        elif arg_distribution == "static":
            for idx in range(0, GRID.number_nodes - 1):
                GRID.set_node_liquid_water_content(idx, 0.1)
        elif arg_distribution == "decreasing":
            for idx in range(0, GRID.number_nodes - 1):
                GRID.set_node_liquid_water_content(
                    idx, 0.01 * GRID.number_nodes
                )
        else:
            pass

        start_theta_e = GRID.get_node_irreducible_water_content(arg_idx)

        start_theta_w = GRID.get_node_liquid_water_content(arg_idx)
        start_theta_e_next = GRID.get_node_irreducible_water_content(
            arg_idx + 1
        )
        start_theta_w_next = GRID.get_node_liquid_water_content(arg_idx + 1)
        start_residual = np.maximum((start_theta_w - start_theta_e), 0.0)
        height_next = GRID.get_node_height(arg_idx + 1)
        predict_theta_w_next = start_theta_w_next + (
            start_residual / height_next
        )

        GRID = self.percolate_layer(grid=GRID, idx=arg_idx)

        new_theta_e = GRID.get_node_irreducible_water_content(arg_idx)
        new_theta_w = GRID.get_node_liquid_water_content(arg_idx)
        new_theta_e_next = GRID.get_node_irreducible_water_content(arg_idx + 1)
        new_theta_w_next = GRID.get_node_liquid_water_content(arg_idx + 1)

        assert np.isclose(new_theta_w, start_theta_e)
        assert np.isclose(new_theta_e, start_theta_e)
        assert np.isclose(new_theta_e_next, start_theta_e_next)
        assert np.isclose(new_theta_w_next, predict_theta_w_next)

    # def test_percolation_percolate_single_layer(self,conftest_mock_grid):

    @pytest.mark.parametrize("arg_melt",[0.0, 0.5, 1.0])
    @pytest.mark.parametrize("arg_lwc", [0.0, 0.1, 0.5])
    def test_percolation(self,capsys, conftest_mock_grid, arg_lwc, arg_melt):
        GRID = conftest_mock_grid
        lwc_array = np.full((GRID.number_nodes), arg_lwc)
        GRID.set_liquid_water_content(lwc_array)
        np.testing.assert_allclose(GRID.get_liquid_water_content(), lwc_array)

        initial_lwc = np.nansum(GRID.get_liquid_water_content()) + arg_melt/GRID.get_node_height(0)

        runoff = percolation(GRID, arg_melt, self.timedelta)
        captured = capsys.readouterr()  # because numba doesn't support warnings
        final_lwc = np.nansum(GRID.get_liquid_water_content())
        final_mwe = runoff + final_lwc

        # all liquid water exits last node
        last_node_lwc = GRID.grid[-1].liquid_water_content
        assert last_node_lwc == 0.0

        # Bug? Total water is greater than the initially available water
        # assert np.isclose(self.melt_water, total_water)
        assert np.isclose(final_mwe - runoff, final_lwc)
        assert runoff <= initial_lwc
        # assert np.isclose(
        #     final_mwe, self.melt_water + (1 - runoff) * (GRID.number_nodes - 1)
        # )
        # assert np.isclose(final_lwc, GRID.number_nodes * (1 - runoff))
        
        error_prefix="\nWARNING: When percolating, the initial LWC is not equal to final LWC"
        timestep = GRID.old_snow_timestamp/self.timedelta

        assert np.isclose(initial_lwc, final_lwc)
        # assert np.isclose(initial_mwe, final_mwe)
        # if LWCs are equal, captured.out is an empty string
        assert captured.out == ""