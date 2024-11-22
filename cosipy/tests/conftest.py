"""Provides shared fixtures and methods for tests.

Use these to replace duplicated code.

For generating objects within a test function's scope, call a fixture
directly:

    .. code-block:: python
        
        def test_foobar(self, conftest_mock_grid):
            grid_object = conftest_mock_grid
            grid_object.set_foo(foo=bar)
            ...
"""

from types import ModuleType
from typing import Any
from unittest.mock import patch

import matplotlib.pyplot as plt
import numba
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from cosipy.cpkernel.grid import Grid


# Function patches
@pytest.fixture(scope="function", autouse=False)
def conftest_mock_check_file_exists():
    """Override checks when mocking files."""

    patcher = patch("os.path.exists")
    mock_exists = patcher.start()
    mock_exists.return_value = True


@pytest.fixture(scope="function", autouse=False)
def conftest_disable_jit():
    # numba.config.DISABLE_JIT = True
    raise NotImplementedError(
        "Disabling JIT for tests is not yet implemented."
    )


@pytest.fixture(scope="function", autouse=False)
def conftest_mock_open_dataset(conftest_mock_xr_dataset):
    """Override xr.open_dataset with mock dataset."""

    patcher = patch("xarray.open_dataset")
    mock_exists = patcher.start()
    dataset = conftest_mock_xr_dataset.copy(deep=True)
    mock_exists.return_value = dataset


@pytest.fixture(scope="function", autouse=False)
def conftest_hide_plot():
    """Suppress plt.show(). Does not close plots."""

    patcher = patch("matplotlib.pyplot.show")
    mock_exists = patcher.start()
    mock_exists.return_value = True


@pytest.fixture(name="conftest_rng_seed", scope="function", autouse=False)
def fixture_conftest_rng_seed() -> np.random.Generator:
    """Set seed for random number generator to 444.

    Returns:
        Random number generator with seed=444.
    """

    random_generator = np.random.default_rng(seed=444)
    assert isinstance(random_generator, np.random.Generator)

    yield random_generator


# Mock GRID data
@pytest.fixture(
    name="conftest_mock_grid_values", scope="function", autouse=False
)
def fixture_conftest_mock_grid_values():
    """Constructs the layer values used to generate Grid objects.

    Returns:
        Generator[dict] Numba arrays for layers' heights, snowpack
        densities, temperatures, and liquid water content.
    """

    layer_values = {}
    layer_values["layer_heights"] = numba.float64([0.05, 0.1, 0.3, 0.5, 0.5])
    layer_values["layer_densities"] = numba.float64([250, 250, 250, 917, 917])
    layer_values["layer_temperatures"] = numba.float64(
        [260, 270, 271, 271.5, 272]
    )
    layer_values["layer_liquid_water_content"] = numba.float64(
        [0.0, 0.0, 0.0, 0.0, 0.0]
    )

    assert isinstance(layer_values, dict)
    for array in layer_values.values():
        assert isinstance(array, np.ndarray)
        assert len(array) == 5

    yield layer_values


@pytest.fixture(name="conftest_mock_grid", scope="function", autouse=False)
def fixture_conftest_mock_grid(conftest_mock_grid_values: dict):
    """Constructs a Grid object.

    .. note:: Use with caution, as this fixture assumes Grid objects are
        correctly instantiated.

    Returns:
        Generator[Grid]: Grid object with numba arrays for the layers'
        heights, densities, temperatures, and liquid water content.
    """

    data = conftest_mock_grid_values.copy()
    grid_object = Grid(
        layer_heights=data["layer_heights"],
        layer_densities=data["layer_densities"],
        layer_temperatures=data["layer_temperatures"],
        layer_liquid_water_content=data["layer_liquid_water_content"],
    )
    assert isinstance(grid_object, Grid)
    assert grid_object.number_nodes == len(data["layer_heights"])

    yield grid_object


@pytest.fixture(name="conftest_mock_grid_ice", scope="function", autouse=False)
def fixture_conftest_mock_grid_ice(conftest_mock_grid_values: dict):
    """Constructs a Grid object for ice layers.

    .. note:: Use with caution, as this fixture assumes Grid objects are
        correctly instantiated.

    Returns:
        Generator[Grid]: Grid object for ice layers with numba arrays
        for layer heights, layer densities, layer temperatures, and
        layer liquid water content.
    """

    data = conftest_mock_grid_values.copy()
    grid_object = Grid(
        layer_heights=data["layer_heights"][3:4],
        layer_densities=data["layer_densities"][3:4],
        layer_temperatures=data["layer_temperatures"][3:4],
        layer_liquid_water_content=data["layer_liquid_water_content"][3:4],
    )
    assert isinstance(grid_object, Grid)

    yield grid_object


"""Mock xarray Dataset"""
@pytest.fixture(
    name="conftest_mock_xr_dataset_dims", scope="function", autouse=False
)
def fixture_conftest_mock_xr_dataset_dims():
    """Yields dimensions for constructing an xr.Dataset.
    
    Returns:
        Generator[dict]: Spatiotemporal dimensions.
    """

    dimensions = {}
    reference_time = pd.Timestamp("2009-01-01T12:00:00")
    dimensions["time"] = pd.date_range(reference_time, periods=4, freq="6h")
    dimensions["latitude"] = [30.460, 30.463, 30.469, 30.472]
    dimensions["longitude"] = [90.621, 90.624, 90.627, 90.630, 90.633]
    dimensions["name"] = ["time", "lat", "lon"]

    yield dimensions


@pytest.fixture(
    name="conftest_mock_xr_dataset", scope="function", autouse=False
)
def fixture_conftest_mock_xr_dataset(
    conftest_mock_xr_dataset_dims: dict, conftest_rng_seed: np.random.Generator
):
    """Constructs mock xarray Dataset of output .nc file.

    Returns:
        Generator[xr.Dataset]: Dataset with data for elevation and
        surface data.
    """

    _ = conftest_rng_seed
    dims = conftest_mock_xr_dataset_dims.copy()
    lengths = [
        len(dims["time"]),
        len(dims["latitude"]),
        len(dims["longitude"]),
    ]
    elevation = xr.Variable(
        data=1000 + 10 * np.random.rand(lengths[1], lengths[2]),
        dims=dims["name"][1:],
        attrs={"long_name": "Elevation", "units": "m"},
    )
    temperature = xr.Variable(
        data=15 + 8 * np.random.randn(lengths[0], lengths[1], lengths[2]),
        dims=dims["name"],
        attrs={"long_name": "Surface temperature", "units": "K"},
    )

    dataset = xr.Dataset(
        data_vars=dict(HGT=elevation, TS=temperature),
        coords=dict(
            time=dims["time"],
            lat=(["lat"], dims["latitude"]),
            lon=(["lon"], dims["longitude"]),
            reference_time=dims["time"][0],
        ),
        attrs=dict(
            description="Weather related data.",
            Full_fiels="True",  # match typo in io.py
        ),
    )
    assert isinstance(dataset, xr.Dataset)

    for key, length in zip(dims["name"], lengths):
        assert dataset[key].shape == (length,)

    assert "time" not in dataset.HGT.dims
    assert dataset.HGT.shape == (lengths[1], lengths[2])
    assert dataset.HGT.long_name == "Elevation"
    for key in ["TS"]:  # in case we add more variables
        assert "time" in dataset[key].dims
        assert dataset[key].shape == (lengths[0], lengths[1], lengths[2])
        assert dataset[key][0].shape == (lengths[1], lengths[2])

    assert dataset.Full_fiels

    yield dataset


class TestBoilerplate:
    """Provides boilerplate methods for serialising tests.

    The class is instantiated via the `conftest_boilerplate` fixture.
    The fixture is autoused, and can be called directly within a test::

    ..code-block:: python

        def test_foo(self, conftest_boilerplate):

            foobar = [...]
            conftest_boilerplate.bar(foobar)

    Methods are arranged with their appropriate test::

    .. code-block:: python

        def foo(self, ...):
            pass

        def test_foo(self ...):
            pass
    """

    def check_plot(self, plot_params: dict, title: str = ""):
        """Check properties of figure/axis pairs.

        Args:
            plot_params: Contains matplotlib objects with supported keys:
                "plot", "title", "x_label", "y_label".
            title: Expected matplotlib figure title.
        """

        assert "plot" in plot_params
        assert isinstance(plot_params["plot"][0], plt.Figure)
        assert isinstance(plot_params["plot"][1], plt.Axes)
        compare_ax = plot_params["plot"][1]
        if "title" in plot_params:
            compare_title = f"{plot_params['title']}{title}"
            assert compare_ax.get_title("center") == compare_title
        if "x_label" in plot_params:
            assert compare_ax.xaxis.get_label_text() == plot_params["x_label"]
        if "y_label" in plot_params:
            assert compare_ax.yaxis.get_label_text() == plot_params["y_label"]

    def test_check_plot(self):
        """Validate tests for plot attributes."""

        test_title = ", Inner Scope"
        plt.close("all")
        figure_strings = plt.figure()
        axis_strings = plt.gca()
        plt.title(f"Test Title{test_title}")
        plt.xlabel("Test x-label")
        plt.ylabel("Test y-label")

        figure_none = plt.figure()
        axis_none = plt.gca()

        test_params = {
            "strings": {
                "plot": (figure_strings, axis_strings),
                "title": "Test Title",
                "x_label": "Test x-label",
                "y_label": "Test y-label",
            },
            "none": {"plot": (figure_none, axis_none)},
        }

        for test_pair in test_params.values():
            self.check_plot(plot_params=test_pair, title=test_title)
        plt.close("all")

    def set_timestamp(self, day: bool) -> str:
        """Set timestamp string for a day or for a timestep."""

        if day:
            timestamp = "2009-01-01"
        else:
            timestamp = "2009-01-01T12:00:00"

        return timestamp

    def test_set_timestamp(self):
        for arg_day in [True, False]:
            compare_time = self.set_timestamp(day=arg_day)
            assert isinstance(compare_time, str)
            if arg_day:
                assert compare_time == "2009-01-01"
            else:
                assert compare_time == "2009-01-01T12:00:00"

    def set_rng_seed(self, seed: int = 444) -> np.random.Generator:
        """Set seed for random number generator to 444.

        Args:
            seed: Seed value for generator.

        Returns:
            Random number generator with seed=444.
        """

        random_generator = np.random.default_rng(seed=seed)

        return random_generator

    def test_set_rng_seed(self):
        rng_none = self.set_rng_seed()
        assert isinstance(rng_none, np.random.Generator)
        rng_none_state = rng_none.__getstate__()["state"]["state"]
        rng_123 = self.set_rng_seed(seed=123)
        assert isinstance(rng_123, np.random.Generator)
        rng_123_state = rng_123.__getstate__()["state"]["state"]
        rng_444 = self.set_rng_seed(seed=444)
        assert isinstance(rng_444, np.random.Generator)
        rng_444_state = rng_444.__getstate__()["state"]["state"]

        assert all(
            isinstance(state, int)
            for state in [rng_none_state, rng_123_state, rng_444_state]
        )
        assert rng_none_state == rng_444_state
        assert not rng_444_state == rng_123_state

    def regenerate_grid_values(
        self, grid: Grid, distribution: str = "zero"
    ) -> Grid:
        """Set liquid water content to match a distribution.

        Args:
            grid: Glacier data mesh.
            distribution: Data distribution. Default "zero". Supports::

            - random: Random uniform between 0.01 and 0.05.
            - static: All liquid water content set to 0.1.
            - decreasing: Follows `1 - (0.01 * node_index)`.
            - increasing: Follows `0.01 * node_index`.
            - zero: All liquid water content set to 0.

        Returns:
            Glacier data mesh with an updated distribution of liquid
            water content.
        """

        if distribution == "random":
            rng = self.set_rng_seed()
            for idx in range(0, grid.number_nodes - 1):
                grid.set_node_liquid_water_content(
                    idx, rng.uniform(low=0.01, high=0.05)
                )
        elif distribution == "static":
            for idx in range(0, grid.number_nodes - 1):
                grid.set_node_liquid_water_content(idx, 0.1)
        elif distribution == "decreasing":
            for idx in range(0, grid.number_nodes - 1):
                grid.set_node_liquid_water_content(idx, 1 - (0.01 * idx))
        elif distribution == "increasing":
            for idx in range(0, grid.number_nodes - 1):
                grid.set_node_liquid_water_content(idx, 0.01 * idx)
        elif distribution == "zero":
            for idx in range(0, grid.number_nodes - 1):
                grid.set_node_liquid_water_content(idx, 0)
        else:
            raise ValueError("Distribution not supported!")

        return grid

    def test_regenerate_grid_values(self, conftest_mock_grid):
        grid = conftest_mock_grid
        distribution_list = [
            "static",
            "decreasing",
            "increasing",
            "random",
            "zero",
        ]
        for distribution in distribution_list:
            grid = self.regenerate_grid_values(
                grid=grid, distribution=distribution
            )
            assert isinstance(grid, Grid)
            test_lwc = grid.get_liquid_water_content()
            assert all(isinstance(i, float) for i in test_lwc)

    def check_output(self, variable: Any, x_type: Any, x_value: Any) -> bool:
        """Check a variable matches an expected type and value.

        Args:
            variable: Variable to check.
            x_type: Expected variable type.
            x_value: Expected variable value.

        Returns:
            True when all assertions pass.
        """

        assert isinstance(variable, x_type)
        if np.issubdtype(type(variable), np.number):
            assert np.isclose(variable, x_value)
        else:
            assert variable == x_value

        return True

    def test_check_output(self):
        variable_list = [[1.0, float], ["test", str], [1, int], [True, bool]]

        for pair in variable_list:
            assert self.check_output(
                variable=pair[0], x_type=pair[1], x_value=pair[0]
            )
        test_array = [0.0, 0.5, 0.6]
        test_value = max(test_array)
        assert test_value == 0.6
        assert isinstance(test_value, float)
        assert self.check_output(
            variable=max(test_array), x_type=float, x_value=test_value
        )

    def assert_grid_profiles_equal(self, grid_01: Grid, grid_02: Grid) -> bool:
        """Assert two Grid instances have equal grid profiles.

        Args:
            grid_01: Reference grid instance.
            grid_02: Comparison grid instance.

        Returns:
            True when all assertions pass.
        """
        for grid in (grid_01, grid_02):
            assert isinstance(grid, Grid)

        assert grid_02 is not grid_01  # point to different instances
        assert grid_02.number_nodes == grid_01.number_nodes
        assert grid_02.get_number_layers() == grid_01.get_number_layers()
        assert (
            grid_02.get_number_snow_layers()
            == grid_01.get_number_snow_layers()
        )

        self.check_output(
            grid_02.get_total_height(), float, grid_01.get_total_height()
        )
        self.check_output(
            grid_02.get_total_snowheight(),
            float,
            grid_01.get_total_snowheight(),
        )
        self.check_output(
            sum(grid_02.get_ice_heights()),
            float,
            sum(grid_01.get_ice_heights()),
        )

        return True

    def test_assert_grid_profiles_equal(self, conftest_mock_grid_values):
        data = conftest_mock_grid_values.copy()

        grid_01 = Grid(
            layer_heights=data["layer_heights"],
            layer_densities=data["layer_densities"],
            layer_temperatures=data["layer_temperatures"],
            layer_liquid_water_content=data["layer_liquid_water_content"],
        )
        grid_02 = Grid(
            layer_heights=data["layer_heights"],
            layer_densities=data["layer_densities"],
            layer_temperatures=data["layer_temperatures"],
            layer_liquid_water_content=data["layer_liquid_water_content"],
        )
        self.assert_grid_profiles_equal(grid_01, grid_02)

    def patch_variable(
        self,
        monkeypatch: pytest.MonkeyPatch,
        module: ModuleType,
        new_params: dict,
    ):
        """Patch any variable in a module.

        Patch the module where the variable is used, not where it's
        defined. The patched variable only exists within the test
        function's scope, so test parametrisation is still supported.

        Example:
            To patch constants used by `cpkernel.node.Node`:

                .. code-block:: python

                    patches = {"dt": 7200, "air_density": 1.0}
                    conftest.boilerplate.patch_variable(
                        monkeypatch,
                        cosipy.cpkernel.node.constants,
                        patches,
                        )

        Args:
            monkeypatch: Monkeypatch instance.
            module: Target module for patching.
            new_params: Variable names as keys, desired patched values as values:

                .. code-block:: python

                    new_params = {"foo": 1, "bar": 2.0}
        """

        if not isinstance(new_params, dict):
            note = "Pass dict with variable names and patched values as items."
            raise TypeError(note)
        for key in new_params:
            monkeypatch.setattr(module, key, new_params[key])

    def calculate_irreducible_water_content(
        self, current_ice_fraction: float
    ) -> float:
        """Calculate irreducible water content."""
        if current_ice_fraction <= 0.23:
            theta_e = 0.0264 + 0.0099 * (
                (1 - current_ice_fraction) / current_ice_fraction
            )
        elif (current_ice_fraction > 0.23) & (current_ice_fraction <= 0.812):
            theta_e = 0.08 - 0.1023 * (current_ice_fraction - 0.03)
        else:
            theta_e = 0.0

        return theta_e

    def test_calculate_irreducible_water_content(self):
        ice_fractions = [0.2, 0.5, 0.9]
        for i in ice_fractions:
            theta_e = self.calculate_irreducible_water_content(i)
            assert isinstance(theta_e, float)

    @pytest.fixture(scope="class", autouse=False)
    def test_boilerplate_integration(self):
        """Integration test for boilerplate methods."""

        self.test_check_plot()
        self.test_set_timestamp()
        self.test_set_rng_seed()
        self.test_regenerate_grid_values()
        self.test_check_output()
        self.test_assert_grid_profiles_equal()
        self.test_calculate_irreducible_water_content()


@pytest.fixture(name="conftest_boilerplate", scope="function", autouse=False)
def conftest_boilerplate():
    """Yield class containing methods for common tests."""

    yield TestBoilerplate()
