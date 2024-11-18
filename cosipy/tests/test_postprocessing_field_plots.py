"""Tests for postprocessing of field plots.

Call ``plt.close("all")`` at the end of any test that creates a
matplotlib object.
"""

import argparse
from unittest.mock import patch

import cartopy
import cartopy.crs as ccrs
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import cosipy.postprocessing.field_plots.plot_cosipy_fields as pcf

# import constants
# from COSIPY import start_logging


class TestPostprocessPlotFieldsHandling:
    """Tests data handling before passing to plots."""

    @pytest.mark.parametrize("arg_mean", [True, False])
    def test_get_selection(self, conftest_mock_xr_dataset, arg_mean):
        dataset = conftest_mock_xr_dataset.copy()
        if arg_mean:
            timestamp = "2009-01-01"  # see test_get_selection_error
        else:
            timestamp = "2009-01-01T12:00:00"

        compare_array = pcf.get_selection(
            array=dataset["TS"], timestamp=timestamp, mean=arg_mean
        )

        assert isinstance(compare_array, xr.DataArray)
        if not arg_mean:
            assert compare_array.time == dataset.time[0]
        else:
            assert "time" not in compare_array.dims

    def test_get_selection_error(self, conftest_mock_xr_dataset):
        dataset = conftest_mock_xr_dataset.copy()
        timestamp = "2009-01-01"
        error_message = (
            "Selected time index refers to an entire day,",
            "not a single timestep.",
            "Use `--mean` to plot the daily mean.",
        )
        with pytest.raises(KeyError, match=" ".join(error_message)):
            pcf.get_selection(
                array=dataset["TS"], timestamp=timestamp, mean=False
            )

    def test_check_2d(self):
        dims = {}
        reference_time = pd.Timestamp("2009-01-01T12:00:00")
        dims["name"] = ["time", "lat", "lon"]
        dims["time"] = pd.date_range(reference_time, periods=4, freq="6h")
        elevation = xr.Variable(
            data=1000 + 10 * np.random.rand(2, 1),
            dims=dims["name"][1:],
            attrs={"long_name": "Elevation", "units": "m"},
        )
        dataset = xr.Dataset(
            data_vars=dict(HGT=elevation),
            coords=dict(
                time=dims["time"],
                lat=(["lat"], [30.460, 30.463]),
                lon=(["lon"], [90.621]),
                reference_time=dims["time"][0],
            ),
            attrs=dict(
                description="Weather related data.",
                Full_fiels="True",  # match typo in io.py
            ),
        )
        error_message = "Spatial coordinates are not 2D."
        with pytest.raises(ValueError, match=error_message):
            pcf.check_2d(dataset)


class TestPostprocessPlotFieldsPlotting:
    """Tests data plotting."""

    @pytest.mark.parametrize("arg_x", [1.00, 1.54, 1.21, 1.01])
    @pytest.mark.parametrize("arg_tex", [True, False])
    def test_set_contour_format(self, arg_x: float, arg_tex: bool):
        test_label = f"{arg_x:.1f}"
        if test_label.endswith("0"):
            test_label = f"{arg_x:.0f}"

        matplotlib.rcParams["text.usetex"] = arg_tex
        if not arg_tex:
            assert not matplotlib.rcParams["text.usetex"]
        else:
            assert matplotlib.rcParams["text.usetex"]

        compare_label = pcf.set_contour_format(x=arg_x)
        assert f"{test_label} m" == compare_label

        matplotlib.rcParams.update(matplotlib.rcParamsDefault)
        plt.close("all")

    def test_set_gridlines(self):
        ax = plt.subplot(projection=ccrs.PlateCarree())

        compare_ax = pcf.set_gridlines(ax=ax)
        assert isinstance(compare_ax, cartopy.mpl.geoaxes.GeoAxes)
        gridliners = compare_ax.gridlines()
        print(gridliners)
        artists = gridliners.xline_artists + gridliners.yline_artists
        for gridlines in artists:
            kwargs = gridlines.collection_kwargs
            assert kwargs.get("linewidth", None) == 0.5
            assert kwargs.get("color", None) == "gray"
            assert kwargs.get("alpha", None) == 0.1
            assert kwargs.get("linestyle", None) == "--"

        plt.close("all")

    def test_plot_topography(self, conftest_mock_xr_dataset):
        dataset = conftest_mock_xr_dataset.copy()
        ax = plt.subplot(projection=ccrs.PlateCarree())

        compare_ax = pcf.plot_topography(ax=ax, elevation=dataset.HGT)
        assert isinstance(compare_ax, plt.Axes)

        compare_interval = ax.xaxis.properties()["data_interval"]
        assert np.isclose(compare_interval[0], dataset.lon.min())
        assert np.isclose(compare_interval[1], dataset.lon.max())

        compare_interval = ax.yaxis.properties()["data_interval"]
        assert np.isclose(compare_interval[0], dataset.lat.min())
        assert np.isclose(compare_interval[1], dataset.lat.max())

        plt.close("all")

    @pytest.mark.parametrize("arg_name", ["TS", "HGT"])
    @pytest.mark.parametrize("arg_topography", [True, False])
    @pytest.mark.parametrize("arg_plot_type", ["contour", "mesh"])
    @pytest.mark.parametrize("arg_mean", [True, False])
    def test_plot_axes(
        self,
        conftest_mock_xr_dataset,
        conftest_boilerplate,
        arg_name: str,
        arg_topography: bool,
        arg_plot_type: str,
        arg_mean: bool,
    ):
        dataset = conftest_mock_xr_dataset.copy()
        test_array = dataset[arg_name]
        test_ax = plt.subplot(projection=ccrs.PlateCarree())
        if arg_topography:
            topography = dataset.HGT
        else:
            topography = None
        timestamp = conftest_boilerplate.set_timestamp(day=arg_mean)

        compare_ax = pcf.plot_axes(
            ax=test_ax,
            array=test_array,
            timestamp=timestamp,
            topography=topography,
            plot_type=arg_plot_type,
            mean=arg_mean,
        )

        assert isinstance(compare_ax, plt.Axes)
        assert compare_ax.gridlines
        assert compare_ax.get_title() == dataset[arg_name].long_name.title()

        plt.close("all")

    def test_plot_axes_error(self, conftest_mock_xr_dataset):
        dataset = conftest_mock_xr_dataset.copy()
        test_ax = plt.subplot(projection=ccrs.PlateCarree())
        test_plot_type = "missing_type"

        error_message = (
            f"{test_plot_type} is an invalid plot type.",
            f"Use 'mesh' or 'contour'.",
        )
        with pytest.raises(ValueError, match=" ".join(error_message)):
            pcf.plot_axes(
                ax=test_ax,
                array=dataset.TS,
                timestamp="2009-01-01T12:00:00",
                topography=None,
                plot_type=test_plot_type,
                mean=False,
            )

        plt.close("all")

    @pytest.mark.parametrize("arg_list", [[], None])
    def test_create_subplots_for_variables_error(self, arg_list):
        error_message = "Passed empty variable array to adaptive subplots."
        with pytest.raises(ValueError, match=error_message):
            pcf.create_subplots_for_variables(variables=arg_list)

        plt.close("all")

    def test_create_subplots_for_variables(self):
        test_variables = [["a", "b", "c"], ["d", "e", "f"]]

        compare_figure, compare_axes = pcf.create_subplots_for_variables(
            variables=test_variables
        )
        assert isinstance(compare_figure, plt.Figure)
        assert isinstance(compare_axes, np.ndarray)
        assert compare_axes.shape == (2, 3)  # match test_variables
        for rows in compare_axes:
            assert isinstance(rows, np.ndarray)
            assert rows.shape == (len(test_variables[0]),)
            for subplot in rows:
                assert isinstance(subplot, cartopy.mpl.geoaxes.GeoAxesSubplot)
                assert subplot.projection == ccrs.PlateCarree()

        plt.close("all")

    @pytest.mark.parametrize("arg_mean", [True, False])
    @pytest.mark.parametrize("arg_plot_type", ["contour", "mesh"])
    def test_plot_data_short_name(
        self,
        conftest_mock_xr_dataset,
        conftest_boilerplate,
        arg_mean,
        arg_plot_type,
    ):
        dataset = conftest_mock_xr_dataset.copy(deep=True)
        timestamp = conftest_boilerplate.set_timestamp(day=arg_mean)
        compare_figure, compare_axes = pcf.plot_data(
            data=dataset,
            timestamp=timestamp,
            short_name="TS",
            mean=arg_mean,
            variables=None,
            plot_type=arg_plot_type,
        )
        compare_params = {
            "plot": (compare_figure, compare_axes),
            "title": dataset["TS"].long_name.title(),
        }

        conftest_boilerplate.check_plot(plot_params=compare_params)

        plt.close("all")

    @pytest.mark.parametrize("arg_mean", [True, False])
    @pytest.mark.parametrize("arg_plot_type", ["contour", "mesh"])
    def test_plot_data_variables(
        self,
        conftest_mock_xr_dataset,
        conftest_boilerplate,
        arg_mean,
        arg_plot_type,
    ):
        dataset = conftest_mock_xr_dataset.copy(deep=True)
        timestamp = conftest_boilerplate.set_timestamp(day=arg_mean)
        test_variables = [
            ["TS", "HGT"],
            ["TS", "HGT"],
        ]

        compare_figure, compare_axes = pcf.plot_data(
            data=dataset,
            timestamp=timestamp,
            short_name=None,
            mean=arg_mean,
            variables=test_variables,
            plot_type=arg_plot_type,
        )
        assert isinstance(compare_figure, plt.Figure)
        assert isinstance(compare_axes, np.ndarray)
        assert compare_axes.shape == (2, 2)
        compare_variables = []
        for rows in compare_axes:
            assert isinstance(rows, np.ndarray)
            for subplot in rows:
                assert isinstance(subplot, cartopy.mpl.geoaxes.GeoAxes)
                compare_variables.append(subplot.get_title())

        for title in compare_variables:
            assert isinstance(title, str)
            assert title in ["Surface Temperature", "Elevation"]

        plt.close("all")

    @pytest.mark.parametrize("arg_mean", [True, False])
    def test_plotMesh(
        self,
        conftest_mock_open_dataset,
        conftest_hide_plot,
        conftest_boilerplate,
        arg_mean,
    ):
        timestamp = conftest_boilerplate.set_timestamp(day=arg_mean)
        _ = conftest_mock_open_dataset  # use mock data instead of file
        _ = conftest_hide_plot  # avoid user interaction

        pcf.plotMesh(
            filename="./path/file",
            pdate=timestamp,
            var="TS",
            mean=arg_mean,
            save=False,
        )

        compare_figure = plt.gcf()
        compare_axes = plt.gca()
        compare_params = {
            "plot": (compare_figure, compare_axes),
            "title": "Surface Temperature",
        }
        conftest_boilerplate.check_plot(plot_params=compare_params)

        plt.close("all")

    @pytest.mark.parametrize("arg_mean", [True, False])
    def test_plotContour(
        self,
        conftest_mock_open_dataset,
        conftest_hide_plot,
        conftest_boilerplate,
        arg_mean,
    ):
        timestamp = conftest_boilerplate.set_timestamp(day=arg_mean)
        _ = conftest_mock_open_dataset  # use mock data instead of file
        _ = conftest_hide_plot  # avoid user interaction

        pcf.plotContour(
            filename="./path/file",
            pdate=timestamp,
            var="TS",
            mean=arg_mean,
            save=False,
        )

        compare_figure = plt.gcf()
        compare_axes = plt.gca()
        compare_params = {
            "plot": (compare_figure, compare_axes),
            "title": "Surface Temperature",
        }
        conftest_boilerplate.check_plot(plot_params=compare_params)

        plt.close("all")

    def test_parse_arguments(self):
        with patch(
            "sys.argv",
            ["main", "--input", "./path/file", "--date", "'2009-01-01'"],
        ):
            args = pcf.parse_arguments()
        assert isinstance(args, argparse.Namespace)

        # Required
        assert args.file == "./path/file"
        assert args.pdate == "'2009-01-01'"
        # Defaults
        assert args.variable is None
        assert isinstance(args.plot_type, int) and args.plot_type == 1
        assert isinstance(args.mean, bool) and not args.mean
        assert isinstance(args.save, bool) and not args.save
