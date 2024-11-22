"""
Usage: ``src/plot_cosipy_fields.py [-h] [-f <nc_file>] [-d] [...] [-v]``
"""

import argparse
import re

import cartopy.crs as ccrs
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# matplotlib.use("TkAgg")


def check_2d(array: xr.DataArray):
    """Checks if an input array has 2D spatial coordinates.

    Raises:
        ValueError: Spatial coordinates are not 2D.
    """

    for dimension in array.dims:
        if dimension not in ["time", "layer"] and (array.sizes[dimension]) <= 1:
            raise ValueError("Spatial coordinates are not 2D.")


def get_selection(
    array: xr.DataArray, timestamp: str, mean: bool = False
) -> xr.DataArray:
    """Selects data from array at specific time or as a daily mean.

    Args:
        array: Labelled data array.
        timestamp: Date or datetime index of target data.
        mean: If True, computes and selects the daily mean. Otherwise,
            selects data at ``timestamp``. Default False.

    Returns:
        Array selection at target time.

    Raises:
        KeyError: Selected time index refers to an entire day, not a
            single timestep. Use ``--mean`` to plot the daily mean.
    """

    if not mean:
        data = array.sel(time=timestamp)
        if data.time.size > 1:
            error_message = (
                "Selected time index refers to an entire day,",
                "not a single timestep.",
                "Use `--mean` to plot the daily mean.",
            )
            raise KeyError(" ".join(error_message))
    else:
        data = (
            array.resample(time="1D", skipna=True)
            .mean()
            .sel(time=timestamp, method="nearest")
        )

    return data


def set_contour_format(x: float) -> str:
    """Formats float into contour label."""

    if round(x, 1) % 1:
        elevation = f"{x:.1f}"
    else:
        elevation = f"{int(x)}"

    if plt.rcParams["text.usetex"]:
        label = rf"{elevation} m"
    else:
        label = f"{elevation} m"

    return label


def set_gridlines(ax: plt.Axes) -> plt.Axes:
    """Projects to PlateCarrée, adds gridlines, and formats labels."""

    gridlines = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.5,
        color="gray",
        alpha=0.1,
        linestyle="--",
    )
    gridlines.top_labels = None
    gridlines.right_labels = False

    return ax


def save_image(
    figure: plt.Figure,
    timestamp: str,
    name: str = "",
    suffix: str = "",
    img_format="png",
):
    """Automatically generates file name, and saves image to disk.

    Args:
        figure: Figure object that is saved to disk.
        timestamp: Date or datetime index of target data.
        name: Name of variable. Default empty string.
        suffix: Additional string appended to file name e.g. the type of
            plot. Default empty string.
        img_format: Output image format. Default "png".
    """

    if not isinstance(timestamp, str):
        img_id = timestamp.strftime("%Y%m%d")
    else:
        img_id = timestamp
    img_id = re.sub(r"\W+", "", str(img_id))  # avoid illegal file names

    if name is None:
        img_id = f"{img_id}_stamp_plot"
    elif name:
        img_id = f"{img_id}_{name}"

    if suffix:
        suffix = f"_{suffix}"

    figure.savefig(fname=f"{img_id}{suffix}.{img_format}")


def plot_topography(ax: plt.Axes, elevation: xr.DataArray) -> plt.Axes:
    """Plots elevation contours with labels onto axis."""

    topography = ax.contour(
        elevation.lon, elevation.lat, elevation, colors="gray", linewidths=0.5
    )
    ax.clabel(
        topography,
        topography.levels,
        inline=True,
        fmt=set_contour_format,
        fontsize=10,
        colors="black",
    )

    return ax


def plot_axes(
    ax: plt.Axes,
    array: xr.DataArray,
    timestamp: str,
    topography: xr.DataArray = None,
    plot_type: str = "contour",
    mean: bool = False,
) -> plt.Axes:
    """Plots data for a specific time or as a daily mean.

    Args:
        ax: Target axis.
        array: XYZ data for a single variable.
        timestamp: Date or datetime index of target data.
        topography: Optional elevation data to plot contours. Default
            None.
        plot_type: Plot data as "contour" or "mesh". Default "contour".
        mean: Plot daily mean instead of specific time. Default False.

    Returns:
        Labelled contour or mesh plot of variable, with optional
        contours for topography.

    Raises:
        ValueError: <plot_type> is an invalid plot type. Use 'mesh' or
            'contour'.
    """

    name = array.attrs.get("long_name", None)
    unit_label = array.attrs.get("units", None)
    c_map = matplotlib.colormaps["viridis"]  # coolwarm

    if name.lower() == "elevation":
        data = array  # HGT has no time dimensions
        c_map = matplotlib.colormaps["gist_earth"]
    else:
        data = get_selection(array=array, timestamp=timestamp, mean=mean)

    if topography is not None:
        plot_topography(ax=ax, elevation=topography)

    if plot_type.lower() == "contour":
        data.plot.contourf(
            x="lon", y="lat", ax=ax, cbar_kwargs={"label": unit_label}, cmap=c_map
        )
    elif plot_type.lower() == "mesh":
        data.plot.pcolormesh(
            x="lon", y="lat", ax=ax, cbar_kwargs={"label": unit_label}, cmap=c_map
        )
    else:
        raise ValueError(
            f"{plot_type} is an invalid plot type. Use 'mesh' or 'contour'."
        )
    ax = set_gridlines(ax=ax)
    ax.set_title(name.title())

    return ax


def create_subplots_for_variables(variables: list = None) -> tuple:
    """Creates adaptive subplots for an array of variables.

    Array is shaped as [nrows[ncols]].

    Returns:
        tuple[plt.Figure, np.ndarray]: Figure with subplots for each
        variable.

    Raises:
        ValueError: Passed empty variable array to adaptive subplots.
    """

    if not variables:
        raise ValueError("Passed empty variable array to adaptive subplots.")
    else:
        plot_array = np.array(variables)
        figure, axes = plt.subplots(
            plot_array.shape[0],
            plot_array.shape[1],
            sharex="col",
            sharey="row",
            figsize=(30, 18),
            subplot_kw=dict(projection=ccrs.PlateCarree()),
        )

    return figure, axes


def plot_data(
    data: xr.DataArray,
    timestamp: str,
    short_name: str = None,
    mean: bool = False,
    variables: list = None,
    plot_type: str = "contour",
) -> tuple:
    """Creates stamp plots or a single plot.

    Args:
        data: Labelled data array.
        timestamp: Datetime index of target data.
        short_name: Short name of variable to plot. If None, a stamp
            plot of several variables is produced. Default None.
        mean: Plot daily mean instead of specific time. Default False.
        variables: Array of variable names of shape [nrows[ncols]].
            Default None.
        plot_type: Plot data as "contour" or "mesh". Default "contour".
    """

    if short_name is None:  # stamp plots
        _, axes = create_subplots_for_variables(variables)
        for i in range(len(variables)):
            for j, key in enumerate(variables[i]):
                axes[i][j] = plot_axes(
                    ax=axes[i][j],
                    array=data[key],
                    timestamp=timestamp,
                    mean=mean,
                    topography=data.HGT,
                    plot_type=plot_type,
                )
    else:
        plt.figure(figsize=(18, 10))
        axes = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
        axes = plot_axes(
            ax=axes,
            array=data[short_name],
            timestamp=timestamp,
            mean=mean,
            topography=data.HGT,
            plot_type=plot_type,
        )
    figure = plt.gcf()

    return figure, axes


def plotMesh(
    filename: str,
    pdate: str,
    var: str = None,
    mean: bool = False,
    save: bool = False,
):
    """This creates a simple mesh plot of the 2D fields.

    Args:
        filename: Path to nc data array.
        pdate: Date or datetime index of target data.
        var: Short name of variable to plot. If None, a stamp plot of
            several variables is produced. Default None.
        mean: Plot daily mean instead of specific time. Default False.
        save: Save plot to disk. Default False.

    Raises:
        ValueError: Input data is not 2D.
    """

    data = xr.open_dataset(filename)
    check_2d(data)

    print(data)
    variables = [  # structure of stamp plot
        ["H", "LE", "B"],
        ["SNOWHEIGHT", "surfM", "TS"],
        ["G", "Q", "LWout"],
        ["surfMB", "MB", "REFREEZE"],
    ]

    figure, _ = plot_data(
        data=data,
        variables=variables,
        timestamp=pdate,
        short_name=var,
        mean=mean,
        plot_type="mesh",
    )
    if save:
        save_image(figure=figure, timestamp=pdate, name=var, suffix="mesh")
    else:
        plt.show()


def plotContour(
    filename: str,
    pdate: str,
    var: str = None,
    mean: bool = False,
    save: bool = False,
):
    """This creates a simple contour plot of the 2D fields.

    Args:
        filename: Path to nc data array.
        pdate: Date or datetime index of target data.
        var: Short name of variable to plot. If None, a stamp plot of
            several variables is produced. Default None.
        mean: Plot daily mean instead of specific time. Default False.
        save: Save plot to disk. Default False.

    Raises:
        ValueError: Input data is not 2D.
    """

    data = xr.open_dataset(filename)
    check_2d(data)

    print(data)
    variables = [  # structure of stamp plot
        ["H", "LE", "B"],
        ["SNOWHEIGHT", "surfM", "TS"],
        ["G", "LWin", "LWout"],
        ["MB", "surfMB", "Q"],
    ]

    figure, _ = plot_data(
        data=data,
        variables=variables,
        timestamp=pdate,
        short_name=var,
        mean=mean,
        plot_type="contour",
    )
    if save:
        save_image(figure=figure, timestamp=pdate, name=var, suffix="contour")
    else:
        plt.show()


def parse_arguments() -> argparse.Namespace:
    """Parse user arguments.

    Required arguments:
        -i, --input <path>       Path to .nc file.
        -d, --date <str>         Target date or timestamp.

    Optional switches:
        -h, --help               Show this help message and exit.
        -s, --save               Save plot. File name is automatically
                                     generated. Default False.
        -m, --mean               Plot daily mean instead of timestep.
                                     Default False.

    Optional arguments:
        -v, --var <str>          Variable to plot. If not set, creates a
                                     stamp plot.
        -t, --type <int>         Set plot type. 1: contour, 2: mesh.
                                     Default 1.
    """

    tagline = "Plot results for single or all variables."
    parser = argparse.ArgumentParser(prog="plot_cosipy_fields.py", description=tagline)
    # Required
    parser.add_argument(
        "-i",
        "--input",
        dest="file",
        required=True,
        default=None,
        type=str,
        metavar="<path>",
        help="Path to .nc file",
    )
    parser.add_argument(
        "-d",
        "--date",
        dest="pdate",
        type=str,
        required=True,
        default=None,
        metavar="<str>",
        help="Target date or timestamp",
    )
    # switches
    parser.add_argument(
        "-m",
        "--mean",
        dest="mean",
        action="store_true",
        help="Plot daily mean instead of timestep. Default False",
    )
    parser.add_argument(
        "-s",
        "--save",
        dest="save",
        action="store_true",
        help="Save plot. File name is automatically generated. Default False.",
    )
    # Optional
    parser.add_argument(
        "-v",
        "--var",
        dest="variable",
        type=str,
        default=None,
        metavar="<str>",
        help="Variable to plot. If not set, creates a stamp plot",
    )
    parser.add_argument(
        "-t",
        "--type",
        dest="plot_type",
        type=int,
        default=1,
        choices=[1, 2, None],
        metavar="<int>",
        help="Set plot type. 1: contour, 2: mesh. Default 1",
    )
    arguments = parser.parse_args()

    return arguments


def main():
    """Produce and save plots.

    Produces field plots of single timesteps or daily mean for a single
    or all variables. Passing a date without ``--mean`` leads to an
    error.

    Required arguments:
        -i, --input <path>      Path to .nc file.
        -d, --date <str>        Target date or timestamp.

    Optional switches:
        -h, --help              Show this help message and exit.
        -s, --save              Save plot. File name is automatically
                                    generated. Default False.
        -m, --mean              Plot daily mean instead of timestep.
                                    Default False.

    Optional arguments:
        -v, --var <str>         Variable to plot. If not set, creates a
                                    stamp plot.
        -t, --type <int>        Set plot type. 1: contour, 2: mesh.
                                    Default 1.
    """

    args = parse_arguments()
    if args.plot_type == 1:
        print("Contour")
        plotContour(args.file, args.pdate, args.variable, args.mean, args.save)
    else:
        print("Mesh")
        plotMesh(args.file, args.pdate, args.variable, args.mean, args.save)


if __name__ == "__main__":
    main()
