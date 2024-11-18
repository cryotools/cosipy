"""
Reads the input data (model forcing) and writes the output to a netCDF
file. It supports point models with ``create_1D_input`` and distributed
simulations with ``create_2D_input``.

The 1D input function works without a static file, in which the static
variables are created.

Edit the configuration by supplying a valid .toml file - this includes
lapse rates for both cases. See the sample ``utilities_config.toml`` for
more information.

Usage:

From source:
``python -m cosipy.utilities.aws2cosipy.aws2cosipy -i <input> -o <output> -s <static> [-u <path>] [-b <date>] [-e <date>]``

Entry point:
``cosipy-aws2cosipy -i <input> -o <output> -s <static> [-u <path>] [-b <date>] [-e <date>]``

Options and arguments:

Required arguments:
    -i, --input <path>          Path to .csv file with meteorological data.
    -o, --output <path>         Path to the resulting COSIPY netCDF file.
    -s, --static_file <path>    Path to static file with DEM, slope etc.

Optional arguments:
    -u, --u <path>          Relative path to utilities' configuration
                                file.
    -b, --start_date <int>  Start date.
    -e, --end_date <int>    End date.
    --xl <float>            Left longitude value of the subset.
    --xr <float>            Right longitude value of the subset.
    --yl <float>            Lower latitude value of the subset.
    --yu <float>            Upper latitude value of the subset.
"""

import argparse
from itertools import product

import dateutil
import netCDF4 as nc
import numpy as np
import pandas as pd
import xarray as xr

import cosipy.modules.radCor as mod_radCor
from cosipy.utilities.config_utils import UtilitiesConfig

_args = None
_cfg = None


def read_input_file(input_path: str) -> tuple:
    """Read input data, parse dates, and convert to a dataframe.

    Args:
        input_path: Path to input .csv file.

    Returns:
         tuple: Dataframe of input data and date parser.
    """
    print(f"{'-' * 43}")
    print(f"Create input\nRead input file {input_path}")

    date_parser = lambda x: dateutil.parser.parse(x, ignoretz=True)
    dataframe = pd.read_csv(
        input_path,
        delimiter=_cfg.coords["delimiter"],
        index_col=["TIMESTAMP"],
        parse_dates=["TIMESTAMP"],
        na_values="NAN",
        date_parser=date_parser,
    )

    return dataframe


def convert_to_numeric(series: pd.Series) -> pd.Series:
    """Convert series to numeric type."""
    series = series.apply(pd.to_numeric, errors="coerce")
    return series


def set_order_and_type(
    dataframe: pd.DataFrame, replace_pressure: bool = False
) -> pd.DataFrame:
    """Set dataframe order and convert to numeric type.

    Args:
        dataframe: Contains input data.
        replace_pressure: Set pressure data to 660 hPa if no pressure
            data is available.

    Returns:
        Ordered dataframe.
    """
    for name in ["T2_var", "RH2_var", "U2_var", "G_var", "PRES_var"]:
        dataframe[_cfg.names[name]] = convert_to_numeric(
            series=dataframe[_cfg.names[name]]
        )

    if _cfg.names["RRR_var"] in dataframe:
        dataframe[_cfg.names["RRR_var"]] = convert_to_numeric(
            dataframe[_cfg.names["RRR_var"]]
        )

    if (replace_pressure) and (_cfg.names["PRES_var"] not in dataframe):
        dataframe[_cfg.names["PRES_var"]] = 660.00

    if (_cfg.names["LWin_var"] not in dataframe) and (
        _cfg.names["N_var"] not in dataframe
    ):
        raise ValueError(
            "No data for either incoming longwave radiation or cloud cover."
        )
    elif _cfg.names["LWin_var"] in dataframe:
        dataframe[_cfg.names["LWin_var"]] = convert_to_numeric(
            dataframe[_cfg.names["LWin_var"]]
        )
    elif _cfg.names["N_var"] in dataframe:
        dataframe[_cfg.names["N_var"]] = convert_to_numeric(
            dataframe[_cfg.names["N_var"]]
        )

    if _cfg.names["SNOWFALL_var"] in dataframe:
        dataframe[_cfg.names["SNOWFALL_var"]] = convert_to_numeric(
            dataframe[_cfg.names["SNOWFALL_var"]]
        )

    return dataframe


def check_data(dataset: xr.Dataset, dataframe: pd.DataFrame):
    """Check data is within physically reasonable bounds."""
    check(dataset.T2, 316.16, 223.16)
    check(dataset.RH2, 100.0, 0.0)
    check(dataset.U2, 50.0, 0.0)
    check(dataset.G, 1600.0, 0.0)
    check(dataset.PRES, 1080.0, 200.0)

    if _cfg.names["RRR_var"] in dataframe:
        check(dataset.RRR, 20.0, 0.0)
    if _cfg.names["SNOWFALL_var"] in dataframe:
        check(dataset.SNOWFALL, 0.05, 0.0)
    if _cfg.names["LWin_var"] in dataframe:
        check(dataset.LWin, 400, 0.0)
    if _cfg.names["N_var"] in dataframe:
        check(dataset.N, 1.0, 0.0)


def get_time_slice(dataframe, start_date, end_date):
    if (start_date is not None) & (end_date is not None):
        dataframe = dataframe.loc[start_date:end_date]
    return dataframe


def set_bias(
    data: np.ndarray,
    lapse_type: str,
    altitude: float = 0.0,
    limit: bool = True,
):
    """Apply lapse rate to data.

    Args:
        data: Numerical data.
        lapse_type: ID of lapse rate in config file.
        altitude: The height difference between a location and sensor.
        limit: If True, set negative values to zero. Default True.
    """

    biased_data = data + altitude * _cfg.lapse[lapse_type]
    if limit:
        biased_data = np.maximum(biased_data, 0.0)

    return biased_data


def set_variable_metadata() -> dict:
    """Initialise variable names and units."""
    metadata = {
        "HGT": ("m", "Elevation"),
        "ASPECT": ("degrees", "Aspect of slope"),
        "SLOPE": ("degrees", "Terrain slope"),
        "MASK": ("boolean", "Glacier mask"),
        "T2": ("K", "Temperature at 2 m"),
        "RH2": ("%", "Relative humidity at 2 m"),
        "U2": ("m s\u207b\xb9", "Wind velocity at 2 m"),
        "G": ("W m\u207b\xb2", "Incoming shortwave radiation"),
        "PRES": ("hPa", "Atmospheric Pressure"),
        "RRR": ("mm", "Total precipitation (liquid+solid)"),
        "SNOWFALL": ("m", "Snowfall"),
        "LWin": ("W m\u207b\xb2", "Incoming longwave radiation"),
        "N": ("%", "Cloud cover fraction"),
    }

    return metadata


def get_variable_metadata(name: str) -> dict:
    """Get metadata associated with a variable name."""
    metadata = set_variable_metadata()
    metadata_bundle = {
        "name": name,
        "units": metadata[name][0],
        "long_name": metadata[name][1],
    }

    return metadata_bundle


def set_dataset_coordinates(dataset, latitude, longitude):

    x, y = np.meshgrid(longitude, latitude)
    dataset.coords["lat"] = (("south_north", "west_east"), x)
    dataset.coords["lon"] = (("south_north", "west_east"), y)

    return dataset


def set_zero_field(
    time_index: int, lat_index: int, lon_index: int
) -> np.ndarray:
    """Initialise array and fill with zeros."""
    field = np.zeros([time_index, lat_index, lon_index])

    return field


def get_pressure_bias(data, height):
    slp = data / np.power(
        (1 - (0.0065 * _cfg.station["stationAlt"]) / (288.15)), 5.255
    )
    pressure = slp * np.power((1 - (0.0065 * height) / (288.15)), 5.22)

    return pressure


def write_netcdf(dataset, output_path):
    dataset.to_netcdf(output_path)
    print(f"{'-' * 43}\nInput file created: {output_path}\n{'-' * 43}")


def create_1D_input(cs_file, cosipy_file, static_file, start_date, end_date):
    """Create an input dataset from a csv file with input point data.

    Here you need to define how to interpolate the data.

    .. warning::
        There should be only one header line in the file.

    Previous updates:
    Tobias Sauter 07.07.2019
    Anselm 04.07.2020
    """

    df = read_input_file(input_path=cs_file)

    print(df[pd.isnull(df).any(axis=1)])
    df = df.fillna(method="ffill")
    print(df[pd.isnull(df).any(axis=1)])

    if _cfg.names["LWin_var"] not in df:
        df[_cfg.names["LWin_var"]] = np.nan
    if _cfg.names["N_var"] not in df:
        df[_cfg.names["N_var"]] = np.nan
    if _cfg.names["SNOWFALL_var"] not in df:
        df[_cfg.names["SNOWFALL_var"]] = np.nan

    col_list = [
        _cfg.names["T2_var"],
        _cfg.names["RH2_var"],
        _cfg.names["U2_var"],
        _cfg.names["G_var"],
        _cfg.names["RRR_var"],
        _cfg.names["PRES_var"],
        _cfg.names["LWin_var"],
        _cfg.names["N_var"],
        _cfg.names["SNOWFALL_var"],
    ]
    df = df[col_list]

    df = df.resample("1H").agg(
        {
            _cfg.names["T2_var"]: "mean",
            _cfg.names["RH2_var"]: "mean",
            _cfg.names["U2_var"]: "mean",
            _cfg.names["G_var"]: "mean",
            _cfg.names["PRES_var"]: "mean",
            _cfg.names["RRR_var"]: nansumwrapper,
            _cfg.names["LWin_var"]: "mean",
            _cfg.names["N_var"]: "mean",
            _cfg.names["SNOWFALL_var"]: nansumwrapper,
        }
    )
    df = df.dropna(axis=1, how="all")
    print(df.head())

    # Select time slice
    df = get_time_slice(dataframe=df, start_date=start_date, end_date=end_date)

    # Load static data
    if _cfg.coords["WRF"]:
        ds = xr.Dataset()
        lon, lat = np.meshgrid(_cfg.points["plon"], _cfg.points["plat"])
        ds.coords["lat"] = (("south_north", "west_east"), lon)
        ds.coords["lon"] = (("south_north", "west_east"), lat)

    else:
        if static_file:
            print(f"Read static file {static_file}\n")
            ds = xr.open_dataset(static_file)
            ds = ds.sel(
                lat=_cfg.points["plat"],
                lon=_cfg.points["plon"],
                method="nearest",
            )
            ds.coords["lon"] = np.array([ds.lon.values])
            ds.coords["lat"] = np.array([ds.lat.values])

        else:
            ds = xr.Dataset()
            ds.coords["lon"] = np.array([_cfg.points["plon"]])
            ds.coords["lat"] = np.array([_cfg.points["plat"]])

        ds.lon.attrs["standard_name"] = "lon"
        ds.lon.attrs["long_name"] = "longitude"
        ds.lon.attrs["units"] = "degrees_east"

        ds.coords["lat"] = np.array([_cfg.points["plat"]])
        ds.lat.attrs["standard_name"] = "lat"
        ds.lat.attrs["long_name"] = "latitude"
        ds.lat.attrs["units"] = "degrees_north"

    ds.coords["time"] = (("time"), df.index.values)

    # Order variables
    df = set_order_and_type(df, replace_pressure=True)

    # Get values from file
    sensor_height = _cfg.points["hgt"] - _cfg.station["stationAlt"]
    if _cfg.names["in_K"]:  # Temperature
        T2 = set_bias(
            data=df[_cfg.names["T2_var"]].values,
            lapse_type="lapse_T",
            altitude=sensor_height,
            limit=False,
        )
    else:
        T2 = set_bias(
            data=df[_cfg.names["T2_var"]].values + 273.16,
            lapse_type="lapse_T",
            altitude=sensor_height,
            limit=False,
        )

    check_temperature_bounds(temperature=T2)

    RH2 = set_bias(  # Relative humidity
        data=df[_cfg.names["RH2_var"]].values,
        lapse_type="lapse_RH",
        altitude=sensor_height,
        limit=False,
    )
    U2 = df[_cfg.names["U2_var"]].values  # Wind velocity
    G = df[_cfg.names["G_var"]].values  # Incoming shortwave radiation

    PRES = get_pressure_bias(
        data=df[_cfg.names["PRES_var"]].values, height=_cfg.points["hgt"]
    )  # Pressure

    if _cfg.names["RRR_var"] in df:  # Precipitation
        RRR = set_bias(
            data=df[_cfg.names["RRR_var"]].values,
            lapse_type="lapse_RRR",
            altitude=sensor_height,
            limit=True,
        )

    if _cfg.names["SNOWFALL_var"] in df:
        SNOWFALL = set_bias(
            data=df[_cfg.names["SNOWFALL_var"]].values,
            lapse_type="lapse_SNOWFALL",
            altitude=sensor_height,
            limit=True,
        )

    if _cfg.names["LWin_var"] in df:
        LW = df[_cfg.names["LWin_var"]].values  # Incoming longwave radiation

    if _cfg.names["N_var"] in df:
        N = df[_cfg.names["N_var"]].values  # Cloud cover fraction

    # Change aspect to south==0, east==negative, west==positive
    if static_file:
        aspect = ds["ASPECT"].values - 180.0
        ds["ASPECT"] = aspect

        # Auxiliary variables
        mask = ds.MASK.values
        slope = ds.SLOPE.values
        aspect = ds.ASPECT.values
    else:
        # Auxiliary variables
        mask = 1
        slope = 0
        aspect = 0

    # Limit bounds for relative humidity
    RH2 = set_relative_humidity_bounds(RH2)

    # Add variables to file
    add_variable_along_point(ds=ds, var=_cfg.points["hgt"], **get_variable_metadata("HGT"))
    add_variable_along_point(ds=ds, var=aspect, **get_variable_metadata("ASPECT"))
    add_variable_along_point(ds=ds, var=slope, **get_variable_metadata("SLOPE"))
    add_variable_along_point(ds=ds, var=mask, **get_variable_metadata("MASK"))
    add_variable_along_timelatlon_point(ds=ds, var=T2, **get_variable_metadata("T2"))
    add_variable_along_timelatlon_point(ds=ds, var=RH2, **get_variable_metadata("RH2"))
    add_variable_along_timelatlon_point(ds=ds, var=U2, **get_variable_metadata("U2"))
    add_variable_along_timelatlon_point(ds=ds, var=G, **get_variable_metadata("G"))
    add_variable_along_timelatlon_point(ds=ds, var=PRES, **get_variable_metadata("PRES"))

    if _cfg.names["RRR_var"] in df:
        add_variable_along_timelatlon_point(ds=ds, var=RRR, **get_variable_metadata("RRR"))
    if _cfg.names["SNOWFALL_var"] in df:
        add_variable_along_timelatlon_point(ds, var=SNOWFALL, **get_variable_metadata("SNOWFALL"))
    if _cfg.names["LWin_var"] in df:
        add_variable_along_timelatlon_point(ds=ds, var=LW, **get_variable_metadata("LWin"))
    if _cfg.names["N_var"] in df:
        add_variable_along_timelatlon_point(ds=ds, var=N, **get_variable_metadata("N"))

    # Write file to disk
    check_for_nan_point(ds)
    write_netcdf(dataset=ds, output_path=cosipy_file)

    # Check data
    check_data(dataset=ds, dataframe=df)


def create_2D_input(
    cs_file,
    cosipy_file,
    static_file,
    start_date,
    end_date,
    x0=None,
    x1=None,
    y0=None,
    y1=None,
):
    """Create a 2D input dataset from a .csv file.

    Here you need to define how to interpolate the data.

    .. warning::
        There should be only one header line in the file.

    Previous updates:
    Tobias Sauter 07.07.2019
    Anselm 01.07.2020
    Franziska Temme 03.08.2021
    """
    df = read_input_file(input_path=cs_file)

    # Select time slice
    df = get_time_slice(dataframe=df, start_date=start_date, end_date=end_date)

    # Aggregate data to selected value
    if _cfg.coords["aggregate"]:
        extra_vars = []
        for name in ["N_var", "RRR_var", "LWin_var", "SNOWFALL_var"]:
            if _cfg.names[name] in df:
                extra_vars.append(_cfg.names[name])
        aggregates = {
            _cfg.names["PRES_var"]: "mean",
            _cfg.names["T2_var"]: "mean",
            _cfg.names["RH2_var"]: "mean",
            _cfg.names["G_var"]: "mean",
            _cfg.names["U2_var"]: "mean",
        }

        for name in extra_vars:
            if name in [_cfg.names["N_var"], _cfg.names["LWin_var"]]:
                aggregates[name] = "mean"
            else:
                aggregates[name] = "sum"
        df = df.resample(_cfg.coords["aggregation_step"].agg(aggregates))

    # Load static data
    print(f"Read static file {static_file}\n")
    ds = xr.open_dataset(static_file)

    # Create subset
    ds = ds.sel(lat=slice(y0, y1), lon=slice(x0, x1))

    if _cfg.coords["WRF"]:
        dso = xr.Dataset()
        x, y = np.meshgrid(ds.lon, ds.lat)
        dso.coords["time"] = (("time"), df.index.values)
        dso.coords["lat"] = (("south_north", "west_east"), y)
        dso.coords["lon"] = (("south_north", "west_east"), x)

    else:
        dso = ds
        dso.coords["time"] = df.index.values

    # Order variables
    df = set_order_and_type(df, replace_pressure=False)

    # Get values from file
    RH2 = df[_cfg.names["RH2_var"]]  # Relative humidity
    U2 = df[_cfg.names["U2_var"]]  # Wind velocity
    G = df[_cfg.names["G_var"]]  # Incoming shortwave radiation
    PRES = df[_cfg.names["PRES_var"]]  # Pressure

    if _cfg.names["in_K"]:
        T2 = df[_cfg.names["T2_var"]].values  # Temperature
    else:
        T2 = df[_cfg.names["T2_var"]].values + 273.16
    check_temperature_bounds(temperature=T2)

    # Create numpy arrays for the 2D fields
    time_index = len(dso.time)
    lat_index = len(ds.lat)
    lon_index = len(ds.lon)
    T_interp = set_zero_field(time_index, lat_index, lon_index)
    RH_interp = set_zero_field(time_index, lat_index, lon_index)
    U_interp = set_zero_field(time_index, lat_index, lon_index)
    G_interp = np.full([time_index, lat_index, lon_index], np.nan)
    P_interp = set_zero_field(time_index, lat_index, lon_index)

    if _cfg.names["RRR_var"] in df:
        RRR = df[_cfg.names["RRR_var"]]  # Precipitation
        RRR_interp = set_zero_field(time_index, lat_index, lon_index)

    if _cfg.names["SNOWFALL_var"] in df:
        SNOWFALL = df[_cfg.names["SNOWFALL_var"]]  # Snowfall
        SNOWFALL_interp = set_zero_field(time_index, lat_index, lon_index)

    if _cfg.names["LWin_var"] in df:
        LW = df[_cfg.names["LWin_var"]]  # Incoming longwave radiation
        LW_interp = set_zero_field(time_index, lat_index, lon_index)

    if _cfg.names["N_var"] in df:
        N = df[_cfg.names["N_var"]]  # Cloud cover fraction
        N_interp = set_zero_field(time_index, lat_index, lon_index)

    # Interpolate point data to grid
    print("Interpolate CR file to grid")

    # Interpolate data (T, RH, RRR, U) to grid using lapse rates
    altitude = ds.HGT.values - _cfg.station["stationAlt"]
    for t in range(time_index):
        T_interp[t, :, :] = set_bias(
            data=(T2[t]), lapse_type="lapse_T", altitude=altitude, limit=False
        )
        RH_interp[t, :, :] = set_bias(
            data=RH2[t], lapse_type="lapse_RH", altitude=altitude, limit=False
        )
        U_interp[t, :, :] = U2[t]

        """
        Interpolate pressure using the barometric equation.
        Do not replace with get_pressure_bias() as the arrays won't
        interpolate correctly.
        """
        slp = PRES[t] / np.power(
            (1 - (0.0065 * _cfg.station["stationAlt"]) / (288.15)), 5.255
        )
        P_interp[t, :, :] = slp * np.power(
            (1 - (0.0065 * ds.HGT.values) / (288.15)), 5.255
        )

        if _cfg.names["RRR_var"] in df:
            RRR_interp[t, :, :] = set_bias(
                data=RRR[t],
                lapse_type="lapse_RRR",
                altitude=altitude,
                limit=True,
            )

        if _cfg.names["SNOWFALL_var"] in df:
            SNOWFALL_interp[t, :, :] = set_bias(
                data=SNOWFALL[t],
                lapse_type="lapse_SNOWFALL",
                altitude=altitude,
                limit=True,
            )

        if _cfg.names["LWin_var"] in df:
            LW_interp[t, :, :] = LW[t]

        if _cfg.names["N_var"] in df:
            N_interp[t, :, :] = N[t]

    print(
        f"Number of glacier cells: {int(np.count_nonzero(~np.isnan(ds['MASK'].values))):d}"
    )
    print(f"Number of glacier cells: {int(np.nansum(ds['MASK'].values)):d}")

    # Auxiliary variables
    mask = ds.MASK.values
    heights = ds.HGT.values
    slope = ds.SLOPE.values
    aspect = ds.ASPECT.values
    lats = ds.lat.values
    lons = ds.lon.values
    sw = G.values

    # Run radiation module
    if (
        _cfg.radiation["radiationModule"] in ["Wohlfahrt2016", "none"]
        or _cfg.radiation["radiationModule"] is None
    ):
        print("Run the radiation module Wohlfahrt2016 or no Radiation Module.")

        # Change aspect to south==0, east==negative, west==positive
        aspect = ds["ASPECT"].values - 180.0
        ds["ASPECT"] = (("lat", "lon"), aspect)

        for t in range(time_index):
            doy = df.index[t].dayofyear
            hour = df.index[t].hour
            for i in range(lat_index):
                for j in range(lon_index):
                    if mask[i, j] == 1:
                        if _cfg.radiation["radiationModule"] == "Wohlfahrt2016":
                            G_interp[t, i, j] = np.maximum(
                                0.0,
                                mod_radCor.correctRadiation(
                                    lats[i],
                                    lons[j],
                                    _cfg.radiation["timezone_lon"],
                                    doy,
                                    hour,
                                    slope[i, j],
                                    aspect[i, j],
                                    sw[t],
                                    _cfg.radiation["zeni_thld"],
                                ),
                            )
                        else:
                            G_interp[t, i, j] = sw[t]

    elif _cfg.radiation["radiationModule"] == "Moelg2009":
        print("Run the radiation module Moelg2009")

        # Calculate solar Parameters
        solPars, timeCorr = mod_radCor.solpars(_cfg.station["stationLat"])

        if _cfg.radiation["LUT"]:
            print("Read in look-up-tables")
            ds_LUT = xr.open_dataset(_cfg.radiation["LUT_path"])
            shad1yr = ds_LUT.SHADING.values
            svf = ds_LUT.SVF.values

        else:
            print("Build look-up-tables")

            # Sky view factor
            svf = mod_radCor.LUTsvf(
                np.flipud(heights),
                np.flipud(mask),
                np.flipud(slope),
                np.flipud(aspect),
                lats[::-1],
                lons,
            )
            print("Look-up-table for sky view factor done")

            # Topographic shading
            shad1yr = mod_radCor.LUTshad(
                solPars,
                timeCorr,
                _cfg.station["stationLat"],
                np.flipud(heights),
                np.flipud(mask),
                lats[::-1],
                lons,
                _cfg.radiation["dtstep"],
                _cfg.radiation["tcart"],
            )
            print("Look-up-table for topographic shading done")

            # Save look-up tables
            Nt = int(
                366 * (3600 / _cfg.radiation["dtstep"]) * 24
            )  # number of time steps
            Ny = len(lats)  # number of latitudes
            Nx = len(lons)  # number of longitudes

            f = nc.Dataset(_cfg.radiation["LUT_path"], "w")
            f.createDimension("time", Nt)
            f.createDimension("lat", Ny)
            f.createDimension("lon", Nx)

            LATS = f.createVariable("lat", "f4", ("lat",))
            LATS.units = "degree"
            LONS = f.createVariable("lon", "f4", ("lon",))
            LONS.units = "degree"

            LATS[:] = lats
            LONS[:] = lons

            shad = f.createVariable("SHADING", float, ("time", "lat", "lon"))
            shad.long_name = "Topographic shading"
            shad[:] = shad1yr

            SVF = f.createVariable("SVF", float, ("lat", "lon"))
            SVF.long_name = "sky view factor"
            SVF[:] = svf

            f.close()

        # Run the radiation model in both cases
        for t in range(time_index):
            doy = df.index[t].dayofyear
            hour = df.index[t].hour
            G_interp[t, :, :] = mod_radCor.calcRad(
                solPars,
                timeCorr,
                doy,
                hour,
                _cfg.station["stationLat"],
                T_interp[t, ::-1, :],
                P_interp[t, ::-1, :],
                RH_interp[t, ::-1, :],
                N_interp[t, ::-1, :],
                np.flipud(heights),
                np.flipud(mask),
                np.flipud(slope),
                np.flipud(aspect),
                shad1yr,
                svf,
                _cfg.radiation["dtstep"],
                _cfg.radiation["tcart"],
            )

        # Change aspect to south == 0, east == negative, west == positive
        aspect2 = ds["ASPECT"].values - 180.0
        ds["ASPECT"] = (("lat", "lon"), aspect2)

    else:
        raise ValueError(
            f'Radiation module {_cfg.radiation["radiationModule"]} not available.\nAvailable options are: Wohlfahrt2016, Moelg2009, none.'
        )

    # Limit bounds for relative humidity
    RH_interp = set_relative_humidity_bounds(RH_interp)

    # Add variables to file
    add_variable_along_latlon(ds=dso, var=ds.HGT.values, **get_variable_metadata("HGT"))
    add_variable_along_latlon(ds=dso, var=ds.ASPECT.values, **get_variable_metadata("ASPECT"))
    add_variable_along_latlon(ds=dso, var=ds.SLOPE.values, **get_variable_metadata("SLOPE"))
    add_variable_along_latlon(ds=dso, var=ds.MASK.values, **get_variable_metadata("MASK"))
    add_variable_along_timelatlon(ds=dso, var=T_interp, **get_variable_metadata("T2"))
    add_variable_along_timelatlon(ds=dso, var=RH_interp, **get_variable_metadata("RH2"))
    add_variable_along_timelatlon(ds=dso, var=U_interp, **get_variable_metadata("U2"))
    add_variable_along_timelatlon(ds=dso, var=G_interp, **get_variable_metadata("G"))
    add_variable_along_timelatlon(ds=dso, var=P_interp, **get_variable_metadata("PRES"))

    if _cfg.names["RRR_var"] in df:
        add_variable_along_timelatlon(ds=dso, var=RRR_interp, **get_variable_metadata("RRR"))
    if _cfg.names["SNOWFALL_var"] in df:
        add_variable_along_timelatlon(ds=dso, var=SNOWFALL_interp, **get_variable_metadata("SNOWFALL"))
    if _cfg.names["LWin_var"] in df:
        add_variable_along_timelatlon(ds=dso, var=LW_interp, **get_variable_metadata("LWin"))
    if _cfg.names["N_var"] in df:
        add_variable_along_timelatlon(ds=dso, var=N_interp, **get_variable_metadata("N"))

    # encoding = dict()
    # for var in IO.get_result().data_vars:
    #     dataMin = IO.get_result()[var].min(skipna=True).values
    #     dataMax = IO.get_result()[var].max(skipna=True).values
    #
    #     dtype = 'int16'
    #     FillValue = -9999
    #     scale_factor, add_offset = compute_scale_and_offset(dataMin, dataMax, 16)
    #     encoding[var] = dict(zlib=True, complevel=2, dtype=dtype, scale_factor=scale_factor, add_offset=add_offset, _FillValue=FillValue)

    # Write file to disk
    # dso.to_netcdf(cosipy_file, encoding=encoding)
    check_for_nan(dso)
    write_netcdf(dataset=dso, output_path=cosipy_file)

    # Check data
    check_data(dataset=dso, dataframe=df)


def add_variable_along_timelatlon(ds, var, name, units, long_name):
    """Add spatiotemporal data to a dataset."""
    if _cfg.coords["WRF"]:
        ds[name] = (("time", "south_north", "west_east"), var)
    else:
        ds[name] = (("time", "lat", "lon"), var)
    ds[name].attrs["units"] = units
    ds[name].attrs["long_name"] = long_name
    return ds


def add_variable_along_latlon(ds, var, name, units, long_name):
    """Add spatial data to a dataset."""
    if _cfg.coords["WRF"]:
        ds[name] = (("south_north", "west_east"), var)
    else:
        ds[name] = (("lat", "lon"), var)
    ds[name].attrs["units"] = units
    ds[name].attrs["long_name"] = long_name
    ds[name].encoding["_FillValue"] = -9999
    return ds


def add_variable_along_timelatlon_point(ds, var, name, units, long_name):
    """Add spatiotemporal point data to a dataset."""
    ds[name] = (("time", "lat", "lon"), np.reshape(var, (len(var), 1, 1)))
    ds[name].attrs["units"] = units
    ds[name].attrs["long_name"] = long_name
    return ds


def add_variable_along_point(ds, var, name, units, long_name):
    """Add point data to a dataset."""
    ds[name] = (("lat", "lon"), np.reshape(var, (1, 1)))
    ds[name].attrs["units"] = units
    ds[name].attrs["long_name"] = long_name
    return ds


def check(field, max_bound, min_bound):
    """Check the validity of the input data."""

    if np.nanmax(field) > max_bound or np.nanmin(field) < min_bound:
        msg = f"{str.capitalize(field.name)} MAX: {np.nanmax(field):.2f} MIN: {np.nanmin(field):.2f}"
        print(
            f"\n\nWARNING! Please check the data, it seems they are out of a reasonable range {msg}"
        )


def check_for_nan(ds):
    if _cfg.coords["WRF"] is True:
        for y, x in product(
            range(ds.dims["south_north"]), range(ds.dims["west_east"])
        ):
            mask = ds.MASK.sel(south_north=y, west_east=x)
            if mask == 1:
                if np.isnan(
                    ds.sel(south_north=y, west_east=x).to_array()
                ).any():
                    raise_nan_error()
    else:
        for y, x in product(range(ds.dims["lat"]), range(ds.dims["lon"])):
            mask = ds.MASK.isel(lat=y, lon=x)
            if mask == 1:
                if np.isnan(ds.isel(lat=y, lon=x).to_array()).any():
                    raise_nan_error()


def raise_nan_error():
    """Raise error if NaNs are in the dataset.

    Raises:
        ValueError: There are NaNs in the dataset.
    """
    raise ValueError("ERROR! There are NaNs in the dataset.")


def check_temperature_bounds(temperature: np.ndarray):
    max_temperature = np.nanmax(temperature)
    min_temperature = np.nanmin(temperature)
    check_msg = "Please check the input temperature"
    if max_temperature > 373.16:
        error_message = (
            f"Maximum temperature is: {max_temperature} K. {check_msg}"
        )
        raise ValueError(error_message)
    elif min_temperature < 173.16:
        error_message = (
            f"Minimum temperature is: {min_temperature} K. {check_msg}"
        )
        raise ValueError(error_message)


def set_relative_humidity_bounds(humidity):
    """Limit bounds for relative humidity."""
    humidity[humidity > 100.0] = 100.0
    humidity[humidity < 0.0] = 0.1

    return humidity


def check_for_nan_point(ds):
    if np.isnan(ds.to_array()).any():
        raise_nan_error()


def compute_scale_and_offset(min, max, n):
    # stretch/compress data to the available packed range
    scale_factor = (max - min) / (2**n - 1)
    # translate the range to be symmetric about zero
    add_offset = min + 2 ** (n - 1) * scale_factor
    return (scale_factor, add_offset)


def nansumwrapper(a, **kw_args):
    """Sum dataframe columns which contain NaNs."""
    if np.isnan(a).all():
        return np.nan
    else:
        return np.nansum(a, **kw_args)


def get_user_arguments(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """Get user arguments for converting AWS data.

    Args:
        parser: An initialised argument parser.

    Returns:
        User arguments for conversion.
    """
    parser.description = "Create netCDF input file from a .csv file."
    parser.prog = __package__

    # Required arguments
    parser.add_argument(
        "-i",
        "--input",
        dest="csv_file",
        type=str,
        metavar="<path>",
        required=True,
        help="Path to .csv file with meteorological data",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="cosipy_file",
        type=str,
        metavar="<path>",
        required=True,
        help="Path to the resulting COSIPY netCDF file",
    )
    parser.add_argument(
        "-s",
        "--static_file",
        type=str,
        dest="static_file",
        help="Path to static file with DEM, slope etc.",
    )

    # Optional arguments
    parser.add_argument(
        "-b",
        "--start_date",
        type=str,
        metavar="<yyyy-mm-dd>",
        dest="start_date",
        help="Start date",
    )
    parser.add_argument(
        "-e",
        "--end_date",
        type=str,
        metavar="<yyyy-mm-dd>",
        dest="end_date",
        help="End date",
    )
    parser.add_argument(
        "--xl",
        dest="xl",
        type=float,
        metavar="<float>",
        const=None,
        help="Left longitude value of the subset",
    )
    parser.add_argument(
        "--xr",
        dest="xr",
        type=float,
        metavar="<float>",
        const=None,
        help="Right longitude value of the subset",
    )
    parser.add_argument(
        "--yl",
        dest="yl",
        type=float,
        metavar="<float>",
        const=None,
        help="Lower latitude value of the subset",
    )
    parser.add_argument(
        "--yu",
        dest="yu",
        type=float,
        metavar="<float>",
        const=None,
        help="Upper latitude value of the subset",
    )
    arguments = parser.parse_args()

    return arguments


def load_config(module_name: str) -> tuple:
    """Load configuration for module.

    Args:
        module_name: Name of this module.

    Returns:
        User arguments and configuration parameters.
    """
    params = UtilitiesConfig()
    arguments = get_user_arguments(params.parser)
    params.load(arguments.utilities_path)
    params = params.get_config_expansion(name=module_name)

    return arguments, params


def main():
    global _args  # Yes, it's bad practice
    global _cfg
    _args, _cfg = load_config(module_name="aws2cosipy")

    if _cfg.points["point_model"]:
        create_1D_input(
            _args.csv_file,
            _args.cosipy_file,
            _args.static_file,
            _args.start_date,
            _args.end_date,
        )
    else:
        create_2D_input(
            _args.csv_file,
            _args.cosipy_file,
            _args.static_file,
            _args.start_date,
            _args.end_date,
            _args.xl,
            _args.xr,
            _args.yl,
            _args.yu,
        )


if __name__ == "__main__":
    main()
