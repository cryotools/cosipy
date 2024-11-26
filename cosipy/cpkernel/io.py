"""Read the input data (model forcing) and write the output to netCDF file."""

from __future__ import annotations

import os
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr

from cosipy.config import main_config
from cosipy.constants import constants_config as cc


class IOClass:
    def __init__(self, data: xr.Dataset | None = None):
        """Initialise the IO Class.

        Attributes:
            DATA (xarray.Dataset or None): Meteorological data.
            RESTART (xarray.Dataset or None): Restart file data.
            RESULT (xarray.Dataset or None): Model result data.
        """
        self.atm = self.get_output_variables(main_config.output_atm)
        self.internal = self.get_output_variables(main_config.output_internal)
        self.full = self.get_output_variables(main_config.output_full)

        # Initialize data
        self.DATA = data
        self.RESTART = None
        self.RESULT = None

        # If local IO class is initialized we need to get the dimensions of the dataset
        if self.DATA is not None:
            self.time = self.DATA.sizes["time"]

    def get_output_variables(self, variables: str) -> list:
        """Get output variable names from configuration string."""
        # Sets are unordered -> same config, different checksums
        if not variables:
            return []
        return list(variables.split(","))

    def init_atm_attribute(self, name: str) -> None:
        """Initialise empty atm attribute."""
        if name in self.atm:
            setattr(self, name, self.create_nan_array())

    def init_internal_attribute(self, name: str) -> None:
        """Initialise empty internal attribute."""
        if name in self.internal:
            setattr(self, name, self.create_nan_array())

    def init_full_field_attribute(self, name: str, max_layers: int) -> None:
        """Initialise empty layer attribute."""
        if name in self.full:
            setattr(self, f"LAYER_{name}", self.create_3d_nan_array(max_layers))

    def set_atm_attribute(self, name: str, value: np.ndarray, x: int, y: int) -> None:
        """Set atm attribute if it is a desired output variable.

        Args:
            name: Output variable name.
            value: Output variable data.
            x: Slice x-coordinate.
            y: Slice y-coordinate.
        """
        if name in self.atm:
            getattr(self, name)[:, y, x] = value

    def set_internal_attribute(
        self, name: str, value: np.ndarray, x: int, y: int
    ) -> None:
        """Set internal attribute if it is a desired output variable.

        Args:
            name: Output variable name.
            value: Output variable data.
            x: Slice x-coordinate.
            y: Slice y-coordinate.
        """
        if name in self.internal:
            getattr(self, name)[:, y, x] = value

    def set_full_field_attribute(
        self, name: str, value: np.ndarray, x: int, y: int
    ) -> None:
        """Set layer attribute if it is a desired output variable.

        Args:
            name: Output variable name.
            value: Output variable data.
            x: Slice x-coordinate.
            y: Slice y-coordinate.
        """
        if name in self.full:
            getattr(self, f"LAYER_{name}")[:, y, x, :] = value

    def get_datetime(
        self,
        timestamp: str | datetime | np.datetime64,
        *,
        use_np: bool = True,
        fmt: str = "%Y-%m-%dT%H:%M",
    ) -> datetime | np.datetime64:
        """Get datetime object from a string.

        Args:
            timestamp: Timestamp.
            use_np: Convert to numpy datetime64. Default True.
            fmt: Timestamp format.

        Returns:
            datetime|np.datetime64: Timestamp converted to datetime or
            np.datetime64.
        """
        if isinstance(timestamp, str):
            if use_np:
                return np.datetime64(timestamp)
            return datetime.strptime(timestamp, fmt)
        return timestamp

    def create_data_file(self) -> xr.Dataset:
        """Create the input data and read the restart file if necessary.

        Returns:
            Model input data.
        """
        if not self.DATA:
            self.DATA = self.create_data_file()
        if main_config.restart:
            print(f"{'-'*62}\n\tRESTART FROM PREVIOUS STATE\n{'-'*62}\n")

            # Load the restart file
            time_start = main_config.time_start
            time_end = main_config.time_end
            start_timestamp = datetime.strptime(time_start, "%Y-%m-%dT%H:%M%z")
            end_timestamp = datetime.strptime(time_end, "%Y-%m-%dT%H:%M%z")
            timestamp = start_timestamp.strftime("%Y-%m-%dT%H-%M")
            restart_path = Path(main_config.data_path) / "restart" / f"restart_{timestamp}.nc"
            if not restart_path.is_file():
                msg = f"No restart file available at {restart_path}"
                raise FileNotFoundError(msg)
            if start_timestamp == end_timestamp:
                msg = f"Start date {time_start} equals end date {time_end}"
                raise IndexError(msg)
            try:
                self.GRID_RESTART = xr.open_dataset(restart_path)
                """Get time of the last calculation and add one time step.

                GRID_RESTART.time is an array of np.datetime64 objects.
                """
                self.restart_date = self.GRID_RESTART.time.to_numpy() + np.timedelta64(
                    cc.dt, "s"
                )
                # Read data from the last date to the end of the data file
                self.init_data_dataset()
            except FileNotFoundError:
                msg = f"No restart file available for the given date: {timestamp}"
                raise SystemExit(msg) from None
            except IndexError:
                msg = f"Start date {time_start} equals end date {time_end}\n"
                raise SystemExit(msg) from None
        else:
            # If no restart, read data according to the dates defined in config file
            self.restart_date = None
            self.init_data_dataset()

        if main_config.tile:  # Tile the data if desired
            if main_config.WRF:
                self.DATA = self.DATA.isel(
                    south_north=slice(main_config.ystart, main_config.yend),
                    west_east=slice(main_config.xstart, main_config.xend),
                )
            else:
                self.DATA = self.DATA.isel(
                    lat=slice(main_config.ystart, main_config.yend),
                    lon=slice(main_config.xstart, main_config.xend),
                )

        self.ny = self.DATA.sizes[main_config.northing]
        self.nx = self.DATA.sizes[main_config.easting]
        self.time = self.DATA.sizes["time"]

        return self.DATA

    def create_result_file(self) -> xr.Dataset:
        """Create and initialise the RESULT dataset."""
        self.init_result_dataset()
        return self.RESULT

    def create_restart_file(self) -> xr.Dataset:
        """Create and initialise the RESTART dataset."""
        self.init_restart_dataset()
        return self.RESTART

    def create_grid_restart(self) -> xr.Dataset:
        """Create and initialise the GRID_RESTART structure.

        This contains the layer state of the last time step, which is
        required for the restart option.
        """
        return self.GRID_RESTART

    def create_nan_array(self) -> np.ndarray:
        """Create and fill a NaN array with time, (x,y) dimensions.

        Returns:
            Filled array with time and 2D spatial coordinates.
        """
        return np.full((self.time, self.ny, self.nx), np.nan)

    def create_3d_nan_array(self, max_layers: int) -> np.ndarray:
        """Create and fill a NaN array with time, (x,y,z) dimensions.

        Args:
            The maximum number of model layers.

        Returns:
            Filled array with time and 3D spatial coordinates.
        """
        return np.full((self.time, self.ny, self.nx, max_layers), np.nan)

    def check_field(self, field, _max, _min) -> bool:
        """Check the validity of the input data."""
        if np.nanmax(field) > _max or np.nanmin(field) < _min:
            print(
                f"Please check the input data, it seems they are out of range:"
                f"\n{field.name} ... MAX: {np.nanmax(field):.2f} "
                f"MIN: {np.nanmin(field):.2f}\n"
            )
            return False
        return True

    def check_input_data(self) -> bool:
        """Check the input data is within valid bounds."""
        print(f"{'-'*62}\nChecking input data ....\n")

        data_bounds = {
            "T2": (313.16, 243.16),
            "RH2": (100.0, 0.0),
            "G": (1600.0, 0.0),
            "U2": (50.0, 0.0),
            "RRR": (20.0, 0.0),
            "N": (1.0, 0.0),
            "PRESS": (1080.0, 400.0),
            "LWin": (400.0, 200.0),
            "SNOWFALL": (0.1, 0.0),
            "SLOPE": (0.0, 90.0),
        }

        for key, bounds in data_bounds.items():
            if key in self.DATA and self.check_field(
                self.DATA[key], bounds[0], bounds[1]
            ):
                print(f"{key} ... ok")

        return True

    def init_data_dataset(self) -> None:
        """Read and store the input netCDF data.

        The input data should contain the following variables:
            :PRESS: Air pressure [hPa].
            :N: Cloud cover fraction [-].
            :RH2: 2m relative humidity [%].
            :RRR: Precipitation per time step [mm].
            :SNOWFALL: Snowfall per time step [m].
            :G: Solar radiation per time step [|W m^-2|].
            :T2: 2m air temperature [K].
            :U2: Wind speed (magnitude) [|m s^-1|].
            :HGT: Elevation [m].
        """
        input_path = Path(main_config.data_path) / "input" / main_config.input_netcdf
        try:
            self.DATA = xr.open_dataset(input_path)
        except FileNotFoundError as e:
            msg = f"Input file not found at: {input_path}"
            raise SystemExit(msg) from e

        self.DATA["time"] = np.sort(self.DATA["time"].values)
        minimum_time = str(self.DATA.time.to_numpy()[0])[0:16]
        maximum_time = str(self.DATA.time.to_numpy()[-1])[0:16]
        start_interval = self.get_datetime(minimum_time)
        end_interval = self.get_datetime(maximum_time)
        time_steps = str(self.DATA.sizes["time"])
        print(
            f"\nMaximum available time interval from {minimum_time} "
            f"until {maximum_time}. Time steps: {time_steps}\n\n"
        )

        time_start = main_config.time_start  # avoid repeat calls
        time_end = main_config.time_end
        start_time = self.get_datetime(time_start)
        end_time = self.get_datetime(time_end)

        if (start_time > end_interval) or (end_time < start_interval):
            msg = "Selected period not available in input data.\n"
            raise IndexError(msg)
        if start_time < start_interval:
            warnings.warn(
                "\nWARNING! Selected startpoint before first timestep of input data\n",
            )
        if end_time > end_interval:
            warnings.warn(
                "\nWARNING! Selected endpoint after last timestep of input data\n",
            )

        if self.restart_date is None:  # Check if restart option is set
            print(f"{'-'*62}\n\tIntegration from {time_start} to {time_end}\n{'-'*62}\n")
            self.DATA = self.DATA.sel(time=slice(time_start, time_end))
        # There is nothing to do if the dates are equal
        elif self.restart_date == end_time:
            msg = "Start date equals end date ... no new data ... EXIT"
            raise SystemExit(msg)
        else:
            # otherwise, run the model from the restart date to the defined end date
            print(
                f"Starting from {self.restart_date} (from restart file) "
                f"to {time_end} (from config file)\n"
            )
            self.DATA = self.DATA.sel(time=slice(self.restart_date, time_end))

        self.check_input_data()
        print(f"\nGlacier gridpoints: {np.nansum(self.DATA.MASK >= 1)} \n\n")

    def get_input_metadata(self) -> tuple:
        """Get input variable names and units.

        Returns:
            tuple[dict, dict]: Metadata for spatial and spatiotemporal
            variables, including netCDF keyname, unit, and long name.
        """
        metadata_spatial = {
            "HGT": ("m", "Elevation"),
            "MASK": ("boolean", "Glacier mask"),
            "SLOPE": ("degrees", "Terrain slope"),
            "ASPECT": ("degrees", "Aspect of slope"),
        }
        metadata_spatiotemporal = {
            "T2": ("K", "Air temperature at 2 m"),
            "RH2": ("%", "Relative humidity at 2 m"),
            "U2": ("m s\u207b\xb9", "Wind velocity at 2 m"),
            "PRESS": ("hPa", "Atmospheric pressure"),
            "G": ("W m\u207b\xb2", "Incoming shortwave radiation"),
            "RRR": ("mm", "Total precipitation"),
            "SNOWFALL": ("m", "Snowfall"),
            "N": ("-", "Cloud fraction"),
            "LWin": ("W m\u207b\xb2", "Incoming longwave radiation"),
        }

        return metadata_spatial, metadata_spatiotemporal

    def get_full_field_metadata(self) -> dict[str, tuple[str, str]]:
        """Get layer variable names and units.

        Returns:
            Metadata for full field layer variables, including netCDF
            keyname, unit, and long name.
        """
        return {
            "LAYER_HEIGHT": ("m", "Layer height"),
            "LAYER_RHO": ("kg m^-3", "Layer density"),
            "LAYER_T": ("K", "Layer temperature"),
            "LAYER_LWC": ("kg m^-2", "layer liquid water content"),
            "LAYER_CC": ("J m^-2", "Cold content"),
            "LAYER_POROSITY": ("-", "Porosity"),
            "LAYER_ICE_FRACTION": ("-", "Layer ice fraction"),
            "LAYER_IF": ("-", "Layer ice fraction"),  # RESTART compatibility
            "LAYER_IRREDUCIBLE_WATER": ("-", "Irreducible water"),
            "LAYER_REFREEZE": ("m w.e.", "Refreezing"),
        }


    def get_restart_metadata(self) -> dict[str, tuple[str, str]]:
        field_metadata = self.get_full_field_metadata()
        restart_metadata = {
            "new_snow_height": ("m .w.e", "New snow height"),
            "new_snow_timestamp": ("s", "New snow timestamp"),
            "old_snow_timestamp": ("s", "Old snow timestamp"),
            "NLAYERS": ("-", "Number of layers"),
            "NEWSNOWHEIGHT": ("m .w.e", "New snow height"),
            "NEWSNOWTIMESTAMP": ("s", "New snow timestamp"),
            "OLDSNOWTIMESTAMP": ("s", "Old snow timestamp"),
        }
        return restart_metadata | field_metadata


    def get_result_metadata(self) -> dict[str, tuple[str, str]]:
        """Get all variable names and units.

        Returns:
            Metadata for all input and output variables, including
            netCDF keyname, unit, and long name.
        """
        metadata_spatial, metadata_spatiotemporal = self.get_input_metadata()
        metadata_full = self.get_full_field_metadata()
        metadata_result = {
            "RAIN": ("mm", "Liquid precipitation"),
            "SNOWFALL": ("m w.e.", "Snowfall"),
            "LWin": ("W m\u207b\xb2", "Incoming longwave radiation"),
            "LWout": ("W m\u207b\xb2", "Outgoing longwave radiation"),
            "H": ("W m\u207b\xb2", "Sensible heat flux"),
            "LE": ("W m\u207b\xb2", "Latent heat flux"),
            "B": ("W m\u207b\xb2", "Ground heat flux"),
            "QRR": ("W m\u207b\xb2", "Rain heat flux"),
            "surfMB": ("m w.e.", "Surface mass balance"),
            "MB": ("m w.e.", "Mass balance"),
            "Q": ("m w.e.", "Runoff"),
            "SNOWHEIGHT": ("m", "Snowheight"),
            "TOTALHEIGHT": ("m", "Total domain height"),
            "TS": ("K", "Surface temperature"),
            "ALBEDO": ("-", "Albedo"),
            "LAYERS": ("-", "Number of layers"),
            "ME": ("W m\u207b\xb2", "Available melt energy"),
            "intMB": ("m w.e.", "Internal mass balance"),
            "EVAPORATION": ("m w.e.", "Evaporation"),
            "SUBLIMATION": ("m w.e.", "Sublimation"),
            "CONDENSATION": ("m w.e.", "Condensation"),
            "DEPOSITION": ("m w.e.", "Deposition"),
            "REFREEZE": ("m w.e.", "Refreezing"),
            "subM": ("m w.e.", "Subsurface melt"),
            "Z0": ("m", "Roughness length"),
            "surfM": ("m w.e.", "Surface melt"),
            "MOL": ("m", "Monin Obukhov length"),
        }
        # metadata_result overwrites items in previous union!
        return (
            metadata_spatial | metadata_spatiotemporal | metadata_full | metadata_result
        )


    def init_result_dataset(self) -> xr.Dataset:
        """Create the final dataset to aggregate and store the results.

        Aggregates results from individual COSIPY runs. After the
        dataset is filled with results from all the workers, the dataset
        is written to disk.

        Returns:
            One-dimensional structure with the model output.
        """
        # Coordinates
        self.RESULT = xr.Dataset()
        self.RESULT.coords["time"] = self.DATA.coords["time"]
        self.RESULT.coords["lat"] = self.DATA.coords["lat"]
        self.RESULT.coords["lon"] = self.DATA.coords["lon"]

        # Global attributes from config.py
        self.RESULT.attrs["Start_from_restart_file"] = str(main_config.restart)
        self.RESULT.attrs["Stake_evaluation"] = str(main_config.stake_evaluation)
        self.RESULT.attrs["WRF_simulation"] = str(main_config.WRF)
        self.RESULT.attrs["Compression_level"] = main_config.compression_level
        self.RESULT.attrs["Slurm_use"] = str(main_config.slurm_use)
        self.RESULT.attrs["Full_field"] = str(main_config.full_field)
        self.RESULT.attrs["Force_use_TP"] = str(main_config.force_use_TP)
        self.RESULT.attrs["Force_use_N"] = str(main_config.force_use_N)
        self.RESULT.attrs["Tile_of_glacier_of_interest"] = str(main_config.tile)

        # Global attributes from constants.py
        self.RESULT.attrs["Time_step_input_file_seconds"] = cc.dt
        self.RESULT.attrs["Max_layers"] = cc.max_layers
        self.RESULT.attrs["Z_measurement_height"] = cc.z
        self.RESULT.attrs["Stability_correction"] = cc.stability_correction
        self.RESULT.attrs["Albedo_method"] = cc.albedo_method
        self.RESULT.attrs["Densification_method"] = cc.densification_method
        self.RESULT.attrs["Penetrating_method"] = cc.penetrating_method
        self.RESULT.attrs["Roughness_method"] = cc.roughness_method
        self.RESULT.attrs["Saturation_water_vapour_method"] = (
            cc.saturation_water_vapour_method
        )

        self.RESULT.attrs["Initial_snowheight"] = cc.initial_snowheight_constant
        self.RESULT.attrs["Initial_snow_layer_heights"] = (
            cc.initial_snow_layer_heights
        )
        self.RESULT.attrs["Initial_glacier_height"] = cc.initial_glacier_height
        self.RESULT.attrs["Initial_glacier_layer_heights"] = (
            cc.initial_glacier_layer_heights
        )
        self.RESULT.attrs["Initial_top_density_snowpack"] = (
            cc.initial_top_density_snowpack
        )
        self.RESULT.attrs["Initial_bottom_density_snowpack"] = (
            cc.initial_bottom_density_snowpack
        )
        self.RESULT.attrs["Temperature_bottom"] = cc.temperature_bottom
        self.RESULT.attrs["Const_init_temp"] = cc.const_init_temp

        self.RESULT.attrs["Center_snow_transfer_function"] = (
            cc.center_snow_transfer_function
        )
        self.RESULT.attrs["Spread_snow_transfer_function"] = (
            cc.spread_snow_transfer_function
        )
        self.RESULT.attrs["Multiplication_factor_for_RRR_or_SNOWFALL"] = (
            cc.mult_factor_RRR
        )
        self.RESULT.attrs["Minimum_snow_layer_height"] = (
            cc.minimum_snow_layer_height
        )
        self.RESULT.attrs["Minimum_snowfall"] = cc.minimum_snowfall

        self.RESULT.attrs["Remesh_method"] = cc.remesh_method
        self.RESULT.attrs["First_layer_height_log_profile"] = cc.first_layer_height
        self.RESULT.attrs["Layer_stretching_log_profile"] = cc.layer_stretching

        self.RESULT.attrs["Merge_max"] = cc.merge_max
        self.RESULT.attrs["Layer_stretching_log_profile"] = cc.layer_stretching
        self.RESULT.attrs["Density_threshold_merging"] = (
            cc.density_threshold_merging
        )
        self.RESULT.attrs["Temperature_threshold_merging"] = (
            cc.temperature_threshold_merging
        )

        self.RESULT.attrs["Density_fresh_snow"] = cc.constant_density
        self.RESULT.attrs["Albedo_fresh_snow"] = cc.albedo_fresh_snow
        self.RESULT.attrs["Albedo_firn"] = cc.albedo_firn
        self.RESULT.attrs["Albedo_ice"] = cc.albedo_ice
        self.RESULT.attrs["Albedo_mod_snow_aging"] = cc.albedo_mod_snow_aging
        self.RESULT.attrs["Albedo_mod_snow_depth"] = cc.albedo_mod_snow_depth
        self.RESULT.attrs["Roughness_fresh_snow"] = cc.roughness_fresh_snow
        self.RESULT.attrs["Roughness_ice"] = cc.roughness_ice
        self.RESULT.attrs["Roughness_firn"] = cc.roughness_firn
        self.RESULT.attrs["Aging_factor_roughness"] = cc.aging_factor_roughness
        self.RESULT.attrs["Snow_ice_threshold"] = cc.snow_ice_threshold

        self.RESULT.attrs["lat_heat_melting"] = cc.lat_heat_melting
        self.RESULT.attrs["lat_heat_vaporize"] = cc.lat_heat_vaporize
        self.RESULT.attrs["lat_heat_sublimation"] = cc.lat_heat_sublimation
        self.RESULT.attrs["spec_heat_air"] = cc.spec_heat_air
        self.RESULT.attrs["spec_heat_ice"] = cc.spec_heat_ice
        self.RESULT.attrs["spec_heat_water"] = cc.spec_heat_water
        self.RESULT.attrs["k_i"] = cc.k_i
        self.RESULT.attrs["k_w"] = cc.k_w
        self.RESULT.attrs["k_a"] = cc.k_a
        self.RESULT.attrs["water_density"] = cc.water_density
        self.RESULT.attrs["ice_density"] = cc.ice_density
        self.RESULT.attrs["air_density"] = cc.air_density
        self.RESULT.attrs["sigma"] = cc.sigma
        self.RESULT.attrs["zero_temperature"] = cc.zero_temperature
        self.RESULT.attrs["Surface_emission_coeff"] = cc.surface_emission_coeff

        # Variables given by the input dataset
        spatial, spatiotemporal = self.get_input_metadata()

        for name, metadata in spatial.items():
            if name in self.DATA:
                self.add_variable_along_latlon(
                    self.RESULT, self.DATA[name], name, metadata[0], metadata[1]
                )
        for name, metadata in spatiotemporal.items():
            if name in self.DATA:
                self.add_variable_along_latlontime(
                    self.RESULT, self.DATA[name], name, metadata[0], metadata[1]
                )

        if "RRR" not in self.DATA:
            self.add_variable_along_latlontime(
                self.RESULT,
                np.full_like(self.DATA.T2, np.nan),
                "RRR",
                spatiotemporal["RRR"][1],
                spatiotemporal["RRR"][0],
            )
        if "N" not in self.DATA:
            self.add_variable_along_latlontime(
                self.RESULT,
                np.full_like(self.DATA.T2, np.nan),
                "N",
                spatiotemporal["N"][1],
                spatiotemporal["N"][0],
            )

        print("\nOutput dataset ... ok")

        return self.RESULT

    def create_global_result_arrays(self) -> None:
        """Create the global numpy arrays to store each output variable.

        Each global array is filled with local results from the workers.
        The arrays are then assigned to the RESULT dataset and stored to
        disk (see COSIPY.py).
        """
        if self.atm:
            for atm_var in self.atm:
                self.init_atm_attribute(atm_var)

        if self.internal:
            for internal_var in self.internal:
                self.init_internal_attribute(internal_var)

        if main_config.full_field and self.full:
            max_layers = cc.max_layers  # faster lookup
            for full_field_var in self.full:
                self.init_full_field_attribute(full_field_var, max_layers)

    def copy_local_to_global(
        self,
        y: int,
        x: int,
        local_RAIN: np.ndarray,
        local_SNOWFALL: np.ndarray,
        local_LWin: np.ndarray,
        local_LWout: np.ndarray,
        local_H: np.ndarray,
        local_LE: np.ndarray,
        local_B: np.ndarray,
        local_QRR: np.ndarray,
        local_MB: np.ndarray,
        local_surfMB: np.ndarray,
        local_Q: np.ndarray,
        local_SNOWHEIGHT: np.ndarray,
        local_TOTALHEIGHT: np.ndarray,
        local_TS: np.ndarray,
        local_ALBEDO: np.ndarray,
        local_LAYERS: np.ndarray,
        local_ME: np.ndarray,
        local_intMB: np.ndarray,
        local_EVAPORATION: np.ndarray,
        local_SUBLIMATION: np.ndarray,
        local_CONDENSATION: np.ndarray,
        local_DEPOSITION: np.ndarray,
        local_REFREEZE: np.ndarray,
        local_subM: np.ndarray,
        local_Z0: np.ndarray,
        local_surfM: np.ndarray,
        local_MOL: np.ndarray,
        local_LAYER_HEIGHT: np.ndarray,
        local_LAYER_RHO: np.ndarray,
        local_LAYER_T: np.ndarray,
        local_LAYER_LWC: np.ndarray,
        local_LAYER_CC: np.ndarray,
        local_LAYER_POROSITY: np.ndarray,
        local_LAYER_ICE_FRACTION: np.ndarray,
        local_LAYER_IRREDUCIBLE_WATER: np.ndarray,
        local_LAYER_REFREEZE: np.ndarray,
    ) -> None:
        """Copy the local results from workers to global numpy arrays."""
        self.set_atm_attribute("RAIN", local_RAIN, x, y)
        self.set_atm_attribute("SNOWFALL", local_SNOWFALL, x, y)
        self.set_atm_attribute("LWin", local_LWin, x, y)
        self.set_atm_attribute("LWout", local_LWout, x, y)
        self.set_atm_attribute("H", local_H, x, y)
        self.set_atm_attribute("LE", local_LE, x, y)
        self.set_atm_attribute("B", local_B, x, y)
        self.set_atm_attribute("QRR", local_QRR, x, y)
        self.set_atm_attribute("TS", local_TS, x, y)
        self.set_atm_attribute("ALBEDO", local_ALBEDO, x, y)
        self.set_atm_attribute("Z0", local_Z0, x, y)

        self.set_internal_attribute("surfMB", local_surfMB, x, y)
        self.set_internal_attribute("MB", local_MB, x, y)
        self.set_internal_attribute("Q", local_Q, x, y)
        self.set_internal_attribute("SNOWHEIGHT", local_SNOWHEIGHT, x, y)
        self.set_internal_attribute("TOTALHEIGHT", local_TOTALHEIGHT, x, y)
        self.set_internal_attribute("LAYERS", local_LAYERS, x, y)
        self.set_internal_attribute("ME", local_ME, x, y)
        self.set_internal_attribute("intMB", local_intMB, x, y)
        self.set_internal_attribute("EVAPORATION", local_EVAPORATION, x, y)
        self.set_internal_attribute("SUBLIMATION", local_SUBLIMATION, x, y)
        self.set_internal_attribute("CONDENSATION", local_CONDENSATION, x, y)
        self.set_internal_attribute("DEPOSITION", local_DEPOSITION, x, y)
        self.set_internal_attribute("REFREEZE", local_REFREEZE, x, y)
        self.set_internal_attribute("subM", local_subM, x, y)
        self.set_internal_attribute("surfM", local_surfM, x, y)
        self.set_internal_attribute("MOL", local_MOL, x, y)

        if main_config.full_field:
            self.set_full_field_attribute("HEIGHT", local_LAYER_HEIGHT, x, y)
            self.set_full_field_attribute("RHO", local_LAYER_RHO, x, y)
            self.set_full_field_attribute("T", local_LAYER_T, x, y)
            self.set_full_field_attribute("LWC", local_LAYER_LWC, x, y)
            self.set_full_field_attribute("CC", local_LAYER_CC, x, y)
            self.set_full_field_attribute("POROSITY", local_LAYER_POROSITY, x, y)
            self.set_full_field_attribute("ICE_FRACTION", local_LAYER_ICE_FRACTION, x, y)
            self.set_full_field_attribute(
                "IRREDUCIBLE_WATER", local_LAYER_IRREDUCIBLE_WATER, x, y
            )
            self.set_full_field_attribute("REFREEZE", local_LAYER_REFREEZE, x, y)

    def write_results_to_file(self) -> None:
        """Add the global numpy arrays to the RESULT dataset."""
        metadata = self.get_result_metadata()
        if self.atm:
            for atm_var in self.atm:
                self.add_variable_along_latlontime(
                    self.RESULT,
                    getattr(self, atm_var),
                    atm_var,
                    metadata[atm_var][0],
                    metadata[atm_var][1],
                )

        if self.internal:
            for internal_var in self.internal:
                self.add_variable_along_latlontime(
                    self.RESULT,
                    getattr(self, internal_var),
                    internal_var,
                    metadata[internal_var][0],
                    metadata[internal_var][1],
                )

        if main_config.full_field and self.full:
            for full_field_var in self.full:
                layer_name = f"LAYER_{full_field_var}"
                self.add_variable_along_latlonlayertime(
                    self.RESULT,
                    getattr(self, layer_name),
                    layer_name,
                    metadata[layer_name][0],
                    metadata[layer_name][1],
                )

    def create_empty_restart(self) -> xr.Dataset:
        """Create an empty dataset for the RESTART attribute.

        Returns:
            Empty xarray dataset with coordinates from self.DATA.
        """
        dataset = xr.Dataset()
        dataset.coords["time"] = self.DATA.coords["time"][-1]
        dataset.coords["lat"] = self.DATA.coords["lat"]
        dataset.coords["lon"] = self.DATA.coords["lon"]
        dataset.coords["layer"] = np.arange(cc.max_layers)

        return dataset

    def init_restart_dataset(self) -> xr.Dataset:
        """Initialise the restart dataset.

        Returns:
            The empty restart dataset.
        """
        self.RESTART = self.create_empty_restart()
        print(f"Restart dataset ... ok\n{'-'*62}\n")

        return self.RESTART

    def create_global_restart_arrays(self) -> None:
        """Initialise the global numpy arrays to store layer profiles.

        Each global array will be filled with local results from the
        workers. The arrays will then be assigned to the RESTART dataset
        and stored to disk (see COSIPY.py).
        """
        max_layers = cc.max_layers  # faster lookup

        for name in [
            "NLAYERS",
            "NEWSNOWHEIGHT",
            "NEWSNOWTIMESTAMP",
            "OLDSNOWTIMESTAMP",
        ]:
            setattr(self, f"RES_{name}", np.full((self.ny, self.nx), np.nan))

        for name in ["HEIGHT", "RHO", "T", "LWC", "IF"]:
            setattr(
                self,
                f"RES_LAYER_{name}",
                np.full((self.ny, self.nx, max_layers), np.nan),
            )

    def create_local_restart_dataset(self) -> xr.Dataset:
        """Create the result dataset for a single grid point.

        Returns:
            RESTART dataset initialised with layer profiles.
        """
        self.RESTART = self.create_empty_restart()

        metadata = self.get_restart_metadata()
        for name in [
            "NLAYERS",
            "NEWSNOWHEIGHT",
            "NEWSNOWTIMESTAMP",
            "OLDSNOWTIMESTAMP",
        ]:
            self.add_variable_along_scalar(
                self.RESTART,
                np.full((1), np.nan),
                name,
                metadata[name][0],
                metadata[name][1],
            )

        for layer_name in ["HEIGHT", "RHO", "T", "LWC", "IF"]:
            keyname = f"LAYER_{layer_name}"
            self.add_variable_along_layer(
                self.RESTART,
                np.full((self.RESTART.coords["layer"].shape[0]), np.nan),
                keyname,
                metadata[keyname][0],
                metadata[keyname][1],
            )

        return self.RESTART

    def copy_local_restart_to_global(self, y: int, x: int, local_restart: xr.Dataset) -> None:
        """Copy local restart data from workers to global numpy arrays.

        Args:
            y: Latitude index.
            x: Longitude index.
            local_restart: Local RESTART dataset.
        """
        for name in [
            "NLAYERS",
            "NEWSNOWHEIGHT",
            "NEWSNOWTIMESTAMP",
            "OLDSNOWTIMESTAMP",
        ]:
            getattr(self, f"RES_{name}")[y, x] = getattr(local_restart, name)

        for name in ["HEIGHT", "RHO", "T", "LWC", "IF"]:
            getattr(self, f"RES_LAYER_{name}")[y, x, :] = getattr(
                local_restart, f"LAYER_{name}"
            )

    def write_restart_to_file(self) -> None:
        """Add global numpy arrays to the RESTART dataset."""
        metadata = self.get_restart_metadata()
        for name in [
            "NLAYERS",
            "NEWSNOWHEIGHT",
            "NEWSNOWTIMESTAMP",
            "OLDSNOWTIMESTAMP",
        ]:
            keyname = f"RES_{name}"
            self.add_variable_along_latlon(
                self.RESTART,
                getattr(self, keyname),
                name,
                metadata[name][0],
                metadata[name][1],
            )
        for name in ["HEIGHT", "RHO", "T", "LWC", "IF"]:
            keyname = f"LAYER_{name}"
            self.add_variable_along_latlonlayer(
                self.RESTART,
                getattr(self, f"RES_{keyname}"),
                keyname,
                metadata[keyname][0],
                metadata[keyname][1],
            )

    # TODO: Make it Pythonian - Finish the getter/setter functions
    @property
    def RAIN(self):
        return self.__RAIN

    @property
    def SNOWFALL(self):
        return self.__SNOWFALL

    @property
    def LWin(self):
        return self.__LWin

    @property
    def LWout(self):
        return self.__LWout

    @property
    def H(self):
        return self.__H

    @property
    def LE(self):
        return self.__LE

    @property
    def B(self):
        return self.__B

    @property
    def QRR(self):
        return self.__QRR

    @property
    def MB(self):
        return self.__MB

    @RAIN.setter
    def RAIN(self, x):
        self.__RAIN = x

    @SNOWFALL.setter
    def SNOWFALL(self, x):
        self.__SNOWFALL = x

    @LWin.setter
    def LWin(self, x):
        self.__LWin = x

    @LWout.setter
    def LWout(self, x):
        self.__LWout = x

    @H.setter
    def H(self, x):
        self.__H = x

    @LE.setter
    def LE(self, x):
        self.__LE = x

    @B.setter
    def B(self, x):
        self.__B = x

    @QRR.setter
    def QRR(self, x):
        self.__QRR = x

    @MB.setter
    def MB(self, x):
        self.__MB = x

    def get_result(self) -> xr.Dataset:
        """Get the RESULT data structure."""
        return self.RESULT

    def get_restart(self) -> xr.Dataset:
        """Get the RESTART data structure."""
        return self.RESTART

    def get_grid_restart(self) -> xr.Dataset:
        """Get the GRID_RESTART data structure."""
        return self.GRID_RESTART

    # ==================================================================
    # Auxiliary functions for writing variables to NetCDF files
    # ==================================================================
    def set_variable_metadata(
        self, data: xr.Variable, units: str, long_name: str, fill_value: int = -9999
    ) -> xr.Dataset:
        """Add long name, units, and default fill value to variable.

        Args:
            data: netCDF data structure.
            units: Variable units.
            long_name: Full name of variable.
            fill_value: NaN fill value. Default "-9999".

        Returns:
            Dataset with updated metadata for a specific variable.
        """
        data.attrs["units"] = units
        data.attrs["long_name"] = long_name
        data.encoding["_FillValue"] = fill_value

        return data

    def add_variable_along_scalar(
        self, ds: xr.Dataset, var: np.ndarray, name: str, units: str, long_name: str
    ) -> xr.Dataset:
        """Add scalar data to a dataset.

        Args:
            ds: Target data structure.
            var: New data.
            name: The new variable's abbreviated name.
            units: The new variable's units.
            long_name: The new variable's full name.

        Returns:
            Target dataset with the new scalar variable.
        """
        ds[name] = var.data
        self.set_variable_metadata(ds[name], units, long_name)
        return ds

    def add_variable_along_latlon(
        self, ds: xr.Dataset, var: np.ndarray, name: str, units: str, long_name: str
    ) -> xr.Dataset:
        """Add spatial data to a dataset.

        Args:
            ds (xr.Dataset): Target data structure.
            var (np.ndarray): New spatial data.
            name (str): The new variable's abbreviated name.
            units (str): New variable units.
            long_name (str): The new variable's full name.

        Returns:
            Target dataset with the new spatial variable.
        """
        ds[name] = ((main_config.northing, main_config.easting), var.data)
        self.set_variable_metadata(ds[name], units, long_name)
        return ds

    def add_variable_along_time(
        self, ds: xr.Dataset, var: np.ndarray, name: str, units: str, long_name: str
    ) -> xr.Dataset:
        """Add temporal data to a dataset.

        Args:
            ds: Target data structure.
            var: New temporal data.
            name: The new variable's abbreviated name.
            units: The new variable's units.
            long_name: The new variable's full name.

        Returns:
            Target dataset with the new temporal variable.
        """
        ds[name] = xr.DataArray(var.data, coords=[("time", ds.time)])
        self.set_variable_metadata(ds[name], units, long_name)
        return ds

    def add_variable_along_latlontime(
        self, ds: xr.Dataset, var: np.ndarray, name: str, units: str, long_name: str
    ) -> xr.Dataset:
        """Add spatiotemporal data to a dataset.

        Args:
            ds: Target data structure.
            var: New spatiotemporal data.
            name: The new variable's abbreviated name.
            units: The new variable's units.
            long_name: The new variable's full name.

        Returns:
            Target dataset with the new spatiotemporal variable.
        """
        ds[name] = (("time", main_config.northing, main_config.easting), var.data)
        self.set_variable_metadata(ds[name], units, long_name)
        return ds

    def add_variable_along_latlonlayertime(
        self, ds: xr.Dataset, var: np.ndarray, name: str, units: str, long_name: str
    ) -> xr.Dataset:
        """Add a spatiotemporal mesh to a dataset.

        Args:
            ds: Target data structure.
            var: New spatiotemporal mesh data.
            name: The new variable's abbreviated name.
            units: The new variable's units.
            long_name: The new variable's full name.

        Returns:
            Target dataset with the new spatiotemporal mesh.
        """
        ds[name] = (("time", main_config.northing, main_config.easting, "layer"), var.data)
        self.set_variable_metadata(ds[name], units, long_name)
        return ds

    def add_variable_along_latlonlayer(
        self, ds: xr.Dataset, var: np.ndarray, name: str, units: str, long_name: str
    ) -> xr.Dataset:
        """Add a spatial mesh to a dataset.

        Args:
            ds: Target data structure.
            var: New spatial mesh.
            name: The new variable's abbreviated name.
            units: The new variable's units.
            long_name: The new variable's full name.

        Returns:
            Target dataset with the new spatial mesh.
        """
        ds[name] = ((main_config.northing, main_config.easting, "layer"), var.data)
        self.set_variable_metadata(ds[name], units, long_name)
        return ds

    def add_variable_along_layertime(
        self, ds: xr.Dataset, var: np.ndarray, name: str, units: str, long_name: str
    ) -> xr.Dataset:
        """Add temporal layer data to a dataset.

        Args:
            ds: Target data structure.
            var: New layer data with a time coordinate.
            name: The new variable's abbreviated name.
            units: The new variable's units.
            long_name: The new variable's full name.

        Returns:
            Target dataset with the new layer data.
        """
        ds[name] = (("time", "layer"), var.data)
        self.set_variable_metadata(ds[name], units, long_name)
        return ds

    def add_variable_along_layer(
        self, ds: xr.Dataset, var: np.ndarray, name: str, units: str, long_name: str
    ) -> xr.Dataset:
        """Add layer data to a dataset.

        Args:
            ds: Target data structure.
            var: New layer data.
            name: The new variable's abbreviated name.
            units: The new variable's units.
            long_name: The new variable's full name.

        Returns:
            Target dataset with the new layer data.
        """
        ds[name] = ("layer", var.data)
        self.set_variable_metadata(ds[name], units, long_name)
        return ds
