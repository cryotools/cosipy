"""Convert old config.py and constants.py files into .toml.

This utility must be run from source!

To safely migrate before upgrading to the new configuration system:

.. code-block:: console
    git fetch --all
    git checkout master -- convert_config.py
    python convert_config.py  # generate .toml files

Otherwise, copy and run this file in the top-level directory of a COSIPY
source tree.

You MUST manually add the values for `create_static` in the generated
`utilities_config.toml`, as that utility script cannot be converted.
"""

import configparser
import inspect
import sys

import config
import constants
import toml
import utilities.aws2cosipy.aws2cosipyConfig as aws2cosipyConfig
import utilities.wrf2cosipy.wrf2cosipyConfig as wrf2cosipyConfig


class ModuleWrapper(object):
    """Hacky way to get module attributes."""

    def __init__(self, module):
        self.module = module

    def __getattr__(self, name):
        try:
            return getattr(self.module, name)
        except AttributeError:
            return None


def set_module_params(module, parameters: dict, key_pairs: dict) -> dict:
    """Set parameters from module variables."""
    for key, attributes in key_pairs.items():
        parameters[key] = {}
        for attr in attributes:
            value = module.__getattr__(attr)
            if value is None:
                value = 0
            parameters[key].update({attr: value})

    return parameters


def get_config_params() -> dict:
    """Get module parameters from config.py."""
    module = ModuleWrapper(sys.modules[config.__name__])
    params = {}
    table_keys = {
        "SIMULATION_PERIOD": ("time_start", "time_end"),
        "FILENAMES": ("data_path", "input_netcdf", "output_prefix"),
        "RESTART": ("restart",),
        "STAKE_DATA": (
            "stake_evaluation",
            "stakes_loc_file",
            "stakes_data_file",
            "eval_method",
            "obs_type",
        ),
        "DIMENSIONS": ("WRF", "WRF_X_CSPY", "northing", "easting"),
        "COMPRESSION": ("compression_level",),
        "PARALLELIZATION": ("slurm_use", "workers", "local_port"),
        "FULL_FIELDS": ("full_field",),
        "FORCINGS": ("force_use_TP", "force_use_N"),
        "SUBSET": ("tile", "xstart", "xend", "ystart", "yend"),
    }
    params = set_module_params(
        module=module, parameters=params, key_pairs=table_keys
    )

    # Different name
    if "_" in config.output_netcdf:
        output_split = config.output_netcdf.split("_")  # remove date

    else:
        output_split = config.output_netcdf.split(".")  # remove nc
    output_prefix = "_".join(output_split[:-1])
    params["FILENAMES"]["output_prefix"] = output_prefix

    params["OUTPUT_VARIABLES"] = get_output_params()

    return params
    

def get_output_params() -> dict:
    """Get output variable selection from cosipy/output."""

    output_config = configparser.ConfigParser()
    output_config.read("./cosipy/output")
    variables = output_config["vars"]

    params = {}
    params["output_atm"] = variables["atm"]
    params["output_internal"] = variables["internal"]
    params["output_full"] = variables["full"]

    return params


def get_constants_params() -> dict:
    """Get module parameters from constants.py."""
    module = ModuleWrapper(sys.modules[constants.__name__])
    params = {}
    table_keys = {
        "GENERAL": ("dt", "max_layers", "z"),
        "PARAMETERIZATIONS": (
            "stability_correction",
            "albedo_method",
            "densification_method",
            "penetrating_method",
            "roughness_method",
            "saturation_water_vapour_method",
            "thermal_conductivity_method",
            "sfc_temperature_method",
        ),
        "INITIAL_CONDITIONS": (
            "initial_snowheight_constant",
            "initial_snow_layer_heights",
            "initial_glacier_height",
            "initial_glacier_layer_heights",
            "initial_top_density_snowpack",
            "initial_bottom_density_snowpack",
            "temperature_bottom",
            "const_init_temp",
            "zlt1",
            "zlt2",
        ),
        "PRECIPITATION": (
            "center_snow_transfer_function",
            "spread_snow_transfer_function",
            "mult_factor_RRR",
            "minimum_snow_layer_height",
            "minimum_snowfall",
        ),
        "REMESHING": (
            "remesh_method",
            "first_layer_height",
            "layer_stretching",
            "merge_max",
            "density_threshold_merging",
            "temperature_threshold_merging",
        ),
        "CONSTANTS": (
            "constant_density",
            "albedo_fresh_snow",
            "albedo_firn",
            "albedo_ice",
            "albedo_mod_snow_aging",
            "albedo_mod_snow_depth",
            "t_star_wet",
            "t_star_dry",
            "t_star_K",
            "t_star_cutoff",
            "roughness_fresh_snow",
            "roughness_ice",
            "roughness_firn",
            "aging_factor_roughness",
            "snow_ice_threshold",
            "lat_heat_melting",
            "lat_heat_vaporize",
            "lat_heat_sublimation",
            "spec_heat_air",
            "spec_heat_ice",
            "spec_heat_water",
            "k_i",
            "k_w",
            "k_a",
            "water_density",
            "ice_density",
            "air_density",
            "sigma",
            "zero_temperature",
            "surface_emission_coeff",
        ),
    }
    params = set_module_params(
        module=module, parameters=params, key_pairs=table_keys
    )

    return params


def get_aws2cosipy_params() -> dict:
    """Get module parameters from aws2cosipyConfig.py."""
    module = ModuleWrapper(sys.modules[aws2cosipyConfig.__name__])
    params = {}
    table_keys = {
        "names": (
            "PRES_var",
            "T2_var",
            "in_K",
            "RH2_var",
            "G_var",
            "RRR_var",
            "U2_var",
            "LWin_var",
            "SNOWFALL_var",
            "N_var",
        ),
        "coords": ("WRF", "aggregate", "aggregation_step", "delimiter"),
        "radiation": (
            "radiationModule",
            "LUT",
            "dtstep",
            "tcart",
            "timezone_lon",
            "zeni_thld",
        ),
        "points": ("point_model", "plon", "plat", "hgt"),
        "station": ("stationName", "stationAlt", "stationLat"),
        "lapse": ("lapse_T", "lapse_RH", "lapse_RRR", "lapse_SNOWFALL"),
    }

    params = set_module_params(
        module=module, parameters=params, key_pairs=table_keys
    )

    return params


def get_wrf2cosipy_params() -> dict:
    """Get module parameters from wrf2cosipyConfig.py."""
    module = ModuleWrapper(sys.modules[wrf2cosipyConfig.__name__])
    params = {}
    table_keys = {"constants": ("hu", "lu_class")}
    params = set_module_params(
        module=module, parameters=params, key_pairs=table_keys
    )

    return params


def get_create_static_params() -> dict:
    """Get module parameters from create_static.py.

    As create_static cannot be imported, this sets parameters to a
    default value instead.
    """
    params = {}
    table_keys = {
        "paths": ("static_folder", "dem_path", "shape_path", "output_file"),
        "coords": (
            "tile",
            "aggregate",
            "aggregate_degree",
            "longitude_upper_left",
            "latitude_upper_left",
            "longitude_lower_right",
            "latitude_lower_right",
        ),
    }
    for key, attributes in table_keys.items():
        params[key] = {}
        for attr in attributes:
            if key == "paths":
                value = ""
            elif key == "coords" and attr in ["tile", "aggregate"]:
                value = True
            else:
                value = 0.0
            params[key].update({attr: value})

    return params


def get_utilities_params() -> dict:
    """Aggregate paramters for all utilities."""
    params = {}
    params["aws2cosipy"] = get_aws2cosipy_params()
    params["create_static"] = get_create_static_params()
    params["wrf2cosipy"] = get_wrf2cosipy_params()

    return params


def write_toml(parameters: dict, filename: str):
    """Write parameters to .toml file."""
    
    with open(f"{filename}.toml", "w") as f:
        toml.dump(parameters, f)
    

    print(f"Generated {filename}.toml")


def print_warning():
    tag = (
        f"\n{79*'-'}\n"
        "Configuration for create_static must be manually added to the generated `utilities_config.toml`.",
        "All custom configuration variables not present in the master branch will be lost!",
        "Make sure to add these back manually." f"\n{79*'-'}\n",
    )
    print("\n".join(tag))


def main():

    print_warning()

    script_path = inspect.getfile(inspect.currentframe())
    toml_suffix = script_path.split("/")[-2]  # avoid overwrite

    config_params = get_config_params()
    write_toml(parameters=config_params, filename=f"config")
    constants_params = get_constants_params()
    write_toml(parameters=constants_params, filename=f"constants")
    utilities_params = get_utilities_params()
    write_toml(parameters=utilities_params, filename=f"utilities_config")


if __name__ == "__main__":
    main()
