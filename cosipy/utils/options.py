from constants import *
from config import *
# Dict to hold valid options.
OPTIONS = {
    'stability_correction': stability_correction,
    'albedo_method': albedo_method,
    'densification_method': densification_method,
    'penetrating_method': penetrating_method,
    'roughness_method': roughness_method,
    'saturation_water_vapour_method': saturation_water_vapour_method,
    'thermal_conductivity_method': thermal_conductivity_method,
    'sfc_temperature_method': sfc_temperature_method,
    'albedo_fresh_snow': albedo_fresh_snow,
    'albedo_firn': albedo_firn,
    'albedo_ice': albedo_ice,
    'roughness_fresh_snow': roughness_fresh_snow,
    'roughness_ice': roughness_ice,
    'roughness_firn': roughness_firn,
    'time_end': time_end,
    'time_start': time_start,
    'input_netcdf': input_netcdf,
    'output_netcdf': output_netcdf
    }

def read_opt(opt_dict, glob):
    """ Reads the opt_dict and overwrites the key-value pairs in glob - the calling function's
    globals() dictionary."""
    if opt_dict is not None:
        for key in opt_dict:
            if key in OPTIONS.keys(): 
                glob[key] = opt_dict[key]
            else:
                print(f'ATTENTION: {key} is not a valid option. Default will be used!')
