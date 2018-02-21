import numpy as np
from datetime import datetime

from core.core_1D import core_1D
from core.io import *
from confs_dicts_constants.dictionaries_distributred import *

def check_1D_or_distributed_and_run():

    start_time = datetime.now()

    ### read input data
    air_pressure, cloud_cover, initial_snow_height, relative_humidity, snowfall, solar_radiation, temperature_2m, \
        wind_speed = read_input()

    if temperature_2m.ndim == 2:
        print("1D run")

        ### run model in 1D version
        albedo_all, condensation_all, depostion_all, evaporation_all, ground_heat_flux_all, \
             longwave_in_all, longwave_out_all, latent_heat_flux_all, melt_heigt_all, number_layers_all, \
             sensible_heat_flux_all, snowHeight_all, shortwave_net_all, sublimation_all, surface_melt_all, \
             surface_temperature_all, u2_all, sw_in_all, T2_all, rH2_all, snowfall_all, pressure_all, cloud_all, \
             sh_all, rho_all, Lv_all, Cs_all, q0_all, q2_all, qdiff_all, phi_all = core_1D(air_pressure, cloud_cover, \
             initial_snow_height, relative_humidity, snowfall, solar_radiation,  temperature_2m, wind_speed)

        ### write 1D output variables to netcdf file!

        write_output_1D(albedo_all, condensation_all, depostion_all, evaporation_all, ground_heat_flux_all, \
             longwave_in_all, longwave_out_all, latent_heat_flux_all, melt_heigt_all, number_layers_all, \
             sensible_heat_flux_all, snowHeight_all, shortwave_net_all, sublimation_all, surface_melt_all, \
             surface_temperature_all, u2_all, sw_in_all, T2_all, rH2_all, snowfall_all, pressure_all, cloud_all, \
             sh_all, rho_all, Lv_all, Cs_all, q0_all, q2_all, qdiff_all, phi_all)

    elif temperature_2m.ndim == 3:

         ### rund model in distributed version (multiple 1D versions)
         print("distributed run")

         ### loop in x direction"""
         for x in range(0, temperature_2m.shape[0], 1):

             ### loop in y direction
             for y in range(0, temperature_2m.shape[1], 1):

                 ### if temperature for point is reasonable
                 if 0.0 <= temperature_2m[x, y, 0] <= 400:
                     print("gridpoint match")

                     ### run 1D core version
                     albedo_all, condensation_all, depostion_all, evaporation_all, ground_heat_flux_all, \
                     longwave_in_all, longwave_out_all, latent_heat_flux_all, melt_heigt_all, number_layers_all, \
                     sensible_heat_flux_all, snowHeight_all, shortwave_net_all, sublimation_all, surface_melt_all, \
                     surface_temperature_all, u2_all, sw_in_all, T2_all, rH2_all, snowfall_all, pressure_all, \
                     cloud_all, sh_all, rho_all, Lv_all, Cs_all, q0_all, q2_all, qdiff_all, phi_all =  \
                     core_1D(air_pressure[x, y, :], cloud_cover[x, y, :] , initial_snow_height[x, y, :], \
                     relative_humidity[x, y, :], snowfall[x, y, :], solar_radiation[x, y, :], temperature_2m[x, y, :], \
                     wind_speed[x, y, :])

                     albedo_distributed[x, y, :] = albedo_all
                     condensation_distributed[x, y, :] = condensation_all
                     depostion_distributed[x, y, :] = depostion_all
                     evaporation_distributed[x, y, :] = evaporation_all
                     ground_heat_flux_distributed[x, y, :] = ground_heat_flux_all
                     longwave_in_distributed[x, y, :] =  longwave_in_all
                     longwave_out_distributed[x, y, :] = longwave_out_all
                     latent_heat_flux_distributed[x, y, :] = latent_heat_flux_all
                     # mass_balance_distributed[x, y, :] = mass_balance_all
                     melt_heigt_distributed[x, y, :] = melt_heigt_all
                     number_layers_distributed[x, y, :] = number_layers_all
                     sensible_heat_flux_distributed[x, y, :] = sensible_heat_flux_all
                     snowHeight_distributed[x, y, :] = snowHeight_all
                     shortwave_net_distributed[x, y, :] = shortwave_net_all
                     sublimation_distributed[x, y, :] = sublimation_all
                     surface_melt_distributed[x, y, :] = surface_melt_all
                     surface_temperature_distributed[x, y, :] = surface_temperature_all
                     # runoff_distributed[x, y, :] = runoff_all

                 else:
                     print("no glacier")

         write_output_distributed(albedo_distributed, condensation_distributed, depostion_distributed, \
                      evaporation_distributed, ground_heat_flux_distributed, longwave_in_distributed, \
                      longwave_out_distributed, latent_heat_flux_distributed, melt_heigt_distributed, \
                      number_layers_distributed, sensible_heat_flux_distributed, snowHeight_distributed, \
                      shortwave_net_distributed, sublimation_distributed, surface_melt_distributed, \
                      surface_temperature_distributed)
    else:
         print("input not suitable")

    duration_run = datetime.now() - start_time
    print("run duration in seconds ", duration_run.total_seconds())
