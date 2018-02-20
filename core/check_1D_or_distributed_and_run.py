import numpy as np
from datetime import datetime

from core.core_1D import core_1D
from core.io import *

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

        write_output(albedo_all, condensation_all, depostion_all, evaporation_all, ground_heat_flux_all, \
             longwave_in_all, longwave_out_all, latent_heat_flux_all, melt_heigt_all, number_layers_all, \
             sensible_heat_flux_all, snowHeight_all, shortwave_net_all, sublimation_all, surface_melt_all, \
             surface_temperature_all, u2_all, sw_in_all, T2_all, rH2_all, snowfall_all, pressure_all, cloud_all, \
             sh_all, rho_all, Lv_all, Cs_all, q0_all, q2_all, qdiff_all, phi_all)

    elif temperature_2m.ndim == 3:
         print("distributed run")

    else:
         print("input not suitable")

###sort following and delete everything which is not needed!!!
####check and run 2D or 1D and create needed variables!!!





# result_sensible_heat_flux = xr.DataArray(np.full_like(temperature_2m_mask, "nan"))
# result_latent_heat_flux = xr.DataArray(np.full_like(temperature_2m_mask, "nan"))
# result_lw_radiation_in = xr.DataArray(np.full_like(temperature_2m_mask, "nan"))
# result_lw_radiation_out = xr.DataArray(np.full_like(temperature_2m_mask, "nan"))
# result_ground_heat_flux = xr.DataArray(np.full_like(temperature_2m_mask, "nan"))
# result_sw_radiation_net = xr.DataArray(np.full_like(temperature_2m_mask, "nan"))
# result_surface_temperature = xr.DataArray(np.full_like(temperature_2m_mask, "nan"))
# result_albedo = xr.DataArray(np.full_like(temperature_2m_mask, "nan"))
# result_snow_height = xr.DataArray(np.full_like(temperature_2m_mask, "nan"))
# result_runnoff = xr.DataArray(np.full_like(temperature_2m_mask, "nan"))
# result_mass_balance = xr.DataArray(np.full_like(temperature_2m_mask, "nan"))
#
# # for parallel computing; does not work at the moment!
# pool = multiprocessing.Pool()
#
# """loop in x direction"""
# for x in range(0, temperature_2m_mask.shape[0], 1):
#     """loop in y direction"""
#
#     for y in range(0, temperature_2m_mask.shape[1], 1):
#         if 0.0 <= temperature_2m_mask[x, y, 0] <= 400:
#             print("match")
#             wind_speed = wind_speed_mask[x, y, :]
#             solar_radiation = solar_radiation_mask[x, y, :]
#             temperature_2m = temperature_2m_mask[x, y, :]
#             relative_humidity = relative_humidity_mask[x, y, :]
#             snowfall = snowfall_mask[x, y, :]
#             air_pressure = air_pressure_mask[x, y, :]
#             cloud_cover = cloud_cover_mask[x, y, :]
#             initial_snow_height = initial_snow_height_mask[x, y, :]
#
#             # write grid point variables to output variables (mask)
#             result_sensible_heat_flux[x, y, t] = sensible_heat_flux
#             result_lw_radiation_in[x, y, t] = lw_radiation_in
#             result_lw_radiation_out[x, y, t] = lw_radiation_out
#             result_sensible_h;eat_flux[x, y, t] = sensible_heat_flux
#             result_sensible_heat_flux[x, y, t] = sensible_heat_flux
#             result_latent_heat_flux[x, y, t] = latent_heat_flux
#             result_ground_heat_flux[x, y, t] = ground_heat_flux
#             result_surface_temperature[x, y, t] = surface_temperature
#             result_sw_radiation_net[x, y, t] = sw_radiation_net
#             result_albedo[x, y, t] = alpha
#             result_snow_height[x, y, t] = np.sum((GRID.get_height()))
#             result_mass_balance[x, y, t] = mass_balance
#
#             # GRID.info()
#             print("grid_point_done")
#         else:
#             print("no glacier")
#             ### needed???
#         if 'GRID' in locals():
#             print('GRID exists')
#             del GRID, air_pressure, alpha, cloud_cover, condensation, deposition, evaporation, fun, gradient, \
#                 ground_heat_flux, hours_since_snowfall, i, initial_snow_height, latent_heat_flux, layer_heights, \
#                 liquid_water_content, lw_radiation_in, lw_radiation_out, melt, melt_energy, relative_humidity, rho, \
#                 sensible_heat_flux, snow_height, snowfall, solar_radiation, sublimation, surface_temperature, \
#                 sw_radiation_net, t, temperature_2m, temperature_surface, wind_speed, z0
#
#         write_output(result_lw_radiation_in, result_lw_radiation_out, result_sensible_heat_flux,
#                      result_latent_heat_flux,
#                      result_ground_heat_flux, result_surface_temperature, result_sw_radiation_net, result_albedo,
#                      result_snow_height)


    duration_run = datetime.now() - start_time
    print("run duration in seconds ", duration_run.total_seconds())
