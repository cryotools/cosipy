import numpy as np
from core.io import *
air_pressure, cloud_cover, initial_snow_height, relative_humidity, snowfall, solar_radiation, temperature_2m, \
wind_speed = read_input()

'''
Arrays for 1D output variables
# todo variable documentation and change names to be more precise and skip saving input variables
'''

albedo_all = xr.DataArray(np.full_like(temperature_2m, "nan"))
condensation_all = xr.DataArray(np.full_like(temperature_2m, "nan"))
depostion_all = xr.DataArray(np.full_like(temperature_2m, "nan"))
evaporation_all = xr.DataArray(np.full_like(temperature_2m, "nan"))
ground_heat_flux_all = xr.DataArray(np.full_like(temperature_2m, "nan"))
longwave_in_all = xr.DataArray(np.full_like(temperature_2m, "nan"))
longwave_out_all = xr.DataArray(np.full_like(temperature_2m, "nan"))
latent_heat_flux_all = xr.DataArray(np.full_like(temperature_2m, "nan"))
number_layers_all = xr.DataArray(np.full_like(temperature_2m, "nan"))
sensible_heat_flux_all = xr.DataArray(np.full_like(temperature_2m, "nan"))
snowHeight_all = xr.DataArray(np.full_like(temperature_2m, "nan"))
shortwave_net_all = xr.DataArray(np.full_like(temperature_2m, "nan"))
sublimation_all = xr.DataArray(np.full_like(temperature_2m, "nan"))
surface_melt_all = xr.DataArray(np.full_like(temperature_2m, "nan"))
surface_temperature_all = xr.DataArray(np.full_like(temperature_2m, "nan"))
#mass_balance_all = xr.DataArray(np.full_like(temperature_2m, "nan"))
melt_heigt_all = xr.DataArray(np.full_like(temperature_2m, "nan"))
# runoff_all = xr.DataArray(np.full_like(temperature_2m, "nan"))

###input variables save in same file for comparision; not needed in longtime???
u2_all = xr.DataArray(np.full_like(temperature_2m, "nan"))
sw_in_all = xr.DataArray(np.full_like(temperature_2m, "nan"))
T2_all = xr.DataArray(np.full_like(temperature_2m, "nan"))
rH2_all = xr.DataArray(np.full_like(temperature_2m, "nan"))
snowfall_all = xr.DataArray(np.full_like(temperature_2m, "nan"))
pressure_all = xr.DataArray(np.full_like(temperature_2m, "nan"))
cloud_all = xr.DataArray(np.full_like(temperature_2m, "nan"))
sh_all = xr.DataArray(np.full_like(temperature_2m, "nan"))

###interim for investigations; not needed in longterm???
rho_all = xr.DataArray(np.full_like(temperature_2m, "nan"))            ### air density
Lv_all = xr.DataArray(np.full_like(temperature_2m, "nan"))           ### used latent heat fusion (melting or vaporisation
Cs_all = xr.DataArray(np.full_like(temperature_2m, "nan"))            ### bulk transfer coeff
q0_all = xr.DataArray(np.full_like(temperature_2m, "nan"))           ### mixing ratio surface
q2_all = xr.DataArray(np.full_like(temperature_2m, "nan"))            ### mixing ratio 2m
qdiff_all = xr.DataArray(np.full_like(temperature_2m, "nan"))          ### difference mixing ratio
phi_all = xr.DataArray(np.full_like(temperature_2m, "nan"))           ### stabilty parameter
#output_list_1D=[albedo, condensation, depostion, evaporation, ground_heat_flux, longwave_in, longwave_out, \
#                latent_heat_flux, number_layers, sensible_heat_flux, snowHeight, shortwave_net, sublimation, \
#                surface_melt, surface_temperature, melt_heigt, u2, sw_in, T2, rH2, snowfall, pressure, cloud, sh, rho, \
#                Lv, Cs, q0, q2, qdiff, phi]

'''
Arrays for distributed output variables
'''

albedo_distributed = xr.DataArray(np.full_like(temperature_2m, "nan"))
condensation_distributed = xr.DataArray(np.full_like(temperature_2m, "nan"))
depostion_distributed = xr.DataArray(np.full_like(temperature_2m, "nan"))
evaporation_distributed = xr.DataArray(np.full_like(temperature_2m, "nan"))
ground_heat_flux_distributed = xr.DataArray(np.full_like(temperature_2m, "nan"))
longwave_in_distributed = xr.DataArray(np.full_like(temperature_2m, "nan"))
longwave_out_distributed = xr.DataArray(np.full_like(temperature_2m, "nan"))
latent_heat_flux_distributed = xr.DataArray(np.full_like(temperature_2m, "nan"))
number_layers_distributed = xr.DataArray(np.full_like(temperature_2m, "nan"))
sensible_heat_flux_distributed = xr.DataArray(np.full_like(temperature_2m, "nan"))
snowHeight_distributed = xr.DataArray(np.full_like(temperature_2m, "nan"))
shortwave_net_distributed = xr.DataArray(np.full_like(temperature_2m, "nan"))
sublimation_distributed = xr.DataArray(np.full_like(temperature_2m, "nan"))
surface_melt_distributed = xr.DataArray(np.full_like(temperature_2m, "nan"))
surface_temperature_distributed = xr.DataArray(np.full_like(temperature_2m, "nan"))
#mass_balance_distributed = xr.DataArray(np.full_like(temperature_2m, "nan"))
melt_heigt_distributed = xr.DataArray(np.full_like(temperature_2m, "nan"))
# runoff_distributed = xr.DataArray(np.full_like(temperature_2m, "nan"))

# output_list_distributed = [albedo_distributed, condensation_distributed, depostion_distributed, \
#                            evaporation_distributed, ground_heat_flux_distributed, longwave_in_distributed,\
#                            longwave_out_distributed, latent_heat_flux_distributed, number_layers_distributed, \
#                            sensible_heat_flux_distributed, snowHeight_distributed, shortwave_net_distributed, \
#                            sublimation_distributed, surface_melt_distributed, surface_temperature_distributed, \
#                            melt_heigt_distributed]