import numpy as np
from core.io import *
air_pressure, cloud_cover, initial_snow_height, relative_humidity, snowfall, solar_radiation, temperature_2m, \
wind_speed = read_input()

'''
Arrays for distributed version
# todo variable documentation and change names to be more precise and skip saving input variables
'''
###resulting arrays
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