import numpy as np
from core.io import *
air_pressure, cloud_cover, initial_snow_height, relative_humidity, snowfall, solar_radiation, temperature_2m, \
wind_speed = read_input()

'''
Arrays for 1D output variables
# todo variable documentation and change names to be more precise and skip saving input variables
'''
### all for all timesteps>
def create_1D_output_numpy_arrays(temperature_2m):
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
    return albedo_all, condensation_all, depostion_all, evaporation_all, ground_heat_flux_all, longwave_in_all, \
           longwave_out_all, latent_heat_flux_all, melt_heigt_all, number_layers_all, sensible_heat_flux_all, \
           snowHeight_all, shortwave_net_all, sublimation_all, surface_melt_all, surface_temperature_all, u2_all, \
           sw_in_all, T2_all, rH2_all, snowfall_all, pressure_all, cloud_all, sh_all, rho_all, Lv_all, Cs_all, q0_all, \
           q2_all, qdiff_all, phi_all
#output_list_1D=[albedo, condensation, depostion, evaporation, ground_heat_flux, longwave_in, longwave_out, \
#                latent_heat_flux, number_layers, sensible_heat_flux, snowHeight, shortwave_net, sublimation, \
#                surface_melt, surface_temperature, melt_heigt, u2, sw_in, T2, rH2, snowfall, pressure, cloud, sh, rho, \
#                Lv, Cs, q0, q2, qdiff, phi]
