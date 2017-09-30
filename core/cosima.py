import numpy as np
from datetime import datetime
import multiprocessing
from functools import partial

from modules.heatEquation import solveHeatEquation
from modules.penetratingRadiation import penetrating_radiation
from modules.roughness import updateRoughness
from modules.surfaceTemperature import update_surface_temperature
from modules.albedo import updateAlbedo
from modules.percolation import percolation

import core.grid as grd
from core.io import *

from constants import *
from config import *


def cosima():
    """ COSIMA main routine """
    start_time = datetime.now()

    # read input data
    wind_speed_mask, solar_radiation_mask, temperature_2m_mask, relative_humidity_mask, snowfall_mask,\
    air_pressure_mask, cloud_cover_mask, initial_snow_height_mask = read_input()
    pool=multiprocessing.Pool()
    """loop in x direction"""
    for x in range(0, temperature_2m_mask.shape[0]-1,1):
        """loop in y direction"""

        partial_func=partial()
        for y in range(0, temperature_2m_mask.shape[1]-1,1):
            if 0.0 <= temperature_2m_mask[x,y,0] <= 400:
                print("match")
                wind_speed = wind_speed_mask[x,y,:]
                solar_radiation = solar_radiation_mask[x,y,:]
                temperature_2m = temperature_2m_mask[x,y,:]
                relative_humidity = relative_humidity_mask[x,y,:]
                snowfall = snowfall_mask[x,y,:]
                air_pressure = air_pressure_mask[x,y,:]
                cloud_cover = cloud_cover_mask[x,y,:]
                initial_snow_height = initial_snow_height_mask[x,y,:]

                ''' INITIALIZATION '''
                hours_since_snowfall = 0
                # Init layers
                layer_heights = 0.1 * np.ones(number_layers)

                rho = ice_density * np.ones(number_layers)
                rho[0] = 250.
                rho[1] = 400.
                rho[2] = 550.

                # Init temperature
                temperature_surface = temperature_bottom * np.ones(number_layers)
                for i in range(len(temperature_surface)):
                    gradient = ((temperature_2m[0] - temperature_bottom) / number_layers)
                    temperature_surface[i] = temperature_2m[0] - gradient * i

                # Init liquid water content
                liquid_water_content = np.zeros(number_layers)

                # if merging_level == 0:
                #     print('Merging level 0')
                # else:
                #     print('Merge in action!')

                # Initialize grid, the grid class contains all relevant grid information
                GRID = grd.Grid(layer_heights, rho, temperature_surface, liquid_water_content, debug_level)
                # todo params handling?

                # Get some information on the grid setup
                GRID.info()

                # Merge grid layers, if necessary
                GRID.update_grid(merging_level)

                result_sensible_heat_flux = []
                result_latent_heat_flux = []
                result_lw_radiation_in = []
                result_lw_radiation_out = []
                result_ground_heat_flux = []
                result_sw_radiation_net = []
                result_surface_temperature = []
                result_albedo = []
                result_snow_height = []

                if initial_snow_height[0]:
                    snow_height = initial_snow_height
                else:
                    snow_height = 0

                ' TIME LOOP '

                for t in range(time_start, time_end, 1):
                    #print(t)
                    # Add snowfall
                    snow_height = snow_height + snowfall[t]

                    if snowfall[t] > 0.0:
                        # TODO: Better use weq than snowheight

                        # Add a new snow node on top
                        GRID.add_node(float(snowfall[t]), density_fresh_snow, float(temperature_2m[t]), 0)
                        GRID.merge_new_snow(merge_snow_threshold)

                    if snowfall[t] < 0.005:
                        hours_since_snowfall += dt / 3600.0
                    else:
                        hours_since_snowfall = 0

                    # Calculate albedo and roughness length changes if first layer is snow
                    # Update albedo values
                    alpha = updateAlbedo(GRID, hours_since_snowfall)

                    # Update roughness length
                    z0 = updateRoughness(GRID, hours_since_snowfall)

                    # Merge grid layers, if necessary
                    GRID.update_grid(merging_level)

                    # Solve the heat equation
                    solveHeatEquation(GRID, dt)

                    # Find new surface temperature
                    fun, surface_temperature, lw_radiation_in, lw_radiation_out, sensible_heat_flux, latent_heat_flux, \
                        ground_heat_flux, sw_radiation_net \
                        = update_surface_temperature(GRID, alpha, z0, t, x, y)

                    # Surface fluxes [m w.e.q.]
                    if GRID.get_node_temperature(0) < zero_temperature:
                        sublimation = max(latent_heat_flux / (1000.0 * lat_heat_sublimation), 0) * dt
                        deposition = min(latent_heat_flux / (1000.0 * lat_heat_sublimation), 0) * dt
                        evaporation = 0
                        condensation = 0
                    else:
                        evaporation = max(latent_heat_flux / (1000.0 * lat_heat_vaporize), 0) * dt
                        condensation = min(latent_heat_flux / (1000.0 * lat_heat_vaporize), 0) * dt
                        sublimation = 0
                        deposition = 0

                    # Melt energy in [m w.e.q.]
                    melt_energy = max(0, sw_radiation_net + lw_radiation_in + lw_radiation_out - ground_heat_flux -
                                      sensible_heat_flux - latent_heat_flux)  # W m^-2 / J s^-1 ^m-2

                    melt = melt_energy * dt / (1000 * lat_heat_melting)  # m w.e.q. (ice)

                    # Remove melt height from surface and store as runoff (R)
                    GRID.remove_melt_energy(melt + sublimation + deposition + evaporation + condensation)

                    # Merge first layer, if too small (for model stability)
                    GRID.merge_new_snow(merge_snow_threshold)

                    # Account layer temperature due to penetrating SW radiation
                    penetrating_radiation(GRID, sw_radiation_net, dt)

                    # todo Percolation, fluid retention (liquid_water_content) & refreezing of melt water
                    # and rain
                    percolation(GRID, melt, dt)

                    # write single variables to output variables
                    result_lw_radiation_in.append(lw_radiation_in)
                    result_lw_radiation_out.append(lw_radiation_out)
                    result_sensible_heat_flux.append(sensible_heat_flux)
                    result_latent_heat_flux.append(latent_heat_flux)
                    result_ground_heat_flux.append(ground_heat_flux)
                    result_surface_temperature.append(surface_temperature)
                    result_sw_radiation_net.append(sw_radiation_net)
                    result_albedo.append(alpha)
                    result_snow_height.append(np.sum((GRID.get_height())))

                    #GRID.info()
                    #print("grid_point_done")
            else:
                print("no glacier")
                #print(temperature_2m[x,y,0])



    #write_output_1D(result_lw_radiation_in,result_lw_radiation_out,result_sensible_heat_flux,result_latent_heat_flux,
     #               result_ground_heat_flux,result_surface_temperature,result_sw_radiation_net,result_albedo,result_snow_height)


    duration_run=datetime.now()-start_time
    print("run duration in seconds ", duration_run.total_seconds())
