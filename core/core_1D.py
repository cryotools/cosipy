import numpy as np

from constants import *
from config import *

from modules.albedo import updateAlbedo
from modules.heatEquation import solveHeatEquation
from modules.penetratingRadiation import penetrating_radiation
from modules.percolation_incl_refreezing import percolation
from modules.roughness import updateRoughness
from modules.surfaceTemperature import update_surface_temperature
from modules.init import *

from core.io import *
from core.grid import *


def core_1D(DATA):

    ''' INITIALIZATION '''

    # Initialize snowpack
    GRID = init_snowpack(DATA)
    GRID.grid_info()

    # Merge grid layers, if necessary
    GRID.update_grid(merging_level)

    # hours since the last snowfall (albedo module)
    hours_since_snowfall = 0

    ' TIME LOOP '
    # For development
    for t in np.arange(len(DATA.time)):
        
        # Rainfall is given as mm, so we convert m. w.e.q. snowheight
        if (DATA.RRR[t]<274.0):
            SNOWFALL = (DATA.RRR[t].values/1000.0) * (ice_density/density_fresh_snow)
        else:
            SNWOFALL = 0.0

        if SNOWFALL > 0.0:
            # TODO: Better use weq than snowheight

            # Add a new snow node on top
            GRID.add_node(SNOWFALL, density_fresh_snow, float(DATA.T2[t]), 0.0, 0.0, 0.0, 0.0)
            GRID.merge_new_snow(merge_snow_threshold)

        if SNOWFALL < 0.005:
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
        cpi = solveHeatEquation(GRID, dt)

        # Find new surface temperature

        fun, surface_temperature, lw_radiation_in, lw_radiation_out, sensible_heat_flux, latent_heat_flux, \
            ground_heat_flux, sw_radiation_net, rho, Lv, Cs, q0, q2, qdiff, phi \
            = update_surface_temperature(GRID, alpha, z0, DATA.T2[t].values, DATA.RH2[t].values, DATA.N[t].values, \
                                         DATA.PRES[t].values, DATA.G[t].values, DATA.U2[t].values)
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

        # Remove melt height 
        GRID.remove_melt_energy(melt + sublimation + deposition + evaporation + condensation)

        # Merge first layer, if too small (for model stability)
        GRID.merge_new_snow(merge_snow_threshold)
        
        # Account layer temperature due to penetrating SW radiation
        penetrating_radiation(GRID, sw_radiation_net, dt)

        ### when freezing work:
        percolation(GRID, melt, dt, debug_level)

        # Print some grid information
    GRID.get_total_snowheight()
