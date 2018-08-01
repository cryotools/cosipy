import numpy as np

from constants import *
from config import *

from modules.albedo import updateAlbedo
from modules.heatEquation import solveHeatEquation
from modules.penetratingRadiation import penetrating_radiation
from modules.percolation_incl_refreezing import percolation
from modules.roughness import updateRoughness
from modules.surfaceTemperature import update_surface_temperature
from modules.radCor import correctRadiation

from cpkernel.init import *
from cpkernel.io import *
from cpkernel.grid import *
import cProfile

def cosipy_core(DATA, GRID_RESTART=None):

    ''' INITIALIZATION '''
    
    # Initialize snowpack or load restart grid
    if GRID_RESTART is None:
        GRID = init_snowpack(DATA)
    else:
        GRID = load_snowpack(GRID_RESTART)

    # Initialize local result array
    RESULT = init_result_dataset_point(DATA, max_layers)

    # Plot some grid information, if necessary
    #GRID.grid_info()

    # Merge grid layers, if necessary
    GRID.update_grid(merging_level, merge_snow_threshold)

    # hours since the last snowfall (albedo module)
    hours_since_snowfall = 0

    # Get data from file
    T2 = DATA.T2.values
    RH2 = DATA.RH2.values
    N = DATA.N.values
    PRES = DATA.PRES.values
    G = DATA.G.values
    U2 = DATA.U2.values
    RRR = DATA.RRR.values

    # Check whether longwave data is availible -> used for surface temperature calculations
    if ('LWin' in DATA):
        LWin = DATA.LWin.values
    else:
        LWin = None


    cp = cProfile.Profile()
    ' TIME LOOP '
    # For development
    for t in np.arange(len(DATA.time)):
        
        # Rainfall is given as mm, so we convert m. w.e.q. snowheight
        # TODO: Insert transition function for precipitation-snowfall, i.e. sigmoid function
        if ((T2[t]<=274.0) & (RRR[t]>0.0)):
            SNOWFALL = (RRR[t].values/1000.0) * (ice_density/density_fresh_snow)
        else:
            SNOWFALL = 0.0

        if SNOWFALL > 0.0:
            # Add a new snow node on top
            GRID.add_node(SNOWFALL, density_fresh_snow, float(T2[t]), 0.0, 0.0, 0.0, 0.0)
            GRID.merge_new_snow(merge_snow_threshold)

        # Get hours since last snowfall for the albedo calculations
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
        GRID.update_grid(merging_level, merge_snow_threshold)

        # Solve the heat equation
        cpi = solveHeatEquation(GRID, dt)

        if LWin is not None:
            # Find new surface temperature
            fun, surface_temperature, lw_radiation_in, lw_radiation_out, sensible_heat_flux, latent_heat_flux, \
                ground_heat_flux, sw_radiation_net, rho, Lv, Cs, q0, q2, qdiff, phi \
                = update_surface_temperature(GRID, alpha, z0, T2[t], RH2[t], N[t], PRES[t], G[t], U2[t], LWin=LWin[t])
        else:
            # Find new surface temperature
            fun, surface_temperature, lw_radiation_in, lw_radiation_out, sensible_heat_flux, latent_heat_flux, \
                ground_heat_flux, sw_radiation_net, rho, Lv, Cs, q0, q2, qdiff, phi \
                = update_surface_temperature(GRID, alpha, z0, T2[t], RH2[t], N[t], PRES[t], G[t], U2[t])
        
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

        # Melt energy in [W m^-2 or J s^-1 m^-2]
        melt_energy = max(0, sw_radiation_net + lw_radiation_in + lw_radiation_out - ground_heat_flux -
                          sensible_heat_flux - latent_heat_flux) 

        # Convert melt energy to m w.e.q.   
        melt = melt_energy * dt / (1000 * lat_heat_melting)  

        # Remove melt m w.e.q.
        GRID.remove_melt_energy(melt + sublimation + deposition + evaporation + condensation)

        # Merge first layer, if too small (for model stability)
        GRID.merge_new_snow(merge_snow_threshold)
        
        # Account layer temperature due to penetrating SW radiation
        penetrating_radiation(GRID, sw_radiation_net, dt)

        # Refreezing
        percolation(GRID, melt, dt, debug_level)
        
        # Write results
        RESULT.SNOWHEIGHT[t] = GRID.get_total_snowheight()
        RESULT.EVAPORATION[t] = evaporation
        RESULT.SUBLIMATION[t] = sublimation
        RESULT.MELT[t] = melt
        RESULT.LWin[t] = lw_radiation_in
        RESULT.LWout[t] = lw_radiation_out
        RESULT.H[t] = sensible_heat_flux
        RESULT.LE[t] = latent_heat_flux
        RESULT.B[t] = ground_heat_flux
        RESULT.TS[t] = surface_temperature
        RESULT.RH2[t] = RH2[t]
        RESULT.T2[t] = T2[t]
        RESULT.G[t] = G[t]
        RESULT.U2[t] = U2[t]
        RESULT.N[t] = N[t]
        RESULT.Z0[t] = z0
        RESULT.ALBEDO[t] = alpha
        
        RESULT.NLAYERS[t] = GRID.get_number_layers()
        RESULT.LAYER_HEIGHT[0:GRID.get_number_layers(),t] = GRID.get_height() 
        RESULT.LAYER_RHO[0:GRID.get_number_layers(),t] = GRID.get_density() 
        RESULT.LAYER_T[0:GRID.get_number_layers(),t] = GRID.get_temperature() 
        RESULT.LAYER_LWC[0:GRID.get_number_layers(),t] = GRID.get_liquid_water_content() 
        RESULT.LAYER_CC[0:GRID.get_number_layers(),t] = GRID.get_cold_content() 
        RESULT.LAYER_POROSITY[0:GRID.get_number_layers(),t] = GRID.get_porosity() 
        RESULT.LAYER_VOL[0:GRID.get_number_layers(),t] = GRID.get_vol_ice_content() 

    # Return results
    return RESULT
