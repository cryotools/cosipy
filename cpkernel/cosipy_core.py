import numpy as np
import logging

from constants import *
from config import *

from modules.albedo import updateAlbedo
from modules.heatEquation import solveHeatEquation
from modules.penetratingRadiation import penetrating_radiation
from modules.percolation_incl_refreezing import percolation
from modules.roughness import updateRoughness
from modules.densification import densification
from modules.surfaceTemperature import update_surface_temperature

from cpkernel.init import *
from cpkernel.io import *
from cpkernel.grid import *
import cProfile

def cosipy_core(DATA, GRID_RESTART=None):

    ''' INITIALIZATION '''
    
    # Start logging
    logger = logging.getLogger(__name__)

    #--------------------------------------------
    # Initialize snowpack or load restart grid
    #--------------------------------------------
    if GRID_RESTART is None:
        GRID = init_snowpack(DATA)
    else:
        GRID = load_snowpack(GRID_RESTART)

    # Create the local output datasets
    logger.debug('Create local datasets')
    IO = IOClass(DATA)
    RESULT = IO.create_local_result_dataset()
    RESTART = IO.create_local_restart_dataset()

    # Merge grid layers, if necessary
    logger.debug('Create local datasets')
    GRID.update_grid(merging, density_threshold_merging, temperature_threshold_merging, merge_snow_threshold)

    # hours since the last snowfall (albedo module)
    hours_since_snowfall = 0

    #--------------------------------------------
    # Get data from file
    #--------------------------------------------
    T2 = DATA.T2.values
    RH2 = DATA.RH2.values
    PRES = DATA.PRES.values
    G = DATA.G.values
    U2 = DATA.U2.values

    # Check whether snowfall data is availible 
    if ('SNOWFALL' in DATA) and ('RRR' in DATA):
        SNOWF = DATA.SNOWFALL.values
        RRR = DATA.RRR.values
        print("You can select between total precipitation and snowfall (default)\n")
    elif ('SNOWFALL' in DATA):
        SNOWF = DATA.SNOWFALL.values
    else:
        SNOWF = None
        RRR = DATA.RRR.values
    if force_use_TP is True:
        SNOWF = None

    # Check whether longwave data is availible -> used for surface temperature calculations
    if ('LWin' in DATA) and ('N' in DATA):
        LWin = DATA.LWin.values
        N = DATA.N.values
    elif ('LWin' in DATA):
        LWin = DATA.LWin.values
    else:
        LWin = None
        N = DATA.N.values

    # Profiling with bokeh
    cp = cProfile.Profile()
    
    #--------------------------------------------
    # TIME LOOP 
    #--------------------------------------------
    logger.debug('Start time loop')
    
    for t in np.arange(len(DATA.time)):

        if (SNOWF is not None):
        # Snowfall is given in m
            SNOWFALL = SNOWF[t]

        else:
        # Rainfall is given as mm, so we convert to m snowheight
            SNOWFALL = (RRR[t]/1000.0)*(ice_density/density_fresh_snow)*(0.5*(-np.tanh(((T2[t]-zero_temperature)-2.5)*2.5)+1))
            if SNOWFALL<0.0:        
                SNOWFALL = 0.0

        if SNOWFALL > 0.0:
            # Add a new snow node on top
            GRID.add_node(SNOWFALL, density_fresh_snow, float(T2[t]), 0.0, 0.0, 0.0, 0.0, 0.0)
            GRID.merge_new_snow(merge_snow_threshold)
     
        # Calculate rain
        RAIN = RRR[t]-SNOWFALL*(density_fresh_snow/ice_density)

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

        # Calculate new density to densification
        densification(GRID)

        # Merge grid layers, if necessary
        GRID.update_grid(merging, temperature_threshold_merging, density_threshold_merging, merge_snow_threshold)

        # Solve the heat equation
        cpi = solveHeatEquation(GRID, dt)

        if LWin is not None:
            # Find new surface temperature
            fun, surface_temperature, lw_radiation_in, lw_radiation_out, sensible_heat_flux, latent_heat_flux, \
                ground_heat_flux, sw_radiation_net, rho, Lv, Cs, q0, q2, qdiff, phi \
                = update_surface_temperature(GRID, alpha, z0, T2[t], RH2[t], PRES[t], G[t], U2[t], LWin=LWin[t])
        else:
            # Find new surface temperature
            fun, surface_temperature, lw_radiation_in, lw_radiation_out, sensible_heat_flux, latent_heat_flux, \
                ground_heat_flux, sw_radiation_net, rho, Lv, Cs, q0, q2, qdiff, phi \
                = update_surface_temperature(GRID, alpha, z0, T2[t], RH2[t], PRES[t], G[t], U2[t], N=N[t])
        
        # Surface fluxes [m w.e.q.]
        if surface_temperature < zero_temperature:
            sublimation = max(latent_heat_flux / (1000.0 * lat_heat_sublimation), 0) * dt
            deposition = min(latent_heat_flux / (1000.0 * lat_heat_sublimation), 0) * dt
            evaporation = 0
            condensation = 0
        else:
            sublimation = 0
            deposition = 0
            evaporation = max(latent_heat_flux / (1000.0 * lat_heat_vaporize), 0) * dt
            condensation = min(latent_heat_flux / (1000.0 * lat_heat_vaporize), 0) * dt

        # Melt energy in [W m^-2 or J s^-1 m^-2]
        melt_energy = max(0, sw_radiation_net + lw_radiation_in + lw_radiation_out - ground_heat_flux -
                          sensible_heat_flux - latent_heat_flux) 

        # Convert melt energy to m w.e.q.   
        melt = melt_energy * dt / (1000 * lat_heat_melting)  

        # Remove melt m w.e.q.
        GRID.remove_melt_energy(melt + sublimation + deposition + evaporation + condensation)

        # Merge first layer, if too small (for model stability)
        GRID.merge_new_snow(merge_snow_threshold)

        # Account layer temperature due to penetrating SW radiation and subsurface melt
        subsurface_melt = penetrating_radiation(GRID, sw_radiation_net, dt)

        # Percolation/Refreezing
        Q, water_refreezed, LWCchange  = percolation(GRID, melt, dt, debug_level)
        
        # Write results
        logger.debug('Write data into local result structure')

        # Calculate mass balance
        surface_mass_balance = SNOWFALL*(density_fresh_snow/ice_density) - melt - sublimation - deposition - evaporation - condensation 
        internal_mass_balance = water_refreezed - subsurface_melt
        mass_balance = surface_mass_balance + internal_mass_balance

        internal_mass_balance2 = melt-Q  #+ subsurface_melt
        mass_balance_check = surface_mass_balance + internal_mass_balance2

        # Save results 
        RESULT.T2[t] = T2[t]
        RESULT.RH2[t] = RH2[t]
        RESULT.U2[t] = U2[t]
        RESULT.RAIN[t] = RAIN
        RESULT.SNOWFALL[t] = SNOWFALL
        RESULT.PRES[t] = PRES[t]
        RESULT.G[t] = G[t]
        RESULT.LWin[t] = lw_radiation_in
        RESULT.LWout[t] = lw_radiation_out
        RESULT.H[t] = -sensible_heat_flux
        RESULT.LE[t] = -latent_heat_flux
        RESULT.B[t] = ground_heat_flux
        RESULT.ME[t] = melt_energy
        RESULT.MB[t] = mass_balance
        RESULT.surfMB[t] = surface_mass_balance
        RESULT.intMB[t] = internal_mass_balance
        RESULT.EVAPORATION[t] = evaporation
        RESULT.SUBLIMATION[t] = sublimation
        RESULT.CONDENSATION[t] = condensation
        RESULT.DEPOSITION[t] = deposition
        RESULT.surfM[t] = melt
        RESULT.subM[t] = subsurface_melt
        RESULT.Q[t] = Q 
        RESULT.REFREEZE[t] = water_refreezed 
        RESULT.SNOWHEIGHT[t] = GRID.get_total_snowheight()
        RESULT.TOTALHEIGHT[t] = GRID.get_total_height()
        RESULT.TS[t] = surface_temperature
        RESULT.ALBEDO[t] = alpha
        RESULT.Z0[t] = z0
        RESULT.NLAYERS[t] = GRID.get_number_layers()

        if LWin is None:
            RESULT.N[t] = N[t]

        if full_field:
            RESULT.LAYER_HEIGHT[t, 0:GRID.get_number_layers()] = GRID.get_height()
            RESULT.LAYER_RHO[t, 0:GRID.get_number_layers()] = GRID.get_density()
            RESULT.LAYER_T[t, 0:GRID.get_number_layers()] = GRID.get_temperature()
            RESULT.LAYER_LWC[t, 0:GRID.get_number_layers()] = GRID.get_liquid_water_content()
            RESULT.LAYER_CC[t, 0:GRID.get_number_layers()] = GRID.get_cold_content()
            RESULT.LAYER_POROSITY[t, 0:GRID.get_number_layers()] = GRID.get_porosity()
            RESULT.LAYER_VOL[t, 0:GRID.get_number_layers()] = GRID.get_vol_ice_content()
            RESULT.LAYER_REFREEZE[t, 0:GRID.get_number_layers()] = GRID.get_refreeze()

    # TODO Restart
    logger.debug('Write restart data into local restart structure')
    RESTART['NLAYERS'] = GRID.get_number_layers()
    RESTART.LAYER_HEIGHT[0:GRID.get_number_layers()] = GRID.get_height() 
    RESTART.LAYER_RHO[0:GRID.get_number_layers()] = GRID.get_density() 
    RESTART.LAYER_T[0:GRID.get_number_layers()] = GRID.get_temperature() 
    RESTART.LAYER_LWC[0:GRID.get_number_layers()] = GRID.get_liquid_water_content() 
    RESTART.LAYER_CC[0:GRID.get_number_layers()] = GRID.get_cold_content() 
    RESTART.LAYER_POROSITY[0:GRID.get_number_layers()] = GRID.get_porosity() 
    RESTART.LAYER_VOL[0:GRID.get_number_layers()] = GRID.get_vol_ice_content() 
    RESTART.LAYER_REFREEZE[0:GRID.get_number_layers()] = GRID.get_refreeze() 

    # Return results
    return RESULT, RESTART