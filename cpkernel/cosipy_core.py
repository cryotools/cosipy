import numpy as np
import logging

from constants import *
from config import *

from modules.albedo import updateAlbedo
from modules.heatEquation import solveHeatEquation
from modules.penetratingRadiation import penetrating_radiation
from modules.percolation import percolation
from modules.refreezing import refreezing 
from modules.roughness import updateRoughness
from modules.densification import densification
from modules.surfaceTemperature import update_surface_temperature

from cpkernel.init import *
from cpkernel.io import *
from cpkernel.grid import *
import cProfile


def cosipy_core(DATA, indY, indX, GRID_RESTART=None):
        
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
    RESTART = IO.create_local_restart_dataset()
    RESULT = IO.create_local_result_arrays()

    # Merge grid layers, if necessary
    logger.debug('Create local datasets')

    # hours since the last snowfall (albedo module)
    hours_since_snowfall = 0

    
    # Get data from file
    #--------------------------------------------
    T2 = DATA.T2.values
    RH2 = DATA.RH2.values
    PRES = DATA.PRES.values
    G = DATA.G.values
    U2 = DATA.U2.values

    #--------------------------------------------
    # Checks for optional input variables
    #--------------------------------------------
    if ('SNOWFALL' in DATA) and ('RRR' in DATA):
        SNOWF = DATA.SNOWFALL.values
        RRR = DATA.RRR.values
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

    if ('SLOPE' in DATA):
        SLOPE = DATA.SLOPE.values

    else:
        SLOPE = 0.0

    # Profiling with bokeh
    cp = cProfile.Profile()
    
    #--------------------------------------------
    # TIME LOOP 
    #--------------------------------------------
    logger.debug('Start time loop')
   
    for t in np.arange(len(DATA.time)):
        
        GRID.grid_check()

        if (SNOWF is not None):
            SNOWFALL = SNOWF[t]

        else:
        # , else convert rainfall [mm] to snowheight [m]
            SNOWFALL = (RRR[t]/1000.0)*(ice_density/density_fresh_snow)*(0.5*(-np.tanh(((T2[t]-zero_temperature) / center_snow_transfer_function) * spread_snow_transfer_function) + 1.0))
            if SNOWFALL<0.0:        
                SNOWFALL = 0.0

        ## TODO DELETE
        SNOWFALL=SNOWFALL*1.5

        if SNOWFALL > 0.0:
            # Add a new snow node on top
            GRID.add_node(SNOWFALL, density_fresh_snow, np.minimum(float(T2[t]),zero_temperature), 0.0)
        
        #--------------------------------------------
        # RAINFALL = Total precipitation - SNOWFALL in mm w.e.
        #--------------------------------------------
        RAIN = RRR[t]-SNOWFALL*(density_fresh_snow/ice_density) * 1000

        #--------------------------------------------
        # Get hours since last snowfall for the albedo calculations
        #--------------------------------------------
        if SNOWFALL < minimum_snow_to_reset_albedo:
            hours_since_snowfall += dt / 3600.0
        else:
            hours_since_snowfall = 0

        #--------------------------------------------
        # Merge grid layers, if necessary
        #--------------------------------------------
        GRID.update_grid(merging, temperature_threshold_merging, density_threshold_merging, merge_snow_threshold, merge_max, split_max)
        
        #--------------------------------------------
        # Calculate albedo and roughness length changes if first layer is snow
        #--------------------------------------------
        alpha = updateAlbedo(GRID, hours_since_snowfall)

        #--------------------------------------------
        # Update roughness length
        #--------------------------------------------
        z0 = updateRoughness(GRID, hours_since_snowfall)
        
        #--------------------------------------------
        # Solve the heat equation
        #--------------------------------------------
        solveHeatEquation(GRID, dt)
        #--------------------------------------------
        # Surface Energy Balance 
        #--------------------------------------------
        # Calculate net shortwave radiation
        SWnet = G[t] * (1 - alpha)
        
        # Penetrating SW radiation and subsurface melt
        if SWnet > 0.0:
            subsurface_melt, G_penetrating = penetrating_radiation(GRID, SWnet, dt)
        else:
            subsurface_melt = 0.0
            G_penetrating = 0.0

        # Calculate residual incoming shortwave radiation (penetrating part removed)
        G_resid = G[t] - G_penetrating

        if LWin is not None:
            # Find new surface temperature (LW is used from the input file)
            fun, surface_temperature, lw_radiation_in, lw_radiation_out, sensible_heat_flux, latent_heat_flux, \
                ground_heat_flux, sw_radiation_net, rho, Lv, Cs_t, Cs_q, q0, q2, qdiff, phi \
                = update_surface_temperature(GRID, alpha, z0, T2[t], RH2[t], PRES[t], G_resid, U2[t], SLOPE, LWin=LWin[t])
        else:
            # Find new surface temperature (LW is parametrized using cloud fraction)
            fun, surface_temperature, lw_radiation_in, lw_radiation_out, sensible_heat_flux, latent_heat_flux, \
                ground_heat_flux, sw_radiation_net, rho, Lv, Cs_t, Cs_q, q0, q2, qdiff, phi \
                = update_surface_temperature(GRID, alpha, z0, T2[t], RH2[t], PRES[t], G_resid, U2[t], SLOPE, N=N[t])
        
        #--------------------------------------------
        # Surface mass fluxes [m w.e.q.]
        #--------------------------------------------
        if surface_temperature < zero_temperature:
            sublimation = min(latent_heat_flux / (1000.0 * lat_heat_sublimation), 0) * dt
            deposition = max(latent_heat_flux / (1000.0 * lat_heat_sublimation), 0) * dt
            evaporation = 0
            condensation = 0
        else:
            sublimation = 0
            deposition = 0
            evaporation = min(latent_heat_flux / (1000.0 * lat_heat_vaporize), 0) * dt
            condensation = max(latent_heat_flux / (1000.0 * lat_heat_vaporize), 0) * dt

        
        #--------------------------------------------
        # Melt process - mass changes of snowpack (melting, sublimation, deposition, evaporation, condensation)
        #--------------------------------------------
        # Melt energy in [W m^-2 or J s^-1 m^-2]
        melt_energy = max(0, sw_radiation_net + lw_radiation_in + lw_radiation_out + ground_heat_flux +
                          sensible_heat_flux + latent_heat_flux) 

        # Convert melt energy to m w.e.q.   
        melt = melt_energy * dt / (1000 * lat_heat_melting)  

        # Remove melt m w.e.q.
        GRID.remove_melt_energy(melt + sublimation + deposition + evaporation + condensation)

        #--------------------------------------------
        # Percolation
        #--------------------------------------------
        Q  = percolation(GRID, melt + condensation, dt, debug_level) 
        
        #--------------------------------------------
        # Refreezing
        #--------------------------------------------
        water_refreezed = refreezing(GRID)
        
        #--------------------------------------------
        # Calculate new density to densification
        #--------------------------------------------
        # densification(GRID,SLOPE)

        #--------------------------------------------
        # Calculate mass balance
        #--------------------------------------------
        surface_mass_balance = SNOWFALL * (density_fresh_snow / ice_density) - melt - sublimation - deposition - evaporation
        internal_mass_balance = water_refreezed - subsurface_melt
        mass_balance = surface_mass_balance + internal_mass_balance

        internal_mass_balance2 = melt-Q  #+ subsurface_melt
        mass_balance_check = surface_mass_balance + internal_mass_balance2

        GRID.grid_check()

        # Write results
        logger.debug('Write data into local result structure')

        # Save results 
        IO.local_RAIN[t] = RAIN
        IO.local_SNOWFALL[t] = SNOWFALL
        IO.local_LWin[t] = lw_radiation_in
        IO.local_LWout[t] = lw_radiation_out
        IO.local_H[t] = sensible_heat_flux
        IO.local_LE[t] = latent_heat_flux
        IO.local_B[t] = ground_heat_flux
        IO.local_MB[t] = mass_balance
        IO.local_surfMB[t] = surface_mass_balance
        IO.local_Q[t] = Q 
        IO.local_SNOWHEIGHT[t] = GRID.get_total_snowheight()
        IO.local_TOTALHEIGHT[t] = GRID.get_total_height()
        IO.local_TS[t] = surface_temperature
        IO.local_ALBEDO[t] = alpha
        IO.local_NLAYERS[t] = GRID.get_number_layers()
        IO.local_ME[t] = melt_energy
        IO.local_intMB[t] = internal_mass_balance
        IO.local_EVAPORATION[t] = evaporation
        IO.local_SUBLIMATION[t] = sublimation
        IO.local_CONDENSATION[t] = condensation
        IO.local_DEPOSITION[t] = deposition
        IO.local_REFREEZE[t] = water_refreezed 
        IO.local_subM[t] = subsurface_melt
        IO.local_Z0[t] = z0
        IO.local_surfM[t] = melt

        if full_field:
            if GRID.get_number_layers()>max_layers:
                logger.error('Maximum number of layers reached')
            else:                    
                IO.local_LAYER_HEIGHT[t, 0:GRID.get_number_layers()] = GRID.get_height()
                IO.local_LAYER_RHO[t, 0:GRID.get_number_layers()] = GRID.get_density()
                IO.local_LAYER_T[t, 0:GRID.get_number_layers()] = GRID.get_temperature()
                IO.local_LAYER_LWC[t, 0:GRID.get_number_layers()] = GRID.get_liquid_water_content()
                IO.local_LAYER_CC[t, 0:GRID.get_number_layers()] = GRID.get_cold_content()
                IO.local_LAYER_POROSITY[t, 0:GRID.get_number_layers()] = GRID.get_porosity()
                IO.local_LAYER_LW[t, 0:GRID.get_number_layers()] = GRID.get_liquid_water()
                IO.local_LAYER_ICE_FRACTION[t, 0:GRID.get_number_layers()] = GRID.get_ice_fraction()
                IO.local_LAYER_IRREDUCIBLE_WATER[t, 0:GRID.get_number_layers()] = GRID.get_irreducible_water_content()
                IO.local_LAYER_REFREEZE[t, 0:GRID.get_number_layers()] = GRID.get_refreeze()

    # Restart
    logger.debug('Write restart data into local restart structure')
    RESTART['NLAYERS'] = GRID.get_number_layers()
    RESTART.LAYER_HEIGHT[0:GRID.get_number_layers()] = GRID.get_height() 
    RESTART.LAYER_RHO[0:GRID.get_number_layers()] = GRID.get_density() 
    RESTART.LAYER_T[0:GRID.get_number_layers()] = GRID.get_temperature() 
    RESTART.LAYER_LW[0:GRID.get_number_layers()] = GRID.get_liquid_water() 

    return (indY,indX,RESTART,IO)
