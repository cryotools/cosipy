import numpy as np
import logging

from constants import *
from config import *

from cosipy.modules.albedo import updateAlbedo
from cosipy.modules.heatEquation import solveHeatEquation
from cosipy.modules.penetratingRadiation import penetrating_radiation
from cosipy.modules.percolation import percolation
from cosipy.modules.refreezing import refreezing
from cosipy.modules.roughness import updateRoughness
from cosipy.modules.densification import densification
from cosipy.modules.evaluation import evaluate
from cosipy.modules.surfaceTemperature import update_surface_temperature

from cosipy.cpkernel.init import *
from cosipy.cpkernel.io import *
from cosipy.cpkernel.grid import *
import cProfile


def cosipy_core(DATA, indY, indX, GRID_RESTART=None, stake_names=None, stake_data=None):

    # Local variables
    _RRR = np.full(len(DATA.time), np.nan)
    _RAIN = np.full(len(DATA.time), np.nan)
    _SNOWFALL = np.full(len(DATA.time), np.nan)
    _LWin = np.full(len(DATA.time), np.nan)
    _LWout = np.full(len(DATA.time), np.nan)
    _H = np.full(len(DATA.time), np.nan)
    _LE = np.full(len(DATA.time), np.nan)
    _B = np.full(len(DATA.time), np.nan)
    _MB = np.full(len(DATA.time), np.nan)
    _surfMB = np.full(len(DATA.time), np.nan)
    _MB = np.full(len(DATA.time), np.nan)
    _Q = np.full(len(DATA.time), np.nan)
    _SNOWHEIGHT = np.full(len(DATA.time), np.nan)
    _TOTALHEIGHT = np.full(len(DATA.time), np.nan)
    _TS = np.full(len(DATA.time), np.nan)
    _ALBEDO = np.full(len(DATA.time), np.nan)
    _ME = np.full(len(DATA.time), np.nan)
    _intMB = np.full(len(DATA.time), np.nan)
    _EVAPORATION = np.full(len(DATA.time), np.nan)
    _SUBLIMATION = np.full(len(DATA.time), np.nan)
    _CONDENSATION = np.full(len(DATA.time), np.nan)
    _DEPOSITION = np.full(len(DATA.time), np.nan)
    _REFREEZE = np.full(len(DATA.time), np.nan)
    _NLAYERS = np.full(len(DATA.time), np.nan)
    _subM = np.full(len(DATA.time), np.nan)
    _Z0 = np.full(len(DATA.time), np.nan)
    _surfM = np.full(len(DATA.time), np.nan)

    _LAYER_HEIGHT = np.full((len(DATA.time),max_layers), np.nan)
    _LAYER_RHO = np.full((len(DATA.time),max_layers), np.nan)
    _LAYER_T = np.full((len(DATA.time),max_layers), np.nan)
    _LAYER_LWC = np.full((len(DATA.time),max_layers), np.nan)
    _LAYER_CC = np.full((len(DATA.time),max_layers), np.nan)
    _LAYER_POROSITY = np.full((len(DATA.time),max_layers), np.nan)
    _LAYER_ICE_FRACTION = np.full((len(DATA.time),max_layers), np.nan)
    _LAYER_IRREDUCIBLE_WATER = np.full((len(DATA.time),max_layers), np.nan)
    _LAYER_REFREEZE = np.full((len(DATA.time),max_layers), np.nan)


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

    # Merge grid layers, if necessary
    logger.debug('Create local datasets')

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

    #--------------------------------------------
    # Checks for optional input variables
    #--------------------------------------------
    if ('SNOWFALL' in DATA) and ('RRR' in DATA):
        SNOWF = DATA.SNOWFALL.values * mult_factor_RRR
        RRR = DATA.RRR.values * mult_factor_RRR

    elif ('SNOWFALL' in DATA):
        SNOWF = DATA.SNOWFALL.values * mult_factor_RRR
        RRR = None
        RAIN = None

    else:
        SNOWF = None
        RRR = DATA.RRR.values * mult_factor_RRR

    # Use RRR rather than snowfall?
    if force_use_TP is True:
        SNOWF = None

    if ('LWin' in DATA) and ('N' in DATA):
        LWin = DATA.LWin.values
        N = DATA.N.values

    elif ('LWin' in DATA):
        LWin = DATA.LWin.values

    else:
        LWin = None
        N = DATA.N.values

    # Use N rather than LWin
    if force_use_N is True:
        LWin = None

    if ('SLOPE' in DATA):
        SLOPE = DATA.SLOPE.values

    else:
        SLOPE = 0.0

    # Initial cumulative mass balance variable
    MB_cum = 0

    if stake_evaluation is True:
        # Create pandas dataframe for stake evaluation
        _df = pd.DataFrame(index=stake_data.index, columns=['mb','snowheight'], dtype='float')

    # Profiling with bokeh
    cp = cProfile.Profile()

    #--------------------------------------------
    # TIME LOOP
    #--------------------------------------------
    logger.debug('Start time loop')
    
    for t in np.arange(len(DATA.time)):

        # Check grid
        GRID.grid_check()

        # get seconds since start
        timestamp = dt*t

        # Calc fresh snow density
        density_fresh_snow = np.maximum(109.0+6.0*(T2[t]-273.16)+26.0*np.sqrt(U2[t]), 50.0)

        # Derive snowfall [m] and rain rates [m w.e.]
        if (SNOWF is not None) and (RRR is not None):
            SNOWFALL = SNOWF[t]
            RAIN = RRR[t]-SNOWFALL*(density_fresh_snow/ice_density) * 1000.0
        elif (SNOWF is not None):
            SNOWFALL = SNOWF[t]
        else:
            # Else convert total precipitation [mm] to snowheight [m]; liquid/solid fraction
            SNOWFALL = (RRR[t]/1000.0)*(ice_density/density_fresh_snow)*(0.5*(-np.tanh(((T2[t]-zero_temperature) - center_snow_transfer_function) * spread_snow_transfer_function) + 1.0))
            RAIN = RRR[t]-SNOWFALL*(density_fresh_snow/ice_density) * 1000.0

        # if snowfall is smaller than the threshold
        if SNOWFALL<minimum_snow_layer_height:
            SNOWFALL = 0.0

        # if rainfall is smaller than the threshold
        if RAIN<(minimum_snow_layer_height*(density_fresh_snow/ice_density)*1000.0):
            RAIN = 0.0

        if SNOWFALL > 0.0:
            # Add a new snow node on top
           GRID.add_fresh_snow(SNOWFALL, density_fresh_snow, np.minimum(float(T2[t]),zero_temperature), 0.0, timestamp)

        # Guarantee that solar radiation is greater equal zero
        if (G[t]<0.0):
            G[t] = 0.0

        #--------------------------------------------
        # Merge grid layers, if necessary
        #--------------------------------------------
        GRID.update_grid()

        #--------------------------------------------
        # Calculate albedo and roughness length changes if first layer is snow
        #--------------------------------------------
        alpha = updateAlbedo(GRID, timestamp)

        #--------------------------------------------
        # Update roughness length
        #--------------------------------------------
        z0 = updateRoughness(GRID, timestamp)

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
                ground_heat_flux, sw_radiation_net, rho, Lv, Cs_t, Cs_q, q0, q2, phi \
                = update_surface_temperature(GRID, alpha, z0, T2[t], RH2[t], PRES[t], G_resid, U2[t], SLOPE, LWin=LWin[t])
        else:
            # Find new surface temperature (LW is parametrized using cloud fraction)
            fun, surface_temperature, lw_radiation_in, lw_radiation_out, sensible_heat_flux, latent_heat_flux, \
                ground_heat_flux, sw_radiation_net, rho, Lv, Cs_t, Cs_q, q0, q2, phi \
                = update_surface_temperature(GRID, alpha, z0, T2[t], RH2[t], PRES[t], G_resid, U2[t], SLOPE, N=N[t])

        #--------------------------------------------
        # Surface mass fluxes [m w.e.q.]
        #--------------------------------------------
        if surface_temperature < zero_temperature:
            sublimation = min(latent_heat_flux / (water_density * lat_heat_sublimation), 0) * dt
            deposition = max(latent_heat_flux / (water_density * lat_heat_sublimation), 0) * dt
            evaporation = 0
            condensation = 0
        else:
            sublimation = 0
            deposition = 0
            evaporation = min(latent_heat_flux / (water_density * lat_heat_vaporize), 0) * dt
            condensation = max(latent_heat_flux / (water_density * lat_heat_vaporize), 0) * dt

        #--------------------------------------------
        # Melt process - mass changes of snowpack (melting, sublimation, deposition, evaporation, condensation)
        #--------------------------------------------
        # Melt energy in [W m^-2 or J s^-1 m^-2]
        melt_energy = max(0, sw_radiation_net + lw_radiation_in + lw_radiation_out + ground_heat_flux +
                          sensible_heat_flux + latent_heat_flux)

        # Convert melt energy to m w.e.q.
        melt = melt_energy * dt / (water_density * lat_heat_melting)

        # Remove melt [m w.e.q.]
        GRID.remove_melt_weq(melt - sublimation - deposition - evaporation)

        #--------------------------------------------
        # Percolation
        #--------------------------------------------
        Q  = percolation(GRID, melt - condensation, dt)

        #--------------------------------------------
        # Refreezing
        #--------------------------------------------
        water_refreezed = refreezing(GRID)
        
        #--------------------------------------------
        # Solve the heat equation
        #--------------------------------------------
        solveHeatEquation(GRID, dt)

        #--------------------------------------------
        # Calculate new density to densification
        #--------------------------------------------
        densification(GRID,SLOPE)

        #--------------------------------------------
        # Calculate mass balance
        #--------------------------------------------
        surface_mass_balance = SNOWFALL * (density_fresh_snow / ice_density) - melt - sublimation - deposition - evaporation
        internal_mass_balance = water_refreezed - subsurface_melt
        mass_balance = surface_mass_balance + internal_mass_balance

        internal_mass_balance2 = melt-Q  #+ subsurface_melt
        mass_balance_check = surface_mass_balance + internal_mass_balance2

        #GRID.grid_check()

        # Write results
        logger.debug('Write data into local result structure')

        # TOBI
        # Cumulative mass balance for stake evaluation 
        MB_cum = MB_cum + mass_balance
        
        # Store cumulative MB in pandas frame for validation
        if stake_names:
            if (DATA.isel(time=t).time.values in stake_data.index):
                _df['mb'].loc[DATA.isel(time=t).time.values] = MB_cum 
                _df['snowheight'].loc[DATA.isel(time=t).time.values] = GRID.get_total_snowheight() 
        
        # Save results
        _RAIN[t] = RAIN
        _SNOWFALL[t] = SNOWFALL
        _LWin[t] = lw_radiation_in
        _LWout[t] = lw_radiation_out
        _H[t] = sensible_heat_flux
        _LE[t] = latent_heat_flux
        _B[t] = ground_heat_flux
        _MB[t] = mass_balance
        _surfMB[t] = surface_mass_balance
        _Q[t] = Q
        _SNOWHEIGHT[t] = GRID.get_total_snowheight()
        _TOTALHEIGHT[t] = GRID.get_total_height()
        _TS[t] = surface_temperature
        _ALBEDO[t] = alpha
        _NLAYERS[t] = GRID.get_number_layers()
        _ME[t] = melt_energy
        _intMB[t] = internal_mass_balance
        _EVAPORATION[t] = evaporation
        _SUBLIMATION[t] = sublimation
        _CONDENSATION[t] = condensation
        _DEPOSITION[t] = deposition
        _REFREEZE[t] = water_refreezed
        _subM[t] = subsurface_melt
        _Z0[t] = z0
        _surfM[t] = melt

        if full_field:
            if GRID.get_number_layers()>max_layers:
                logger.error('Maximum number of layers reached')
            else:
                _LAYER_HEIGHT[t, 0:GRID.get_number_layers()] = GRID.get_height()
                _LAYER_RHO[t, 0:GRID.get_number_layers()] = GRID.get_density()
                _LAYER_T[t, 0:GRID.get_number_layers()] = GRID.get_temperature()
                _LAYER_LWC[t, 0:GRID.get_number_layers()] = GRID.get_liquid_water_content()
                _LAYER_CC[t, 0:GRID.get_number_layers()] = GRID.get_cold_content()
                _LAYER_POROSITY[t, 0:GRID.get_number_layers()] = GRID.get_porosity()
                _LAYER_ICE_FRACTION[t, 0:GRID.get_number_layers()] = GRID.get_ice_fraction()
                _LAYER_IRREDUCIBLE_WATER[t, 0:GRID.get_number_layers()] = GRID.get_irreducible_water_content()
                _LAYER_REFREEZE[t, 0:GRID.get_number_layers()] = GRID.get_refreeze()
        else:
            _LAYER_HEIGHT = None
            _LAYER_RHO = None
            _LAYER_T = None
            _LAYER_LWC = None
            _LAYER_CC = None
            _LAYER_POROSITY = None
            _LAYER_ICE_FRACTION = None
            _LAYER_IRREDUCIBLE_WATER = None
            _LAYER_REFREEZE = None

    if stake_evaluation is True:
        # Evaluate stakes
        _stat = evaluate(stake_names, stake_data, _df)
    else:
        _stat = None
        _df = None

    # Restart
    logger.debug('Write restart data into local restart structure')
    RESTART['NLAYERS'] = GRID.get_number_layers()
    RESTART.LAYER_HEIGHT[0:GRID.get_number_layers()] = GRID.get_height()
    RESTART.LAYER_RHO[0:GRID.get_number_layers()] = GRID.get_density()
    RESTART.LAYER_T[0:GRID.get_number_layers()] = GRID.get_temperature()
    RESTART.LAYER_LWC[0:GRID.get_number_layers()] = GRID.get_liquid_water_content()

    return (indY,indX,RESTART,_RAIN,_SNOWFALL,_LWin,_LWout,_H,_LE,_B, \
            _MB,_surfMB,_Q,_SNOWHEIGHT,_TOTALHEIGHT,_TS,_ALBEDO,_NLAYERS, \
            _ME,_intMB,_EVAPORATION,_SUBLIMATION,_CONDENSATION,_DEPOSITION,_REFREEZE, \
            _subM,_Z0,_surfM, \
            _LAYER_HEIGHT,_LAYER_RHO,_LAYER_T,_LAYER_LWC,_LAYER_CC,_LAYER_POROSITY,_LAYER_ICE_FRACTION, \
            _LAYER_IRREDUCIBLE_WATER,_LAYER_REFREEZE,stake_names,_stat,_df)
