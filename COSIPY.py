#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    This is the main code file of the 'COupled Snowpack and Ice surface energy
    and MAss balance glacier model' (COSIMA). The model is originally written
    and developed in Matlab code by Huintjes et al. (2015).

    The Python translation and model improvement of COSIMA was done by
    Tobias Sauter and Bjoern Sass under the umbrella of the Institute of
    Geography, Friedrich-Alexander-University Erlangen-Nuernberg.
    The python version of the model is subsequently called COSIPY.

    You are allowed to use and modify this code in a noncommercial manner and by
    appropriately citing the above mentioned developers.

    The code is available on bitbucket. For more information read the README.

    The model is written in Python 2.7 and is tested on Anaconda2-4.3.0 64-bit
    distribution with additional packages.

    Correspondence: bjoern.sass@fau.de.
"""

# from jackedCodeTimerPY import JackedTiming
import numpy as np
import matplotlib.pylab as plt
from config import *
from Constants import *
from inputData import *
import Grid as grd
from heatEquationLagrange import solveHeatEquation
from albedo import updateAlbedo
from roughness import updateRoughness
from surfaceTemperature import updateSurfaceTemperature
from penetratingRadiation import penetratingRadiation
from percofreeze import percolation


' model function '

def main():

    # JTime = JackedTiming()

    evdiff = 0  # Auxiliary Variables - hours since last snowfall

    ' Initialization '

    nlay = 10

    # Init layers
    hlayers = 0.1*np.ones(nlay)

    rho = pice*np.ones(nlay)
    rho[0] = 250.
    rho[1] = 400.
    rho[2] = 550.
    
    # Init temp
    Ts = Tb*np.ones(nlay)
    for i in range(len(Ts)):
        gradient = ((T2[0] - Tb) / (nlay))
        Ts[i] = T2[0] - gradient*i

    # Init LWC
    LWC = np.zeros(nlay)

    if (mergingLevel == 0):
        print('Merging level 0')
    else:
        print('Merge in action!')

    # Initialize grid, the grid class contains all relevant grid information
    GRID = grd.Grid(hlayers, rho, Ts, LWC, debug_level)
    # todo params handling?

    # Get some information on the grid setup
    GRID.info()

    # Merge grid layers, if necessary
    GRID.update_grid(mergingLevel)

    Hall = []               # todo variable documentation
    Lall = []
    Liall = []
    Loall = []
    Ball = []
    SWall = []
    T0all = []
    Alphaall = []
    snowHeightall = []

    Hsnow = 0

    ' Time Loop '

    # JTime.start('while')
    for t in range(tstart, tend, 1):

        # Add snowfall
        Hsnow = Hsnow + snowfall[t]

        if (snowfall[t] > 0.0):

            # TODO: Better use weq than snowheight

            # Add a new snow node on top
            GRID.add_node(float(DATA['snowfall'][t]), rho_new, float(DATA['T2'][t]), 0)
            GRID.mergeNewSnow(mergeNewSnowThreshold)


        if (snowfall[t] < 0.005):
            evdiff = evdiff + (dt/3600.0)
        else:
            evdiff = 0

        # Calculate albedo and roughness length changes if first layer is snow
        # Update albedo values
        alpha = updateAlbedo(GRID, evdiff)

        # Update roughness length
        z0 = updateRoughness(GRID, evdiff)

        # Merge grid layers, if necessary
        GRID.update_grid(mergingLevel)

        # Solve the heat equation
        solveHeatEquation(GRID, dt)    

        # Find new surface temperature
        fun, T0, Li, Lo, H, L, B, SWnet = updateSurfaceTemperature(GRID, alpha, z0, t)

        # Surface fluxes [m w.e.q.]
        if (GRID.get_T_node(0) < zeroT):
            sublimation = max(L/(1000.0*L_ms), 0) * dt
            deposition = min(L/(1000.0*L_ms), 0) * dt
            evaporation = 0
            condensation = 0
        else:
            evaporation = max(L/(1000.0*L_mv), 0) * dt
            condensation = min(L/(1000.0*L_mv), 0) * dt
            sublimation = 0
            deposition = 0

        # Melt energy in [m w.e.q.]
        meltEnergy =  max(0, SWnet+Li+Lo-B-H-L)     # W m^-2 / J s^-1 ^m-2
        melt = meltEnergy*dt/(1000*L_m)             # m w.e.q. (ice)

        # Remove melt height from surface and store as runoff (R)
        GRID.removeMeltEnergy(melt+sublimation+deposition+evaporation+condensation)

        # Merge first layer, if too small (for model stability)
        GRID.mergeNewSnow(mergeNewSnowThreshold)

        # Account layer temperature due to penetrating SW radiation
        penetratingRadiation(GRID, SWnet, dt)

        # todo Percolation, fluid retention (LWC) & refreezing of melt water
        # and rain
        percolation(GRID, melt, dt)

        # write single variables to output variables
        Liall.append(Li)  # in long rad
        Loall.append(Lo)  # out long rad
        Hall.append(H)    # sensible heat flux *
        Lall.append(L)    # latent heat flux   *
        Ball.append(B)    # ground heat flux
        T0all.append(T0)    # surface Temperature
        SWall.append(SWnet)  # Surface net radiation
        Alphaall.append(alpha)  # albedo
        snowHeightall.append(np.sum((GRID.get_hlayer())))

    # JTime.stop('while')

    GRID.info()

    # print(JTime.report())

''' model execution '''

if __name__ == "__main__":
    main()
