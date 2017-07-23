#!/usr/bin/python

""" Declaration of constants """

L_m     = 3.34e5            # latent heat for melting [J kg-1]
L_mv    = 2.5e6             # latent heat for vaporization [J kg-1]
L_ms    = 2.834e6           # latent heat for sublimation [J kg-1]
c_p     = 1004.67           # specific heat of air [J kg-1 K-1]
c_pi    = 2050.00           # specific heat of ice [J Kg-1 K-1]
sigma   = 5.67e-8           # Stefan-Bolzmann constant [W m-2 K-4]
eps     = 0.97              # surface emmision coefficient [-]
g       = 9.81              # acceleration of gravity (Braithwaite 1995) [m s-1]

snowIceThres    = 900.0     # pore close of density [kg m^(-3)]
pice            = 917.      # density of ice [kg m^(-3)]

zeroT = 273.16              # Kelvin [K]
rhoH2O = 1000.0             # density of water [kg m^(-3)]
LWCfrac = 0.05              # irreducible water content of a snow layer;
                            # fraction of total mass of the layer [%/100]
Vp = 0.0006                 # percolation velocity for unsaturated layers [m s-1] (0.06 cm s-1)
                            # how does it change with density?
                            # Martinec, J.: Meltwater percolation through an alpine snowpack, Avalanche Formation,
                            # Movement and Effects, Proceedings of the Davos Symposium, 162, 1987.

alphaFirn = 0.55            # constant for albedo of firn [-] (Moelg etal. 2012, TC)
alphaFreshSnow = 0.90       # constant for albedo of fresh snow [-] (Moelg etal. 2012, TC)
alphaIce = 0.3              # constant for albedo of ice [-] (Moelg etal. 2012, TC)
tscale = 6                  # constant for the effect of ageing on snow albedo [days] (Moelg etal. 2012, TC)
depscale = 8                # constant for the effect of snow depth on albedo [cm] (Moelg etal. 2012, TC)
roughnessFreshSnow = 0.24   # surface roughness length for fresh snow [mm] (Moelg etal. 2012, TC)
roughnessIce = 1.7          # surface roughness length for ice [mm] (Moelg etal. 2012, TC)
roughnessFirn = 4.0         # surface roughness length for aged snow [mm] (Moelg etal. 2012, TC)