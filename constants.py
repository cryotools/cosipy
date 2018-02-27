"""
    Declaration of constants
    Do not modify unless you are absolutely sure what you are doing.
"""

' PHYSICAL CONSTANTS '

lat_heat_melting = 3.34e5           # latent heat for melting [J kg-1]
lat_heat_vaporize = 2.5e6           # latent heat for vaporization [J kg-1]
lat_heat_sublimation = 2.834e6      # latent heat for sublimation [J kg-1]
spec_heat_air = 1004.67             # specific heat of air [J kg-1 K-1]
spec_heat_ice = 2050.00             # specific heat of ice [J Kg-1 K-1]
sigma = 5.67e-8                     # Stefan-Bolzmann constant [W m-2 K-4]
surface_emission_coeff = 0.97       # surface emision coefficient [-]
gravity_acceleration = 9.81         # acceleration of gravity (Braithwaite 1995) [m s-1]

snow_ice_threshold = 900.0          # pore close of density [kg m^(-3)]
ice_density = 917.                  # density of ice [kg m^(-3)]

zero_temperature = 273.16           # Kelvin [K]
water_density = 1000.0              # density of water [kg m^(-3)]
liquid_water_fraction = 0.05        # irreducible water content of a snow layer;
                                    #  fraction of total mass of the layer [%/100]
perolation_velocity = 0.0006        # percolation velocity for unsaturated layers [m s-1] (0.06 cm s-1)
                                    # how does it change with density?
                                    # Martinec, J.: Meltwater percolation through an alpine snowpack, Avalanche Formation,
                                    # Movement and Effects, Proceedings of the Davos Symposium, 162, 1987.

albedo_firn = 0.55                  # albedo of firn [-] (Moelg etal. 2012, TC)
albedo_fresh_snow = 0.90            # albedo of fresh snow [-] (Moelg etal. 2012, TC)
albedo_ice = 0.3                    # albedo of ice [-] (Moelg etal. 2012, TC)
albedo_mod_snow_aging = 6           # effect of ageing on snow albedo [days] (Moelg etal. 2012, TC)
albedo_mod_snow_depth = 8           # effect of snow depth on albedo [cm] (Moelg etal. 2012, TC)
roughness_fresh_snow = 0.24         # surface roughness length for fresh snow [mm] (Moelg etal. 2012, TC)
roughness_ice = 1.7                 # surface roughness length for ice [mm] (Moelg etal. 2012, TC)
roughness_firn = 4.0                # surface roughness length for aged snow [mm] (Moelg etal. 2012, TC)

density_fresh_snow = 250.           # density of freshly fallen snow [kg m-3]

' MODEL CONSTANTS '

number_layers = 10
initial_snowheight = 0.2           
initial_snow_layer_heights = 0.1       # Initial snow layer heights
initial_glacier_layer_heights = 1.0    # Initial glacier layer heights
initial_glacier_height = 20.0               # total domain height (snow+glacier)

