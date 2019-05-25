"""
    Declaration of constants
    Do not modify unless you are absolutely sure what you are doing.
"""

'Choose parameterisations'
albedo_method = 'Oerlemans98'                   # possibilities: 'Oerlemans98'
densification_method = 'Herron80'               # possibilities: 'Herron80'
penetrating_method = 'Bintanja95'               # possibilities: 'Bintanja95'
roughness_method = 'Moelg12'                    # possibilities: 'Moelg12'
saturation_water_vapour_method = 'Sonntag90'    # possibilities: 'Sonntag90'

' Inital constants'
initial_snowheight_constant = 0.8                        # Inital snowheigt
initial_snow_layer_heights = 0.05               # Initial thickness of snow layers
initial_glacier_height = 40.0                   # Inital glacier heigt without snowlayers
initial_glacier_layer_heights = 1.0             # Initial thickness of glacier ice layers

initial_top_density_snowpack = 300.             # Top density for inital snowpack
initial_botton_density_snowpack = 600.          # Botton density for inital snowpack

temperature_top_constant = 268.15                        # Upper boudary conditation for inital temperature profile (K)
temperature_bottom = 270.15                       # Lower boundary condition for inital tempeature profile (K)
const_init_temp = 0.1                           # constant for init temperature profile used in exponential function (exponential decay)

'Remeshing options'
first_layer_height = 0.02                       # The first layer will always have the defined height (m)
layer_stretching = 1.5                          # Stretching factor used by the log_profile method (e.g. 1.1 mean the subsequent layer is 10% greater than the previous

minimum_snow_to_reset_albedo = 0.01             # minimum snowfall to reset hours since last snowfall! Default was 0.005

density_fresh_snow = 250.                       # density of freshly fallen snow [kg m-3]

albedo_fresh_snow = 0.90                        # albedo of fresh snow [-] (Moelg etal. 2012, TC)
albedo_firn = 0.55                              # albedo of firn [-] (Moelg etal. 2012, TC)
albedo_ice = 0.3                                # albedo of ice [-] (Moelg etal. 2012, TC)
albedo_mod_snow_aging = 22.                     # effect of ageing on snow albedo [days] (Moelg etal. 2012, TC)
albedo_mod_snow_depth = 1.                      # effect of snow depth on albedo [cm] (Moelg etal. 2012, TC)
roughness_fresh_snow = 0.24                     # surface roughness length for fresh snow [mm] (Moelg etal. 2012, TC)
roughness_ice = 1.7                             # surface roughness length for ice [mm] (Moelg etal. 2012, TC)
roughness_firn = 4.0                            # surface roughness length for aged snow [mm] (Moelg etal. 2012, TC)
aging_factor_roughness = 0.0026                 # effect of ageing on roughness lenght (hours) 60 days from 0.24 to 4.0 => 0.0026

surface_emission_coeff = 0.97                   # surface emision coefficient [-]

snow_ice_threshold = 900.0                      # pore close of density [kg m^(-3)]
snow_firn_threshold = 555.0                     #

center_snow_transfer_function = 2.5             # center (50/50) when total precipitation is transfered to snow and rain
spread_snow_transfer_function = 2.5             # 2: +-2.5; 2: +- 1.5; 3: +- 0.5

' PHYSICAL CONSTANTS '

lat_heat_melting = 3.34e5                       # latent heat for melting [J kg-1]
lat_heat_vaporize = 2.5e6                       # latent heat for vaporization [J kg-1]
lat_heat_sublimation = 2.834e6                  # latent heat for sublimation [J kg-1]
spec_heat_air = 1004.67                         # specific heat of air [J kg-1 K-1]
spec_heat_ice = 2050.00                         # specific heat of ice [J Kg-1 K-1]
spec_heat_water = 4217.00                       # specific heat of water [J Kg-1 K-1]
sigma = 5.67e-8                                 # Stefan-Bolzmann constant [W m-2 K-4]

' MODEL CONSTANTS '
water_density = 1000.0                          # density of water [kg m^(-3)]
ice_density = 917.                              # density of ice [kg m^(-3)]
air_density = 1.1                               # density of air [kg m^(-3)]

zero_temperature = 273.16                       # Kelvin [K]

' Densification constants '
K0   = 11                                       # rate factors [-]
K1   = 575
E0   = 10260                                    # activation energy
E1   = 21400
R    = 8.3144                                   # universal gas constant [J K-1 mol-1]
