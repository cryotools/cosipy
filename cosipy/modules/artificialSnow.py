
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from constants import *
from config import *
import sys


def artificialSnow(WET_BULB_TEMPERATURE, U2, SNOWFALL, density_fresh_snow, water_consumption_sum):
    artificial_snow_allowed = ['constant', 'linear']
    if artificial_snow_method == 'constant':
        SNOWFALL, ARTIFICIAL_SNOW, WATER_CONSUMPTION, output_density_artificial_snow, density_fresh_and_artificial_snow, water_consumption_sum = method_constant(WET_BULB_TEMPERATURE, U2, SNOWFALL, density_fresh_snow, water_consumption_sum)
    elif artificial_snow_method == 'linear':
        SNOWFALL, ARTIFICIAL_SNOW, WATER_CONSUMPTION, output_density_artificial_snow, density_fresh_and_artificial_snow, water_consumption_sum = method_linear(WET_BULB_TEMPERATURE, U2, SNOWFALL, density_fresh_snow, water_consumption_sum)
        pass
    else:
        raise ValueError("Artificial Snow method = \"{:s}\" is not allowed, must be one of {:s}".format(artificial_snow_method, ", ".join(artificial_snow_allowed)))
    return SNOWFALL, ARTIFICIAL_SNOW, WATER_CONSUMPTION, output_density_artificial_snow, density_fresh_and_artificial_snow, water_consumption_sum

def method_constant(WET_BULB_TEMPERATURE, U2, SNOWFALL, density_fresh_snow, water_consumption_sum):
    """ Description: add constant amount of artificial snow at each grid point
        with one condition for wet bulb temperature and wind speed
    """
    
    if (WET_BULB_TEMPERATURE <= wet_bulb_temperature_artificial_snow) & (U2 >= wind_speed_artificial_snow):
        if (use_density_artificial_snow):
            # calculate new density for fresh snow weighted with the mass of SNOWFALL and ARTIFICIAL_SNOW
            ARTIFICIAL_SNOW = mass_artificial_snow / density_artificial_snow
            mass_fresh_snow = SNOWFALL * density_fresh_snow
            mass_all_snow = mass_artificial_snow + mass_fresh_snow
            density_fresh_and_artificial_snow = (mass_artificial_snow/mass_all_snow)*density_artificial_snow + (mass_fresh_snow/mass_all_snow)*density_fresh_snow
            output_density_artificial_snow = density_artificial_snow
        else:
            ARTIFICIAL_SNOW = mass_artificial_snow / density_fresh_snow
            density_fresh_and_artificial_snow = density_fresh_snow
            output_density_artificial_snow = density_fresh_snow
        SNOWFALL = SNOWFALL + ARTIFICIAL_SNOW
        WATER_CONSUMPTION = mass_artificial_snow * 0.001
    else:
        ARTIFICIAL_SNOW = 0.0
        WATER_CONSUMPTION = 0.0
        output_density_artificial_snow = 0.0
        density_fresh_and_artificial_snow = density_fresh_snow
    water_consumption_sum = water_consumption_sum +  WATER_CONSUMPTION
    
    return SNOWFALL, ARTIFICIAL_SNOW, WATER_CONSUMPTION, output_density_artificial_snow, density_fresh_and_artificial_snow, water_consumption_sum

def method_linear(WET_BULB_TEMPERATURE, U2, SNOWFALL, density_fresh_snow, water_consumption_sum):
    """Descripton: add linear fitted amount of snow in between two threshold values
       two conditions for 0% and 100% of snow in between linear fit for wetbulb temperatur and wind speed
       The conditions for artificial snow production are defined by the glacier simulation matlab modell 
    """
    if(WET_BULB_TEMPERATURE <= wet_bulb_temperature_artificial_snow_100_perc) & (U2 >= wind_speed_artificial_snow_100_perc):
        factor_artificial_snow = 1
    elif(WET_BULB_TEMPERATURE >= wet_bulb_temperature_artificial_snow_0_perc) | (U2 <= wind_speed_artificial_snow_0_perc):
        factor_artificial_snow = 0
    else:
        if(WET_BULB_TEMPERATURE > wet_bulb_temperature_artificial_snow_100_perc):
            factor_wet_bulb_temperature = factor_wet_bulb_temperature_linear
        else:
            factor_wet_bulb_temperature = 1
        if(U2 < wind_speed_artificial_snow_100_perc):
            factor_wind_speed = factor_windspeed_linear
        else:
            factor_wind_speed = 1
        factor_artificial_snow = factor_wet_bulb_temperature * factor_wind_speed

    mass_artificial_snow_linear = mass_artificial_snow * factor_artificial_snow
    if(mass_artificial_snow_linear > 0.0):
        if(use_density_artificial_snow):
            ARTIFICIAL_SNOW = mass_artificial_snow_linear / density_artificial_snow
            mass_fresh_snow = SNOWFALL * density_fresh_snow
            mass_all_snow = mass_artificial_snow_linear + mass_fresh_snow
            density_fresh_and_artificial_snow = (mass_artificial_snow_linear/mass_all_snow)*density_artificial_snow + (mass_fresh_snow/mass_all_snow)*density_fresh_snow
            output_density_artificial_snow = density_artificial_snow
        else:
            ARTIFICIAL_SNOW = mass_artificial_snow_linear / density_fresh_snow
            output_density_artificial_snow = density_fresh_snow
            density_fresh_and_artificial_snow = density_fresh_snow
    else:
        ARTIFICIAL_SNOW = 0.0
        WATER_CONSUMPTION = 0.0
        output_density_artificial_snow = 0.0
        density_fresh_and_artificial_snow = density_fresh_snow

    SNOWFALL = SNOWFALL + ARTIFICIAL_SNOW
    WATER_CONSUMPTION = mass_artificial_snow_linear * 0.001
    water_consumption_sum = water_consumption_sum +  WATER_CONSUMPTION

    return SNOWFALL, ARTIFICIAL_SNOW, WATER_CONSUMPTION, output_density_artificial_snow, density_fresh_and_artificial_snow, water_consumption_sum
