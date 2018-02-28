"""
 This file reads the input data (model forcing) and write the output to netcdf file
"""

import xarray as xr
import time
from config import input_netcdf, output_netcdf, plots
from postprocessing_modules.plots import plots_fluxes
import numpy as np

def read_input():
    DATA = xr.open_dataset(input_netcdf)
    air_pressure = DATA.p.values                # Air Pressure [hPa]
    cloud_cover = DATA.N.values                 # Cloud cover  [fraction][%/100]
    initial_snow_height = DATA.sh.values        # Initial snow height [m]
    relative_humidity = DATA.rH2.values         # Relative humidity (2m over ground)[%]
    snowfall = DATA.snowfall.values             # Snowfall per time step [m]
    solar_radiation = DATA.G.values             # Solar radiation at each time step [W m-2]
    temperature_2m = DATA.T2.values             # Air temperature (2m over ground) [K]
    wind_speed = DATA.u2.values                 # Wind speed (magnitude) m/s

    '''
    only for local ldevelopment
    '''

    return air_pressure, cloud_cover, initial_snow_height, relative_humidity, snowfall, solar_radiation, \
                temperature_2m, wind_speed

def write_output_1D(albedo, condensation, deposition, evaporation, ground_heat_flux, longwave_in, longwave_out, \
                 latent_heat_flux, mass_balance, melt_height, number_layers, refreezing, sensible_heat_flux, \
                 snowHeight, shortwave_net, sublimation, subsurface_melt, surface_melt, surface_temperature, u2, \
                 sw_in, T2, rH2, snowfall, pressure, cloud, sh, rho, Lv, Cs, q0, q2, qdiff, phi, cpi):
    COSIPY_output = xr.Dataset({
                            "albedo": albedo,
                            "condensation": condensation,
                            "deposition": deposition,
                            "evaporation": evaporation,
                            "ground_heat_flux": ground_heat_flux,
                            "longwave_in": longwave_in,
                            "longwave_out": longwave_out,
                            "latent_heat_flux": latent_heat_flux,
                            "mass_balance" : mass_balance,
                            "melt_height" : melt_height,
                            "number_layers" : number_layers,
                            # "runoff" : runoff,
                            "refreezing": refreezing,
                            "sensible_heat_flux" : sensible_heat_flux,
                            "snowHeight" : snowHeight,
                            "shortwave_net" : shortwave_net,
                            "sublimation" : sublimation,
                            "subsurface_melt": subsurface_melt,
                            "surface_melt" : surface_melt,
                            "surface_temperature" : surface_temperature,

                            ###input variables save in same file for comparision; not needed in longtime???
                            "u2" : u2,
                            "sw_in" : sw_in,
                            "T2" : T2,
                            "rH2": rH2,
                            "snowfall" : snowfall,
                            "pressure": pressure,
                            "cloud" : cloud,
                            "sh" : sh,

                            ###interim for investigations; not needed in longterm???
                            "rho" : rho,
                            "Lv" : Lv,
                            "Cs" : Cs,
                            "q0" : q0,
                            "q2" : q2,
                            "qdiff" : qdiff,
                            "phi" : phi,
                            "cpi": cpi,
    })
    timestr = time.strftime("%Y%m%d-%H%M%S")
    COSIPY_output.to_netcdf(output_netcdf+'-'+timestr+'.nc')
    if plots == 1:
        plots_fluxes(COSIPY_output,timestr)


### TODO netcdf Dataset has to have attributres!!!

'''
    data.attrs['TITLE'] = 'COSIPY 1D results'
    data.attrs['CREATION_DATE'] = str(today)
    data.to_netcdf(output_netcdf)
'''

def write_output_distributed(albedo, condensation, deposition, evaporation, ground_heat_flux, longwave_in, longwave_out, \
                 latent_heat_flux, melt_height, number_layers, sensible_heat_flux, snowHeight, shortwave_net, \
                 sublimation, surface_melt, surface_temperature):
    COSIPY_output = xr.Dataset({
                            "albedo": albedo,
                            "condensation": condensation,
                            "deposition": deposition,
                            "evaporation": evaporation,
                            "ground_heat_flux": ground_heat_flux,
                            "longwave_in": longwave_in,
                            "longwave_out": longwave_out,
                            "latent_heat_flux": latent_heat_flux,
                            # "mass_balance" : mass_balance,
                            "melt_height" : melt_height,
                            "number_layers" : number_layers,
                            # "runoff" : runoff,
                            "sensible_heat_flux" : sensible_heat_flux,
                            "snowHeight" : snowHeight,
                            "shortwave_net" : shortwave_net,
                            "sublimation" : sublimation,
                            "surface_melt" : surface_melt,
                            "surface_temperature" : surface_temperature
                             })
    timestr = time.strftime("%Y%m%d-%H%M%S")
    print("Timestamp: ",timestr)
    COSIPY_output.to_netcdf(output_netcdf+'-'+timestr+'.nc')

