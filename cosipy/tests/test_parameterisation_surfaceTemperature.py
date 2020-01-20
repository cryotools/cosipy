from pytest import approx
from cosipy.cpkernel.grid import *
from cosipy.modules.surfaceTemperature import update_surface_temperature
from COSIPY import start_logging


layer_heights = [0.1, 0.2, 0.3, 0.5, 0.5]
layer_densities = [250, 250, 250, 917, 917]
layer_temperatures = [260, 270, 271, 271.5, 272]
layer_liquid_water = [0.0, 0.0, 0.0, 0.0, 0.0]

GRID = Grid(layer_heights, layer_densities, layer_temperatures, layer_liquid_water)

def test_surface_Temperature_parameterisation():

    fun, surface_temperature, lw_radiation_in, lw_radiation_out, sensible_heat_flux, latent_heat_flux, \
    ground_heat_flux, sw_radiation_net, rho, Lv, Cs_t, Cs_q, q0, q2 \
        = update_surface_temperature(GRID, 0.6, (0.24/1000), 275, 0.6, 789, 1000, 4.5, 0.0, 0.1)

    assert surface_temperature <= zero_temperature and surface_temperature >= 220.0
    assert lw_radiation_in <= 400 and lw_radiation_in >= 0
    assert lw_radiation_out <= 0 and lw_radiation_out >= -400
    assert sensible_heat_flux <= 250 and sensible_heat_flux >= -250
    assert latent_heat_flux <= 200 and latent_heat_flux >= -200
    assert ground_heat_flux <= 100 and ground_heat_flux >= -100
    assert sw_radiation_net <= 1000 and sw_radiation_net >= 0
