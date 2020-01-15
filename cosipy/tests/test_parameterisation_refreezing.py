from pytest import approx
from cosipy.cpkernel.grid import *
from cosipy.modules.refreezing import refreezing
from COSIPY import start_logging

layer_heights = [0.1, 0.2, 0.3, 0.5, 0.5]
layer_densities = [250, 250, 250, 917, 917]
layer_temperatures = [260, 270, 271, 271.5, 272]
layer_liquid_water = [0.01, 0.01, 0.01, 0.01, 0.01]

GRID = Grid(layer_heights, layer_densities, layer_temperatures, layer_liquid_water)

def test_refreezing_parametrisation():
    water_content_before = np.nansum(GRID.get_liquid_water_content())
    water_refreezed = refreezing(GRID)
    #assert water_content_before == np.nansum(GRID.get_liquid_water_content()) + water_refreezed