from pytest import approx
from cosipy.cpkernel.grid import *
from cosipy.modules.percolation import percolation
from COSIPY import start_logging

layer_heights = [0.1, 0.2, 0.3, 0.5, 0.5]
layer_densities = [250, 250, 250, 917, 917]
layer_temperatures = [260, 270, 271, 271.5, 272]
layer_liquid_water = [0.0, 0.0, 0.0, 0.0, 0.0]

GRID = Grid(layer_heights, layer_densities, layer_temperatures, layer_liquid_water)
melt_water = 1.0
timestamp = 7200

def test_percolation_parameterisation():
    q = percolation(GRID, melt_water, timestamp)
    assert melt_water == np.nansum(GRID.get_liquid_water_content()) + q