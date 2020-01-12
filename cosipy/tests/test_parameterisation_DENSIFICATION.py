from cosipy.modules.densification import densification
from cosipy.cpkernel.grid import *
from COSIPY import start_logging

layer_heights = [0.1, 0.2, 0.3, 0.5, 0.5]
layer_densities = [250, 250, 250, 917, 917]
layer_temperatures = [260, 270, 271, 271.5, 272]
layer_liquid_water = [0.0, 0.0, 0.0, 0.0, 0.0]
#
GRID = Grid(layer_heights, layer_densities, layer_temperatures, layer_liquid_water)
GRID_ice = Grid(layer_heights[3:4], layer_densities[3:4], layer_temperatures[3:4],layer_liquid_water[3:4])

def test_densification_parameterisation():
    GRID.grid_info_screen()
    densification(GRID,0.0)
    GRID.grid_info_screen()

