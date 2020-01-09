import os; import yaml; import logging
from cosipy.cpkernel.grid import *
from COSIPY import start_logging

layer_heights = [0.1, 0.2, 0.3, 0.5, 0.5]
layer_densities = [250, 250, 250, 917, 917]
layer_temperatures = [260, 270, 271, 271.5, 272]
layer_liquid_water = [0.0, 0.0, 0.0, 0.0, 0.0]
#
#start_logging()

#logging.basicConfig(level=logging.INFO)

def test_grid_getter_functions():
    GRID = Grid(layer_heights, layer_densities, layer_temperatures, layer_liquid_water)
    GRID.remove_melt_weq(0.01)

    #########REMOVE_NODE
    #number_nodes_before = GRID.get_number_layers()
    #GRID.remove_node()
    #assert GRID.get_number_nodes() == number_nodes_before -1


#    assert np.nanmean(GRID.get_density()) == np.nanmean(layer_densities)
#    assert GRID.get_density() == layer_densities
