from pytest import approx
from cosipy.cpkernel.grid import *
from COSIPY import start_logging

layer_heights = [0.1, 0.2, 0.3, 0.5, 0.5]
layer_densities = [250, 250, 250, 917, 917]
layer_temperatures = [260, 270, 271, 271.5, 272]
layer_liquid_water = [0.0, 0.0, 0.0, 0.0, 0.0]

def test_grid_getter_functions():
     GRID = Grid(layer_heights, layer_densities, layer_temperatures, layer_liquid_water)
     assert GRID.get_height() == layer_heights
     assert GRID.get_density() == approx(layer_densities, abs=1e-3)
     assert GRID.get_temperature() == layer_temperatures
     assert GRID.get_liquid_water_content() == layer_liquid_water
     assert GRID.get_snow_heights() == layer_heights[0:3]
     assert GRID.get_ice_heights() == layer_heights[3:5]
     assert GRID.get_node_height(0) == layer_heights[0]
     assert GRID.get_node_density(0) == approx(layer_densities[0], abs=1e-3)
     assert GRID.get_node_temperature(0) == layer_temperatures[0]

    # GRID.remove_melt_weq(0.01)
    #########REMOVE_NODE
    #number_nodes_before = GRID.get_number_layers()
    #GRID.remove_node()
    #assert GRID.get_number_nodes() == number_nodes_before -1
    #assert np.nanmean(GRID.get_density()) == np.nanmean(layer_densities)
    #assert GRID.get_density() == layer_densities
