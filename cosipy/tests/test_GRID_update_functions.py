from pytest import approx
from cosipy.cpkernel.grid import *
from COSIPY import start_logging

layer_heights = [0.1, 0.2, 0.3, 0.5, 0.5]
layer_densities = [250, 250, 250, 917, 917]
layer_temperatures = [260, 270, 271, 271.5, 272]
layer_liquid_water = [0.0, 0.0, 0.0, 0.0, 0.0]

def test_grid_getter_functions():
     GRID = Grid(layer_heights, layer_densities, layer_temperatures, layer_liquid_water)
     GRID.set_node_liquid_water_content(0, 0.04)
     GRID.set_node_liquid_water_content(1, 0.03)
     GRID.set_node_liquid_water_content(2, 0.03)
     GRID.set_node_liquid_water_content(3, 0.02)
     GRID.set_node_liquid_water_content(4, 0.01)

     SWE_before = np.array(GRID.get_height()) / np.array(GRID.get_density())
     SWE_before_sum = np.nansum(SWE_before)

     GRID.update_grid()
     SWE_after = np.array(GRID.get_height()) / np.array(GRID.get_density())
     SWE_after_sum = np.nansum(SWE_after)

     GRID.adaptive_profile()
     SWE_after_adaptive = np.array(GRID.get_height()) / np.array(GRID.get_density())
     SWE_after_adaptive_sum = np.nansum(SWE_after_adaptive)

     #assert SWE_before_sum == SWE_after_sum
     #assert SWE_after_sum == SWE_after_adaptive_sum

     assert SWE_before_sum == approx(SWE_after_sum, abs=1e-3)
     assert SWE_after_sum == approx(SWE_after_adaptive_sum, abs=1e-3)