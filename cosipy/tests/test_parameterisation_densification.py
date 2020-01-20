from pytest import approx
from cosipy.modules.densification import densification
from cosipy.cpkernel.grid import *
from COSIPY import start_logging

layer_heights = [10.0, 10.0, 10.0, 0.2, 0.2]
layer_densities = [450, 450, 450, 100, 100]
layer_temperatures = [260, 270, 271, 271.5, 272]
layer_liquid_water = [0.0, 0.0, 0.0, 0.0, 0.0]

GRID = Grid(layer_heights, layer_densities, layer_temperatures, layer_liquid_water)
#
def test_densification_parameterisation():
    SWE_before = np.array(GRID.get_height()) / np.array(GRID.get_density())
    SWE_before_sum = np.nansum(SWE_before)

    densification(GRID, 0.0)

    SWE_after = np.array(GRID.get_height()) / np.array(GRID.get_density())
    SWE_after_sum = np.nansum(SWE_after)

    #assert SWE_before_sum == approx(SWE_after_sum, abs=1e-3)
    #assert SWE_before_sum == SWE_after_sum
