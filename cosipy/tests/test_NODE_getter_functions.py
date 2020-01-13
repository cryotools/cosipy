from cosipy.cpkernel.node import *

height = 0.1
density = 200.
temperature = 270
lwc = 0.2
ice_fraction = 0.4

test_node = Node(height, density, temperature, lwc)
test_node.set_layer_ice_fraction(ice_fraction)

def calculate_irreducible_water_content(current_ice_fraction):
    if (current_ice_fraction <= 0.23):
        theta_e = 0.0264 + 0.0099 * ((1 - current_ice_fraction) / current_ice_fraction)
    elif (current_ice_fraction > 0.23) & (current_ice_fraction <= 0.812):
        theta_e = 0.08 - 0.1023 * (current_ice_fraction - 0.03)
    else:
        theta_e = 0.0
    return theta_e

def test_node_getter_functions():
    assert test_node.get_layer_height() == height

    assert test_node.get_layer_temperature() == temperature

    assert test_node.get_layer_ice_fraction() == ice_fraction

    assert test_node.get_layer_refreeze() == 0.0

    assert test_node.get_layer_density() == test_node.get_layer_ice_fraction()*ice_density + \
           test_node.get_layer_liquid_water_content()*water_density + test_node.get_layer_air_porosity()*air_density

    assert test_node.get_layer_air_porosity() == 1 - lwc - ice_fraction

    assert test_node.get_layer_specific_heat() == (1 - lwc - ice_fraction) * spec_heat_air + ice_fraction * spec_heat_ice \
           + lwc * spec_heat_water

    assert test_node.get_layer_liquid_water_content() == lwc

    assert test_node.get_layer_irreducible_water_content() == calculate_irreducible_water_content(test_node.get_layer_ice_fraction())

    assert test_node.get_layer_cold_content() == - test_node.get_layer_specific_heat() * test_node.get_layer_density() \
           * height * (temperature - zero_temperature)

    assert test_node.get_layer_porosity() == 1 - test_node.get_layer_ice_fraction() - test_node.get_layer_liquid_water_content()

    assert test_node.get_layer_thermal_conductivity() == ice_fraction * k_i + test_node.get_layer_porosity() * k_a + lwc * k_w

    assert test_node.get_layer_thermal_diffusivity() == test_node.get_layer_thermal_conductivity() \
           / (test_node.get_layer_density() * test_node.get_layer_specific_heat())

def test_node_getter_functions_other_cases():
    test_node.set_layer_ice_fraction(0.1)
    assert test_node.get_layer_irreducible_water_content() == calculate_irreducible_water_content(test_node.get_layer_ice_fraction())

    test_node.set_layer_ice_fraction(0.9)
    assert test_node.get_layer_irreducible_water_content() == calculate_irreducible_water_content(test_node.get_layer_ice_fraction())
