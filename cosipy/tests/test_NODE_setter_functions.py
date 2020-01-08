from cosipy.cpkernel.node import *
height = 0.1
density = 200.
temperature = 270
lwc = 0.3
ice_fraction = 0.4
test_node = Node(height, density, temperature, lwc, ice_fraction)

new_height = 1.
new_temperature = 260.
new_liquidwater_content = 0.5
new_ice_fraction = 0.9
new_refreeze = 0.01

def test_node_setter_functions():

    test_node.set_layer_height(new_height)
    assert test_node.get_layer_height() == new_height

    test_node.set_layer_temperature(new_temperature)
    assert test_node.get_layer_temperature() == new_temperature

    test_node.set_layer_liquid_water_content(new_liquidwater_content)
    assert test_node.get_layer_liquid_water_content() == new_liquidwater_content

    test_node.set_layer_ice_fraction(new_ice_fraction)
    assert test_node.get_layer_ice_fraction() == new_ice_fraction

    test_node.set_layer_refreeze(new_refreeze)
    assert test_node.get_layer_refreeze() == new_refreeze