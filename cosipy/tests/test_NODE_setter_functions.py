# import constants
from cosipy.cpkernel.node import Node


class TestNodeSetter:
    """Tests set methods for Node objects.

    Attributes:
        height (float): Layer height [:math:`m`].
        density (float): Snow density [:math:`kg~m^{-3}`].
        temperature (int): Layer temperature [:math:`K`].
        lwc (float): Liquid water content [:math:`m~w.e.`].
        ice_fraction (float): Volumetric ice fraction [-].
        new_height (float): New layer height [:math:`m`].
        new_temperature (int): New layer temperature [:math:`K`].
        new_lwc (float): New liquid water content [:math:`m~w.e.`].
        new_ice_fraction (float): New volumetric ice fraction [-].
    """

    height = 0.1
    density = 200.0
    temperature = 270
    lwc = 0.3
    ice_fraction = 0.4
    new_height = 1.0
    new_temperature = 260.0
    new_lwc = 0.5
    new_ice_fraction = 0.9
    new_refreeze = 0.01

    def create_node(
        self,
        height: float = height,
        density: float = density,
        temperature: float = temperature,
        lwc: float = lwc,
        ice_fraction: float = ice_fraction,
    ) -> Node:
        """Instantiate a Node."""

        node = Node(
            height=height,
            snow_density=density,
            temperature=temperature,
            liquid_water_content=lwc,
            ice_fraction=ice_fraction,
        )
        return node

    def test_create_node(self, conftest_boilerplate):
        node = self.create_node()
        assert isinstance(node, Node)
        conftest_boilerplate.check_output(node.height, float, self.height)
        conftest_boilerplate.check_output(
            node.temperature, float, self.temperature
        )
        conftest_boilerplate.check_output(
            node.liquid_water_content, float, self.lwc
        )
        conftest_boilerplate.check_output(
            node.ice_fraction, float, self.ice_fraction
        )
        conftest_boilerplate.check_output(node.refreeze, float, 0.0)

    def test_node_set_layer_height(self, conftest_boilerplate):
        node = self.create_node()
        node.set_layer_height(self.new_height)
        conftest_boilerplate.check_output(
            node.get_layer_height(), float, self.new_height
        )

    def test_node_set_layer_temperature(self, conftest_boilerplate):
        node = self.create_node()
        node.set_layer_temperature(self.new_temperature)
        conftest_boilerplate.check_output(
            node.get_layer_temperature(), float, self.new_temperature
        )

    def test_node_set_layer_liquid_water_content(self, conftest_boilerplate):
        node = self.create_node()
        node.set_layer_liquid_water_content(self.new_lwc)
        conftest_boilerplate.check_output(
            node.get_layer_liquid_water_content(), float, self.new_lwc
        )

    def test_node_set_layer_ice_fraction(self, conftest_boilerplate):
        node = self.create_node()
        node.set_layer_ice_fraction(self.new_ice_fraction)
        conftest_boilerplate.check_output(
            node.get_layer_ice_fraction(), float, self.new_ice_fraction
        )

    def test_node_set_layer_refreeze(self, conftest_boilerplate):
        node = self.create_node()
        node.set_layer_refreeze(self.new_refreeze)
        conftest_boilerplate.check_output(
            node.get_layer_refreeze(), float, self.new_refreeze
        )
