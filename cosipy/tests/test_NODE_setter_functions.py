import numpy as np

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
        refreeze (float): Amount of refreezed water [:math:`m~w.e.`].
        new_height (float): New layer height [:math:`m`].
        new_density (float): New snow density [:math:`kg~m^{-3}`].
        new_temperature (int): New layer temperature [:math:`K`].
        new_lwc (float): New liquid water content [:math:`m~w.e.`].
        new_ice_fraction (float): New volumetric ice fraction [-].

    """

    height = 0.1
    density = 200.0
    temperature = 270
    lwc = 0.3
    ice_fraction = 0.4
    refreeze = 0.01
    new_height = 1.0
    new_temperature = 260.0
    new_lwc = 0.5
    new_ice_fraction = 0.9

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
        )
        assert isinstance(node, Node)
        node.set_layer_ice_fraction(ice_fraction)

        return node

    def test_create_node(self):
        node = self.create_node()
        assert isinstance(node, Node)

    def test_node_setter_functions(self):
        node = self.create_node()
        node.set_layer_height(self.new_height)
        assert np.isclose(node.get_layer_height(), self.new_height)

        node.set_layer_temperature(self.new_temperature)
        assert np.isclose(node.get_layer_temperature(), self.new_temperature)

        node.set_layer_liquid_water_content(self.new_lwc)
        assert np.isclose(node.get_layer_liquid_water_content(), self.new_lwc)

        node.set_layer_ice_fraction(self.new_ice_fraction)
        assert np.isclose(node.get_layer_ice_fraction(), self.new_ice_fraction)

        node.set_layer_refreeze(self.refreeze)
        assert np.isclose(node.get_layer_refreeze(), self.refreeze)
