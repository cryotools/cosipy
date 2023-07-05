import numpy as np
import pytest

import constants
from cosipy.cpkernel.node import Node


class TestNodeGetter:
    """Tests get methods for Node objects.

    Attributes:
        height (float): Layer height [:math:`m`]
        density (float): Snow density [:math:`kg~m^{-3}`]
        temperature (int): Layer temperature [:math:`K`]
        lwc (float): Liquid water content [:math:`m~w.e.`]
        ice_fraction (float): Volumetric ice fraction [-]
    """

    height = 0.1
    density = 200.0
    temperature = 270.0
    lwc = 0.2
    ice_fraction = 0.4

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

    def calculate_irreducible_water_content(
        self, current_ice_fraction: float
    ) -> float:
        """Calculate irreducible water content."""
        if current_ice_fraction <= 0.23:
            theta_e = 0.0264 + 0.0099 * (
                (1 - current_ice_fraction) / current_ice_fraction
            )
        elif (current_ice_fraction > 0.23) & (current_ice_fraction <= 0.812):
            theta_e = 0.08 - 0.1023 * (current_ice_fraction - 0.03)
        else:
            theta_e = 0.0

        return theta_e

    @pytest.mark.parametrize("arg_ice_fraction", [0.2, 0.5, 0.9])
    def test_calculate_irreducible_water_content(self, arg_ice_fraction):
        theta_e = self.calculate_irreducible_water_content(arg_ice_fraction)
        assert isinstance(theta_e, float)

    def test_node_getter_functions(self):
        node = self.create_node()
        assert np.isclose(node.get_layer_height(), self.height)
        assert np.isclose(node.get_layer_temperature(), self.temperature)
        assert np.isclose(node.get_layer_ice_fraction(), self.ice_fraction)
        assert np.isclose(node.get_layer_refreeze(), 0.0)

        test_density = (
            node.get_layer_ice_fraction() * constants.ice_density
            + node.get_layer_liquid_water_content() * constants.water_density
            + node.get_layer_air_porosity() * constants.air_density
        )
        assert np.isclose(node.get_layer_density(), test_density)
        assert np.isclose(
            node.get_layer_air_porosity(),
            1 - self.lwc - self.ice_fraction,
        )

        test_specific_heat = (
            (1 - self.lwc - self.ice_fraction) * constants.spec_heat_air
            + self.ice_fraction * constants.spec_heat_ice
            + self.lwc * constants.spec_heat_water
        )
        assert np.isclose(node.get_layer_specific_heat(), test_specific_heat)
        assert node.get_layer_liquid_water_content() == self.lwc

        test_irreducible_water_content = (
            self.calculate_irreducible_water_content(
                node.get_layer_ice_fraction()
            )
        )
        assert np.isclose(
            node.get_layer_irreducible_water_content(),
            test_irreducible_water_content,
        )

        test_cold_content = (
            -node.get_layer_specific_heat()
            * node.get_layer_density()
            * self.height
            * (self.temperature - constants.zero_temperature)
        )
        assert np.isclose(node.get_layer_cold_content(), test_cold_content)

        test_porosity = (
            1
            - node.get_layer_ice_fraction()
            - node.get_layer_liquid_water_content()
        )
        assert np.isclose(node.get_layer_porosity(), test_porosity)

        test_thermal_conductivity = (
            self.ice_fraction * constants.k_i
            + node.get_layer_porosity() * constants.k_a
            + self.lwc * constants.k_w
        )
        assert np.isclose(
            node.get_layer_thermal_conductivity(),
            test_thermal_conductivity,
        )

        test_thermal_diffusivity = node.get_layer_thermal_conductivity() / (
            node.get_layer_density() * node.get_layer_specific_heat()
        )
        assert np.isclose(
            node.get_layer_thermal_diffusivity(), test_thermal_diffusivity
        )

    @pytest.mark.parametrize("arg_ice_fraction", [0.1, 0.9])
    def test_node_getter_functions_other_cases(self, arg_ice_fraction):
        node = self.create_node()
        node.set_layer_ice_fraction(arg_ice_fraction)

        test_irreducible_water_content = (
            self.calculate_irreducible_water_content(
                node.get_layer_ice_fraction()
            )
        )
        assert np.isclose(
            node.get_layer_irreducible_water_content(),
            test_irreducible_water_content,
        )
