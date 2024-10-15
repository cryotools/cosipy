from collections import OrderedDict

import numpy as np
from numba import float64
from numba.experimental import jitclass

from cosipy.constants import Constants

# only required for jitclass/njit
ice_density = Constants.ice_density
water_density = Constants.water_density
air_density = Constants.air_density
spec_heat_ice = Constants.spec_heat_ice
spec_heat_air = Constants.spec_heat_air
spec_heat_water = Constants.spec_heat_water
k_i = Constants.k_i
k_a = Constants.k_a
k_w = Constants.k_w
thermal_conductivity_method = Constants.thermal_conductivity_method
zero_temperature = Constants.zero_temperature

spec = OrderedDict()
spec["height"] = float64
spec["temperature"] = float64
spec["liquid_water_content"] = float64
spec["ice_fraction"] = float64
spec["refreeze"] = float64


@jitclass(spec)
class Node:
    """A ``Node`` class stores a layer's state variables.
    
    A node stores the information of an individual layer. The class
    provides various setter/getter functions to read or overwrite the
    state of an individual layer.

    Attributes:
        height (float): Layer height [m].
        snow_density (float): Layer snow density [|kg m^-3|].
        temperature (float): temperature [K].
        liquid_water_content (float): Liquid water content [|m w.e.|].
        ice_fraction (float): Volumetric ice fraction [-].
        refreeze (float): Amount of refrozen liquid water [|m w.e.|].
    """

    def __init__(
        self,
        height: float,
        snow_density: float,
        temperature: float,
        liquid_water_content: float,
        ice_fraction: float = None,
    ):
        # Initialises state variables.
        self.height = height
        self.temperature = temperature
        self.liquid_water_content = liquid_water_content

        if ice_fraction is None:
            # Remove weight of air from density
            a = snow_density - (1 - (snow_density / ice_density)) * air_density
            self.ice_fraction = a / ice_density
        else:
            self.ice_fraction = ice_fraction

        self.refreeze = 0.0


    """GETTER FUNCTIONS"""

    # -----------------------------------------
    # Getter-functions for state variables
    # -----------------------------------------
    def get_layer_height(self) -> float:
        """Get the node's layer height.
        
        Returns:
            Snow layer height [m].
        """
        return self.height

    def get_layer_temperature(self) -> float:
        """Get the node's snow layer temperature.
        
        Returns:
            Snow layer temperature [K].
        """
        return self.temperature

    def get_layer_ice_fraction(self) -> float:
        """Get the node's volumetric ice fraction.
        
        Returns:
            The volumetric ice fraction, |theta_i| [-].
        """
        return self.ice_fraction

    def get_layer_refreeze(self) -> float:
        """Get the amount of refrozen water in the node.
        
        Returns:
            Amount of refrozen water per time step [|m w.e.|].
        """
        return self.refreeze

    # ---------------------------------------------
    # Getter-functions for derived state variables
    # ---------------------------------------------
    def get_layer_density(self) -> float:
        """Get the node's mean density including ice and liquid.

        Returns:
            Layer density [|kg m^-3|].
        """
        return (
            self.get_layer_ice_fraction() * ice_density
            + self.get_layer_liquid_water_content() * water_density
            + self.get_layer_air_porosity() * air_density
        )

    def get_layer_air_porosity(self) -> float:
        """Get the fraction of air in the node.

        Returns:
            Air porosity, |phi_v| [-].
        """
        return max(0.0, 1 - self.get_layer_liquid_water_content() - self.get_layer_ice_fraction())

    def get_layer_specific_heat(self) -> float:
        """Get the node's volume-weighted specific heat capacity.

        Returns:
            Specific heat capacity [|J kg^-1 K^-1|].
        """
        return self.get_layer_ice_fraction()*spec_heat_ice + self.get_layer_air_porosity()*spec_heat_air + self.get_layer_liquid_water_content()*spec_heat_water

    def get_layer_liquid_water_content(self) -> float:
        """Get the node's liquid water content.

        Returns:
            Liquid water content, [m w.e].
        """
        return self.liquid_water_content

    def get_layer_irreducible_water_content(self) -> float:
        """Get the node's irreducible water content.

        Returns:
            Irreducible water content, |theta_e| [-].
        """

        ice_fraction = self.get_layer_ice_fraction()
        if ice_fraction <= 0.23:
            theta_e = 0.0264 + 0.0099 * ((1 - ice_fraction) / ice_fraction)
        elif (ice_fraction > 0.23) & (ice_fraction <= 0.812):
            theta_e = 0.08 - 0.1023 * (ice_fraction - 0.03)
        else:
            theta_e = 0.0
        return theta_e

    def get_layer_cold_content(self) -> float:
        """Get the node's cold content.

        Returns:
            Cold content [|J m^-2|].
        """
        return -self.get_layer_specific_heat() * self.get_layer_density() * self.get_layer_height() * (self.get_layer_temperature()-zero_temperature)

    def get_layer_porosity(self) -> float:
        """Get the node's porosity.

        Returns:
            Layer porosity, |phi| [-].
        """
        return 1-self.get_layer_ice_fraction()-self.get_layer_liquid_water_content()

    def get_layer_thermal_conductivity(self) -> float:
        """Get the node's volume-weighted thermal conductivity.

        Returns:
            Thermal conductivity, |kappa| [|W m^-1 K^-1|].
        """
        methods_allowed = ['bulk', 'empirical']
        if thermal_conductivity_method == 'bulk':
            kappa = self.get_layer_ice_fraction()*k_i + self.get_layer_air_porosity()*k_a + self.get_layer_liquid_water_content()*k_w
        elif thermal_conductivity_method == 'empirical':
            kappa = 0.021 + 2.5 * np.power((self.get_layer_density()/1000),2)
        else:
            message = ("Thermal conductivity method =",
                       f"{thermal_conductivity_method}",
                       f"is not allowed, must be one of",
                       f"{', '.join(methods_allowed)}")
            raise ValueError(" ".join(message))
        return kappa

    def get_layer_thermal_diffusivity(self) -> float:
        """Get the node's thermal diffusivity.

        Returns:
            Thermal diffusivity, |lambda| [|m^2 s^-1|].
        """
        lam = self.get_layer_thermal_conductivity()/(self.get_layer_density()*self.get_layer_specific_heat())
        return lam

    """SETTER FUNCTIONS"""

    # ---------------------------------------------
    # Setter-functions for derived state variables
    # ---------------------------------------------
    def set_layer_height(self, height: float):
        """Set the node's layer height.
        
        Args:
            height: Layer height [m].
        """
        self.height = height

    def set_layer_temperature(self, T: float):
        """Set the node's mean temperature.

        Args:
            T: Layer temperature [K].
        """
        self.temperature = T

    def set_layer_liquid_water_content(self, lwc: float):
        """Set the node's liquid water content.

        Args:
            lwc: Liquid water content [m w.e].
        """
        self.liquid_water_content = lwc

    def set_layer_ice_fraction(self, ifr: float):
        """Set the node's volumetric ice fraction.

        Args:
            ifr: Volumetric ice fraction, |theta_i| [-].
        """
        self.ice_fraction = ifr

    def set_layer_refreeze(self, refr: float):
        """Set the amount of refrozen water in the node.

        Args:
            refr: Amount of refrozen water [|m w.e.|].
        """
        self.refreeze = refr
