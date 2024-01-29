from collections import OrderedDict

import numpy as np
from numba import float64
from numba.experimental import jitclass

import constants

spec = OrderedDict()
spec["height"] = float64
spec["temperature"] = float64
spec["liquid_water_content"] = float64
spec["ice_fraction"] = float64
spec["refreeze"] = float64

@jitclass(spec)
class Node:
    """A `Node` class stores a layer's state variables.
    
    The numerical grid consists of a list of nodes which store the
    information of individual layers. The class provides various
    setter/getter functions to read or overwrite the state of these
    individual layers. 

    Attributes
    ----------
    height : float
        Layer height [:math:`m`].
    snow_density : float
        Layer snow density [:math:`kg~m^{-3}`].
    temperature: float
        Layer temperature [:math:`K`].
    liquid_water_content : float
        Liquid water content [:math:`m~w.e.`].
    ice_fraction : float
        Volumetric ice fraction [-].
    refreeze : float
        Amount of refrozen liquid water [:math:`m~w.e.`].

    Returns
    -------
    Node : :py:class:`cosipy.cpkernel.node` object.
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
            a = snow_density - (1 - (snow_density / constants.ice_density)) * constants.air_density
            self.ice_fraction = a / constants.ice_density
        else:
            self.ice_fraction = ice_fraction

        self.refreeze = 0.0


    """GETTER FUNCTIONS"""

    # -----------------------------------------
    # Getter-functions for state variables
    # -----------------------------------------
    def get_layer_height(self) -> float:
        """Gets the node's layer height.
        
        Returns
        -------
        height : float
            Snow layer height [:math:`m`].
        """
        return self.height

    def get_layer_temperature(self) -> float:
        """Gets the node's snow layer temperature.
        
        Returns
        -------
        T : float
            Snow layer temperature [:math:`K`].
        """
        return self.temperature

    def get_layer_ice_fraction(self) -> float:
        """Gets the node's volumetric ice fraction.
        
        Returns
        -------
        ice_fraction : float
            The volumetric ice fraction [-].
        """
        return self.ice_fraction

    def get_layer_refreeze(self) -> float:
        """Gets the amount of refrozen water in the node.
        
        Returns
        -------
        refreeze : float
            Amount of refrozen water per time step [:math:`m~w.e.`].
        """
        return self.refreeze

    # ---------------------------------------------
    # Getter-functions for derived state variables
    # ---------------------------------------------
    def get_layer_density(self) -> float:
        """Gets the node's mean density including ice and liquid.

        Returns
        -------
        rho : float
            Snow density [:math:`kg~m^{-3}`].
        """
        return (
            self.get_layer_ice_fraction() * constants.ice_density
            + self.get_layer_liquid_water_content() * constants.water_density
            + self.get_layer_air_porosity() * constants.air_density
        )

    def get_layer_air_porosity(self) -> float:
        """Gets the fraction of air in the node.

        Returns
        -------
        porosity : float
            Air porosity [:math:`m`].
        """
        return max(0.0, 1 - self.get_layer_liquid_water_content() - self.get_layer_ice_fraction())

    def get_layer_specific_heat(self) -> float:
        """Gets the node's volumetric averaged specific heat capacity.

        Returns
        -------
        cp : float
            Specific heat capacity [:math:`J~kg^{-1}~K^{-1}`].
        """
        return self.get_layer_ice_fraction()*constants.spec_heat_ice + self.get_layer_air_porosity()*constants.spec_heat_air + self.get_layer_liquid_water_content()*constants.spec_heat_water

    def get_layer_liquid_water_content(self) -> float:
        """Gets the node's liquid water content.

        Returns
        -------
        lwc : float
            Liquid water content [-].
        """
        return self.liquid_water_content

    def get_layer_irreducible_water_content(self) -> float:
        """Gets the node's irreducible water content.

        Returns
        -------
        theta_e : float
            Irreducible water content [-].
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
        """Gets the node's cold content.

        Returns
        -------
        cc : float
            Cold content [:math:`J~m^{-2}`].
        """
        return -self.get_layer_specific_heat() * self.get_layer_density() * self.get_layer_height() * (self.get_layer_temperature()-constants.zero_temperature)

    def get_layer_porosity(self) -> float:
        """Gets the node's porosity.

        Returns
        -------
        porosity : float
            Air porosity [-].
        """
        return 1-self.get_layer_ice_fraction()-self.get_layer_liquid_water_content()

    def get_layer_thermal_conductivity(self) -> float:
        """Gets the node's volumetric weighted thermal conductivity.

        Returns
        -------
        lam : float
            Thermal conductivity [:math:`W~m^{-1}~K^{-1}`].
        """
        methods_allowed = ['bulk', 'empirical']
        if constants.thermal_conductivity_method == 'bulk':
            lam = self.get_layer_ice_fraction()*constants.k_i + self.get_layer_air_porosity()*constants.k_a + self.get_layer_liquid_water_content()*constants.k_w
        elif constants.thermal_conductivity_method == 'empirical':
            lam = 0.021 + 2.5 * np.power((self.get_layer_density()/1000),2)
        else:
            message = ("Thermal conductivity method =",
                       f"{constants.thermal_conductivity_method}",
                       f"is not allowed, must be one of",
                       f"{', '.join(methods_allowed)}")
            raise ValueError(" ".join(message))
        return lam

    def get_layer_thermal_diffusivity(self) -> float:
        """Gets the node's thermal diffusivity.

        Returns
        -------
        K : float
            Thermal diffusivity [:math:`m^{2}~s^{-1}`].
        """
        K = self.get_layer_thermal_conductivity()/(self.get_layer_density()*self.get_layer_specific_heat())
        return K

    """SETTER FUNCTIONS"""

    # ---------------------------------------------
    # Setter-functions for derived state variables
    # ---------------------------------------------
    def set_layer_height(self, height: float):
        """Sets the node's layer height.
        
        Parameters
        ----------
        height : float
            Layer height [:math:`m`].
        """
        self.height = height

    def set_layer_temperature(self, T: float):
        """Sets the node's mean temperature.

        Parameters
        ----------
        T : float
            Layer temperature [:math:`K`].
        """
        self.temperature = T

    def set_layer_liquid_water_content(self, lwc: float):
        """Sets the node's liquid water content.

        Parameters
        ----------
        lwc : float
            Liquid water content [-].
        """
        self.liquid_water_content = lwc

    def set_layer_ice_fraction(self, ifr: float):
        """Sets the node's volumetric ice fraction.

        Parameters
        ----------
        ifr : float
            Volumetric ice fraction [-].
        """
        self.ice_fraction = ifr

    def set_layer_refreeze(self, refr: float):
        """Sets the amount of refrozen water in the node.

        Parameters
        ----------
        refr : float
            Amount of refrozen water [:math:`m~w.e.`].
        """
        self.refreeze = refr
