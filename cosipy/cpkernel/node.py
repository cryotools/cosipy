from constants import *
import numpy as np

from collections import OrderedDict
from numba import float64  
from numba.experimental import jitclass

spec = OrderedDict()
spec['height'] = float64              
spec['temperature'] = float64     
spec['liquid_water_content'] = float64     
spec['ice_fraction'] = float64
spec['refreeze'] = float64

@jitclass(spec)
class Node:
    """ The Node-class stores the state variables of layer. 
    
    The numerical grid consists of a list of nodes that store the information 
    of individual layers. The class provides various setter/getter functions
    to read or overwrite the state of individual layers. 

    Parameters
    ----------
        height : float
            Height of the layer [:math:`m`]
        density : float
            snow density [:math:`kg~m^{-3}`]
        temperature : float
            temperature of the layer [:math:`K`]
        liquid_water : float
            liquid water [:math:`m~w.e.`]

    Returns
    -------
        Node : :py:class:`cosipy.cpkernel.node` object

    """

    def __init__(self, height, snow_density, temperature, liquid_water_content, ice_fraction=None):

        # Initialize state variables 
        self.height = height
        self.temperature = temperature
        self.liquid_water_content = liquid_water_content
        
        if ice_fraction is None:
            # Remove weight of air from density
            a = snow_density - (1-(snow_density/ice_density))*air_density
            self.ice_fraction = a/ice_density
        else:
            self.ice_fraction = ice_fraction

        self.refreeze = 0.0 


    ''' GETTER FUNCTIONS '''
    
    #------------------------------------------
    # Getter-functions for state variables
    #------------------------------------------
    def get_layer_height(self):
        """ Returns the layer height of the node.
        
        Returns
        -------
            height : float
                Snow layer height [:math:`m`]
        """
        return self.height

    def get_layer_temperature(self):
        """ Returns the snow layer temperature of the node. 
        
        Returns
        -------
            T : float
                Snow layer temperature [:math:`K`]
        """
        return self.temperature
    
    def get_layer_ice_fraction(self):
        """ Returns the volumetric ice fraction of the node. 
        
        Returns
        -------
            ice_fraction : float
                The volumetric ice fraction [-] 
        """
        return self.ice_fraction 
    
    def get_layer_refreeze(self):
        """ Returns the amount of refreezing of the node. 
        
        Returns
        -------
            refreeze : float
                Amount of water that has refreezed per time step [:math:`m~w.e.`]
        """
        return self.refreeze


    #----------------------------------------------
    # Getter-functions for derived state variables
    #----------------------------------------------
    def get_layer_density(self):
        """ Returns the mean density including ice and liquid of the node. 

        Returns
        -------
            rho : float
                Snow density [:math:`kg~m^{-3}`]
        """
        return self.get_layer_ice_fraction()*ice_density + self.get_layer_liquid_water_content()*water_density + self.get_layer_air_porosity()*air_density
    
    def get_layer_air_porosity(self):
        """ Returnis the ice fraction of the node.

        Returns
        -------
            porosity : float
                Air porosity [:math:`m`]
        """
        return max(0.0, 1 - self.get_layer_liquid_water_content() - self.get_layer_ice_fraction())
    
    def get_layer_specific_heat(self):
        """ Returns the volumetric averaged specific heat of the node. 

        Returns
        -------
            cp : float
                Specific heat [:math:`J~kg^{-1}~K^{-1}`]
        """
        return self.get_layer_ice_fraction()*spec_heat_ice + self.get_layer_air_porosity()*spec_heat_air + self.get_layer_liquid_water_content()*spec_heat_water

    def get_layer_liquid_water_content(self):
        """ Returns the liquid water content of the node.

        Returns
        -------
            lwc : float
                Liquid water content [-]
        """
        return self.liquid_water_content 
    
    def get_layer_irreducible_water_content(self):
        """ Returns the irreducible water content of the node. 

        Returns
        -------
            ret : float
                Irreducible water content [-]
        """
        if (self.get_layer_ice_fraction() <= 0.23):
            theta_e = 0.0264 + 0.0099*((1-self.get_layer_ice_fraction())/self.get_layer_ice_fraction()) 
        elif (self.get_layer_ice_fraction() > 0.23) & (self.get_layer_ice_fraction() <= 0.812):
            theta_e = 0.08 - 0.1023*(self.get_layer_ice_fraction()-0.03)
        else:
            theta_e = 0.0
        return theta_e 
    
    def get_layer_cold_content(self):
        """ Returns the cold content of the node. 

        Returns
        -------
            cc : float
                Cold content [:math:`J~m^{-2}`]
        """
        return -self.get_layer_specific_heat() * self.get_layer_density() * self.get_layer_height() * (self.get_layer_temperature()-zero_temperature)
    
    def get_layer_porosity(self):
        """ Returns the porosity of the node. 

        Returns
        -------
            por : float
                Air porosity [-]
        """
        return 1-self.get_layer_ice_fraction()-self.get_layer_liquid_water_content()
   
    def get_layer_thermal_conductivity(self):
        """ Returns the volumetric weighted thermal conductivity of the node.

        Returns
        -------
            lambda : float
                Thermal conductivity [:math:`W~m^{-1}~K^{-1}`]
        """
        methods_allowed = ['bulk', 'empirical']
        if thermal_conductivity_method == 'bulk':
            lam = self.get_layer_ice_fraction()*k_i + self.get_layer_air_porosity()*k_a + self.get_layer_liquid_water_content()*k_w
        elif thermal_conductivity_method == 'empirical':
            lam = 0.021 + 2.5 * np.power((self.get_layer_density()/1000),2)
        else:
            raise ValueError("Thermal conductivity method = \"{:s}\" is not allowed, must be one of {:s}".format(thermal_conductivity_method, ", ".join(methods_allowed)))
        return lam

    def get_layer_thermal_diffusivity(self):
        """ Returns the thermal diffusivity of the node. 

        Returns
        -------
            K : float
                Thermal diffusivity [:math:`m^{2}~s^{-1}`]
        """
        K = self.get_layer_thermal_conductivity()/(self.get_layer_density()*self.get_layer_specific_heat())
        return K


    ''' SETTER FUNCTIONS '''

    #----------------------------------------------
    # Setter-functions for derived state variables
    #----------------------------------------------
    def set_layer_height(self, height):
        """ Sets the layer height of the node. 
        
        Parameters
        ----------
            height : float
                Layer height [:math:`m`]
        """
        self.height = height

    def set_layer_temperature(self, T):
        """ Sets the mean temperature of the node.

        Parameters
        ----------
            T : float
                Layer temperature [:math:`K`]
        """
        self.temperature = T

    def set_layer_liquid_water_content(self, lwc):
        """ Sets the liquid water content of the node.

        Parameters
        ----------
            lwc : float
                Liquid water content [-]
        """
        self.liquid_water_content = lwc
    
    def set_layer_ice_fraction(self, ifr):
        """ Sets the ice fraction of the node. 

        Parameters
        ----------
            ifr : float
                Ice fraction [-]
        """
        self.ice_fraction = ifr
    
    def set_layer_refreeze(self, refr):
        """ Sets the amount of water refreezed of the node.

        Parameters
        ----------
            refr : float
                Amount of refreezed water [:math:`m~w.e.`]
        """
        self.refreeze = refr
