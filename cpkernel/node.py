from constants import *

class Node:
    """ This is the basic class which contains information of individual grid point. """

    def __init__(self, height, snow_density, temperature, liquid_water_content, ice_fraction=None):
        """ Initialize the node with:
            height                  :      height of the layer [m]
            density                 :      snow density  [kg m^-3]
            temperature             :      temperature of the layer [K]
            liquid_water            :      liquid water [m w.e.]

            Calculates:

            liquid_water_content    :       liquid water content [-]
            ice_fraction            :       ice fraction [-]
            porosity                :       porosity [-]
            cold_content            :       cold content [J m^-2]
            refreeze                :       how much water is refreezed [m]

        """

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
        """ Return the layer height """
        return self.height

    def get_layer_temperature(self):
        """ Return the mean temperature of the layer """
        return self.temperature
    
    def get_layer_ice_fraction(self):
        """ Return the ice fraction of the layer """
        return self.ice_fraction 
    
    def get_layer_refreeze(self):
        """ Return the volumetric ice content of the layer """
        return self.refreeze


    #----------------------------------------------
    # Getter-functions for derived state variables
    #----------------------------------------------
    def get_layer_density(self):
        """ Return the mean density including ice and liquid of the layer """
        return self.get_layer_ice_fraction()*ice_density + self.get_layer_liquid_water_content()*water_density + self.get_layer_air_porosity()*air_density
    
    def get_layer_air_porosity(self):
        """ Return the ice fraction of the layer """
        return max(0.0, 1 - self.get_layer_liquid_water_content() - self.get_layer_ice_fraction())
    
    def get_layer_specific_heat(self):
        """ Return the mean temperature of the layer """
        return self.get_layer_ice_fraction()*spec_heat_ice + self.get_layer_air_porosity()*spec_heat_air + self.get_layer_liquid_water_content()*spec_heat_water

    def get_layer_liquid_water_content(self):
        """ Return the liquid water [-] content of the layer """
        return self.liquid_water_content 
    
    def get_layer_irreducible_water_content(self):
        """ Return the irreducible water content of the layer """
        if (self.get_layer_ice_fraction() <= 0.23):
            theta_e = 0.0264 + 0.0099*((1-self.get_layer_ice_fraction())/self.get_layer_ice_fraction()) 
        elif (self.get_layer_ice_fraction() > 0.23) & (self.get_layer_ice_fraction() <= 0.812):
            theta_e = 0.08 - 0.1023*(self.get_layer_ice_fraction()-0.03)
        else:
            theta_e = 0.0
        return theta_e 
    
    def get_layer_cold_content(self):
        """ Return the liquid water content of the layer """
        return -self.get_layer_specific_heat() * self.get_layer_density() * self.get_layer_height() * (self.get_layer_temperature()-zero_temperature)
    
    def get_layer_porosity(self):
        """ Return the porosity of the layer """
        return 1-self.get_layer_ice_fraction()-self.get_layer_liquid_water_content()
   
    def get_layer_thermal_conductivity(self):
        """ Return the volumetic weighted thermal conductivity of the layer Sturm et al. (1997) and Paterson (1994)"""
        return self.get_layer_ice_fraction()*k_i + self.get_layer_air_porosity()*k_a + self.get_layer_liquid_water_content()*k_w

    def get_layer_thermal_diffusivity(self):
        """ Returns the thermal diffusivity of the layer"""
        K = self.get_layer_thermal_conductivity()/(self.get_layer_density()*self.get_layer_specific_heat())
        return K


    ''' SETTER FUNCTIONS '''

    #----------------------------------------------
    # Setter-functions for derived state variables
    #----------------------------------------------
    def set_layer_height(self, height):
        """ Set the layer height """
        self.height = height

    def set_layer_temperature(self, temperature):
        """ Set the mean temperature of the layer """
        self.temperature = temperature

    def set_layer_liquid_water_content(self, liquid_water_content):
        """ Set the liquid water content of the layer """
        self.liquid_water_content = liquid_water_content
    
    def set_layer_ice_fraction(self, ice_fraction):
        """ Set the ice fraction of the layer """
        self.ice_fraction = ice_fraction
    
    def set_layer_refreeze(self, refreeze):
        """ Set the amount of water refreezed of the layer """
        self.refreeze = refreeze 
