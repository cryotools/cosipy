class Node:
    """ This is the basic class which contains information of individual grid point. """

    def __init__(self, height, density, temperature, liquid_water_content, cold_content, porosity, max_vol_ice_content, refreeze):
        """ Initialize the node with:
            loc                     :      location of the node 0 = lowest node
            height                  :      height of the layer [m]
            temperature             :      temperature of the layer [K]
            liquid_water_content    :      liquid water content [m w.e.]
        """

        # Define class variables
        self.height = height
        self.density = density
        self.temperature = temperature
        self.liquid_water_content = liquid_water_content
        self.cold_content = cold_content
        self.porosity = porosity
        self.max_vol_ice_content = max_vol_ice_content
        self.refreeze = refreeze

    def __del__(self):
        """ Remove node """
        # TODO: Delete node function

    ''' GETTER FUNCTIONS '''

    def get_layer_height(self):
        """ Return the layer height """
        return self.height

    def get_layer_density(self):
        """ Return the mean density of the layer """
        return self.density

    def get_layer_temperature(self):
        """ Return the mean temperature of the layer """
        return self.temperature

    def get_layer_liquid_water_content(self):
        """ Return the liquid water content of the layer """
        return self.liquid_water_content
    
    def get_layer_cold_content(self):
        """ Return the liquid water content of the layer """
        return self.cold_content
    
    def get_layer_porosity(self):
        """ Return the porosity of the layer """
        return self.porosity
    
    def get_layer_max_vol_ice_content(self):
        """ Return the volumetric ice content of the layer """
        return self.max_vol_ice_content
    
    def get_layer_refreeze(self):
        """ Return the volumetric ice content of the layer """
        return self.refreeze

    ''' SETTER FUNCTIONS '''

    def set_layer_height(self, height):
        """ Set the layer height """
        self.height = height

    def set_layer_density(self, density):
        """ Set the mean density of the layer """
        self.density = density

    def set_layer_temperature(self, temperature):
        """ Set the mean temperature of the layer """
        self.temperature = temperature

    def set_layer_liquid_water_content(self, liquid_water_content):
        """ Set the liquid water content of the layer """
        self.liquid_water_content = liquid_water_content
    
    def set_layer_cold_content(self, cold_content):
        """ Set the cold content of the layer """
        self.cold_content = cold_content
    
    def set_layer_porosity(self, porosity):
        """ Set the porosity of the layer """
        self.porosity = porosity

    def set_layer_max_vol_ice_content(self, max_vol_ice_content):
        """ Set the volumetric ice content of the layer """
        self.max_vol_ice_content = max_vol_ice_content

    def set_layer_refreeze(self, refreeze):
        """ Set the volumetric ice content of the layer """
        self.refreeze = refreeze 
