import numpy as np
from constants import *
from config import * 
from cpkernel.node import *
import sys
import logging
import yaml
import os


class Grid:

    def __init__(self, layer_heights, layer_densities, layer_temperatures, liquid_water, debug):
        """ Initialize numerical grid 
        
        Input:         
        layer_heights           : numpy array with the layer height
        layer_densities         : numpy array with density values for each layer
        layer_temperatures      : numpy array with temperature values for each layer
        liquid_water            : numpy array with liquid water [m] for each layer
        debug                   : Debug level (0, 10, 20, 30) """

        # Start logging
        ''' Start the python logging'''
    
        if os.path.exists('./cosipy.yaml'):
            with open('./cosipy.yaml', 'rt') as f:
                config = yaml.safe_load(f.read())
            logging.config.dictConfig(config)
        else:
            logging.basicConfig(level=logging.DEBUG)
   
        self.logger = logging.getLogger(__name__)
        

        # Set class variables
        self.layer_heights = layer_heights
        self.layer_densities = layer_densities
        self.layer_temperatures = layer_temperatures
        self.liquid_water = liquid_water
        self.debug = debug
        
        # Number of total nodes
        self.number_nodes = len(layer_heights)
        
        # Print some information on initialized grid
        if self.debug > 0:
            print("Init grid with %d nodes \t" % self.number_nodes)
            print("Total domain depth is %4.2f m \n" % np.sum(layer_heights))

        # Do the grid initialization
        self.init_grid()


    def init_grid(self):
        """ Initialize the grid with according to the input data """

        # Init list with nodes
        self.grid = []

        # Fill the list with node instances and fill it with user defined data
        for idxNode in range(self.number_nodes):
            self.grid.append(Node(self.layer_heights[idxNode], self.layer_densities[idxNode],
                        self.layer_temperatures[idxNode], self.liquid_water[idxNode]))



    def add_node(self, height, density, temperature, liquid_water):
        """ Add a new node at the beginning of the node list (upper layer) """

        self.logger.debug('Add  node')

        # Add new node
        self.grid.insert(0, Node(height, density, temperature, liquid_water))
        
        # Increase node counter
        self.number_nodes += 1



    def add_node_idx(self, idx, height, density, temperature, liquid_water):
        """ Add a new node below node idx """

        # Add new node
        self.grid.insert(idx, Node(height, density, temperature, liquid_water))

        # Increase node counter
        self.number_nodes += 1


    def remove_node(self, pos=None):
        """ Removes a node or a list of nodes at pos from the node list """

        self.logger.debug('Remove node')

        # Remove node from list when there is at least one node
        if not self.grid:
            pass
        else:
            if pos is None:
                self.grid.pop(0)
            else:
                for index in sorted(pos, reverse=True):
                    del self.grid[index]

            # Decrease node counter
            self.number_nodes -= len(pos)


    def merge_nodes(self, idx):
        """ This function merges the nodes at location idx and idx+1. The node at idx is updated 
        with the new properties (height, liquid water content, ice fraction, temperature), while the node
        at idx+1 is deleted after merging"""

        # New layer height by adding up the height of the two layers
        new_height = self.grid[idx].get_layer_height() + self.grid[idx+1].get_layer_height()
        
        # Update liquid water
        new_liquid_water = self.grid[idx].get_layer_liquid_water() + self.grid[idx+1].get_layer_liquid_water()
        
        # Update ice fraction
        new_ice_fraction = ((self.grid[idx].get_layer_ice_fraction()*self.grid[idx].get_layer_height() + \
                            self.grid[idx+1].get_layer_ice_fraction()*self.grid[idx+1].get_layer_height())/new_height)
        
        # New volume fractions and density
        new_liquid_water_content = new_liquid_water/new_height
        new_air_porosity = 1 - new_liquid_water_content - new_ice_fraction
        
        if abs(1-new_ice_fraction-new_air_porosity-new_liquid_water_content)>1e-8:
            print('Merging is not mass consistent (%2.7f)' % (new_ice_fraction+new_air_porosity+new_liquid_water_content))
       
        # Calc new temperature
        new_temperature = (self.grid[idx].get_layer_height()/new_height)*self.grid[idx].get_layer_temperature() + \
                            (self.grid[idx+1].get_layer_height()/new_height)*self.grid[idx+1].get_layer_temperature()

        # Update node properties
        self.update_node(idx, new_height, new_temperature, new_ice_fraction, new_liquid_water)
        
        # Remove the second layer
        self.remove_node([idx+1])



    def split_node(self, pos):
        """ Split node at position pos """

        self.logger.debug('Split node')

        self.grid.insert(pos+1, Node(self.get_node_height(pos)/2.0, self.get_node_density(pos), self.get_node_temperature(pos), self.get_node_liquid_water(pos)/2.0, self.get_node_ice_fraction(pos)))
        self.update_node(pos, self.get_node_height(pos)/2.0, self.get_node_temperature(pos), self.get_node_ice_fraction(pos), self.get_node_liquid_water(pos)/2.0)
        self.number_nodes += 1



    def update_node(self, no, height, temperature, ice_fraction, liquid_water):
        """ Update properties of a specific node """

        self.logger.debug('Update node')
        
        self.grid[no].set_layer_height(height)
        self.grid[no].set_layer_temperature(temperature)
        self.grid[no].set_layer_ice_fraction(ice_fraction)
        self.grid[no].set_layer_liquid_water(liquid_water)



    def update_grid(self, merge, threshold_temperature, threshold_density, merge_snow_threshold, merge_max, split_max):
        """ Merge the similar layers according to certain criteria. Users can determine different levels:
            
            merging can be False or True
            temperature threshold
            density threshold
            merge new snow threshold is minimum heihgt of snow layers

        The iterative process starts from the upper grid point and goes through 
        all nodes. If two subsequent nodes are similar, the layers are merged. 
        In this case the next merging step starts again from the top. This loop
        is repeated until the last node is reached and all similar layers merged."""

        self.logger.debug('--------------------------')
        self.logger.debug('Update grid')

        #-------------------------------------------------------------------------
        # Merging 
        #
        # Layers are merged, if:
        # (1) the density difference between the layer and the subsequent layer is smaller than the user defined threshold
        # (2) the temperature difference is smaller than the user defined threshold
        # (3) the new layer height does not exceed a height of 0.5 m
        #-------------------------------------------------------------------------
        if merge:
            
            # Check for merging 
            for i in range(merge_max):
                # Get number of snow layers
                nlayers = self.get_number_snow_layers()

                # Check if there are at least two layers
                if nlayers > 1:
                    
                    # Calc differences between a layer and the subsequent layer
                    dT = np.diff(self.get_temperature()[0:nlayers+1])
                    dRho = np.diff(self.get_density()[0:nlayers+1])

                    # Sort the by differences in ascending order, and merge if criteria is met
                    ind = np.lexsort((abs(dRho),abs(dT)))
                    if (abs(dT[ind[0]])<threshold_temperature) & (abs(dRho[ind[0]])<threshold_density):
                        self.merge_nodes(ind[i])
            
         
        #---------------------
        # Splitting
        #---------------------
        if merge: 

            # Only split when maximum layers not reached
            if self.get_number_layers()<max_layers:

                #--------------------------------------
                # Snow splitting
                # Check for merging (user defined number of splittings) 
                #--------------------------------------
                for i in range(split_max):

                    # Get number of snow layers
                    nlayers = self.get_number_snow_layers()
   
                    # Check if there are still snow layers
                    if nlayers > 0:
                        # Get layer heights
                        h = np.asarray(self.get_height()[0:nlayers])
    
                        # Calc differences between a layer and the subsequent layer
                        dT = np.diff(self.get_temperature()[0:nlayers+1])
                        dRho = np.diff(self.get_density()[0:nlayers+1])

                        # Sort the by differences in ascending order
                        ind = np.lexsort((abs(dT),abs(dRho),h))[::-1]
                   
                        if (h[ind[0]]>max_snow_layer_height) & (abs(dT[ind[0]])>5*threshold_temperature) & (abs(dRho[ind[0]])>5*threshold_density):
                            self.split_node(ind[i])

                #--------------------------------------
                # Guarantee that the first layer is not greater than 2 cm
                #--------------------------------------
                while self.grid[0].get_layer_height() > 0.02:
                    self.split_node(0)

                #--------------------------------------
                # Split mesh at glacier-snow interface
                #--------------------------------------
                # Get number of snow layers
                nlayers = self.get_number_snow_layers()
               
                while self.grid[nlayers].get_layer_height() - self.grid[nlayers-1].get_layer_height() >= 0.02:
                    self.split_node(nlayers)                
                



    def merge_new_snow(self, height_diff):
        """ Merge first layers according to certain criteria """

        self.logger.debug('Merge new snow')

        ### Merge snow layers if thickness thinner threshold or if only one snow layer thinner minimum snow height
        if ((self.grid[1].get_layer_density() < snow_ice_threshold) & (self.grid[0].get_layer_height() <= height_diff)) \
            or ((self.grid[1].get_layer_density() >= snow_ice_threshold) & (self.grid[0].get_layer_height() < minimum_snow_height)):

            # If only one snow layer is left and is smaller than the minimum snowheight, merge to ice (reduct to ice density)
            if ((self.grid[1].get_layer_density() >= snow_ice_threshold) & (self.grid[0].get_layer_height() < minimum_snow_height)):
                height_first_layer_tmp = (self.grid[0].get_layer_density()/ice_density) * self.grid[0].get_layer_height()
                density_first_layer_tmp = ice_density

            else:
                height_first_layer_tmp = self.grid[0].get_layer_height()
                density_first_layer_tmp = self.grid[0].get_layer_density()

            # Add up height of the two layer
            new_height = height_first_layer_tmp + self.grid[1].get_layer_height()
            
            # Update liquid water
            new_liquid_water = self.grid[0].get_layer_liquid_water() + self.grid[1].get_layer_liquid_water()

            # Update ice fraction
            new_ice_fraction = (self.grid[0].get_layer_ice_fraction()*self.grid[0].get_layer_height() + \
                                self.grid[1].get_layer_ice_fraction()*self.grid[1].get_layer_height()) / new_height

            # New volume fractions and density
            new_liquid_water_content = new_liquid_water/new_height
            new_air_porosity = 1 - new_liquid_water_content - new_ice_fraction
            new_density = new_ice_fraction*ice_density + new_liquid_water_content*water_density + new_air_porosity*air_density
            
            if abs(1-new_ice_fraction-new_air_porosity-new_liquid_water_content)>1e-8:
                print('Merging is not mass consistent (%2.7f)' % (new_ice_fraction+new_air_porosity+new_liquid_water_content))
                
            
            # Calc new temperature
            new_temperature = (self.grid[0].get_layer_height()/new_height)*self.grid[0].get_layer_temperature() + \
                            (self.grid[1].get_layer_height()/new_height)*self.grid[1].get_layer_temperature()


            # Update node properties
            self.update_node(1, new_height, new_temperature, new_ice_fraction, new_liquid_water)
            
            # Remove the second layer
            self.remove_node([0])

            # Write merging steps if debug level is set >= 10
            self.logger.debug("Merging new snow (merge_new_snow) ....")



    def remove_melt_energy(self, melt):

        """ removes every iteration the surface layer if melt energy is large enough """

        self.logger.debug('Remove melt energy')        

        # Convert melt (m w.e.q.) to m height
        height_diff = float(melt) / (self.get_node_density(0) / 1000.0)   # m (snow) - negative = melt

        if height_diff != 0.0:
            remove = True
        else:
            remove = False

        while remove:
               
            # How much energy required to melt first layer
            melt_required = self.get_node_height(0) * (self.get_node_density(0) / 1000.0)

            # How much energy is left
            melt_rest = melt - melt_required

            # If not enough energy to remove first layer, first layers height is reduced by melt height
            if melt_rest <= 0:
                self.set_node_height(0, self.get_node_height(0) - height_diff)
                remove = False

            # If entire layer is removed
            else:
                self.remove_node([0])
                melt -= melt_required
                remove = True



    def set_node_temperature(self, idx, temperature):
        """ Returns temperature of node idx """
        return self.grid[idx].set_layer_temperature(temperature)



    def set_temperature(self, temperature):
        """ Set temperature of profile """
        for idx in range(self.number_nodes):
            self.grid[idx].set_layer_temperature(temperature[idx])



    def set_node_height(self, idx, height):
        """ Set height of node idx """
        return self.grid[idx].set_layer_height(height)



    def set_height(self, height):
        """ Set height of profile """
        for idx in range(self.number_nodes):
            self.grid[idx].set_layer_height(height[idx])



    def set_node_liquid_water(self, idx, liquid_water):
        """ Set liquid water of node idx """
        return self.grid[idx].set_layer_liquid_water(liquid_water)



    def set_liquid_water(self, liquid_water):
        """ Set the liquid water profile """
        for idx in range(self.number_nodes):
            self.grid[idx].set_layer_liquid_water(liquid_water[idx])

    
    def set_node_liquid_water_content(self, idx, liquid_water_content):
        """ Set liquid water content of node idx """
        return self.grid[idx].set_layer_liquid_water_content(liquid_water_content)



    def set_liquid_water_content(self, liquid_water_content):
        """ Set the liquid water content profile """
        for idx in range(self.number_nodes):
            self.grid[idx].set_layer_liquid_water_content(liquid_water_content[idx])

    
    def set_node_ice_fraction(self, idx, ice_fraction):
        """ Set liquid ice_fraction of node idx """
        return self.grid[idx].set_layer_ice_fraction(ice_fraction)


    def set_ice_fraction(self, ice_fraction):
        """ Set the ice_fraction profile """
        for idx in range(self.number_nodes):
            self.grid[idx].set_layer_ice_fraction(ice_fraction[idx])


    def set_node_refreeze(self, idx, refreeze):
        """ Set refreezing of node idx """        
        return self.grid[idx].set_layer_refreeze(refreeze)



    def set_refreeze(self, refreeze):
        """ Set the refreezing profile """
        for idx in range(self.number_nodes):
            self.grid[idx].set_refreeze(refreeze[idx])



    def get_temperature(self):
        """ Returns the temperature profile """
        T = []
        for idx in range(self.number_nodes):
            T.append(self.grid[idx].get_layer_temperature())
        return T


    def get_node_temperature(self, idx):
        """ Returns temperature of node idx """
        return self.grid[idx].get_layer_temperature()

    
    def get_specific_heat(self):
        """ Returns the specific heat (air+water+ice) profile """
        cp = []
        for idx in range(self.number_nodes):
            cp.append(self.grid[idx].get_layer_specific_heat())
        return cp

    def get_node_specific_heat(self, idx):
        """ Returns specific heat (air+water+ice) of node idx """
        return self.grid[idx].get_layer_specific_heat()


    def get_height(self):
        """ Returns the heights of the layers """
        hlayer = []
        for idx in range(self.number_nodes):
            hlayer.append(self.grid[idx].get_layer_height())
        return hlayer



    def get_node_height(self, idx):
        """ Returns layer height of node idx """
        return self.grid[idx].get_layer_height()



    def get_node_density(self, idx):
        """ Returns density of node idx """
        return self.grid[idx].get_layer_density()


        
    def get_density(self):
        """ Returns the rho profile """
        rho = []
        for idx in range(self.number_nodes):
            rho.append(self.grid[idx].get_layer_density())
        return rho


    def get_node_liquid_water_content(self, idx):
        """ Returns density of node idx """
        return self.grid[idx].get_layer_liquid_water_content()


        
    def get_liquid_water_content(self):
        """ Returns the rho profile """
        LWC = []
        for idx in range(self.number_nodes):
            LWC.append(self.grid[idx].get_layer_liquid_water_content())
        return LWC


    def get_node_liquid_water(self, idx):
        """ Returns liquid water of node idx """
        return self.grid[idx].get_layer_liquid_water()


    def get_liquid_water(self):
        """ Returns the liquid water profile """
        LW = []
        for idx in range(self.number_nodes):
            LW.append(self.grid[idx].get_layer_liquid_water())
        return LW


    def get_node_ice_fraction(self, idx):
        """ Returns ice fraction of node idx """
        return self.grid[idx].get_layer_ice_fraction()


    def get_ice_fraction(self):
        """ Returns the liquid water profile """
        theta_i = []
        for idx in range(self.number_nodes):
            theta_i.append(self.grid[idx].get_layer_ice_fraction())
        return theta_i

    
    def get_node_irreducible_water_content(self, idx):
        """ Returns irreducible water content of node idx """
        return self.grid[idx].get_layer_irreducible_water_content()
    
    
    def get_irreducible_water_content(self):
        """ Returns the irreducible water content profile """
        theta_e = []
        for idx in range(self.number_nodes):
            theta_e.append(self.grid[idx].get_layer_irreducible_water_content())
        return theta_e
        

    def get_node_cold_content(self, idx):
        """ Returns cold content of node idx """
        return self.grid[idx].get_layer_cold_content()


        
    def get_cold_content(self):
        """ Returns the cold content profile """
        CC = []
        for idx in range(self.number_nodes):
            CC.append(self.grid[idx].get_layer_cold_content())
        return CC


    
    def get_node_porosity(self, idx):
        """ Returns porosity of node idx """
        return self.grid[idx].get_layer_porosity()


        
    def get_porosity(self):
        """ Returns the porosity profile """
        por = []
        for idx in range(self.number_nodes):
            por.append(self.grid[idx].get_layer_porosity())
        return por


    
    def get_node_refreeze(self, idx):
        """ Returns refreezing of node idx """
        return self.grid[idx].get_layer_refreeze()


        
    def get_refreeze(self):
        """ Returns the refreezing profile """
        ref = []
        for idx in range(self.number_nodes):
            ref.append(self.grid[idx].get_layer_refreeze())
        return ref



    def get_total_snowheight(self, verbose=False):
        """ Get the total snowheight (density<snow_ice_threshold)"""
        
        total = 0
        snowheight = 0
        for i in range(self.number_nodes):
            if (self.grid[i].get_layer_density()<threshold_for_snowheight):
                snowheight = snowheight + self.grid[i].get_layer_height()
            total = total + self.grid[i].get_layer_height()

        if verbose:
            print("******************************")
            print("Number of nodes: %d" % self.number_nodes)
            print("******************************")

            print("Grid consists of %d nodes \t" % self.number_nodes)
            print("Total snow depth is %4.2f m \n" % snowheight)
            print("Total domain depth is %4.2f m \n" % total)
        
        return snowheight

    
    def get_total_height(self, verbose=False):
        """ Get the total domain height """
        
        total = 0
        snowheight = 0
        for i in range(self.number_nodes):
            if (self.grid[i].get_layer_density()<snow_ice_threshold):
                snowheight = snowheight + self.grid[i].get_layer_height()
            total = total + self.grid[i].get_layer_height()

        if verbose:
            print("******************************")
            print("Number of nodes: %d" % self.number_nodes)
            print("******************************")

            print("Grid consists of %d nodes \t" % self.number_nodes)
            print("Total snow depth is %4.2f m \n" % snowheight)
            print("Total domain depth is %4.2f m \n" % total)
        
        return total

        
    def get_number_snow_layers(self):
        """ Get the number of snow layers (density<snow_ice_threshold)"""
        
        nlayers = 0
        for i in range(self.number_nodes):
            if (self.grid[i].get_layer_density()<threshold_for_snowheight):
                nlayers = nlayers+1
        return nlayers


    def get_number_layers(self):
        """ Get the number of layers"""
        return (self.number_nodes)



    def info(self):
        """ Print some information on grid """
        
        print("******************************")
        print("Number of nodes: %d" % self.number_nodes)
        print("******************************")

        tmp = 0
        for i in range(self.number_nodes):
            tmp = tmp + self.grid[i].get_layer_height()

#        print("Grid consists of %d nodes \t" % self.number_nodes)
#        print("Total domain depth is %4.2f m \n" % tmp)


    def grid_info(self, n=-999):
        """ The function prints the state of the snowpack 
            Args:
                n   : nuber of nodes to plot (from top)
        """

        if (n==-999):
            n = self.number_nodes
        
        self.logger.debug("Node no. \t\t  Layer height [m] \t Temperature [K] \t Density [kg m^-3] \t LWC [-] \t LW [m] \t CC [J m^-2] \t Porosity [-] \t Refreezing [m w.e.] \ Irreducible water content [-]")

        for i in range(n):
            self.logger.debug("%d %3.2f \t %3.2f \t %4.2f \t %2.7f \t %2.7f \t %10.4f \t %4.4f \t  %4.8f \t %2.7f" % (i, self.grid[i].get_layer_height(), self.grid[i].get_layer_temperature(),
                  self.grid[i].get_layer_density(), self.grid[i].get_layer_liquid_water_content(), self.grid[i].get_layer_liquid_water(), self.grid[i].get_layer_cold_content(),
                  self.grid[i].get_layer_porosity(), self.grid[i].get_layer_refreeze(), self.grid[i].get_layer_irreducible_water_content()))
        self.logger.debug('\n\n')


    def grid_info_screen(self, n=-999):
        """ The function prints the state of the snowpack 
            Args:
                n   : nuber of nodes to plot (from top)
        """

        if (n==-999):
            n = self.number_nodes
        
        print("Node no. \t\t  Layer height [m] \t Temperature [K] \t Density [kg m^-3] \t LWC [-] \t LW [m] \t CC [J m^-2] \t Porosity [-] \t Refreezing [m w.e.]")

        for i in range(n):
            print("%d %3.2f \t %3.2f \t %4.2f \t %2.7f \t %2.7f \t %10.4f \t %4.4f \t  %4.8f" % (i, self.grid[i].get_layer_height(), self.grid[i].get_layer_temperature(),
                  self.grid[i].get_layer_density(), self.grid[i].get_layer_liquid_water_content(), self.grid[i].get_layer_liquid_water(), self.grid[i].get_layer_cold_content(),
                  self.grid[i].get_layer_porosity(), self.grid[i].get_layer_refreeze()))
        print('\n\n')


