import numpy as np
from constants import *
from core.node import *


class Grid:

    def __init__(self, layer_heights, layer_densities, layer_temperatures, liquid_water_contents, debug):
        """ Initialize numerical grid 
        
        Input:         
        layer_heights           : numpy array with the layer height
        layer_densities         : numpy array with density values for each layer
        layer_temperatures      : numpy array with temperature values for each layer
        liquid_water_contents   : numpy array with liquid water content for each layer
        debug                   : Debug level (0, 10, 20, 30) """

        # Set class variables
        self.layer_heights = layer_heights
        self.layer_densities = layer_densities
        self.layer_temperatures = layer_temperatures
        self.liquid_water_contents = liquid_water_contents
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
                                  self.layer_temperatures[idxNode], self.liquid_water_contents[idxNode]))

    def add_node(self, height, density, temperature, liquid_water_content):
        """ Add a new node at the beginning of the node list (upper layer) """

        # Add new node
        self.grid.insert(0, Node(height, density, temperature, liquid_water_content))
        
        # Increase node counter
        self.number_nodes += 1

    def add_node_idx(self, idx, height, density, temperature, liquid_water_content):
        """ Add a new node below node idx """

        # Add new node
        self.grid.insert(idx, Node(height, density, temperature, liquid_water_content))

        # Increase node counter
        self.number_nodes += 1

    def remove_node(self, pos=None):
        """ Removes the node at position pos from the node list """

        # Remove node from list when there is at least one node
        if not self.grid:
            pass
        else:
            if pos is None:
                self.grid.pop(0)
            else:
                self.grid.pop(pos)
            
            # Decrease node counter
            self.number_nodes -= 1

    def update_node(self, no, height, density, temperature, liquid_water_content):
        """ Update properties of a specific node """

        self.grid[no].set_layer_height(height)
        self.grid[no].set_layer_density(density)
        self.grid[no].set_layer_temperature(temperature)
        self.grid[no].set_layer_liquid_water_content(liquid_water_content)

    def update_grid(self, level):
        """ Merge the similar layers according to certain criteria. Users can 
        determine different levels: 
            
            level 0 :   Layers are never merged
            level 1 :   Layers are merged when density and T are similar
            level 2 :   Layers are merged when density and T are very similar 
            
        The iterative process starts from the upper grid point and goes through 
        all nodes. If two subsequent nodes are similar, the layers are merged. 
        In this case the next merging step starts again from the top. This loop
        is repeated until the last node is reached and all similar layers merged."""

        # Define the thresholds for merging levels
        if level == 0:
            # print("Merging level 0")
            merge = False
        elif level == 1:
            threshold_density = 5.
            threshold_temperature = 0.05
            merge = True
        elif level == 2:
            threshold_density = 10.
            threshold_temperature = 0.1
            merge = True
        else:
            print("Invalid merging level")
            pass

        # Auxilary variables
        idx = self.number_nodes - 1
        
        # Iterate over grid and check for similarity
        while merge: 
            
            if ((np.abs(self.grid[idx].get_layer_density() - self.grid[idx-1].get_layer_density())
                <= threshold_density) &
                    (np.abs(self.grid[idx-1].get_layer_density() - self.grid[idx-2].get_layer_density()) <= 100.) &
                    (np.abs(self.grid[idx].get_layer_temperature() - self.grid[idx-1].get_layer_temperature())
                <= threshold_temperature) &
                    (self.grid[idx].get_layer_height() + self.grid[idx-1].get_layer_height() <= 0.5)):
                
                # Total height of both layer which are merged
                total_height = self.grid[idx].get_layer_height() + self.grid[idx - 1].get_layer_height()

                # Add up height of the two layer
                new_height = self.grid[idx].get_layer_height() + self.grid[idx - 1].get_layer_height()
                
                # Get the new density, weighted by the layer heights
                new_density = (self.grid[idx].get_layer_height() / total_height) * self.grid[idx].get_layer_density() + \
                          (self.grid[idx-1].get_layer_height() / total_height) * self.grid[idx - 1].get_layer_density()

                # specific heat of ice (i) and air (p) [J kg-1 K-1] TODO: NEED TO BE CORRECTED
                # c_i = spec_heat_air

                # First calculate total energy
                new_total_energy = (self.grid[idx].get_layer_height() * spec_heat_air *
                                    self.grid[idx].get_layer_density() * self.grid[idx].get_layer_temperature()) + \
                                   (self.grid[idx-1].get_layer_height() * spec_heat_air *
                                    self.grid[idx - 1].get_layer_density() * self.grid[idx - 1].get_layer_temperature())
                
                # Convert total energy to temperature according to the new density
                new_temperature = new_total_energy/(spec_heat_air*new_density*new_height)
                
                # Todo: CHECK IF RIGHT!!!!!
                new_liquid_water_content = self.grid[idx].get_layer_liquid_water_content() + self.grid[idx].get_layer_liquid_water_content()

                # Update node properties
                self.update_node(idx, new_height, new_density, new_temperature, new_liquid_water_content)

                # Remove the second layer
                self.remove_node(idx-1)
           
                # Move to next layer 
                idx -= 1

                # Write merging steps if debug level is set >= 10
                if self.debug >= 20:
                    print("Merging ....")
                    for i in range(self.number_nodes):
                        print(self.grid[i].get_layer_height(), self.grid[i].get_layer_temperature(),
                              self.grid[i].get_layer_density())
                    print("End merging .... \n")

            else:

                # Stop merging process, if iterated over entire grid
                idx -= 1
                if idx == 0:
                    merge = False

    def merge_new_snow(self, height_diff):
        """ Merge first layers according to certain criteria """

        if self.grid[0].get_layer_height() <= height_diff:

                # Total height of both layer which are merged
                total_height = self.grid[0].get_layer_height() + self.grid[1].get_layer_height()

                # Add up height of the two layer
                new_height = self.grid[0].get_layer_height() + self.grid[1].get_layer_height()
                
                # Get the new density, weighted by the layer heights
                new_density = (self.grid[0].get_layer_height() / total_height) * self.grid[0].get_layer_density() + \
                    (self.grid[1].get_layer_height() / total_height) * self.grid[1].get_layer_density()

                # TODO: NEED TO BE CORRECTED
                # spec_heat_air

                # First calculate total energy
                new_total_energy = (self.grid[0].get_layer_height() * spec_heat_air * self.grid[0].get_layer_density() *
                                    self.grid[0].get_layer_temperature()) + \
                                   (self.grid[1].get_layer_height() * spec_heat_air * self.grid[1].get_layer_density() *
                                    self.grid[1].get_layer_temperature())
                
                # Convert total energy to temperature according to the new density
                new_temperature = new_total_energy/(spec_heat_air*new_density*new_height)
                
                # Todo: CHECK IF RIGHT!!!!!
                new_liquid_water_content = self.grid[0].get_layer_liquid_water_content() + \
                                           self.grid[1].get_layer_liquid_water_content()

                # Update node properties
                self.update_node(1, new_height, new_density, new_temperature, new_liquid_water_content)

                # Remove the second layer
                self.remove_node(0)
           
                # Write merging steps if debug level is set >= 10
                if self.debug >= 20:
                    print("New Snow Merging ....")
                    for i in range(self.number_nodes):
                        print(self.grid[i].get_layer_height(), self.grid[i].get_layer_temperature(),
                              self.grid[i].get_layer_density())
                    print("End merging .... \n")

    def remove_melt_energy(self, melt):

        """ removes every iteration the surface layer if melt energy is large enough """

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
                    self.remove_node(0)
                    melt -= melt_required
                    remove = True
                # todo store removed layer height as runoff (R)
                # return R

    def set_node_temperature(self, idx, temperature):
        """ Returns temperature of node idx """
        
        return self.grid[idx].set_layer_temperature(temperature)
    
    def set_node_height(self, idx, height):
        """ Set height of node idx """
        
        return self.grid[idx].set_layer_height(height)
    
    def set_node_liquid_water_content(self, idx, liquid_water_content):
        """ Set liquid water content of node idx """
        
        return self.grid[idx].set_layer_liquid_water_content(liquid_water_content)

    def set_liquid_water_content(self, liquid_water_content):
        """ Returns the temperature profile """

        for idx in range(self.number_nodes):
            self.grid[idx].set_layer_liquid_water_content(liquid_water_content[idx])

    def get_temperature(self):
        """ Returns the temperature profile """
        T = []
        for idx in range(self.number_nodes):
            T.append(self.grid[idx].get_layer_temperature())
        
        return T

    def get_node_temperature(self, idx):
        """ Returns temperature of node idx """
        
        return self.grid[idx].get_layer_temperature()

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

