import numpy as np
from constants import *
from config import *
from cpkernel.node import *
import sys
import logging
import yaml
import os


class Grid:

    def __init__(self, layer_heights, layer_densities, layer_temperatures, layer_liquid_water_content):
        """ Initialize numerical grid

        Input:
        layer_heights           : numpy array with the layer height
        layer_densities         : numpy array with density values for each layer
        layer_temperatures      : numpy array with temperature values for each layer
        layer_liquid_water      : numpy array with liquid water content [-] for each layer """

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
        self.layer_liquid_water_content = layer_liquid_water_content

        # Number of total nodes
        self.number_nodes = len(layer_heights)

        # Track the fresh snow layer (new_snow_height, new_snow_timestamp) as well as the old
        # snow layer age (old_snow_timestamp)
        self.new_snow_height = 0.0      # meter snow accumulation
        self.new_snow_timestamp = 0.0   # seconds since snowfall
        self.old_snow_timestamp = 0.0   # snow age below fresh snow layer

        # Do the grid initialization
        self.init_grid()


    def init_grid(self):
        """ Initialize the grid with according to the input data """

        # Init list with nodes
        self.grid = []

        # Fill the list with node instances and fill it with user defined data
        for idxNode in range(self.number_nodes):
            self.grid.append(Node(self.layer_heights[idxNode], self.layer_densities[idxNode],
                        self.layer_temperatures[idxNode], self.layer_liquid_water_content[idxNode]))



    
    def add_fresh_snow(self, height, density, temperature, liquid_water_content, timestamp):
        """ Add a new node at the beginning of the node list (upper layer) """

        # Make entry in the log file
        self.logger.debug('Add  node')

        # Add new node
        self.grid.insert(0, Node(height, density, temperature, liquid_water_content))

        # Increase node counter
        self.number_nodes += 1

        # Set the fresh snow properties for albedo calculation (height and timestamp)
        self.set_fresh_snow_props(height, timestamp)



    def remove_node(self, pos=None):
        """ Removes a node or a list of nodes at pos from the node list """

        # Make entry in the log file
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
        new_height = self.get_node_height(idx) + self.get_node_height(idx+1)

        # Update liquid water
        new_liquid_water_content = self.get_node_liquid_water_content(idx) + self.get_node_liquid_water_content(idx+1)

        # Update ice fraction
        new_ice_fraction = ((self.get_node_ice_fraction(idx)*self.get_node_height(idx) + \
                            self.get_node_ice_fraction(idx+1)*self.get_node_height(idx+1))/new_height)

        # New air porosity
        new_air_porosity = 1 - new_liquid_water_content - new_ice_fraction

        if abs(1-new_ice_fraction-new_air_porosity-new_liquid_water_content)>1e-8:
            self.logger.error('Merging is not mass consistent (%2.7f)' % (new_ice_fraction+new_air_porosity+new_liquid_water_content))

        # Calc new temperature
        new_temperature = (self.get_node_height(idx)/new_height)*self.get_node_temperature(idx) + \
                            (self.get_node_height(idx+1)/new_height)*self.get_node_temperature(idx+1)

        # Update node properties
        self.update_node(idx, new_height, new_temperature, new_ice_fraction, new_liquid_water_content)

        # Remove the second layer
        self.remove_node([idx+1])



    def correct_layer(self, idx, min_height):
        """ This function adjusts layer idx to a given height (min_height). First, the layers below are merged until
            the height is sufficiently large to allow the adjustment. Then the layer is merged with the subsequent layer. """

        # New layer height by adding up the height of the two layers
        total_height = self.get_node_height(idx)

        # Merge subsequent layer with underlying layers until height of the layer is greater than the given height
        while ((total_height<min_height) & (idx+1<self.get_number_layers())):
            if (self.get_node_density(idx)<snow_ice_threshold) & (self.get_node_density(idx+1)<snow_ice_threshold):
                self.merge_nodes(idx)
            elif (self.get_node_density(idx)>=snow_ice_threshold) & (self.get_node_density(idx+1)>=snow_ice_threshold):
                self.merge_nodes(idx)
            else:
                break

            # Recalculate total height
            total_height = self.get_node_height(idx)

        # Only merge snow-snow or glacier-glacier, and if the height is greater than the minimum height
        if (total_height>min_height):

            # Get new heights for layer 0 and 1
            h0 = min_height
            h1 = total_height - min_height

            # Update liquid water
            # First the upper layer is filled with water until the max.
            # retention. The rest is assigned to the second layer.
            # if LWC exceeds the irreducible water content of the first layer,
            # then fill the first layer and assign the rest to the second layer
            if ((self.get_node_liquid_water_content(idx)-self.get_node_irreducible_water_content(idx))>0):
                lw0 = self.get_node_irreducible_water_content(idx)
                lw1 = self.get_node_liquid_water_content(idx)-self.get_node_irreducible_water_content(idx)
            # if LWC<WC_irr, then assign all water to the first layer
            else:   
                lw0 = self.get_node_liquid_water_content(idx)
                lw1 = 0.0

            # Update ice fraction
            if0 = self.get_node_ice_fraction(idx)
            if1 = self.get_node_ice_fraction(idx)

            # Temperature
            T0 = self.get_node_temperature(idx)
            T1 = self.get_node_temperature(idx)

            # New volume fractions and density
            lwc0 = lw0/h0
            lwc1 = lw1/h1
            por0 = 1 - lwc0 - if0
            por1 = 1 - lwc1 - if1

            # Check for consistency
            if (abs(1-if0-por0-lwc0)>1e-8) | (abs(1-if1-por1-lwc1)>1e-8):
                self.logger.error('Correct layer is not mass consistent (%2.7f) [Layer 0]' % (if0,por0,lwc0))
                self.logger.error('Correct layer is not mass consistent (%2.7f) [Layer 1]' % (if0,por0,lwc0))

            # Update node properties
            self.update_node(idx, h0, T0, if0, lw0)
            self.grid.insert(idx+1, Node(h1, self.get_node_density(idx), T1, lw1, if1))

            # Update node counter
            self.number_nodes += 1


    def log_profile(self):
        """ Logarithmic remeshing """

        last_layer_height = first_layer_height

        # Total snowheight
        hsnow = self.get_total_snowheight()

        # How much snow is not remeshed
        hrest = hsnow

        # First, the snowpack is remeshed
        idx = 0
        while (idx < self.get_number_snow_layers()):

            if (hrest>=last_layer_height):
                # Correct first layer
                self.correct_layer(idx,last_layer_height)

                hrest = hrest - last_layer_height

                # Height for the next layer
                last_layer_height = layer_stretching*last_layer_height

            # if the last layer is smaller than the required height, then merge
            # with the previous layer
            elif ((hrest<last_layer_height) & (idx>0)):
                self.merge_nodes(idx-1)

            idx = idx+1


        # get the glacier depth 
        hrest = self.get_total_height()-self.get_total_snowheight()

        # get number of snow layers
        idx = self.get_number_snow_layers()

        # then the glacier
        while (idx < self.get_number_layers()):

            if (hrest>=last_layer_height):
                # Correct first layer
                self.correct_layer(idx,last_layer_height)

                hrest = hrest - last_layer_height

                # Height for the next layer
                last_layer_height = layer_stretching*last_layer_height

            # if the last layer is smaller than the required height, then merge
            # with the previous layer
            elif ((hrest<last_layer_height)):
                self.merge_nodes(idx-1)

            idx = idx+1



    def adaptive_profile(self):
        """ Remesh according to certain layer state criteria"""

        #-------------------------------------------------------------------------
        # Merging
        #
        # Layers are merged, if:
        # (1) the density difference between the layer and the subsequent layer is smaller than the user defined threshold
        # (2) the temperature difference is smaller than the user defined threshold
        #-------------------------------------------------------------------------

        #-------------------------------------------------------------------------
        # Check for merging due to density and temperature
        #-------------------------------------------------------------------------

        # Correct first layer
        #self.correct_layer(0 ,first_layer_height)

        # First, the snowpack is remeshed
        idx = 0
        while (idx < self.get_number_snow_layers()-1):

            dT = np.abs(self.get_node_temperature(idx)-self.get_node_temperature(idx+1))
            dRho = np.abs(self.get_node_density(idx)-self.get_node_density(idx+1))

            if ((dT<temperature_threshold_merging) & (dRho<density_threshold_merging)) | (self.get_node_height(idx)<minimum_snow_layer_height):
                self.merge_nodes(idx)
            else:
                idx += 1
        
        print('++++++++++++++++++++')
        print(self.grid_info_screen(8))
        # Correct first layer
        self.correct_layer(0 ,first_layer_height)

        if (self.get_number_layers()==2) & (self.get_node_height(1)<minimum_snow_layer_height):
                self.merge_nodes(0)
        print(self.grid_info_screen(8))
    
        # get the glacier depth 
        hrest = self.get_total_height()-self.get_total_snowheight()

        # get number of snow layers
        idx = self.get_number_snow_layers()

        if (self.get_number_snow_layers()>0):
            last_layer_height = np.maximum(minimum_snow_layer_height,self.get_node_height(self.get_number_snow_layers()-1))
        else:
            last_layer_height = minimum_snow_layer_height

        # then the glacier
        while (idx < self.get_number_layers()):

            if (hrest>=last_layer_height):
                # Correct first layer
                self.correct_layer(idx,last_layer_height)

                hrest = hrest - last_layer_height

                # Height for the next layer
                last_layer_height = layer_stretching*last_layer_height

            # if the last layer is smaller than the required height, then merge
            # with the previous layer
            elif ((hrest<last_layer_height)):
                self.merge_nodes(idx-1)

            idx = idx+1
        
        print(self.grid_info_screen(8))

#        for i in range(merge_max):
#            # Get number of snow layers
#            nlayers = self.get_number_snow_layers()
#
#            # Check if there are at least two layers
#            if nlayers > 1:
#
#                # Calc differences between a layer and the subsequent layer
#                dT = np.diff(self.get_temperature()[0:nlayers])
#                dRho = np.diff(self.get_density()[0:nlayers])
#
#                # Sort the by differences in ascending order, and merge if criteria is met
#                ind = np.lexsort((abs(dRho),abs(dT)))
#                if ( (abs(dT[ind[0]])<temperature_threshold_merging) & (abs(dRho[ind[0]])<density_threshold_merging) ):
#                    self.merge_nodes(ind[0])
#
#        
#        # New layer height by adding up the height of the two layers
#        total_height = self.get_node_height(idx)
#
#        # Merge subsequent layer with underlying layers until height of the layer is greater than the given height
#        while ((total_height<min_height) & (idx+1<self.get_number_layers())):
#            if (self.get_node_density(idx)<snow_ice_threshold) & (self.get_node_density(idx+1)<snow_ice_threshold):
#                self.merge_nodes(idx)
#            elif (self.get_node_density(idx)>=snow_ice_threshold) & (self.get_node_density(idx+1)>=snow_ice_threshold):
#                self.merge_nodes(idx)
#            else:
#                break
#
#            # Recalculate total height
#            total_height = self.get_node_height(idx)
#
#        print('before', self.grid_info_screen(10))
#        self.correct_layer(0 ,first_layer_height)
#        print(self.grid_info_screen(10))
#
#        self.check('MERGE')



    def split_node(self, pos):
        """ Split node at position pos """
        self.grid.insert(pos+1, Node(self.get_node_height(pos)/2.0, self.get_node_density(pos), self.get_node_temperature(pos), \
                                     self.get_node_liquid_water_content(pos)/2.0, self.get_node_ice_fraction(pos)))
        self.update_node(pos, self.get_node_height(pos)/2.0, self.get_node_temperature(pos), \
                                     self.get_node_ice_fraction(pos), self.get_node_liquid_water_content(pos)/2.0)

        self.number_nodes += 1



    def update_node(self, no, height, temperature, ice_fraction, liquid_water_content):
        """ Update properties of a specific node """

        self.logger.debug('Update node')
        self.set_node_height(no,height)
        self.set_node_temperature(no,temperature)
        self.set_node_ice_fraction(no,ice_fraction)
        self.set_node_liquid_water_content(no,liquid_water_content)



    def check(self, name):
        """ Function checks whether temperature and layer heights are within the valid range """
        if np.min(self.get_height()) < 0.01:
            self.logger.error(name)
            self.logger.error('Layer height is smaller than the user defined minimum new_height')
            self.logger.error(self.get_height())
            self.logger.error(self.get_density())
        if np.max(self.get_temperature()) > 273.2:
            self.logger.error(name)
            self.logger.error('Layer temperature exceeds 273.16 K')
            self.logger.error(self.get_temperature())
            self.logger.error(self.get_density())
        if np.max(self.get_height()) > 1.0:
            self.logger.error(name)
            self.logger.error('Layer height exceeds 1.0 m')
            self.logger.error(self.get_height())
            self.logger.error(self.get_density())



    def update_grid(self):
        """
            The first step is to ensure that the first layer always has a
            defined height. The underlying layers are then adjusted, whereby
            two different options are available:

                (i)  log_profile
                (ii) adaptive_profile
                (iii) None

            (i)  The log-profile algorithm arranges the mesh logarithmically.
                 The user gives a stretching factor (layer_stretching) that
                 determines the increase in layer heights.

            (ii) The adjustment of the profile by means of the XX method is
                 done on the basis of the similarity of layers. Layers with very
                 similar states (temperature and density) are joined together. The
                 similarity is determined by user-specified threshold values
                 (temperature_threshold_merging, density_threshold_merging). In
                 addition, the maximum number of merging steps per time step
                 can be specified (merge_max).

           (iii) This option only guarantees that layer are not smaller than the
                 user specific minimum layer height
        """

        self.logger.debug('--------------------------')
        self.logger.debug('Update grid')

        #-------------------------------------------------------------------------
        # Remeshing options
        #-------------------------------------------------------------------------
        if (remesh_method=='log_profile'):
            self.log_profile()
        elif (remesh_method=='adaptive_profile'):
            self.adaptive_profile()

        # if first layer becomes very small, remove it
        if (self.get_node_height(0)<minimum_snow_layer_height):
            self.remove_node([0])


    def merge_snow_with_glacier(self, idx):

        if (self.get_node_density(idx) < snow_ice_threshold) & (self.get_node_density(idx+1) >= snow_ice_threshold):

            # Update node properties
            first_layer_height = self.get_node_height(idx)*(self.get_node_density(idx)/ice_density)
            self.update_node(idx+1, self.get_node_height(idx+1)+first_layer_height, self.get_node_temperature(idx+1), self.get_node_ice_fraction(idx+1), 0.0)

            # Remove the second layer
            self.remove_node([idx])

            #self.check('Merge snow with glacier function')



    def remove_melt_weq(self, melt, idx=0):

        """ If melting occurs, the function reduces the height of the first layer """

        self.logger.debug('Remove melt energy')

        # Convert melt (m w.e.q.) to m height
        height_diff = float(melt) / (self.get_node_density(idx) / 1000.0)   # m (snow) - negative = melt

        if height_diff != 0.0:
            remove = True
        else:
            remove = False

        while remove:

            # How much energy required to melt first layer
            melt_required = self.get_node_height(idx) * (self.get_node_density(idx) / 1000.0)

            # How much energy is left
            melt_rest = melt - melt_required

            # If not enough energy to remove first layer, first layers height is reduced by melt height
            if melt_rest <= 0:
                self.set_node_height(idx, self.get_node_height(idx) - height_diff)
                remove = False

            # If entire layer is removed
            else:
                self.remove_node([idx])
                melt -= melt_required
                height_diff = float(melt) / (self.get_node_density(idx) / 1000.0)
                remove = True

        # TOBI
        # Keep track of the fresh snow layer
        if (idx==0):
            self.set_fresh_snow_props_height(self.new_snow_height-melt)


    #===============================================================================
    # Getter and setter functions
    #===============================================================================

    # TOBI
    def set_fresh_snow_props(self, height, timestamp):
        """ Keeps track of the new snowheight """
        self.new_snow_height = height
        # Keep track of the old snow age
        self.old_snow_timestamp = self.new_snow_timestamp
        # Set the timestamp when fresh snowfall occured
        self.new_snow_timestamp = timestamp

    def set_fresh_snow_props_to_old_props(self):
        """ Old snowpack reappers at surface ... change snow age to old snow age """
        self.new_snow_timestamp = self.old_snow_timestamp

    def set_fresh_snow_props_height(self, height):
        """ Update fresh snow properties """
        self.new_snow_height = height

    def get_fresh_snow_props(self):
        return self.new_snow_height, self.new_snow_timestamp


    def set_node_temperature(self, idx, temperature):
        """ Returns temperature of node idx """
        self.grid[idx].set_layer_temperature(temperature)



    def set_temperature(self, temperature):
        """ Set temperature of profile """
        for idx in range(self.number_nodes):
            self.grid[idx].set_layer_temperature(temperature[idx])



    def set_node_height(self, idx, height):
        """ Set height of node idx """
        self.grid[idx].set_layer_height(height)



    def set_height(self, height):
        """ Set height of profile """
        for idx in range(self.number_nodes):
            self.grid[idx].set_layer_height(height[idx])



    def set_node_liquid_water_content(self, idx, liquid_water_content):
        """ Set liquid water content of node idx """
        self.grid[idx].set_layer_liquid_water_content(liquid_water_content)



    def set_liquid_water_content(self, liquid_water_content):
        """ Set the liquid water content profile """
        for idx in range(self.number_nodes):
            self.grid[idx].set_layer_liquid_water_content(liquid_water_content[idx])


    def set_node_ice_fraction(self, idx, ice_fraction):
        """ Set liquid ice_fraction of node idx """
        self.grid[idx].set_layer_ice_fraction(ice_fraction)


    def set_ice_fraction(self, ice_fraction):
        """ Set the ice_fraction profile """
        for idx in range(self.number_nodes):
            self.grid[idx].set_layer_ice_fraction(ice_fraction[idx])


    def set_node_refreeze(self, idx, refreeze):
        """ Set refreezing of node idx """
        self.grid[idx].set_layer_refreeze(refreeze)



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

    def get_snow_heights(self):
        """ Returns the heights of the snow layers """
        hlayer = []
        for idx in range(self.get_number_snow_layers()):
            hlayer.append(self.grid[idx].get_layer_height())
        return hlayer

    def get_ice_heights(self):
        """ Returns the heights of the ice layers """
        hlayer = []
        for idx in range(self.get_number_layers()):
            if (self.get_layer_density(idx)>=snow_ice_threshold):
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


    def get_node_thermal_conductivity(self, idx):
        """ Returns the thermal conductivity of node idx """
        return self.grid[idx].get_layer_thermal_conductivity()


    def get_thermal_conductivity(self):
        """ Returns the thermal conductivity profile """
        keff = []
        for idx in range(self.number_nodes):
            keff.append(self.grid[idx].get_layer_thermal_conductivity())
        return keff


    def get_node_thermal_diffusivity(self, idx):
        """ Returns the thermal diffusivityof node idx """
        return self.grid[idx].get_layer_thermal_diffusivity()


    def get_thermal_diffusivity(self):
        """ Returns the thermal diffusivity profile """
        K = []
        for idx in range(self.number_nodes):
            K.append(self.grid[idx].get_layer_thermal_diffusivity())
        return K


    def get_node_refreeze(self, idx):
        """ Returns refreezing of node idx """
        return self.grid[idx].get_layer_refreeze()


    def get_refreeze(self):
        """ Returns the refreezing profile """
        ref = []
        for idx in range(self.number_nodes):
            ref.append(self.grid[idx].get_layer_refreeze())
        return ref

    def get_node_depth(self, idx):
        d = 0
        for i in range(idx+1):
            if i==0:
                d = d + self.get_node_height(i)/2.0
            else:
                d = d + self.get_node_height(i-1)/2.0 + self.get_node_height(i)/2.0
        return d


    def get_depth(self):
        """ Returns depth profile """
        d = []
        for idx in range(self.number_nodes):
            d.append(self.get_node_depth(idx))
        return d


    def get_total_snowheight(self, verbose=False):
        """ Get the total snowheight (density<snow_ice_threshold)"""

        total = 0
        snowheight = 0
        for i in range(self.number_nodes):
            if (self.get_node_density(i)<snow_ice_threshold):
                snowheight = snowheight + self.get_node_height(i)
            total = total + self.get_node_height(i)

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
            if (self.get_node_density(i)<snow_ice_threshold):
                snowheight = snowheight + self.get_node_height(i)
            total = total + self.get_node_height(i)

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
            if (self.get_node_density(i)<snow_ice_threshold):
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
            tmp = tmp + self.get_node_height(i)

        print("Grid consists of %d nodes \t" % self.number_nodes)
        print("Total domain depth is %4.2f m \n" % tmp)



    def grid_info(self, n=-999):
        """ The function prints the state of the snowpack
            Args:
                n   : nuber of nodes to plot (from top)
        """
        if (n==-999):
            n = self.number_nodes

        self.logger.debug("Node no. \t\t  Layer height [m] \t Temperature [K] \
                          \t Density [kg m^-3] \t LWC [-] \t LW [m] \t CC [J \
                          m^-2] \t Porosity [-] \t Refreezing [m w.e.] \
                          Irreducible water content [-]")

        for i in range(n):
            self.logger.debug("%d %3.2f \t %3.2f \t %4.2f \t %2.7f \t %10.4f \t %4.4f \t  %4.8f \t %2.7f" % (i, self.get_node_height(i), self.get_node_temperature(i),
                  self.get_node_density(i), self.get_node_liquid_water_content(i), self.get_node_cold_content(i),
                  self.get_node_porosity(i), self.get_node_refreeze(i), self.get_node_irreducible_water_content(i)))
        self.logger.debug('\n\n')



    def grid_info_screen(self, n=-999):
        """ The function prints the state of the snowpack
            Args:
                n   : nuber of nodes to plot (from top)
        """
        if (n==-999):
            n = self.number_nodes

        print("Node no. \t\t  Layer height [m] \t Temperature [K] \t Density [kg m^-3] \t LWC [-] \t Retention [-] \t CC [J m^-2] \t Porosity [-] \t Refreezing [m w.e.]")

        for i in range(n):
            print("%d %3.3f \t %3.2f \t %4.2f \t %2.7f \t %2.7f \t %10.4f \t %4.4f \t  %4.8f" % (i, self.get_node_height(i), self.get_node_temperature(i),
                  self.get_node_density(i), self.get_node_liquid_water_content(i), self.get_node_irreducible_water_content(i), self.get_node_cold_content(i),
                  self.get_node_porosity(i), self.get_node_refreeze(i)))
        print('\n\n')



    def grid_check(self, level=1):
        """ The function checks the grid
            Args:
                n   : nuber of nodes to plot (from top)
        """
        #if level == 1:
        #    self.check_layer_property(self.get_height(), 'thickness', 1.01, -0.001)
        #    self.check_layer_property(self.get_temperature(), 'temperature', 273.2, 100.0)
        #    self.check_layer_property(self.get_density(), 'density', 918, 100)
        #    self.check_layer_property(self.get_liquid_water_content(), 'LWC', 1.0, 0.0)
        #    #self.check_layer_property(self.get_cold_content(), 'CC', 1000, -10**8)
        #    self.check_layer_property(self.get_porosity(), 'Porosity', 0.8, -0.00001)
        #    self.check_layer_property(self.get_refreeze(), 'Refreezing', 0.5, 0.0)



    def check_layer_property(self, property, name, maximum, minimum, n=-999, level=1):
        if np.nanmax(property) > maximum or np.nanmin(property) < minimum:
            print('%s max: %.2f min: %.2f' %(str.capitalize(name), np.nanmax(property), np.nanmin(property)))
            os._exit()
