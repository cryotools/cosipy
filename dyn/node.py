#!/usr/bin/python

class Node:
    """ This is the basic class which contains information of individual grid point. """


    def __init__(self, hlayer, density, T, LWC):
        """ Initialize the node with:
            loc      :      location of the node 0 = lowest node
            hlayer   :      height of the layer [m]
            T        :      temperature of the layer [K]
            LWD      :      liquid water content [m w.e.]
        """

        # Define class variables
        self.hlayer = hlayer
        self.rho = density
        self.T = T
        self.LWC = LWC


    def __del__(self):
        """ Remove node """


    #-------------------------------
    # Define getter functions
    #-------------------------------
    def get_hlayer(self):
        """ Return the layer height """
        return self.hlayer

    def get_rho(self):
        """ Return the mean density of the layer """
        return self.rho

    def get_T(self):
        """ Return the mean temperature of the layer """
        return self.T

    def get_LWC(self):
        """ Return the liquid water content of the layer """
        return self.LWC

    #-------------------------------
    # Define setter functions
    #-------------------------------
    def set_hlayer(self, hlayer):
        """ Set the layer height """
        self.hlayer = hlayer

    def set_rho(self, rho):
        """ Set the mean density of the layer """
        self.rho = rho

    def set_T(self, T):
        """ Set the mean temperature of the layer """
        self.T = T

    def set_LWC(self, LWC):
        """ Set the liquid water content of the layer """
        self.LWC = LWC
       
