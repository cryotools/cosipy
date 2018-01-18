import numpy as np
from constants import *
from config import *


def percolation(GRID, water, t):
    """ Percolation and refreezing of melt water through the snow- and firn pack

    t  ::  integration time/ length time step

    """

    # Courant-Friedrich Lewis Criteria
    curr_t = 0
    Tnew = 0

    Q = 0  # initial runoff [m w.e.]

    # Get height of snow layers
    hlayers = GRID.get_height()

    # Check stability criteria for diffusion
    dt_stab = c_stab * min(hlayers) / perolation_velocity

    ### array for refreezing per layer
    node_freezing = np.zeros(GRID.number_nodes)

    ### array for melting per layer
    node_melting = np.zeros(GRID.number_nodes)

    ### temporray array for liquid water per layer melted?
    liquid_water_temp = np.zeros(GRID.number_nodes)

    ### potential freezing (>0) or melting per layer (<0) depending on densities layer heights but not one actual liquid water content
    potential_energy_ice = ((zero_temperature-GRID.layer_temperatures)*GRID.layer_densities*GRID.layer_heights*spec_heat_ice)/(lat_heat_melting*water_density)

    ### idices freezing
    idx_freezing = np.where(potential_energy_ice > 0)

    ### idices melting
    idx_melting = np.where(potential_energy_ice < 0)

    # array with only the freezing layers
    node_potential_freezing=potential_energy_ice[idx_freezing]

    # fill node freezing with potential freezing afters wirds it is substracted; to complicated?
    #node_freezing[idx_freezing]=node_potential_freezing

    # upwind scheme to adapt liquid water content of entire GRID
    while curr_t < t:

        # Select appropriate time step
        dt_use = min(dt_stab, t - curr_t)

        # set liquid water content of top layer (idx, LWCnew)
        GRID.set_node_liquid_water_content(0, (float(water) / t) * dt_use)

        # melt water to add to liquid water content per time step
        liquid_water_temp[idx_melting] = (potential_energy_ice[idx_melting] / t) * dt_use

        # substract sub melt temp? do not understand?
        node_melting[idx_melting] = -(liquid_water_temp[idx_melting])*(t/dt_use)

        # Get a copy of the GRID liquid water content profile
        LWCtmp = np.copy(GRID.get_liquid_water_content())

        # Loop over all internal grid points
        for idxNode in range(1, GRID.number_nodes-1, 1):
            # Percolation
            if (GRID.get_node_liquid_water_content(idxNode - 1) - GRID.get_node_liquid_water_content(idxNode)) != 0:
                ux = (GRID.get_node_liquid_water_content(idxNode - 1) - GRID.get_node_liquid_water_content(idxNode)) / \
                     np.abs((GRID.get_node_height(idxNode - 1) / 2.0) + (GRID.get_node_height(idxNode) / 2.0))

                uy = (GRID.get_node_liquid_water_content(idxNode + 1) - GRID.get_node_liquid_water_content(idxNode)) / \
                     np.abs((GRID.get_node_height(idxNode + 1) / 2.0) + (GRID.get_node_height(idxNode) / 2.0))

                # Calculate new liquid water content
                LWCtmp[idxNode] = GRID.get_node_liquid_water_content(idxNode) + dt_use * (ux * perolation_velocity + uy * perolation_velocity)

            if idxNode == GRID.number_nodes-2:
                runoff = uy * perolation_velocity * dt_use

            # Update GRID with new liquid water content
            GRID.set_liquid_water_content(LWCtmp)

        # see if liquid water content is lower or potential freezing and use the minimum for refreezing at time step
        node_freezing_temp = np.minimum(GRID.liquid_water_contents[idx_freezing],node_potential_freezing)

        #print(node_freezing_temp)

        # substract freezing at time step from potential freezing
        node_potential_freezing -= node_freezing_temp

        #print(node_freezing_temp)

        # substract from liquid water meltted at time step? do not understand?
        # liquid_water_temp[idx_freezing] = node_freezing_temp

        #print(GRID.liquid_water_contents.shape)
        #print(liquid_water_temp.shape)
        # substract frozen water from liquid water content
        #GRID.liquid_water_contents = GRID.liquid_water_contents - liquid_water_temp

        # enlarge layer density if there is refreezing correct?
        #GRID.layer_densities = (GRID.layer_densities * GRID.layer_heights + liquid_water_temp)/GRID.layer_heights

        # Add the time step to current time
        curr_t = curr_t + dt_use

    # node freezing and node potential freezing is same at the begining; only if freezing per iteration is substracted from
    # node potential freezing node freezing which will be return is not empty
    #node_freezing[idx_freezing]=node_freezing[idx_freezing]-node_potential_freezing

    # update layer termpature when water is frozen temperature must be higher because after node potential freezing is lower
    # after one hour the termpature is raised; when there is no refreezing the temperatures has to be the same as before
    GRID.layer_temperatures[idx_freezing] = zero_temperature - ((node_potential_freezing*lat_heat_melting*water_density) \
                / (GRID.layer_densities[idx_freezing]*GRID.layer_heights[idx_freezing]*spec_heat_ice))

    #print(GRID.layer_temperatures)

    #return node_freezing, node_melting

    '''
    percolation, saturated/unsaturated, retention, refreezing, runoff
    '''

#    for idxNode in range(1, GRID.number_nodes, 1):

        ### Cold content energy needed to raise Temperature to melting point

        # Cold Content ### old? shy like that?
        # cc = -spec_heat_ice * water_density * GRID.get_node_height(idxNode) * (GRID.get_node_density(idxNode) / 1000.0) \
        ###* (GRID.get_node_temperature(idxNode) - 273.16)

        # Cold Conten new; I need SWE? Or
        #SWE = (GRID.get_node_height(idxNode) * (GRID.get_node_density(idxNode) / 1000))  ###unit SEW: (m)

        #cc = -spec_heat_ice * water_density * SWE * (
        #GRID.get_node_temperature(idxNode) - 273.16)  ###unit cc: (J m^-2) cc durch zeit ist net flux!

        # print(cc)             ### Jennings2017 maximum 20 MJ/m²


        #if cc > 0 and GRID.get_node_liquid_water_content(idxNode) > 0:
         #   energy_water = lat_heat_melting * GRID.get_node_liquid_water_content(idxNode)  ### unit energy water J m
            # print("energy_water", energy_water)
            # print("cold_content", cc, "node temperature", GRID.get_node_temperature(idxNode))
            #refreezing = cc / energy_water
            # print("refreezing:", refreezing)
            # print("stop")

            # if GRID.get_node_liquid_water_content(idxNode) > 0 and GRID.get_node_temperature(idxNode) < zero_temperature:
            #     print("match")
            #     print("CC ", cc, "energy water: ", energy_water, "LWC: ", GRID.get_node_liquid_water_content(idxNode))
            #     print(GRID.get_node_temperature(idxNode))
            ###frozen_water=


            ### if termpature below zero; refreeze water if persist and enlarge temperature


            # print("Runoff", runoff)
            # print(GRID.get_LWC())

    # for idxNode in range(0,GRID.number_nodes-1,1):
    #
    #     # absolute irreducible water content
    #     LWCmin = GRID.get_LWC_node(idxNode)*LWCfrac
    #
    #     if GRID.get_LWC_node(idxNode) > LWCmin:
    #         percofreeze = True
    #         print idxNode, 'refreeze', GRID.get_T_node(idxNode)
    #     else:
    #         percofreeze = False
    #
    #     while percofreeze:
    #
    #         percofreeze = False
    #
    #         # how much water the layer can hold?
    #
    #         # how much water must/can be shifted to next layer?
    #
    #         # if T<zeroT, how much water refreezes until T>zeroT
    #
    #         # todo How to accouont for rain and its added water, melt and energy?
    #         # percolation and refreezing module in COSIMA matlab

'''
refreezing wird an die Methode aus GRID.removeMeltEnergy() angelehnt
refreezing setzt energie frei!
refreezing in layers < 273.16 K aber neue Energie beachten!

remove certain amount of refreezing meltwater (subFreeze [m w.e.]) from the LWC[idxnode] ([m w.e.])
and add node with density p_ice [kg m^-3] below the reduced node (which then is impermeable)
'''