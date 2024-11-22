import os
from collections import OrderedDict

import numpy as np
from numba import float64, intp, optional, typed, types
from numba.experimental import jitclass

from cosipy.constants import Constants
from cosipy.cpkernel.node import Node

node_type = Node.class_type.instance_type

spec = OrderedDict()
spec["layer_heights"] = float64[:]
spec["layer_densities"] = float64[:]
spec["layer_temperatures"] = float64[:]
spec["layer_liquid_water_content"] = float64[:]
spec["layer_ice_fraction"] = optional(float64[:])
spec["number_nodes"] = intp
spec["new_snow_height"] = float64
spec["new_snow_timestamp"] = float64
spec["old_snow_timestamp"] = float64
spec["grid"] = types.ListType(node_type)

# only required for njitted functions
snow_ice_threshold = Constants.snow_ice_threshold
first_layer_height = Constants.first_layer_height
layer_stretching = Constants.layer_stretching
temperature_threshold_merging = Constants.temperature_threshold_merging
density_threshold_merging = Constants.density_threshold_merging
merge_max = Constants.merge_max
minimum_snowfall = Constants.minimum_snowfall
minimum_snow_layer_height = Constants.minimum_snow_layer_height
remesh_method = Constants.remesh_method
ice_density = Constants.ice_density
water_density = Constants.water_density
albedo_method = Constants.albedo_method


@jitclass(spec)
class Grid:
    """A ``Grid`` structure controls and stores the numerical mesh.

    The grid attribute consists of a list of nodes that each store
    information on an individual layer. The class provides various
    setter/getter functions to add, read, overwrite, merge, split,
    update or re-mesh the layers.

    Attributes:
        layer_heights (np.ndarray): Height of the snowpack layers [m].
        layer_densities (np.ndarray): Snow density of the snowpack
            layers [|kg m^-3|].
        layer_temperatures (np.ndarray): Layer temperatures [K].
        layer_liquid_water_content (np.ndarray): Liquid water content of
            the layers [|m w.e.|].
        layer_ice_fraction (np.ndarray): Volumetric ice fraction of the
            layers [-]. Default None.
        new_snow_height (float): Height of the fresh snow layer [m].
            Default None.
        new_snow_timestamp (float): Time elapsed since the last
            snowfall [s]. Default None.
        old_snow_timestamp (float): Time elapsed between the last and
            penultimate snowfalls [s]. Default None.
        grid (typed.List): Numerical mesh for glacier data.
        number_nodes (int): Number of layers in the numerical mesh.
    """

    def __init__(
        self,
        layer_heights,
        layer_densities,
        layer_temperatures,
        layer_liquid_water_content,
        layer_ice_fraction=None,
        new_snow_height=None,
        new_snow_timestamp=None,
        old_snow_timestamp=None,
    ):
        # Set class variables
        self.layer_heights = layer_heights
        self.layer_densities = layer_densities
        self.layer_temperatures = layer_temperatures
        self.layer_liquid_water_content = layer_liquid_water_content
        self.layer_ice_fraction = layer_ice_fraction

        # Number of total nodes
        self.number_nodes = len(layer_heights)

        # Track the fresh snow layer (new_snow_height, new_snow_timestamp)
        # as well as the old snow layer age (old_snow_timestamp)
        if (
            (new_snow_height is not None)
            and (new_snow_timestamp is not None)
            and (old_snow_timestamp is not None)
        ):
            self.new_snow_height = new_snow_height  # meter snow accumulation
            self.new_snow_timestamp = (
                new_snow_timestamp  # seconds since snowfall
            )
            self.old_snow_timestamp = (
                old_snow_timestamp  # snow age below fresh snow layer
            )
        else:
            # TO DO: pick better initialization values
            self.new_snow_height = 0.0
            self.new_snow_timestamp = 0.0
            self.old_snow_timestamp = 0.0

        # Do the grid initialization
        self.grid = typed.List.empty_list(node_type)

        self.init_grid()

    def init_grid(self):
        """Initialize the grid with the input data."""
        # Fill the list with node instances and fill it with user defined data
        for idxNode in range(self.number_nodes):
            layer_IF = None
            if self.layer_ice_fraction is not None:
                layer_IF = self.layer_ice_fraction[idxNode]
            self.grid.append(
                Node(
                    self.layer_heights[idxNode],
                    self.layer_densities[idxNode],
                    self.layer_temperatures[idxNode],
                    self.layer_liquid_water_content[idxNode],
                    layer_IF,
                )
            )

    def add_fresh_snow(
        self, height, density, temperature, liquid_water_content, dt
    ):
        """Add a fresh snow layer (node).

        Adds a fresh snow layer to the beginning of the node list (upper
        layer).

        Args:
            height (float): Layer height [m].
            density (float): Layer density [|kg m^-3|].
            temperature (float): Layer temperature [K].
            liquid_water_content (float): Liquid water content of the
                layer [|m w.e.|].
            dt (int): Integration time [s].
        """

        # Add new node
        self.grid.insert(
            0, Node(height, density, temperature, liquid_water_content, None)
        )

        # Increase node counter
        self.number_nodes += 1

        if height < minimum_snowfall:
            # Ignore impact of small snowfall on fresh snow layer properties
            self.set_fresh_snow_props_update_time(dt)
        else:
            # Set the fresh snow properties for albedo calculation (height and timestamp)
            self.set_fresh_snow_props(height)

    def remove_node(self, idx: list = None):
        """Remove a layer (node) from the grid (node list).

        Args:
            idx: Indices of the node to be removed. The first node is
                removed if empty or None. Default ``None``.
        """

        # Remove node from list when there is at least one node
        if self.grid:
            if idx is None or not idx:
                self.grid.pop(0)
                self.number_nodes -= 1  # Decrease node counter
            else:
                for index in sorted(idx, reverse=True):
                    del self.grid[index]
                self.number_nodes -= len(idx)

    def merge_nodes(self, idx: int):
        """Merge two consecutive nodes.

        Merges the two nodes at location ``idx`` and ``idx+1``. The node
        at ``idx`` is updated with the new properties (height, liquid
        water content, ice fraction, temperature). The node at ``idx+1``
        is deleted after merging.

        Args:
            idx: Index of the node to be removed. The first node is
                removed if no index is provided.
        """
        # Get overburden pressure for consistency check
        # w0 = self.get_node_height(idx) * self.get_node_density(
        #     idx
        # ) + self.get_node_height(idx + 1) * self.get_node_density(idx + 1)

        # New layer height by adding up the height of the two layers
        new_height = self.get_node_height(idx) + self.get_node_height(idx + 1)

        # Update liquid water
        # new_liquid_water_content = self.get_node_liquid_water_content(idx) + self.get_node_liquid_water_content(idx+1)
        new_liquid_water_content = (
            self.get_node_liquid_water_content(idx) * self.get_node_height(idx)
            + self.get_node_liquid_water_content(idx + 1)
            * self.get_node_height(idx + 1)
        ) / new_height

        # Update ice fraction
        new_ice_fraction = (
            self.get_node_ice_fraction(idx) * self.get_node_height(idx)
            + self.get_node_ice_fraction(idx + 1)
            * self.get_node_height(idx + 1)
        ) / new_height

        # New air porosity
        new_air_porosity = 1 - new_liquid_water_content - new_ice_fraction

        if (
            abs(
                1
                - new_ice_fraction
                - new_air_porosity
                - new_liquid_water_content
            )
            > 1e-8
        ):
            print(
                "Merging is not mass consistent",
                (
                    new_ice_fraction
                    + new_air_porosity
                    + new_liquid_water_content
                ),
            )

        # Calc new temperature
        new_temperature = (
            self.get_node_height(idx) / new_height
        ) * self.get_node_temperature(idx) + (
            self.get_node_height(idx + 1) / new_height
        ) * self.get_node_temperature(
            idx + 1
        )

        # Update node properties
        self.update_node(
            idx,
            new_height,
            new_temperature,
            new_ice_fraction,
            new_liquid_water_content,
        )

        # Remove the second layer
        self.remove_node([idx + 1])

    def correct_layer(self, idx: int, min_height: float):
        """Adjust the height of a given layer.

        Adjusts the height of the layer at index ``idx`` to the given
        height ``min_height``. First, the layers below are merged until
        the height is large enough to allow for the adjustment. Then the
        layer is merged with the following layer.

        Args:
            idx: Index of the node to be removed.
            min_height: New layer height [m].

        """
        # New layer height by adding up the height of the two layers
        total_height = self.get_node_height(idx)

        # Merge with underlying layers until the height of the layer is
        # greater than the given height.
        while (total_height < min_height) & (
            idx + 1 < self.get_number_layers()
        ):
            if (self.get_node_density(idx) < snow_ice_threshold) & (
                self.get_node_density(idx + 1) < snow_ice_threshold
            ):
                self.merge_nodes(idx)
            elif (self.get_node_density(idx) >= snow_ice_threshold) & (
                self.get_node_density(idx + 1) >= snow_ice_threshold
            ):
                self.merge_nodes(idx)
            else:
                break

            # Recalculate total height
            total_height = self.get_node_height(idx)

        # Only merge snow-snow or glacier-glacier, and if the height is
        # greater than the minimum height
        if total_height > min_height:
            # Get new heights for layer 0 and 1
            h0 = min_height
            h1 = total_height - min_height

            """Update liquid water content.
            Fills the upper layer with water until maximum retention.
            The remaining water is assigned to the second layer.
            If LWC exceeds the irreducible water content of the first
            layer, then the first layer is filled and the rest assigned
            to the second layer.
            """
            if (
                self.get_node_liquid_water_content(idx)
                - self.get_node_irreducible_water_content(idx)
            ) > 0:
                lw0 = self.get_node_irreducible_water_content(
                    idx
                ) * self.get_node_height(idx)
                lw1 = self.get_node_liquid_water_content(
                    idx
                ) * self.get_node_height(
                    idx
                ) - self.get_node_irreducible_water_content(
                    idx
                ) * self.get_node_height(
                    idx
                )
            # if LWC<WC_irr, then assign all water to the first layer
            else:
                lw0 = self.get_node_liquid_water_content(
                    idx
                ) * self.get_node_height(idx)
                lw1 = 0.0

            # Update ice fraction
            if0 = self.get_node_ice_fraction(idx)
            if1 = self.get_node_ice_fraction(idx)

            # Temperature
            T0 = self.get_node_temperature(idx)
            T1 = self.get_node_temperature(idx)

            # New volume fractions and density
            lwc0 = lw0 / h0
            lwc1 = lw1 / h1
            por0 = 1 - lwc0 - if0
            por1 = 1 - lwc1 - if1

            # Check for consistency
            if (abs(1 - if0 - por0 - lwc0) > 1e-8) | (
                abs(1 - if1 - por1 - lwc1) > 1e-8
            ):
                print(
                    "Correct layer is not mass consistent [Layer 0]",
                    (if0, por0, lwc0),
                )
                print(
                    "Correct layer is not mass consistent [Layer 1]",
                    (if0, por0, lwc0),
                )

            self.update_node(idx, h0, T0, if0, lwc0)  # Update node properties
            self.grid.insert(
                idx + 1, Node(h1, self.get_node_density(idx), T1, lwc1, if1)
            )

            self.number_nodes += 1  # Update node counter

    def log_profile(self):
        """Remesh the layer heights logarithmically.

        This algorithm remeshes the layer heights (numerical grid)
        logarithmically using a given stretching factor and first layer
        height. Both are defined in ``constants.py``:

        * The stretching factor is defined by ``layer_stretching``.
        * The first layer height is defined by ``first_layer_height``.

        E.g. for the stretching factor, a value of 1.1 corresponds to a
        10% stretching from one layer to the next.
        """
        last_layer_height = first_layer_height

        hsnow = self.get_total_snowheight()  # Total snowheight

        hrest = hsnow  # How much snow is not remeshed

        # First remesh the snowpack
        idx = 0

        while idx < self.get_number_snow_layers():
            if hrest >= last_layer_height:
                # Correct first layer
                self.correct_layer(idx, last_layer_height)

                hrest = hrest - last_layer_height

                # Height for the next layer
                last_layer_height = layer_stretching * last_layer_height

            # if the last layer is smaller than the required height,
            # then merge with the previous layer
            elif (hrest < last_layer_height) & (idx > 0):
                self.merge_nodes(idx - 1)

            idx = idx + 1

        # get the glacier depth
        hrest = self.get_total_height() - self.get_total_snowheight()

        # get number of snow layers
        idx = self.get_number_snow_layers()

        # then the glacier
        while idx < self.get_number_layers():
            if hrest >= last_layer_height:
                # Correct first layer
                self.correct_layer(idx, last_layer_height)

                hrest = hrest - last_layer_height

                # Height for the next layer
                last_layer_height = layer_stretching * last_layer_height

            # if the last layer is smaller than the required height,
            # then merge with the previous layer
            elif hrest < last_layer_height:
                self.merge_nodes(idx - 1)

            idx = idx + 1

    def adaptive_profile(self):
        """Remesh according to certain layer state criteria.

        This algorithm is an alternative to logarithmic remeshing.
        It checks the similarity of two consecutive layers. Layers are
        merged if:

        (1) the density difference between one layer and the next is
            smaller than the user defined threshold.
        (2) the temperature difference is smaller than the user defined
            threshold.
        (3) the number of merges per time step does not exceed the user
            defined threshold.

        The thresholds are defined by ``temperature_threshold_merging``,
        ``density_threshold_merging``, and ``merge_max`` in
        ``constants.py``.
        """
        # First remesh the snowpack
        idx = 0
        merge_counter = 0
        while idx < self.get_number_snow_layers() - 1:
            dT = np.abs(
                self.get_node_temperature(idx)
                - self.get_node_temperature(idx + 1)
            )
            dRho = np.abs(
                self.get_node_density(idx) - self.get_node_density(idx + 1)
            )

            if (
                (dT <= temperature_threshold_merging)
                & (dRho <= density_threshold_merging)
                & (self.get_node_height(idx) <= 0.1)
                & (merge_counter <= merge_max)
            ):
                self.merge_nodes(idx)
                merge_counter = merge_counter + 1
            # elif ((self.get_node_height(idx)<=minimum_snow_layer_height) & (dRho<=density_threshold_merging)):
            elif self.get_node_height(idx) <= minimum_snow_layer_height:
                self.remove_node([idx])
            else:
                idx += 1

        # Remesh ice
        # remeshing layer 0 done by correct_layer above
        min_ice_idx = max(1, self.get_number_snow_layers())
        # Ensure top ice layer has first_layer_height when thin snow layers will be removed in update_grid
        if (min_ice_idx == 1) & (self.get_node_height(0) < first_layer_height):
            self.correct_layer(min_ice_idx, first_layer_height)
            min_ice_idx += 1

        idx = min_ice_idx
        while idx < self.get_number_layers() - 1:
            # Correct first layer
            if self.get_node_height(idx) < minimum_snow_layer_height:
                self.merge_nodes(idx)
            else:
                idx += 1
        self.correct_layer(0, first_layer_height)

    def split_node(self, pos: int):
        """Split node at position.

        Splits a node at a location index ``pos`` into two similar
        nodes. The new nodes at location ``pos`` and ``pos+1`` will have
        the same properties (height, liquid water content, ice fraction,
        temperature).

        Args:
            pos: Index of the node to split.
        """
        self.grid.insert(
            pos + 1,
            Node(
                self.get_node_height(pos) / 2.0,
                self.get_node_density(pos),
                self.get_node_temperature(pos),
                self.get_node_liquid_water_content(pos) / 2.0,
                self.get_node_ice_fraction(pos),
            ),
        )
        self.update_node(
            pos,
            self.get_node_height(pos) / 2.0,
            self.get_node_temperature(pos),
            self.get_node_ice_fraction(pos),
            self.get_node_liquid_water_content(pos) / 2.0,
        )

        self.number_nodes += 1

    def update_node(
        self, idx, height, temperature, ice_fraction, liquid_water_content
    ):
        """Update properties of a specific node.

        Updates a layer's attributes for ``height``, ``temperature``,
        ``ice_fraction``, and ``liquid_water_content``. The density
        cannot be updated as it is derived from air porosity, liquid
        water content, and ice fraction.

        Args:
            idx (int): Index of the layer to be updated.
            height (float): Layer's new snowpack height [m].
            temperature (float): Layer's new temperature [K].
            ice_fraction (float): Layer's new ice fraction [-].
            liquid_water_content (float): Layer's new liquid water
                content [|m w.e.|].
        """
        self.set_node_height(idx, height)
        self.set_node_temperature(idx, temperature)
        self.set_node_ice_fraction(idx, ice_fraction)
        self.set_node_liquid_water_content(idx, liquid_water_content)

    def check(self, name):
        """Check layer temperature and height are within a valid range."""
        if np.min(self.get_height()) < 0.01:
            print(name)
            print(
                "Layer height is smaller than the user defined minimum new_height"
            )
            print(self.get_height())
            print(self.get_density())
        if np.max(self.get_temperature()) > 273.2:
            print(name)
            print("Layer temperature exceeds 273.16 K")
            print(self.get_temperature())
            print(self.get_density())
        if np.max(self.get_height()) > 1.0:
            print(name)
            print("Layer height exceeds 1.0 m")
            print(self.get_height())
            print(self.get_density())

    def update_grid(self):
        """Remesh the layers (numerical grid).

        Two algorithms are currently implemented to remesh layers:

            (i)  log_profile
            (ii) adaptive_profile

        (i)  The log-profile algorithm arranges the mesh
             logarithmically. The user specifies the stretching factor
             ``layer_stretching`` in ``constants.py`` to determine the
             increase in layer heights.

        (ii) Profile adjustment uses layer similarity. Layers with very
             similar temperature and density states are joined.
             Similarity is determined from the user-specified threshold
             values ``temperature_threshold_merging`` and
             ``density_threshold_merging`` in ``constants.py``. The
             maximum number of merging steps per time step is specified
             by ``merge_max``.
        """
        if remesh_method == "log_profile":
            self.log_profile()
        elif remesh_method == "adaptive_profile":
            self.adaptive_profile()

        # remove the first layer if it is too small
        if self.get_node_height(0) < minimum_snow_layer_height:
            self.remove_node([0])

    def merge_snow_with_glacier(self, idx: int):
        """Merge a snow layer with an ice layer.

        Merges a snow layer at location ``idx`` (density smaller than
        the ``snow_ice_threshold`` value in ``constants.py``) with an
        ice layer at location ``idx+1``.

        Args:
            idx: Index of the snow layer.
        """
        if (self.get_node_density(idx) < snow_ice_threshold) & (
            self.get_node_density(idx + 1) >= snow_ice_threshold
        ):
            # Update node properties
            surface_layer_height = self.get_node_height(idx) * (
                self.get_node_density(idx) / ice_density
            )
            self.update_node(
                idx + 1,
                self.get_node_height(idx + 1) + surface_layer_height,
                self.get_node_temperature(idx + 1),
                self.get_node_ice_fraction(idx + 1),
                0.0,
            )

            self.remove_node([idx])  # Remove the second layer

            # self.check('Merge snow with glacier function')

    def remove_melt_weq(self, melt: float, idx: int = 0) -> float:
        """Remove mass from a layer.

        Reduces the mass/height of layer ``idx`` by the available melt
        energy.

        Args:
            melt: Snow water equivalent of melt [|m w.e.|].
            idx: Index of the layer. If no value is given, the function
                acts on the first layer.

        Returns:
            Liquid water content from removed layers.
        """
        lwc_from_layers = 0.0

        while melt > 0.0 and idx < self.number_nodes:
            # Get SWE of layer
            SWE = self.get_node_height(idx) * (
                self.get_node_density(idx) / water_density
            )
            # Remove melt from layer and set new snowheight
            if melt < SWE:
                self.set_node_height(
                    idx,
                    (SWE - melt)
                    / (self.get_node_density(idx) / water_density),
                )
                melt = 0.0
            # remove layer otherwise and continue loop
            elif melt >= SWE:
                lwc_from_layers = (
                    lwc_from_layers
                    + self.get_node_liquid_water_content(idx)
                    * self.get_node_height(idx)
                )
                self.remove_node([idx])
                melt = melt - SWE

        # Keep track of the fresh snow layer
        if idx == 0:
            self.set_fresh_snow_props_height(self.new_snow_height - melt)

        return lwc_from_layers

    # ===============================================================================
    # Getter and setter functions
    # ===============================================================================

    def set_fresh_snow_props(self, height: float):
        """Track the new snowheight.

        Args:
            height: Height of the fresh snow layer [m].
        """
        self.new_snow_height = height
        # Keep track of the old snow age
        self.old_snow_timestamp = self.new_snow_timestamp
        # Set the timestamp to zero
        self.new_snow_timestamp = 0

    def set_fresh_snow_props_to_old_props(self):
        """Revert the timestamp of fresh snow properties.

        Reverts the timestamp of fresh snow properties to that of the
        underlying snow layer. This is used internally to track the
        albedo properties of the first snow layer.
        """
        self.new_snow_timestamp = self.old_snow_timestamp

    def set_fresh_snow_props_update_time(self, seconds: float):
        """Update the timestamp of the snow properties.

        Args:
            seconds: seconds without snowfall [s].
        """
        self.old_snow_timestamp = self.old_snow_timestamp + seconds
        # Set the timestamp to zero
        self.new_snow_timestamp = self.new_snow_timestamp + seconds

    def set_fresh_snow_props_height(self, height: float):
        """Update the fresh snow layer height.

        This is used internally to track the albedo properties of the
        first snow layer.
        """
        self.new_snow_height = height

    def get_fresh_snow_props(self) -> tuple:
        """Get the first snow layer's properties.

        This is used internally to track the albedo properties of the
        first snow layer.

        Returns:
            First snow layer's updated height, time elapsed since the
            last snowfall, and the time elapsed between the last and
            penultimate snowfall.
        """
        return (
            self.new_snow_height,
            self.new_snow_timestamp,
            self.old_snow_timestamp,
        )

    def set_node_temperature(self, idx: int, temperature: float):
        """Set the temperature of a layer (node) at location ``idx``.

        Args:
            idx: Index of the layer.
            temperature: Layer's new temperature [K].
        """
        self.grid[idx].set_layer_temperature(temperature)

    def set_temperature(self, temperature: np.ndarray):
        """Set all layer temperatures.

        Args:
            temperature: New layer temperatures [K].
        """
        for idx in range(self.number_nodes):
            self.grid[idx].set_layer_temperature(temperature[idx])

    def set_node_height(self, idx: int, height: float):
        """Set a node's height."""
        self.grid[idx].set_layer_height(height)

    def set_height(self, height: np.ndarray):
        """Set the height profile."""
        for idx in range(self.number_nodes):
            self.grid[idx].set_layer_height(height[idx])

    def set_node_liquid_water_content(
        self, idx: int, liquid_water_content: float
    ):
        """Set a node's liquid water content."""
        self.grid[idx].set_layer_liquid_water_content(liquid_water_content)

    def set_liquid_water_content(self, liquid_water_content: np.ndarray):
        """Set the liquid water content profile."""
        for idx in range(self.number_nodes):
            self.grid[idx].set_layer_liquid_water_content(
                liquid_water_content[idx]
            )

    def set_node_ice_fraction(self, idx: int, ice_fraction: float):
        """Set a node's ice fraction."""
        self.grid[idx].set_layer_ice_fraction(ice_fraction)

    def set_ice_fraction(self, ice_fraction: np.ndarray):
        """Set the ice fraction profile."""
        for idx in range(self.number_nodes):
            self.grid[idx].set_layer_ice_fraction(ice_fraction[idx])

    def set_node_refreeze(self, idx: int, refreeze: float):
        """Set the amount of refrozen water in a node."""
        self.grid[idx].set_layer_refreeze(refreeze)

    def set_refreeze(self, refreeze: np.ndarray):
        """Set the refrozen water profile."""
        for idx in range(self.number_nodes):
            self.grid[idx].set_layer_refreeze(refreeze[idx])

    def get_temperature(self) -> list:
        """Get the temperature profile."""
        return [
            self.grid[idx].get_layer_temperature()
            for idx in range(self.number_nodes)
        ]

    def get_node_temperature(self, idx: int):
        """Get a node's temperature."""
        return self.grid[idx].get_layer_temperature()

    def get_specific_heat(self) -> list:
        """Get the specific heat capacity profile (air+water+ice)."""
        return [
            self.grid[idx].get_layer_specific_heat()
            for idx in range(self.number_nodes)
        ]

    def get_node_specific_heat(self, idx: int):
        """Get a node's specific heat capacity (air+water+ice)."""
        return self.grid[idx].get_layer_specific_heat()

    def get_height(self) -> list:
        """Get the heights of all the layers."""
        return [
            self.grid[idx].get_layer_height()
            for idx in range(self.number_nodes)
        ]

    def get_snow_heights(self) -> list:
        """Get the heights of the snow layers."""
        return [
            self.grid[idx].get_layer_height()
            for idx in range(self.get_number_snow_layers())
        ]

    def get_ice_heights(self) -> list:
        """Get the heights of the ice layers."""
        return [
            self.grid[idx].get_layer_height()
            for idx in range(self.number_nodes)
            if (self.get_node_density(idx) >= snow_ice_threshold)
        ]

    def get_node_height(self, idx: int):
        """Get a node's layer height."""
        return self.grid[idx].get_layer_height()

    def get_node_density(self, idx: int):
        """Get a node's density."""
        return self.grid[idx].get_layer_density()

    def get_density(self) -> list:
        """Get the density profile."""
        return [
            self.grid[idx].get_layer_density()
            for idx in range(self.number_nodes)
        ]

    def get_node_liquid_water_content(self, idx: int):
        """Get a node's liquid water content."""
        return self.grid[idx].get_layer_liquid_water_content()

    def get_liquid_water_content(self) -> list:
        """Get a profile of the liquid water content."""
        return [
            self.grid[idx].get_layer_liquid_water_content()
            for idx in range(self.number_nodes)
        ]

    def get_node_ice_fraction(self, idx: int):
        """Get a node's ice fraction."""
        return self.grid[idx].get_layer_ice_fraction()

    def get_ice_fraction(self) -> list:
        """Get a profile of the ice fraction."""
        return [
            self.grid[idx].get_layer_ice_fraction()
            for idx in range(self.number_nodes)
        ]

    def get_node_irreducible_water_content(self, idx: int):
        """Get a node's irreducible water content."""
        return self.grid[idx].get_layer_irreducible_water_content()

    def get_irreducible_water_content(self) -> list:
        """Get a profile of the irreducible water content."""
        return [
            self.grid[idx].get_layer_irreducible_water_content()
            for idx in range(self.number_nodes)
        ]

    def get_node_cold_content(self, idx: int):
        """Get a node's cold content."""
        return self.grid[idx].get_layer_cold_content()

    def get_cold_content(self) -> list:
        """Get the cold content profile."""
        return [
            self.grid[idx].get_layer_cold_content()
            for idx in range(self.number_nodes)
        ]

    def get_node_porosity(self, idx: int):
        """Get a node's porosity."""
        return self.grid[idx].get_layer_porosity()

    def get_porosity(self) -> list:
        """Get the porosity profile."""
        return [
            self.grid[idx].get_layer_porosity()
            for idx in range(self.number_nodes)
        ]

    def get_node_thermal_conductivity(self, idx: int):
        """Get a node's thermal conductivity."""
        return self.grid[idx].get_layer_thermal_conductivity()

    def get_thermal_conductivity(self) -> list:
        """Get the thermal conductivity profile."""
        return [
            self.grid[idx].get_layer_thermal_conductivity()
            for idx in range(self.number_nodes)
        ]

    def get_node_thermal_diffusivity(self, idx: int):
        """Get a node's thermal diffusivity."""
        return self.grid[idx].get_layer_thermal_diffusivity()

    def get_thermal_diffusivity(self) -> list:
        """Get the thermal diffusivity profile."""
        return [
            self.grid[idx].get_layer_thermal_diffusivity()
            for idx in range(self.number_nodes)
        ]

    def get_node_refreeze(self, idx: int):
        """Get the amount of refrozen water in a node."""
        return self.grid[idx].get_layer_refreeze()

    def get_refreeze(self) -> list:
        """Get the profile of refrozen water."""
        return [
            self.grid[idx].get_layer_refreeze()
            for idx in range(self.number_nodes)
        ]

    def get_node_depth(self, idx: int):
        """Get a node's depth relative to the surface."""
        d = self.get_node_height(idx) / 2.0
        if idx != 0:
            for i in range(idx):
                d = d + self.get_node_height(i)
        return d

    def get_depth(self) -> list:
        """Get a depth profile."""
        h = np.array(self.get_height())
        z = np.empty_like(h)  # faster than copy
        z[0] = 0.5 * h[0]
        z[1:] = np.cumsum(h[:-1]) + (0.5 * h[1:])

        return [z[idx] for idx in range(self.number_nodes)]

    def get_total_snowheight(self, verbose=False):
        """Get the total snowheight (density<snow_ice_threshold)."""
        return sum(self.get_snow_heights())

    def get_total_height(self, verbose=False):
        """Get the total domain height."""
        return sum(self.get_height())

    def get_number_snow_layers(self):
        """Get the number of snow layers (density<snow_ice_threshold)."""
        nlayers = [
            1
            for idx in range(self.number_nodes)
            if self.get_node_density(idx) < snow_ice_threshold
        ]
        return sum(nlayers)

    def get_number_layers(self):
        """Get the number of layers."""
        return self.number_nodes

    def info(self):
        """Print some information on grid."""

        print("*" * 30)
        print(f"Number of nodes: {self.number_nodes}")
        print("*" * 30)

        tmp = 0
        for i in range(self.number_nodes):
            tmp = tmp + self.get_node_height(i)

        print(f"Grid consists of {self.number_nodes} nodes")
        print(f"Total domain depth is {tmp} m")

    def grid_info(self, n: int = -999):
        """Print the state of the snowpack.

        Args:
            n: Number of nodes to plot from the top. Default -999.
        """
        if n == -999:
            n = self.number_nodes

        print(
            "Node no., Layer height [m], Temperature [K], Density [kg m^-3], \
               LWC [-], LW [m], CC [J m^-2], Porosity [-], Refreezing [m w.e.], \
               Irreducible water content [-]"
        )

        for i in range(n):
            print(
                i,
                self.get_node_height(i),
                self.get_node_temperature(i),
                self.get_node_density(i),
                self.get_node_liquid_water_content(i),
                self.get_node_cold_content(i),
                self.get_node_porosity(i),
                self.get_node_refreeze(i),
                self.get_node_irreducible_water_content(i),
            )

    def grid_info_screen(self, n: int = -999):
        """Print the state of the snowpack.

        Args:
            n: Number of nodes to plot from the top. Default -999.
        """
        if n == -999:
            n = self.number_nodes

        print(
            "Node no., Layer height [m], Temperature [K], Density [kg m^-3], LWC [-], \
               Retention [-], CC [J m^-2], Porosity [-], Refreezing [m w.e.]"
        )

        for i in range(n):
            print(
                i,
                self.get_node_height(i),
                self.get_node_temperature(i),
                self.get_node_density(i),
                self.get_node_liquid_water_content(i),
                self.get_node_irreducible_water_content(i),
                self.get_node_cold_content(i),
                self.get_node_porosity(i),
                self.get_node_refreeze(i),
            )

    def grid_check(self, level: int = 1):
        """Check the grid for out of range values.

        Args:
            level: Level number.
        """
        # if level == 1:
        #    self.check_layer_property(self.get_height(), 'thickness', 1.01, -0.001)
        #    self.check_layer_property(self.get_temperature(), 'temperature', 273.2, 100.0)
        #    self.check_layer_property(self.get_density(), 'density', 918, 100)
        #    self.check_layer_property(self.get_liquid_water_content(), 'LWC', 1.0, 0.0)
        #    #self.check_layer_property(self.get_cold_content(), 'CC', 1000, -10**8)
        #    self.check_layer_property(self.get_porosity(), 'Porosity', 0.8, -0.00001)
        #    self.check_layer_property(self.get_refreeze(), 'Refreezing', 0.5, 0.0)
        pass

    def check_layer_property(
        self, layer_property, name, maximum, minimum, n=-999, level=1
    ):
        if (
            np.nanmax(layer_property) > maximum
            or np.nanmin(layer_property) < minimum
        ):
            print(
                str.capitalize(name),
                "max",
                np.nanmax(layer_property),
                "min",
                np.nanmin(layer_property),
            )
            os._exit()
