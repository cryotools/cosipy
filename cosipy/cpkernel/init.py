import numpy as np
from numba import njit

from cosipy.constants import Constants
from cosipy.cpkernel.grid import Grid


def init_snowpack(DATA):
    """Initialise the snowpack.

    Args:
        DATA (xarray.Dataset): Glacier data for a single grid point.

    Returns:
        Grid: Initialised glacier data structure with snowpack.
    """

    # Declare locally for faster lookup
    initial_snowheight_constant = Constants.initial_snowheight_constant
    initial_glacier_height = Constants.initial_glacier_height
    initial_glacier_layer_heights = Constants.initial_glacier_layer_heights
    temperature_bottom = Constants.temperature_bottom
    initial_top_density_snowpack = Constants.initial_top_density_snowpack
    initial_bottom_density_snowpack = Constants.initial_bottom_density_snowpack
    ice_density = Constants.ice_density

    # Check for WRF data
    if "SNOWHEIGHT" in DATA:
        initial_snowheight = DATA.SNOWHEIGHT.values
        if np.isnan(initial_snowheight):
            initial_snowheight = 0.0
    else:
        initial_snowheight = initial_snowheight_constant
    temperature_top = np.minimum(DATA.T2.values[0], Constants.zero_temperature)

    # Do the vertical interpolation
    layer_heights = []
    layer_densities = []
    layer_T = []
    layer_liquid_water = []

    if initial_snowheight > 0.0:
        optimal_height = 0.1  # 10 cm
        nlayers = int(min(initial_snowheight / optimal_height, 5))
        dT = (temperature_top - temperature_bottom) / (
            initial_snowheight + initial_glacier_height
        )
        if nlayers == 0:
            layer_heights = np.array([initial_snowheight])
            layer_densities = np.array([initial_top_density_snowpack])
            layer_T = np.array([temperature_top])
            layer_liquid_water = np.array([0.0])
        elif nlayers > 0:
            drho = (
                initial_top_density_snowpack - initial_bottom_density_snowpack
            ) / initial_snowheight
            layer_heights = np.ones(nlayers) * (initial_snowheight / nlayers)
            layer_densities = np.array(
                [
                    initial_top_density_snowpack
                    - drho * (initial_snowheight / nlayers) * i
                    for i in range(1, nlayers + 1)
                ]
            )
            layer_T = np.array(
                [
                    temperature_top - dT * (initial_snowheight / nlayers) * i
                    for i in range(1, nlayers + 1)
                ]
            )
            layer_liquid_water = np.zeros(nlayers)

        # add glacier
        nlayers = int(initial_glacier_height / initial_glacier_layer_heights)
        layer_heights = np.append(
            layer_heights, np.ones(nlayers) * initial_glacier_layer_heights
        )
        layer_densities = np.append(
            layer_densities, np.ones(nlayers) * ice_density
        )
        layer_T = np.append(
            layer_T,
            [
                layer_T[-1] - dT * initial_glacier_layer_heights * i
                for i in range(1, nlayers + 1)
            ],
        )
        layer_liquid_water = np.append(layer_liquid_water, np.zeros(nlayers))

    else:  # only glacier
        nlayers = int(initial_glacier_height / initial_glacier_layer_heights)
        dT = (temperature_top - temperature_bottom) / initial_glacier_height
        layer_heights = np.ones(nlayers) * initial_glacier_layer_heights
        layer_densities = np.ones(nlayers) * ice_density
        layer_T = np.array(
            [
                temperature_top - dT * initial_glacier_layer_heights * i
                for i in range(1, nlayers + 1)
            ]
        )
        layer_liquid_water = np.zeros(nlayers)

    # Initialize grid, the grid class contains all relevant grid information
    GRID = create_grid_jitted(
        np.array(layer_heights, dtype=np.float64),
        np.array(layer_densities, dtype=np.float64),
        np.array(layer_T, dtype=np.float64),
        np.array(layer_liquid_water, dtype=np.float64),
        None,
        None,
        None,
        None,
    )

    return GRID


def load_snowpack(GRID_RESTART):
    """Initialize grid from restart file."""

    # Number of layers
    num_layers = int(GRID_RESTART.NLAYERS.values)

    # Init layer height
    # Weird slicing position to accommodate NestedNamespace in WRF_X_CSPY
    layer_heights = GRID_RESTART.LAYER_HEIGHT.values[0:num_layers]
    layer_density = GRID_RESTART.LAYER_RHO.values[0:num_layers]
    layer_T = GRID_RESTART.LAYER_T.values[0:num_layers]
    layer_LWC = GRID_RESTART.LAYER_LWC.values[0:num_layers]
    layer_IF = GRID_RESTART.LAYER_IF.values[0:num_layers]

    new_snow_height = np.float64(GRID_RESTART.new_snow_height.values)
    new_snow_timestamp = np.float64(GRID_RESTART.new_snow_timestamp.values)
    old_snow_timestamp = np.float64(GRID_RESTART.old_snow_timestamp.values)

    GRID = create_grid_jitted(
        layer_heights,
        layer_density,
        layer_T,
        layer_LWC,
        layer_IF,
        new_snow_height,
        new_snow_timestamp,
        old_snow_timestamp,
    )

    return GRID


@njit
def create_grid_jitted(
    layer_heights: np.ndarray,
    layer_density: np.ndarray,
    layer_T: np.ndarray,
    layer_LWC: np.ndarray,
    layer_IF: np.ndarray,
    new_snow_height: float,
    new_snow_timestamp: float,
    old_snow_timestamp: float,
):
    """Create Grid with JIT.
    
    Returns:
        Grid: Glacier data structure.
    """

    GRID = Grid(
        layer_heights,
        layer_density,
        layer_T,
        layer_LWC,
        layer_IF,
        new_snow_height,
        new_snow_timestamp,
        old_snow_timestamp,
    )

    if np.isnan(layer_T).any():
        GRID.grid_info_screen()
        raise ValueError("NaNs encountered in GRID creation.")

    return GRID
