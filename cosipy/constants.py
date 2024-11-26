import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, model_validator
from typing_extensions import Self

from cosipy.config import (TomlLoader, default_constants_path,
                           get_user_arguments, main_config)

if sys.version_info >= (3, 11):
    import tomllib
else:
    pass  # backwards compatibility


class ConstantsModel(BaseModel):
    dt: int
    max_layers: int
    z: float
    zlt1: float
    zlt2: float
    t_star_K: int
    t_star_cutoff: float
    t_star_wet: int
    t_star_dry: int
    sfc_temperature_method: str
    stability_correction: str
    albedo_method: str
    densification_method: str
    penetrating_method: str
    roughness_method: str
    saturation_water_vapour_method: str
    initial_snowheight_constant: float
    initial_snow_layer_heights: float
    initial_glacier_height: float
    initial_glacier_layer_heights: float
    initial_top_density_snowpack: float
    initial_bottom_density_snowpack: float
    temperature_bottom: float
    const_init_temp: float
    center_snow_transfer_function: float
    spread_snow_transfer_function: float
    mult_factor_RRR: float
    minimum_snow_layer_height: float
    minimum_snowfall: float
    remesh_method: str
    thermal_conductivity_method: Literal["bulk", "empirical"]
    first_layer_height: float
    layer_stretching: float
    density_threshold_merging: float
    temperature_threshold_merging: float
    albedo_fresh_snow: float
    albedo_firn: float
    albedo_ice: float
    albedo_mod_snow_aging: float
    albedo_mod_snow_depth: float
    roughness_fresh_snow: float
    roughness_ice: float
    roughness_firn: float
    aging_factor_roughness: float
    snow_ice_threshold: float
    lat_heat_melting: float
    lat_heat_vaporize: float
    lat_heat_sublimation: float
    spec_heat_air: float
    spec_heat_ice: float
    spec_heat_water: float
    k_i: float
    k_w: float
    k_a: float
    water_density: float
    ice_density: float
    air_density: float
    sigma: float
    zero_temperature: float
    surface_emission_coeff: float
    merge_max: int
    constant_density: float

    @model_validator(mode="after")
    def adjust_config(self) -> Self:
        """Adjust invalid or mutually exclusive configuration values.

        Args:
            config_table: Loaded .toml data.

        Returns:
            Adjusted .toml data.
        """
        # WRF_X_CSPY: for efficiency and consistency
        if main_config.WRF_X_CSPY:
            self.albedo_method = "Oerlemans98"
            self.stability_correction = "MO"

        return self


class Constants(TomlLoader):
    """Constants configuration.

    Loads, parses, and sets constants for COSIPY from a valid .toml
    file.
    """

    def __init__(self, path: Path = default_constants_path) -> None:
        raw_toml = self.get_raw_toml(path)
        self.raw_toml = self.flatten(raw_toml)

    @classmethod
    def set_config(cls, config: dict) -> None:
        for key, value in config.items():
            setattr(cls, key, value)

    def validate(self) -> ConstantsModel:
        """Validate configuration using Pydantic class.

        Returns:
            CosipyConfigModel: Validated configuration.
        """
        return ConstantsModel(**self.raw_toml)

def main():
    args = get_user_arguments()
    return Constants(args.constants_path).validate()


if __name__ == "__main__":
    main()
else:
    constants_config = main()
