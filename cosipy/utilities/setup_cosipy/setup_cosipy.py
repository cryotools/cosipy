"""
Generate sample configuration files for COSIPY.

Usage:

Entry point:
``cosipy-setup [-y] [-o <path>]``

From source:
``python -m cosipy.utilities.setup_cosipy.setup_cosipy -i <input> -o <output> -s <static> [-u <path>] [-b <date>] [-e <date>]``

Options and arguments:

Optional switches:
    -h, --help              Show this help message and exit.
    -y, --yes               Silently overwrite existing configuration
                                files.

Optional arguments:
    -o, --output <path>     Relative path to target configuration
                                directory.
"""

import argparse
import inspect
import os
import shutil


def get_user_arguments() -> argparse.Namespace:
    """Parse user arguments when run as main.

    Optional switches:
        -h, --help              Show this help message and exit.
        -y, --yes               Silently overwrite existing
                                    configuration files.

    Optional arguments:
        -o, --output <path>     Relative path to target configuration
                                    directory.

    Returns:
        Namespace of user arguments.
    """

    tagline = "Generate configuration files for COSIPY"
    parser = argparse.ArgumentParser(
        prog="cosipy.utilities.gen_config", description=tagline
    )

    # Optional switches
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        default=False,
        dest="overwrite",
        help="silently overwrite existing configuration files",
    )

    # Optional arguments
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        dest="output_path",
        type=str,
        metavar="<path>",
        required=False,
        help="relative path to target configuration directory",
    )

    arguments = parser.parse_args()

    return arguments


def check_file_exists(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"{file_path} not found.")


def get_sample_directory() -> str:
    """Get the path to the sample directory.

    Returns:
        Path to sample configuration directory.
    """
    # Package is not installed in working directory
    filename = inspect.getfile(inspect.currentframe())
    filename = filename.rsplit("/", 1)
    src_dir = f"{filename[0]}"

    return src_dir


def copy_file_to_target(
    basename: str,
    source_dir: str,
    target_dir: str,
    silent_overwrite: bool = False,
):
    """Copy a file to a target directory.

    Args:
        basename: Name of file.
        source_dir: Source directory of file.
        target_dir: Target directory of the copied file.
        silent_overwrite: Silently overwrite existing files in target
            directory. Default False.
    """

    target_path = f"{target_dir}/{basename}"
    decision = True

    if not silent_overwrite and os.path.isfile(target_path):
        decision = input(
            f"{basename} already exists in {target_dir}/\nOverwrite?\n"
        )
        decision = strtobool(decision)
    if decision:
        shutil.copyfile(
            f"{source_dir}/{basename}", target_path, follow_symlinks=True
        )


def strtobool(val: str) -> bool:
    """Convert user input to bool."""
    val = val.lower()
    if val in ("y", "yes"):
        return True
    elif val in ("n", "no"):
        return False


def main():

    args = get_user_arguments()

    sample_path = get_sample_directory()
    if not args.output_path:
        target_path = os.getcwd()
    else:
        target_path = args.output_path
        os.makedirs(target_path, exist_ok=True)
    if target_path == sample_path:
        raise ValueError("The target and source paths cannot be identical.")

    config_files = [
        "config.toml",
        "constants.toml",
        "slurm_config.toml",
        "utilities_config.toml",
    ]
    for file in config_files:
        check_file_exists(file_path=f"{sample_path}/{file}")
        copy_file_to_target(
            basename=file,
            source_dir=sample_path,
            target_dir=target_path,
            silent_overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
