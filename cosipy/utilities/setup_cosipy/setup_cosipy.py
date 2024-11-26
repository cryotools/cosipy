"""Generate sample configuration files for COSIPY.

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
from pathlib import Path


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
        type=Path,
        metavar="<path>",
        required=False,
        help="relative path to target configuration directory",
    )

    arguments = parser.parse_args()

    return arguments


def get_sample_directory() -> Path:
    """Get the path to the sample directory.

    Returns:
        Path to sample configuration directory.
    """
    # Package is not installed in working directory
    frame = inspect.currentframe()
    if frame is None:
        msg = "Current frame is None."
        raise ValueError(msg)
    try:
        filename = Path(inspect.getfile(frame)).resolve()
    finally:
        del frame
    return filename.parent


def copy_file_to_target(
    basename: str,
    source_dir: Path,
    target_dir: Path,
    *,
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
    target_path = target_dir / basename
    source_path = source_dir / basename
    overwrite = True  # otherwise no file created if missing

    if not silent_overwrite and target_path.exists(follow_symlinks=True):
        prompt = f"{basename} already exists in {target_dir}/\nReplace target? [y/N] "
        overwrite = get_user_confirmation(prompt)
    if overwrite:
        shutil.copyfile(source_path, target_path, follow_symlinks=True)
    else:
        print("Skipping...")

def get_user_confirmation(prompt: str) -> bool:
    """Get user confirmation.

    Args:
        prompt: Prompt to display to user.

    Returns:
        True if user confirms, False otherwise.
    """
    user_input: str = input(prompt).lower().strip()
    if user_input not in ("y", "yes", "n", "no", ""):
        print("Please enter 'yes' or 'no'.\n")
        return get_user_confirmation(prompt)
    return user_input in ("y", "yes")

def main():
    args = get_user_arguments()

    sample_path = get_sample_directory()
    if not args.output_path:
        target_path = Path().cwd()
    else:
        target_path = args.output_path
        target_path.mkdir(parents=True, exist_ok=True)
    if target_path == sample_path:
        raise ValueError("The target and source paths cannot be identical.")

    config_files = [
        "config.toml",
        "constants.toml",
        "slurm_config.toml",
        "utilities_config.toml",
    ]
    for file in config_files:
        filepath = Path(sample_path) / file
        if not filepath.exists():
            msg = f"{filepath} does not exist."
            raise FileNotFoundError(msg)
        copy_file_to_target(
            basename=file,
            source_dir=sample_path,
            target_dir=target_path,
            silent_overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
