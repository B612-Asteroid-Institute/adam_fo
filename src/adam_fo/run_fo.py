# TODO: Update this to work inside this new repo instead of the adam_impact_study

import logging
import os
import pathlib
import shutil
import subprocess
import tempfile
from typing import Optional, Tuple

import pyarrow.compute as pc
from adam_core.observations.ades import ADESObservations
from adam_core.orbits import Orbits

from . import config
from .conversions import fo_to_adam_orbit_cov, rejected_observations_from_fo

logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("ADAM_LOG_LEVEL", "INFO"))


def _populate_fo_directory(working_dir: str) -> str:
    """Populate a working directory with required find_orb files."""
    config.check_build_exists()
    
    os.makedirs(working_dir, exist_ok=True)
    # List of required files to copy from FO_DIR to current directory
    required_files = [
        "ObsCodes.htm",
        "jpl_eph.txt",
        "orbitdef.sof",
        "rovers.txt",
        "xdesig.txt",
        "cospar.txt",
        "efindorb.txt",
        "odd_name.txt",
        "sigma.txt",
        "mu1.txt",
        "link_def.json",
    ]

    # Copy required files from the build directory
    fo_files_dir = config.BUILD_DIR / "find_orb/find_orb"
    for file in required_files:
        src = fo_files_dir / file
        dst = os.path.join(working_dir, file)
        if not src.exists():
            raise RuntimeError(
                f"Required file {file} not found in {fo_files_dir}. "
                "Please run 'build-fo' command to install find_orb properly."
            )
        shutil.copy2(src, dst)

    # Copy bc405.dat to the current directory
    if not config.BC405_FILENAME.exists():
        raise RuntimeError(
            f"Required file bc405.dat not found in {config.BC405_FILENAME}. "
            "Please run 'build-fo' command to install find_orb properly."
        )
    shutil.copy2(config.BC405_FILENAME, os.path.join(working_dir, "bc405.dat"))

    # Template in the JPL path to environ.dat
    environ_dat_template = pathlib.Path(__file__).parent / "environ.dat.tpl"
    with open(environ_dat_template, "r") as file:
        environ_dat_content = file.read()
    environ_dat_content = environ_dat_content.format(
        LINUX_JPL_FILENAME=config.LINUX_JPL_PATH.absolute(),
    )
    with open(os.path.join(working_dir, "environ.dat"), "w") as file:
        file.write(environ_dat_content)

    return working_dir


def _create_fo_tmp_directory() -> str:
    """
    Creates a temporary directory that avoids /tmp to handle fo locking and directory length limits.
    Uses ~/.cache/adam_fo/ftmp to avoid Find_Orb's special handling of paths containing /tmp/.

    Returns:
        str: The absolute path to the temporary directory populated with necessary FO files
    """
    base_tmp_dir = config.get_cache_dir()
    os.makedirs(base_tmp_dir, mode=0o770, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(dir=base_tmp_dir, prefix="fo_")
    os.chmod(tmp_dir, 0o770)
    tmp_dir = _populate_fo_directory(tmp_dir)
    return tmp_dir


def _de440t_exists():
    if not config.LINUX_JPL_PATH.exists():
        raise Exception(
            f"DE440t file not found at {config.LINUX_JPL_PATH}, find_orb will not work correctly"
        )


def fo(
    ades_string: str,
    clean_up: bool = True,
) -> Tuple[Orbits, ADESObservations, Optional[str]]:
    """Run programmatic Find_Orb orbit determination

    Parameters
    ----------
    ades_string : str
        ADES file as a string
    clean_up : bool, optional
        Whether to clean up the temporary directory after running Find_Orb

    Returns
    -------
    Tuple[Orbits, ADESObservations, Optional[str]]
        Tuple containing:
        - Determined orbit
        - Processed observations
        - Error message (if any)
    """

    _de440t_exists()
    fo_tmp_dir = _create_fo_tmp_directory()

    # Create input file
    input_file = os.path.join(fo_tmp_dir, "observations.ades")
    with open(input_file, "w") as file:
        file.write(ades_string)

    # Get the current logger log level
    current_log_level = logger.getEffectiveLevel()
    # fo uses 2 for INFO, 10 for most debugging
    fo_debug_level = 2
    if current_log_level < 10:
        fo_debug_level = 10
    # Run Find_Orb
    fo_command = (
        f"{config.FO_BINARY_DIR}/fo {input_file} -c "
        f"-d {fo_debug_level} "
        f"-D {fo_tmp_dir}/environ.dat "
        f"-O {fo_tmp_dir}"
    )

    logger.debug(f"fo command: {fo_command}")

    result = subprocess.run(
        fo_command,
        shell=True,
        cwd=fo_tmp_dir,
        text=True,
        capture_output=True,
    )
    logger.debug(f"{result.stdout}\n{result.stderr}")

    if result.returncode != 0:
        logger.warning(
            f"Find_Orb failed with return code {result.returncode} for observations in {fo_tmp_dir}"
        )
        logger.warning(f"{result.stdout}\n{result.stderr}")
        return Orbits.empty(), ADESObservations.empty(), "Find_Orb failed"

    if not os.path.exists(f"{fo_tmp_dir}/covar.json") or not os.path.exists(
        f"{fo_tmp_dir}/total.json"
    ):
        logger.warning("Find_Orb failed, covar.json or total.json file not found")
        return (
            Orbits.empty(),
            ADESObservations.empty(),
            "Find_Orb failed, covar.json or total.json file not found",
        )

    orbit = fo_to_adam_orbit_cov(fo_tmp_dir)
    rejected = rejected_observations_from_fo(fo_tmp_dir)

    if clean_up:
        shutil.rmtree(fo_tmp_dir)

    return orbit, rejected, None
