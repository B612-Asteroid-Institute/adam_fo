import logging
import os
import pathlib
import shutil
import subprocess
import tempfile
from typing import Optional, Tuple

from adam_core.observations.ades import ADESObservations
from adam_core.orbits import Orbits

from . import config
from .conversions import (
    FindOrbFormatError,
    fo_to_adam_orbit_cov,
    rejected_observations_from_fo,
)

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
    for filename in required_files:
        src = fo_files_dir / filename
        dst = os.path.join(working_dir, filename)
        if not src.exists():
            raise RuntimeError(
                f"Required file {filename} not found in {fo_files_dir}. "
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
    with open(environ_dat_template, "r") as template_file:
        environ_dat_content = template_file.read()
    environ_dat_content = environ_dat_content.format(
        LINUX_JPL_FILENAME=config.LINUX_JPL_PATH.absolute(),
    )
    with open(os.path.join(working_dir, "environ.dat"), "w") as environ_file:
        environ_file.write(environ_dat_content)

    return working_dir


def _create_fo_tmp_directory(base_tmp_dir: Optional[str] = None) -> str:
    """
    Creates a temporary directory that avoids /tmp to handle fo locking and directory length limits.
    Uses ~/.cache/adam_fo/ftmp to avoid Find_Orb's special handling of paths containing /tmp/.

    Returns:
        str: The absolute path to the temporary directory populated with necessary FO files
    """
    resolved_base_dir = (
        str(config.get_cache_dir()) if base_tmp_dir is None else base_tmp_dir
    )
    os.makedirs(resolved_base_dir, mode=0o770, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(dir=resolved_base_dir, prefix="fo_")
    try:
        os.chmod(tmp_dir, 0o770)
        return _populate_fo_directory(tmp_dir)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


def _copy_fo_evidence(fo_tmp_dir: str, out_dir: Optional[str]) -> None:
    if out_dir is None:
        return
    shutil.copytree(
        fo_tmp_dir,
        out_dir,
        ignore=shutil.ignore_patterns("bc405.dat", "linux_p1550p2650.440t"),
        dirs_exist_ok=True,
    )


def _de440t_exists() -> None:
    if not config.LINUX_JPL_PATH.exists():
        raise Exception(
            f"DE440t file not found at {config.LINUX_JPL_PATH}, find_orb will not work correctly"
        )


def fo(
    ades_string: str,
    clean_up: bool = True,
    out_dir: Optional[str] = None,
    temp_dir: Optional[str] = None,
    state_vec: Optional[str] = None,
) -> Tuple[Orbits, ADESObservations, Optional[str]]:
    """Run programmatic Find_Orb orbit determination

    Parameters
    ----------
    ades_string : str
        ADES file as a string
    clean_up : bool, optional
        Whether to clean up the temporary directory after running Find_Orb
    out_dir : Optional[str], optional
        If provided, the temporary directory will be copied to this path after running Find_Orb.
        The bc405.dat and DE440t files will not be copied.
    temp_dir : Optional[str], optional
        If provided, the temporary directory will be created at this location.
        If not provided, the default temporary directory in ~/.cache/adam_fo/ will be used.
        It may be useful to explicitly set the path in HPC use cases, where user directories
        often have stricter disk usage quotas.
    state_vec : Optional[str], optional
        Pre-formatted seed for Find_Orb's ``-v`` flag. When supplied, Find_Orb
        bypasses Gauss/Vaisala cold-start and begins least-squares from this
        state vector. Format is the one parsed by
        ``extract_state_vect_from_text()`` in find_orb's ``elem_out.cpp``:
        ``"<epoch>,<x> <y> <z> <vx> <vy> <vz>[,<modifier>...]"`` — units AU
        and AU/day, default frame heliocentric ecliptic J2000, default time
        scale TT (append ``,TDB`` or ``,UTC`` to convert).

    Returns
    -------
    Tuple[Orbits, ADESObservations, Optional[str]]
        Tuple containing:
        - Determined orbit
        - Processed observations
        - Error message (if any)
    """

    _de440t_exists()
    fo_tmp_dir = _create_fo_tmp_directory(temp_dir)

    try:
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        input_file = os.path.join(fo_tmp_dir, "observations.ades")
        with open(input_file, "w") as file:
            file.write(ades_string)

        current_log_level = logger.getEffectiveLevel()
        fo_debug_level = 10 if current_log_level < 10 else 2
        fo_command = [
            str(config.FO_BINARY_DIR / "fo"),
            input_file,
            "-c",
            "-d",
            str(fo_debug_level),
            "-D",
            f"{fo_tmp_dir}/environ.dat",
            "-O",
            fo_tmp_dir,
        ]
        if state_vec is not None:
            fo_command.extend(["-v", state_vec])

        logger.debug("fo command: %s", fo_command)
        result = subprocess.run(
            fo_command,
            cwd=fo_tmp_dir,
            text=True,
            capture_output=True,
        )
        logger.debug("%s\n%s", result.stdout, result.stderr)

        if result.returncode != 0:
            logger.warning(
                "Find_Orb failed with return code %s for observations in %s",
                result.returncode,
                fo_tmp_dir,
            )
            logger.warning("%s\n%s", result.stdout, result.stderr)
            _copy_fo_evidence(fo_tmp_dir, out_dir)
            return Orbits.empty(), ADESObservations.empty(), "Find_Orb failed"

        if not os.path.exists(f"{fo_tmp_dir}/covar.json") or not os.path.exists(
            f"{fo_tmp_dir}/total.json"
        ):
            logger.warning("Find_Orb failed, covar.json or total.json file not found")
            _copy_fo_evidence(fo_tmp_dir, out_dir)
            return (
                Orbits.empty(),
                ADESObservations.empty(),
                "Find_Orb failed, covar.json or total.json file not found",
            )

        _copy_fo_evidence(fo_tmp_dir, out_dir)

        try:
            orbit = fo_to_adam_orbit_cov(fo_tmp_dir)
            rejected = rejected_observations_from_fo(fo_tmp_dir)
        except FindOrbFormatError as error:
            logger.warning("Find_Orb output validation failed: %s", error)
            return Orbits.empty(), ADESObservations.empty(), str(error)

        return orbit, rejected, None
    finally:
        if clean_up:
            shutil.rmtree(fo_tmp_dir)
