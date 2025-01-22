import json
import logging
from typing import Dict, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv
from adam_core.coordinates import (CartesianCoordinates, CoordinateCovariances,
                                   Origin)
from adam_core.observations import ADESObservations
from adam_core.orbits import Orbits
from adam_core.time import Timestamp

logger = logging.getLogger(__name__)


def read_fo_output(fo_output_dir: str) -> Tuple[Dict[str, dict], Dict[str, dict]]:
    """
    Read the find_orb output files from the specified directory into dictionaries
    containing orbital elements and covariances.

    Parameters
    ----------
    fo_output_dir : str
        Directory path where find_orb output files (e.g., total.json and covar.json) are located.

    Returns
    -------
        elements_dict : dict
        Dictionary containing orbital elements for each object,
        keyed by object ID, and with elements:
        - "central body" (center of the coordinate system)
        - "frame" (coordinate frame)
        - "reference" (source of the elements)
        - "epoch_iso" (epoch in ISO format)
        - "epoch" (epoch in Julian date)
        - "P" (orbital period)
        - "P sigma" (uncertainty in orbital period)
        - "M" (mean anomaly)
        - "M sigma" (uncertainty in mean anomaly)
        - "n" (mean motion)
        - "n sigma" (uncertainty in mean motion)
        - "a" (semi-major axis)
        - "a sigma" (uncertainty in semi-major axis)
        - "e" (eccentricity)
        - "e sigma" (uncertainty in eccentricity)
        - "q" (perihelion distance)
        - "q sigma" (uncertainty in perihelion distance)
        - "Q" (aphelion distance)
        - "Q sigma" (uncertainty in aphelion distance)
        - "i" (inclination)
        - "i sigma" (uncertainty in inclination)
        - "arg_per" (argument of perihelion)
        - "arg_per sigma" (uncertainty in argument of perihelion)
        - "asc_node" (ascending node)
        - "asc_node sigma" (uncertainty in ascending node)
        - "Tp" (time of perihelion passage)
        - "Tp sigma" (uncertainty in time of perihelion passage)
        - "Tp_iso" (time of perihelion passage in ISO format)
        - "H" (absolute magnitude)
        - "G" (slope parameter)
        - "rms_residual" (root mean square of residuals)
        - "weighted_rms_residual" (weighted root mean square of residuals)
        - "n_resids" (number of residuals)
        - "U" (uncertainty parameter)
        - "p_NEO" (NEO probability)
        - "MOIDs" (minimum orbit intersection distances)
    covar_json : dict
        Dictionary containing the covariance data from the JSON file.
        Includes the following keys:
        - "covar" (covariance matrix, in cartesian coordinates)
        - "state_vect" (state vector, in cartesian coordinates)
        - "epoch" (epoch in Julian date)
    """
    covar_dict = read_fo_covariance(f"{fo_output_dir}/covar.json")
    elements_dict = read_fo_orbits(f"{fo_output_dir}/total.json")
    # TODO: Read in the observations used and unused and return them
    return elements_dict, covar_dict


def read_fo_covariance(covar_file: str) -> Dict[str, dict]:
    """
    Read the find_orb covariance JSON file into a dictionary.

    Parameters
    ----------
    covar_file : str
        Path to the find_orb covariance JSON file (covar.json).

    Returns
    -------
    covar_json : dict
        Dictionary containing the covariance data from the JSON file.
        Includes the following keys:
        - "covar" (covariance matrix, in cartesian coordinates)
        - "state_vect" (state vector, in cartesian coordinates)
        - "epoch" (epoch in Julian date)
    """
    with open(covar_file, "r") as f:
        covar_json = json.load(f)
    return covar_json


def read_fo_orbits(input_file: str) -> Dict[str, dict]:
    """
    Read the find_orb total.json file into a dictionary of orbital elements.

    Parameters
    ----------
    input_file : str
        Path to the find_orb total.json file.

    Returns
    -------
    elements_dict : dict
        Dictionary containing orbital elements for each object,
        keyed by object ID, and with elements:
        - "central body" (center of the coordinate system)
        - "frame" (coordinate frame)
        - "reference" (source of the elements)
        - "epoch_iso" (epoch in ISO format)
        - "epoch" (epoch in Julian date)
        - "P" (orbital period)
        - "P sigma" (uncertainty in orbital period)
        - "M" (mean anomaly)
        - "M sigma" (uncertainty in mean anomaly)
        - "n" (mean motion)
        - "n sigma" (uncertainty in mean motion)
        - "a" (semi-major axis)
        - "a sigma" (uncertainty in semi-major axis)
        - "e" (eccentricity)
        - "e sigma" (uncertainty in eccentricity)
        - "q" (perihelion distance)
        - "q sigma" (uncertainty in perihelion distance)
        - "Q" (aphelion distance)
        - "Q sigma" (uncertainty in aphelion distance)
        - "i" (inclination)
        - "i sigma" (uncertainty in inclination)
        - "arg_per" (argument of perihelion)
        - "arg_per sigma" (uncertainty in argument of perihelion)
        - "asc_node" (ascending node)
        - "asc_node sigma" (uncertainty in ascending node)
        - "Tp" (time of perihelion passage)
        - "Tp sigma" (uncertainty in time of perihelion passage)
        - "Tp_iso" (time of perihelion passage in ISO format)
        - "H" (absolute magnitude)
        - "G" (slope parameter)
        - "rms_residual" (root mean square of residuals)
        - "weighted_rms_residual" (weighted root mean square of residuals)
        - "n_resids" (number of residuals)
        - "U" (uncertainty parameter)
        - "p_NEO" (NEO probability)
        - "MOIDs" (minimum orbit intersection distances)
    """

    with open(input_file, "r") as f:
        total_json = json.load(f)
    objects = total_json.get("objects", {})
    elements_dict = {}
    for object_id, object_data in objects.items():
        elements = object_data.get("elements", {})
        elements_dict[object_id] = elements
    return elements_dict


def fo_to_adam_orbit_cov(fo_output_folder: str) -> Orbits:
    """
    Convert Find_Orb output to ADAM Orbit objects, including covariance.

    Parameters
    ----------
    fo_output_folder : str
        Path to the folder containing Find_Orb output files.

    Returns
    -------
    orbits : `~adam_core.orbits.orbits.Orbits`
        ADAM Orbit object created from the Find_Orb output data.
    """

    elements_dict, covar_dict = read_fo_output(fo_output_folder)

    orbits = Orbits.empty()
    for object_id, elements in elements_dict.items():
        covar_matrix = np.array([covar_dict["covar"]])
        covar_state_vector = [covar_dict["state_vect"]]

        covariances_cartesian = CoordinateCovariances.from_matrix(covar_matrix)
        # After a lot of searching, we mostly believe that the epoch
        # is defined in TT (TD in find_orb). During reading of the
        # ADES files, find_orb converts the jd times in its struct to
        # TD and does not appear to rescale it again before writing
        # out the total.json and covar.json files.
        times = Timestamp.from_jd([covar_dict["epoch"]], scale="tt")

        cartesian_coordinates = CartesianCoordinates.from_kwargs(
            x=[covar_state_vector[0][0]],
            y=[covar_state_vector[0][1]],
            z=[covar_state_vector[0][2]],
            vx=[covar_state_vector[0][3]],
            vy=[covar_state_vector[0][4]],
            vz=[covar_state_vector[0][5]],
            time=times,
            origin=Origin.from_kwargs(code=["SUN"]),
            frame="ecliptic",
            covariance=covariances_cartesian,
        )
        orbit = Orbits.from_kwargs(
            orbit_id=[object_id],
            object_id=[object_id],
            coordinates=cartesian_coordinates,
        )
        orbits = qv.concatenate([orbits, orbit])
        if orbits.fragmented():
            orbits = qv.defragment(orbits)

    return orbits


def rejected_observations_from_fo(fo_output_folder: str) -> ADESObservations:
    with open(f"{fo_output_folder}/total.json", "r") as f:
        total_json = json.load(f)
    objects = total_json.get("objects", {})
    json_observations = []
    for object_id, object_data in objects.items():
        object_observations = object_data.get("observations", {}).get("residuals", [])
        for observation in object_observations:
            observation.update({"object_id": object_id})
        json_observations.extend(object_observations)

    rejected_observations = []
    for observation in json_observations:
        if observation.get("incl") == 0:
            rejected_observations.append(observation)

    if len(rejected_observations) == 0:
        return ADESObservations.empty()

    ades_rejected_observations = ADESObservations.from_kwargs(
        trkSub=pa.array(
            [observation.get("object_id") for observation in rejected_observations]
        ),
        obsTime=Timestamp.from_jd(
            [observation.get("JD") for observation in rejected_observations],
            scale="utc",
        ),
        ra=pa.array([observation.get("RA") for observation in rejected_observations]),
        dec=pa.array([observation.get("Dec") for observation in rejected_observations]),
        mag=pa.array(
            [observation.get("MagObs") for observation in rejected_observations]
        ),
        rmsRACosDec=pa.array(
            [observation.get("sigma_1") for observation in rejected_observations]
        ),
        rmsDec=pa.array(
            [observation.get("sigma_2") for observation in rejected_observations]
        ),
        band=pa.array(
            [observation.get("MagBand") for observation in rejected_observations]
        ),
        stn=pa.array(
            [observation.get("obscode") for observation in rejected_observations]
        ),
        mode=pa.repeat("NA", len(rejected_observations)),
        astCat=pa.repeat("NA", len(rejected_observations)),
    )
    return ades_rejected_observations
