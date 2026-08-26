import json
import logging
import math
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
import pyarrow as pa
from adam_core.coordinates import CartesianCoordinates, CoordinateCovariances, Origin
from adam_core.observations import ADESObservations
from adam_core.orbits import Orbits
from adam_core.time import Timestamp

logger = logging.getLogger(__name__)

_COMPANION_FILENAMES = (
    "elements.json",
    "elem_short.json",
    "total.json",
    "combined.json",
)
_COVARIANCE_SYMMETRY_RTOL = 1e-7
_MAX_JSON_BYTES = 32 * 1024 * 1024
_MAX_ELEMENT_EPOCH_GAP_DAYS = 45.0
_DEGENERATE_ECCENTRICITY = 1e-4
_DEGENERATE_INCLINATION_DEGREES = 0.1


class FindOrbFormatError(ValueError):
    """Raised when a Find_Orb output bundle is unsafe to canonicalize."""


@dataclass(frozen=True)
class FindOrbMetadata:
    object_id: str
    packed_id: str | None
    companion_file: str
    companion_files: tuple[str, ...]
    covariance_epoch_jd_tt: float
    element_epoch_jd_tt: float
    central_body: str
    frame: str
    reference: str
    absolute_magnitude: float | None
    slope_parameter: float | None
    find_orb_version_jd: float | None
    consistency_check: str


@dataclass(frozen=True)
class FindOrbConversion:
    orbits: Orbits
    metadata: FindOrbMetadata


@dataclass(frozen=True)
class _CompanionSolution:
    filename: str
    object_id: str
    packed_id: str | None
    elements: Mapping[str, object]
    find_orb_version_jd: float | None


def _read_json_object(path: Path) -> dict[str, object]:
    try:
        if path.stat().st_size > _MAX_JSON_BYTES:
            raise FindOrbFormatError(
                f"Find_Orb JSON file {path.name!r} exceeds the 32 MiB limit"
            )
        with path.open("r", encoding="utf-8") as file:
            value = cast(object, json.load(file))
    except FindOrbFormatError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise FindOrbFormatError(
            f"Could not read Find_Orb JSON file {path.name!r}"
        ) from error

    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise FindOrbFormatError(
            f"Find_Orb file {path.name!r} must contain a JSON object"
        )
    return cast(dict[str, object], value)


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise FindOrbFormatError(f"Find_Orb field {field!r} must be an object")
    return cast(Mapping[str, object], value)


def _sequence(value: object, *, field: str) -> Sequence[object]:
    if not isinstance(value, list):
        raise FindOrbFormatError(f"Find_Orb field {field!r} must be an array")
    return cast(Sequence[object], value)


def _finite_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FindOrbFormatError(f"Find_Orb field {field!r} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise FindOrbFormatError(f"Find_Orb field {field!r} must be finite")
    return parsed


def _required_float(mapping: Mapping[str, object], key: str, *, field: str) -> float:
    if key not in mapping:
        raise FindOrbFormatError(f"Find_Orb field {field!r} is required")
    return _finite_float(mapping[key], field=field)


def _optional_float(
    mapping: Mapping[str, object], key: str, *, field: str
) -> float | None:
    value = mapping.get(key)
    if value is None:
        return None
    return _finite_float(value, field=field)


def _invalid_identifier_character(character: str) -> bool:
    return (
        ord(character) < 32
        or ord(character) == 127
        or unicodedata.category(character) == "Cf"
    )


def _required_string(mapping: Mapping[str, object], key: str, *, field: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str):
        raise FindOrbFormatError(f"Find_Orb field {field!r} must be a string")
    parsed = value.strip()
    if (
        not parsed
        or len(parsed) > 256
        or any(_invalid_identifier_character(character) for character in parsed)
    ):
        raise FindOrbFormatError(
            f"Find_Orb field {field!r} contains an invalid identifier"
        )
    return parsed


def _optional_string(
    mapping: Mapping[str, object], key: str, *, field: str
) -> str | None:
    value = mapping.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise FindOrbFormatError(f"Find_Orb field {field!r} must be a string")
    parsed = value.strip()
    if (
        not parsed
        or len(parsed) > 256
        or any(_invalid_identifier_character(character) for character in parsed)
    ):
        raise FindOrbFormatError(
            f"Find_Orb field {field!r} contains an invalid identifier"
        )
    return parsed


def _parse_companion(path: Path) -> _CompanionSolution:
    document = _read_json_object(path)
    num = document.get("num")
    if isinstance(num, bool) or not isinstance(num, int) or num != 1:
        raise FindOrbFormatError(f"Find_Orb companion {path.name!r} must declare num=1")

    ids = _sequence(document.get("ids"), field=f"{path.name}.ids")
    if len(ids) != 1 or not isinstance(ids[0], str):
        raise FindOrbFormatError(
            f"Find_Orb companion {path.name!r} must contain exactly one ID"
        )
    declared_id = ids[0].strip()

    objects = _mapping(document.get("objects"), field=f"{path.name}.objects")
    if len(objects) != 1:
        raise FindOrbFormatError(
            f"Find_Orb companion {path.name!r} must contain exactly one object"
        )
    object_id, raw_object = next(iter(objects.items()))
    object_id = object_id.strip()
    if (
        not object_id
        or len(object_id) > 256
        or any(_invalid_identifier_character(character) for character in object_id)
    ):
        raise FindOrbFormatError(
            f"Find_Orb companion {path.name!r} has an invalid object key"
        )
    if declared_id != object_id:
        raise FindOrbFormatError(
            f"Find_Orb companion {path.name!r} declares ID {declared_id!r} but contains {object_id!r}"
        )

    object_document = _mapping(raw_object, field=f"{path.name}.objects.{object_id}")
    reported_object = _required_string(
        object_document,
        "object",
        field=f"{path.name}.objects.{object_id}.object",
    )
    if reported_object != object_id:
        raise FindOrbFormatError(
            f"Find_Orb companion {path.name!r} object metadata does not match {object_id!r}"
        )

    elements = _mapping(
        object_document.get("elements"),
        field=f"{path.name}.objects.{object_id}.elements",
    )
    return _CompanionSolution(
        filename=path.name,
        object_id=object_id,
        packed_id=_optional_string(
            object_document,
            "packed",
            field=f"{path.name}.objects.{object_id}.packed",
        ),
        elements=elements,
        find_orb_version_jd=_optional_float(
            object_document,
            "Find_Orb_version",
            field=f"{path.name}.objects.{object_id}.Find_Orb_version",
        ),
    )


def _load_companions(
    bundle_path: Path,
) -> tuple[_CompanionSolution, tuple[_CompanionSolution, ...]]:
    companions = tuple(
        _parse_companion(bundle_path / filename)
        for filename in _COMPANION_FILENAMES
        if (bundle_path / filename).is_file()
    )
    if not companions:
        allowed = ", ".join(_COMPANION_FILENAMES)
        raise FindOrbFormatError(
            f"Find_Orb bundle requires one companion file: {allowed}"
        )

    primary = companions[0]
    for companion in companions[1:]:
        _validate_companions_match(primary, companion)
    return primary, companions


def _validate_companions_match(
    expected: _CompanionSolution,
    actual: _CompanionSolution,
) -> None:
    if expected.object_id != actual.object_id:
        raise FindOrbFormatError(
            f"Find_Orb companions disagree on object ID: {expected.object_id!r} != {actual.object_id!r}"
        )

    string_fields = ("central body", "frame", "reference")
    for field in string_fields:
        expected_string = _required_string(
            expected.elements, field, field=f"{expected.filename}.{field}"
        )
        actual_string = _required_string(
            actual.elements, field, field=f"{actual.filename}.{field}"
        )
        if expected_string != actual_string:
            raise FindOrbFormatError(f"Find_Orb companions disagree on {field!r}")

    numeric_fields = ("epoch", "q", "e", "i", "asc_node", "arg_per", "Tp")
    for field in numeric_fields:
        expected_number = _required_float(
            expected.elements, field, field=f"{expected.filename}.{field}"
        )
        actual_number = _required_float(
            actual.elements, field, field=f"{actual.filename}.{field}"
        )
        if not math.isclose(
            expected_number, actual_number, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise FindOrbFormatError(f"Find_Orb companions disagree on {field!r}")


def _parse_state_and_covariance(
    covar_path: Path,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float]:
    document = _read_json_object(covar_path)
    state_values = _sequence(document.get("state_vect"), field="covar.json.state_vect")
    state = np.asarray(
        [
            _finite_float(value, field=f"covar.json.state_vect[{index}]")
            for index, value in enumerate(state_values)
        ],
        dtype=np.float64,
    )
    covariance_rows = _sequence(document.get("covar"), field="covar.json.covar")
    covariance = np.asarray(
        [
            [
                _finite_float(
                    value,
                    field=f"covar.json.covar[{row_index}][{column_index}]",
                )
                for column_index, value in enumerate(
                    _sequence(row, field=f"covar.json.covar[{row_index}]")
                )
            ]
            for row_index, row in enumerate(covariance_rows)
        ],
        dtype=np.float64,
    )

    if state.shape != (6,):
        raise FindOrbFormatError(
            f"Find_Orb state_vect must have shape (6,), got {state.shape!r}"
        )
    if covariance.shape != (6, 6):
        raise FindOrbFormatError(
            f"Find_Orb covariance must have shape (6, 6), got {covariance.shape!r}"
        )
    if not np.all(np.isfinite(state)) or not np.all(np.isfinite(covariance)):
        raise FindOrbFormatError(
            "Find_Orb state and covariance must contain only finite values"
        )

    covariance_norm = float(np.linalg.norm(covariance, ord="fro"))
    asymmetry_norm = float(np.linalg.norm(covariance - covariance.T, ord="fro"))
    if covariance_norm == 0.0:
        raise FindOrbFormatError("Find_Orb covariance must not be all zero")
    if asymmetry_norm > _COVARIANCE_SYMMETRY_RTOL * covariance_norm:
        raise FindOrbFormatError(
            "Find_Orb covariance is not symmetric within serialization tolerance"
        )

    covariance = (covariance + covariance.T) / 2.0
    eigenvalues = np.linalg.eigvalsh(covariance)
    eigenvalue_scale = float(np.max(np.abs(eigenvalues)))
    psd_tolerance = eigenvalue_scale * 1e-10
    if float(eigenvalues[0]) < -psd_tolerance:
        raise FindOrbFormatError("Find_Orb covariance is not positive semidefinite")

    epoch = _required_float(document, "epoch", field="covar.json.epoch")
    return state, covariance, epoch


def _angle_difference_degrees(left: float, right: float) -> float:
    return (left - right + 180.0) % 360.0 - 180.0


def _validate_supported_solution(
    solution: _CompanionSolution,
) -> tuple[float, str, str, str]:
    central_body = _required_string(
        solution.elements,
        "central body",
        field=f"{solution.filename}.central body",
    )
    if central_body.casefold() not in {"sun", "sol"}:
        raise FindOrbFormatError(f"Unsupported Find_Orb central body {central_body!r}")

    frame = _required_string(
        solution.elements, "frame", field=f"{solution.filename}.frame"
    )
    if frame.casefold() != "j2000 ecliptic":
        raise FindOrbFormatError(f"Unsupported Find_Orb element frame {frame!r}")

    reference = _required_string(
        solution.elements,
        "reference",
        field=f"{solution.filename}.reference",
    )
    if reference.casefold() != "find_orb":
        raise FindOrbFormatError(f"Unsupported Find_Orb reference {reference!r}")

    element_epoch = _required_float(
        solution.elements,
        "epoch",
        field=f"{solution.filename}.epoch",
    )
    return element_epoch, central_body, frame, reference


def _validate_state_matches_elements(
    state: npt.NDArray[np.float64],
    covariance_epoch: float,
    element_epoch: float,
    solution: _CompanionSolution,
) -> None:
    coordinates = CartesianCoordinates.from_kwargs(
        x=[float(state[0])],
        y=[float(state[1])],
        z=[float(state[2])],
        vx=[float(state[3])],
        vy=[float(state[4])],
        vz=[float(state[5])],
        time=Timestamp.from_jd([covariance_epoch], scale="tt"),
        origin=Origin.from_kwargs(code=["SUN"]),
        frame="ecliptic",
    )
    cometary = coordinates.to_cometary()
    derived = {
        "q": float(cometary.q[0].as_py()),
        "e": float(cometary.e[0].as_py()),
        "i": float(cometary.i[0].as_py()),
        "asc_node": float(cometary.raan[0].as_py()),
        "arg_per": float(cometary.ap[0].as_py()),
    }
    if not all(math.isfinite(value) for value in derived.values()):
        raise FindOrbFormatError(
            "Find_Orb covar.json state produces non-finite orbital invariants"
        )
    expected = {
        key: _required_float(solution.elements, key, field=f"{solution.filename}.{key}")
        for key in derived
    }

    epoch_gap_days = abs(element_epoch - covariance_epoch)
    if epoch_gap_days > _MAX_ELEMENT_EPOCH_GAP_DAYS:
        raise FindOrbFormatError(
            "Find_Orb companion epoch is too far from the covariance epoch for "
            "a safe same-solution consistency check"
        )

    differences = {
        "q": abs(derived["q"] - expected["q"]),
        "e": abs(derived["e"] - expected["e"]),
        "i": abs(_angle_difference_degrees(derived["i"], expected["i"])),
    }
    tolerances = {
        "q": max(1e-3, 1e-4 * abs(expected["q"])),
        "e": max(1e-3, 1e-3 * abs(expected["e"])),
        "i": 0.1 + 0.02 * epoch_gap_days,
    }
    angle_tolerance = 0.5 + 0.1 * epoch_gap_days
    inclination_is_degenerate = (
        min(
            abs(derived["i"]),
            abs(180.0 - derived["i"]),
            abs(expected["i"]),
            abs(180.0 - expected["i"]),
        )
        < _DEGENERATE_INCLINATION_DEGREES
    )
    eccentricity_is_degenerate = (
        min(derived["e"], expected["e"]) < _DEGENERATE_ECCENTRICITY
    )

    if inclination_is_degenerate:
        if not eccentricity_is_degenerate:
            differences["longitude_perihelion"] = abs(
                _angle_difference_degrees(
                    derived["asc_node"] + derived["arg_per"],
                    expected["asc_node"] + expected["arg_per"],
                )
            )
            tolerances["longitude_perihelion"] = angle_tolerance
    else:
        differences["asc_node"] = abs(
            _angle_difference_degrees(derived["asc_node"], expected["asc_node"])
        )
        tolerances["asc_node"] = angle_tolerance
        if not eccentricity_is_degenerate:
            differences["arg_per"] = abs(
                _angle_difference_degrees(derived["arg_per"], expected["arg_per"])
            )
            tolerances["arg_per"] = angle_tolerance
    failures = [
        field for field in differences if differences[field] > tolerances[field]
    ]
    if failures:
        detail = ", ".join(
            f"{field} delta={differences[field]:.6g}" for field in failures
        )
        raise FindOrbFormatError(
            "Find_Orb covar.json state is inconsistent with the companion solution: "
            + detail
        )


def convert_find_orb_bundle(bundle_directory: str | Path) -> FindOrbConversion:
    """Validate and canonicalize a one-object Find_Orb output bundle.

    The canonical state, covariance, and epoch always come from ``covar.json``.
    Companion elements are a separately propagated representation used to verify
    identity, supported frame/origin, and gross same-solution consistency. That
    heuristic is limited to element epochs within 45 days of the covariance epoch,
    scales angular tolerances with the gap, and avoids singular classical angles.
    Companion H/G values are retained only as fit provenance.
    """

    bundle_path = Path(bundle_directory)
    if not bundle_path.is_dir():
        raise FindOrbFormatError("Find_Orb bundle path must be a directory")
    covar_path = bundle_path / "covar.json"
    if not covar_path.is_file():
        raise FindOrbFormatError("Find_Orb bundle requires covar.json")

    primary, companions = _load_companions(bundle_path)
    element_epoch, central_body, frame, reference = _validate_supported_solution(
        primary
    )
    state, covariance, covariance_epoch = _parse_state_and_covariance(covar_path)
    _validate_state_matches_elements(state, covariance_epoch, element_epoch, primary)

    coordinates = CartesianCoordinates.from_kwargs(
        x=[float(state[0])],
        y=[float(state[1])],
        z=[float(state[2])],
        vx=[float(state[3])],
        vy=[float(state[4])],
        vz=[float(state[5])],
        time=Timestamp.from_jd([covariance_epoch], scale="tt"),
        origin=Origin.from_kwargs(code=["SUN"]),
        frame="ecliptic",
        covariance=CoordinateCovariances.from_matrix(covariance[np.newaxis, :, :]),
    )
    orbits = Orbits.from_kwargs(
        orbit_id=[primary.object_id],
        object_id=[primary.object_id],
        coordinates=coordinates,
    )
    metadata = FindOrbMetadata(
        object_id=primary.object_id,
        packed_id=primary.packed_id,
        companion_file=primary.filename,
        companion_files=tuple(companion.filename for companion in companions),
        covariance_epoch_jd_tt=covariance_epoch,
        element_epoch_jd_tt=element_epoch,
        central_body=central_body,
        frame=frame,
        reference=reference,
        absolute_magnitude=_optional_float(
            primary.elements, "H", field=f"{primary.filename}.H"
        ),
        slope_parameter=_optional_float(
            primary.elements, "G", field=f"{primary.filename}.G"
        ),
        find_orb_version_jd=primary.find_orb_version_jd,
        consistency_check="epoch_bounded_elements_v2",
    )
    return FindOrbConversion(orbits=orbits, metadata=metadata)


def read_fo_output(
    fo_output_dir: str,
) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    """Read the historical ``total.json`` plus ``covar.json`` output pair."""
    covar_dict = read_fo_covariance(f"{fo_output_dir}/covar.json")
    elements_dict = read_fo_orbits(f"{fo_output_dir}/total.json")
    return elements_dict, covar_dict


def read_fo_covariance(covar_file: str) -> dict[str, object]:
    return _read_json_object(Path(covar_file))


def read_fo_orbits(input_file: str) -> dict[str, dict[str, object]]:
    total_json = _read_json_object(Path(input_file))
    objects = _mapping(total_json.get("objects"), field="objects")
    elements_dict: dict[str, dict[str, object]] = {}
    for object_id, object_data in objects.items():
        object_mapping = _mapping(object_data, field=f"objects.{object_id}")
        elements = _mapping(
            object_mapping.get("elements"),
            field=f"objects.{object_id}.elements",
        )
        elements_dict[object_id] = dict(elements)
    return elements_dict


def fo_to_adam_orbit_cov(fo_output_folder: str) -> Orbits:
    """Convert a strict one-object Find_Orb bundle to an ADAM Orbit."""
    return convert_find_orb_bundle(fo_output_folder).orbits


def rejected_observations_from_fo(fo_output_folder: str) -> ADESObservations:
    total_json = _read_json_object(Path(fo_output_folder) / "total.json")
    objects = _mapping(total_json.get("objects"), field="objects")
    json_observations: list[dict[str, object]] = []
    for object_id, object_data in objects.items():
        object_mapping = _mapping(object_data, field=f"objects.{object_id}")
        raw_observations = object_mapping.get("observations")
        if raw_observations is None:
            continue
        observations = _mapping(
            raw_observations,
            field=f"objects.{object_id}.observations",
        )
        raw_residuals = observations.get("residuals")
        if raw_residuals is None:
            continue
        residuals = _sequence(
            raw_residuals,
            field=f"objects.{object_id}.observations.residuals",
        )
        for residual in residuals:
            observation = dict(
                _mapping(
                    residual,
                    field=f"objects.{object_id}.observations.residuals[]",
                )
            )
            observation["object_id"] = object_id
            json_observations.append(observation)

    rejected_observations = [
        observation
        for observation in json_observations
        if _optional_float(observation, "incl", field="residual.incl") == 0
    ]

    if len(rejected_observations) == 0:
        return ADESObservations.empty()

    ades_rejected_observations = ADESObservations.from_kwargs(
        trkSub=pa.array(
            [observation.get("object_id") for observation in rejected_observations]
        ),
        obsTime=Timestamp.from_jd(
            [
                _required_float(observation, "JD", field="residual.JD")
                for observation in rejected_observations
            ],
            scale="utc",
        ),
        ra=pa.array(
            [
                _required_float(observation, "RA", field="residual.RA")
                for observation in rejected_observations
            ]
        ),
        dec=pa.array(
            [
                _required_float(observation, "Dec", field="residual.Dec")
                for observation in rejected_observations
            ]
        ),
        mag=pa.array(
            [
                _optional_float(observation, "MagObs", field="residual.MagObs")
                for observation in rejected_observations
            ]
        ),
        rmsRACosDec=pa.array(
            [
                _optional_float(observation, "sigma_1", field="residual.sigma_1")
                for observation in rejected_observations
            ]
        ),
        rmsDec=pa.array(
            [
                _optional_float(observation, "sigma_2", field="residual.sigma_2")
                for observation in rejected_observations
            ]
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
