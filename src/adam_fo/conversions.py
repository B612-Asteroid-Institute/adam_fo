import hashlib
import json
import logging
import math
import struct
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

_COVARIANCE_SYMMETRY_RTOL = 1e-7
# This deliberately matches the API and Jobs execution trust boundaries. The
# accepted antisymmetric serialization noise is removed before this PSD check.
_COVARIANCE_PSD_RTOL = 1e-10
_MAX_JSON_BYTES = 1024 * 1024
_FIND_ORB_COVARIANCE_FIELDS = frozenset({"state_vect", "covar", "epoch"})
_COVARIANCE_ORDER = ("x", "y", "z", "vx", "vy", "vz")
_FIND_ORB_CONVENTION_SOURCE = (
    "Bill-Gray/find_orb@7e02c585cb4e13130c9b2e9c82b9e95469b738b5 "
    "orb_func.cpp full_improvement covar.json output"
)


class FindOrbFormatError(ValueError):
    """Raised when a Find_Orb covariance file is unsafe to canonicalize."""


@dataclass(frozen=True)
class FindOrbMetadata:
    source_filename: str
    covariance_epoch_jd_tt: float
    origin: str
    frame: str
    time_scale: str
    position_unit: str
    velocity_unit: str
    covariance_order: tuple[str, ...]
    convention_source: str
    fit_verified: bool


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


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            key_hint = repr(key[:100])
            raise FindOrbFormatError(f"Find_Orb JSON contains duplicate key {key_hint}")
        value[key] = item
    return value


def _read_json_object(path: Path) -> dict[str, object]:
    try:
        with path.open("rb") as file:
            encoded = file.read(_MAX_JSON_BYTES + 1)
        if len(encoded) > _MAX_JSON_BYTES:
            raise FindOrbFormatError(
                f"Find_Orb JSON file {path.name!r} exceeds the 1 MiB limit"
            )
        value = cast(
            object,
            json.loads(
                encoded.decode("utf-8"),
                object_pairs_hook=_reject_duplicate_json_keys,
            ),
        )
    except FindOrbFormatError:
        raise
    except (OSError, UnicodeDecodeError, ValueError, RecursionError) as error:
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
    try:
        parsed = float(value)
    except OverflowError as error:
        raise FindOrbFormatError(f"Find_Orb field {field!r} must be finite") from error
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


def _parse_state_and_covariance(
    covar_path: Path,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float]:
    document = _read_json_object(covar_path)
    if set(document) != _FIND_ORB_COVARIANCE_FIELDS:
        raise FindOrbFormatError(
            "Find_Orb covar.json must contain exactly 'state_vect', 'covar', and 'epoch'"
        )
    state_values = _sequence(document.get("state_vect"), field="covar.json.state_vect")
    if len(state_values) != 6:
        raise FindOrbFormatError(
            f"Find_Orb state_vect must have shape (6,), got ({len(state_values)},)"
        )
    state = np.asarray(
        [
            _finite_float(value, field=f"covar.json.state_vect[{index}]")
            for index, value in enumerate(state_values)
        ],
        dtype=np.float64,
    )
    covariance_rows = _sequence(document.get("covar"), field="covar.json.covar")
    if len(covariance_rows) != 6:
        raise FindOrbFormatError(
            f"Find_Orb covariance must have shape (6, 6), got ({len(covariance_rows)}, ...)"
        )
    parsed_rows: list[list[float]] = []
    for row_index, row in enumerate(covariance_rows):
        row_values = _sequence(row, field=f"covar.json.covar[{row_index}]")
        if len(row_values) != 6:
            raise FindOrbFormatError(
                "Find_Orb covariance must have shape (6, 6), "
                f"row {row_index} has {len(row_values)} values"
            )
        parsed_rows.append(
            [
                _finite_float(
                    value,
                    field=f"covar.json.covar[{row_index}][{column_index}]",
                )
                for column_index, value in enumerate(row_values)
            ]
        )
    covariance = np.asarray(parsed_rows, dtype=np.float64)
    if not np.all(np.isfinite(state)) or not np.all(np.isfinite(covariance)):
        raise FindOrbFormatError(
            "Find_Orb state and covariance must contain only finite values"
        )

    covariance_scale = float(np.max(np.abs(covariance)))
    if covariance_scale == 0.0:
        raise FindOrbFormatError("Find_Orb covariance must not be all zero")
    normalized_covariance = covariance / covariance_scale
    covariance_norm = float(np.linalg.norm(normalized_covariance, ord="fro"))
    asymmetry_norm = float(
        np.linalg.norm(normalized_covariance - normalized_covariance.T, ord="fro")
    )
    if asymmetry_norm > _COVARIANCE_SYMMETRY_RTOL * covariance_norm:
        raise FindOrbFormatError(
            "Find_Orb covariance is not symmetric within serialization tolerance"
        )

    normalized_covariance = (normalized_covariance + normalized_covariance.T) / 2.0
    try:
        eigenvalues = np.linalg.eigvalsh(normalized_covariance)
    except np.linalg.LinAlgError as error:
        raise FindOrbFormatError(
            "Find_Orb covariance eigendecomposition did not converge"
        ) from error
    if not np.all(np.isfinite(eigenvalues)):
        raise FindOrbFormatError("Find_Orb covariance eigenvalues must be finite")
    if float(eigenvalues[0]) < -_COVARIANCE_PSD_RTOL:
        raise FindOrbFormatError("Find_Orb covariance is not positive semidefinite")
    with np.errstate(over="raise", invalid="raise"):
        try:
            covariance = (covariance + covariance.T) / 2.0
        except FloatingPointError as error:
            raise FindOrbFormatError(
                "Find_Orb covariance cannot be symmetrized safely"
            ) from error

    epoch = _required_float(document, "epoch", field="covar.json.epoch")
    return state, covariance, epoch


def _canonical_orbit_id(
    state: npt.NDArray[np.float64],
    covariance: npt.NDArray[np.float64],
    epoch: float,
) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(state, dtype="<f8").tobytes(order="C"))
    digest.update(np.asarray(covariance, dtype="<f8").tobytes(order="C"))
    digest.update(struct.pack("<d", epoch))
    return f"find_orb_covar_{digest.hexdigest()[:24]}"


def convert_find_orb_covariance(covar_file: str | Path) -> FindOrbConversion:
    """Validate and canonicalize one modern Find_Orb ``covar.json`` file.

    Find_Orb writes Cartesian state in AU and AU/day with a Cartesian covariance
    ordered x, y, z, vx, vy, vz. The state and covariance share the file's TT
    epoch and use heliocentric J2000 ecliptic coordinates. This function validates
    only the supplied numerical product; it does not verify the original fit or
    infer object identity from another Find_Orb output.
    """

    covar_path = Path(covar_file)
    if not covar_path.is_file():
        raise FindOrbFormatError("Find_Orb covariance input must be a file")

    state, covariance, covariance_epoch = _parse_state_and_covariance(covar_path)
    try:
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
            orbit_id=[_canonical_orbit_id(state, covariance, covariance_epoch)],
            object_id=[None],
            coordinates=coordinates,
        )
    except Exception as error:
        logger.warning(
            "Could not construct ADAM Orbit from Find_Orb covariance: %s", error
        )
        raise FindOrbFormatError(
            "Find_Orb state, covariance, or epoch cannot form an ADAM Orbit"
        ) from error
    metadata = FindOrbMetadata(
        source_filename=covar_path.name,
        covariance_epoch_jd_tt=covariance_epoch,
        origin="SUN",
        frame="ecliptic",
        time_scale="tt",
        position_unit="au",
        velocity_unit="au/day",
        covariance_order=_COVARIANCE_ORDER,
        convention_source=_FIND_ORB_CONVENTION_SOURCE,
        fit_verified=False,
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


def _validate_controlled_solution_conventions(solution: _CompanionSolution) -> None:
    central_body = _required_string(
        solution.elements,
        "central body",
        field=f"{solution.filename}.central body",
    )
    frame = _required_string(
        solution.elements,
        "frame",
        field=f"{solution.filename}.frame",
    )
    reference = _required_string(
        solution.elements,
        "reference",
        field=f"{solution.filename}.reference",
    )
    if central_body.casefold() not in {"sun", "sol"}:
        raise FindOrbFormatError(
            f"Unsupported controlled Find_Orb central body {central_body!r}"
        )
    if frame.casefold() != "j2000 ecliptic":
        raise FindOrbFormatError(f"Unsupported controlled Find_Orb frame {frame!r}")
    if reference.casefold() != "find_orb":
        raise FindOrbFormatError(
            f"Unsupported controlled Find_Orb reference {reference!r}"
        )


def fo_to_adam_orbit_cov(fo_output_folder: str) -> Orbits:
    """Convert controlled one-object Find_Orb output to an identified ADAM Orbit."""
    output_path = Path(fo_output_folder)
    conversion = convert_find_orb_covariance(output_path / "covar.json")
    solution = _parse_companion(output_path / "total.json")
    _validate_controlled_solution_conventions(solution)
    return Orbits.from_kwargs(
        orbit_id=[solution.object_id],
        object_id=[solution.object_id],
        coordinates=conversion.orbits.coordinates,
    )


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
