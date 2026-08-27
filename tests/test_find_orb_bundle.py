import json
from pathlib import Path

import numpy as np
import pytest

from adam_fo import FindOrbFormatError, convert_find_orb_covariance
from adam_fo.conversions import fo_to_adam_orbit_cov, rejected_observations_from_fo

_STATE = [
    2.32620748805888988,
    1.88255959130043937,
    -0.462668513113437874,
    -0.00666748005038758466,
    0.0075569182379197691,
    -0.000672927691282525827,
]
_COVARIANCE = [
    [
        1.69731122712e-05,
        1.30273228265e-05,
        -4.9424919323e-06,
        1.10162825743e-07,
        -1.3070478952e-08,
        -2.16174816568e-08,
    ],
    [
        1.30273225495e-05,
        9.99882267168e-06,
        -3.79349621045e-06,
        8.4554918789e-08,
        -1.00305312914e-08,
        -1.65925634893e-08,
    ],
    [
        -4.94249185392e-06,
        -3.79349623096e-06,
        1.43923093224e-06,
        -3.20791517407e-08,
        3.80588731988e-09,
        6.29495386529e-09,
    ],
    [
        1.10162824916e-07,
        8.45549199501e-08,
        -3.20791520066e-08,
        1.03305830746e-09,
        1.50128974411e-10,
        -2.31906446816e-10,
    ],
    [
        -1.30704787764e-08,
        -1.00305313715e-08,
        3.80588733062e-09,
        1.50128974904e-10,
        1.83670306381e-10,
        -5.10259998703e-11,
    ],
    [
        -2.16174816018e-08,
        -1.65925637993e-08,
        6.29495394852e-09,
        -2.31906447513e-10,
        -5.10259996923e-11,
        5.39319671738e-11,
    ],
]
_EPOCH = 2457696.686981


def _write_covar(
    path: Path,
    *,
    state: list[object] | None = None,
    covariance: list[object] | None = None,
    epoch: object = _EPOCH,
    extra: dict[str, object] | None = None,
) -> Path:
    document: dict[str, object] = {
        "state_vect": _STATE if state is None else state,
        "covar": _COVARIANCE if covariance is None else covariance,
        "epoch": epoch,
    }
    if extra:
        document.update(extra)
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _write_total(
    path: Path,
    *,
    object_id: str = "t41f9bb",
    declared_id: str | None = None,
    observations: dict[str, object] | None = None,
    elements: dict[str, object] | None = None,
) -> None:
    declared_id = object_id if declared_id is None else declared_id
    object_document: dict[str, object] = {
        "object": object_id,
        "packed": object_id,
        "elements": elements
        or {
            "central body": "Sun",
            "frame": "J2000 ecliptic",
            "reference": "Find_Orb",
        },
    }
    if observations is not None:
        object_document["observations"] = observations
    document = {
        "num": 1,
        "ids": [declared_id],
        "objects": {object_id: object_document},
    }
    path.write_text(json.dumps(document), encoding="utf-8")


def test_convert_real_covar_json_preserves_state_covariance_and_exact_epoch(
    tmp_path: Path,
) -> None:
    path = _write_covar(tmp_path / "covar.json")

    result = convert_find_orb_covariance(path)

    assert len(result.orbits) == 1
    assert result.orbits.object_id[0].as_py() is None
    assert result.metadata.source_filename == "covar.json"
    assert result.metadata.covariance_epoch_jd_tt == _EPOCH
    assert result.metadata.origin == "SUN"
    assert result.metadata.frame == "ecliptic"
    assert result.metadata.time_scale == "tt"
    assert result.metadata.position_unit == "au"
    assert result.metadata.velocity_unit == "au/day"
    assert result.metadata.covariance_order == ("x", "y", "z", "vx", "vy", "vz")
    assert "Bill-Gray/find_orb@7e02c585" in result.metadata.convention_source
    assert result.metadata.fit_verified is False
    np.testing.assert_array_equal(result.orbits.coordinates.values[0], _STATE)
    assert result.orbits.coordinates.time.jd()[0].as_py() == _EPOCH
    np.testing.assert_array_equal(
        result.orbits.coordinates.covariance.to_matrix()[0],
        (np.asarray(_COVARIANCE) + np.asarray(_COVARIANCE).T) / 2.0,
    )


def test_direct_covar_json_does_not_require_or_infer_companion_identity(
    tmp_path: Path,
) -> None:
    path = _write_covar(tmp_path / "covar.json", state=[0.5, *_STATE[1:]])

    result = convert_find_orb_covariance(path)

    assert result.orbits.object_id[0].as_py() is None
    assert result.orbits.coordinates.x[0].as_py() == 0.5


def test_controlled_fo_output_attaches_strict_total_json_identity(
    tmp_path: Path,
) -> None:
    _write_covar(tmp_path / "covar.json")
    _write_total(tmp_path / "total.json")

    result = fo_to_adam_orbit_cov(str(tmp_path))

    assert result.orbit_id[0].as_py() == "t41f9bb"
    assert result.object_id[0].as_py() == "t41f9bb"


def test_direct_covar_ids_are_deterministic_and_content_derived(tmp_path: Path) -> None:
    first_path = _write_covar(tmp_path / "first.json")
    second_path = _write_covar(tmp_path / "second.json")

    first = convert_find_orb_covariance(first_path)
    first_again = convert_find_orb_covariance(first_path)
    second = convert_find_orb_covariance(second_path)
    changed_path = _write_covar(tmp_path / "changed.json", state=[0.5, *_STATE[1:]])
    changed = convert_find_orb_covariance(changed_path)

    assert first.orbits.orbit_id[0].as_py() == first_again.orbits.orbit_id[0].as_py()
    assert first.orbits.orbit_id[0].as_py() == second.orbits.orbit_id[0].as_py()
    assert first.orbits.orbit_id[0].as_py() != changed.orbits.orbit_id[0].as_py()


def test_controlled_fo_output_rejects_conflicting_identity(tmp_path: Path) -> None:
    _write_covar(tmp_path / "covar.json")
    _write_total(tmp_path / "total.json", declared_id="other-object")

    with pytest.raises(FindOrbFormatError, match="declares ID"):
        fo_to_adam_orbit_cov(str(tmp_path))


def test_controlled_fo_output_rejects_unsupported_conventions(tmp_path: Path) -> None:
    _write_covar(tmp_path / "covar.json")
    _write_total(
        tmp_path / "total.json",
        elements={
            "central body": "Earth",
            "frame": "J2000 ecliptic",
            "reference": "Find_Orb",
        },
    )

    with pytest.raises(FindOrbFormatError, match="central body"):
        fo_to_adam_orbit_cov(str(tmp_path))


def test_rejects_boolean_and_nonfinite_state_values(tmp_path: Path) -> None:
    boolean_state: list[object] = list(_STATE)
    boolean_state[0] = True
    path = _write_covar(tmp_path / "covar.json", state=boolean_state)
    with pytest.raises(FindOrbFormatError, match="must be numeric"):
        convert_find_orb_covariance(path)

    nonfinite_state: list[object] = list(_STATE)
    nonfinite_state[0] = float("nan")
    _write_covar(path, state=nonfinite_state)
    with pytest.raises(FindOrbFormatError, match="must be finite"):
        convert_find_orb_covariance(path)

    overflowing_integer: list[object] = list(_STATE)
    overflowing_integer[0] = 10**400
    _write_covar(path, state=overflowing_integer)
    with pytest.raises(FindOrbFormatError, match="must be finite"):
        convert_find_orb_covariance(path)


def test_rejects_json_integer_over_interpreter_digit_limit(tmp_path: Path) -> None:
    path = tmp_path / "covar.json"
    prefix = json.dumps(
        {"state_vect": _STATE, "covar": _COVARIANCE},
        separators=(",", ":"),
    )[:-1]
    path.write_text(f'{prefix},"epoch":1{"0" * 5000}}}', encoding="utf-8")

    with pytest.raises(FindOrbFormatError, match="Could not read"):
        convert_find_orb_covariance(path)


def test_rejects_invalid_state_and_covariance_shapes(tmp_path: Path) -> None:
    path = _write_covar(tmp_path / "covar.json", state=_STATE[:-1])
    with pytest.raises(FindOrbFormatError, match=r"shape \(6,\)"):
        convert_find_orb_covariance(path)

    _write_covar(path, covariance=np.eye(7).tolist())
    with pytest.raises(FindOrbFormatError, match=r"shape \(6, 6\)"):
        convert_find_orb_covariance(path)

    ragged = [list(row) for row in _COVARIANCE]
    ragged[2] = ragged[2][:-1]
    _write_covar(path, covariance=ragged)
    with pytest.raises(FindOrbFormatError, match=r"shape \(6, 6\)"):
        convert_find_orb_covariance(path)


def test_rejects_invalid_covariance_science(tmp_path: Path) -> None:
    path = _write_covar(tmp_path / "covar.json", covariance=np.zeros((6, 6)).tolist())
    with pytest.raises(FindOrbFormatError, match="must not be all zero"):
        convert_find_orb_covariance(path)

    asymmetric = np.asarray(_COVARIANCE)
    asymmetric[0, 1] += 1e-4
    _write_covar(path, covariance=asymmetric.tolist())
    with pytest.raises(FindOrbFormatError, match="not symmetric"):
        convert_find_orb_covariance(path)

    non_psd = np.eye(6)
    non_psd[0, 0] = -1.0
    _write_covar(path, covariance=non_psd.tolist())
    with pytest.raises(FindOrbFormatError, match="positive semidefinite"):
        convert_find_orb_covariance(path)

    below_shared_psd_tolerance = np.eye(6)
    below_shared_psd_tolerance[0, 0] = -2e-10
    _write_covar(path, covariance=below_shared_psd_tolerance.tolist())
    with pytest.raises(FindOrbFormatError, match="positive semidefinite"):
        convert_find_orb_covariance(path)

    within_shared_psd_tolerance = np.eye(6)
    within_shared_psd_tolerance[0, 0] = -5e-11
    _write_covar(path, covariance=within_shared_psd_tolerance.tolist())
    assert len(convert_find_orb_covariance(path).orbits) == 1


def test_rejects_nonfinite_epoch_and_non_numeric_covariance(tmp_path: Path) -> None:
    path = _write_covar(tmp_path / "covar.json", epoch=float("inf"))
    with pytest.raises(FindOrbFormatError, match="covar.json.epoch.*finite"):
        convert_find_orb_covariance(path)

    covariance: list[object] = [list(row) for row in _COVARIANCE]
    first_row = list(covariance[0])
    first_row[0] = False
    covariance[0] = first_row
    _write_covar(path, covariance=covariance)
    with pytest.raises(FindOrbFormatError, match="must be numeric"):
        convert_find_orb_covariance(path)

    huge = np.eye(6) * 1e308
    _write_covar(path, covariance=huge.tolist())
    with pytest.raises(FindOrbFormatError, match="symmetrized safely"):
        convert_find_orb_covariance(path)


def test_rejects_missing_or_extra_top_level_fields(tmp_path: Path) -> None:
    path = _write_covar(tmp_path / "covar.json", extra={"object": "hint"})
    with pytest.raises(FindOrbFormatError, match="must contain exactly"):
        convert_find_orb_covariance(path)

    path.write_text(
        json.dumps({"state_vect": _STATE, "epoch": _EPOCH}), encoding="utf-8"
    )
    with pytest.raises(FindOrbFormatError, match="must contain exactly"):
        convert_find_orb_covariance(path)

    path.write_text(
        '{"state_vect": [], "state_vect": [], "covar": [], "epoch": 1}',
        encoding="utf-8",
    )
    with pytest.raises(FindOrbFormatError, match="duplicate key"):
        convert_find_orb_covariance(path)


def test_rejects_missing_non_utf8_oversized_and_deep_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "covar.json"
    with pytest.raises(FindOrbFormatError, match="must be a file"):
        convert_find_orb_covariance(path)

    path.write_bytes(b"\xff\xfe")
    with pytest.raises(FindOrbFormatError, match="Could not read"):
        convert_find_orb_covariance(path)

    _write_covar(path)
    monkeypatch.setattr("adam_fo.conversions._MAX_JSON_BYTES", 10)
    with pytest.raises(FindOrbFormatError, match="1 MiB limit"):
        convert_find_orb_covariance(path)

    monkeypatch.setattr("adam_fo.conversions._MAX_JSON_BYTES", 1024 * 1024)
    path.write_text("[" * 2000, encoding="utf-8")
    with pytest.raises(FindOrbFormatError, match="Could not read"):
        convert_find_orb_covariance(path)


def test_rejected_residual_requires_numeric_astrometry(tmp_path: Path) -> None:
    _write_total(
        tmp_path / "total.json",
        observations={"residuals": [{"incl": 0}]},
    )

    with pytest.raises(FindOrbFormatError, match="residual.JD"):
        rejected_observations_from_fo(str(tmp_path))


def test_missing_residual_diagnostics_are_optional(tmp_path: Path) -> None:
    _write_total(tmp_path / "total.json")

    rejected = rejected_observations_from_fo(str(tmp_path))

    assert len(rejected) == 0
