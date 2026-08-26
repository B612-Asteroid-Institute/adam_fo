import json
from pathlib import Path

import numpy as np
import pytest

from adam_fo import FindOrbFormatError, convert_find_orb_bundle
from adam_fo.conversions import rejected_observations_from_fo

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
_ELEMENTS = {
    "central body": "Sun",
    "frame": "J2000 ecliptic",
    "reference": "Find_Orb",
    "epoch": 2457702.5,
    "q": 2.9955406006568,
    "e": 0.0542028586915,
    "i": 9.7123368192265,
    "arg_per": 282.6442739845374,
    "asc_node": 154.3846948448567,
    "Tp": 2457891.39298828,
    "H": 17.75,
    "G": 0.15,
}


def _write_bundle(
    path: Path,
    *,
    object_id: str = "t41f9bb",
    declared_id: str | None = None,
    state: list[float] | None = None,
    covariance: list[list[float]] | None = None,
    elements: dict[str, object] | None = None,
    companion_filename: str = "elem_short.json",
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    declared_id = object_id if declared_id is None else declared_id
    companion = {
        "num": 1,
        "ids": [declared_id],
        "objects": {
            object_id: {
                "object": object_id,
                "packed": object_id,
                "Find_Orb_version": 2460673.5,
                "elements": dict(_ELEMENTS if elements is None else elements),
            }
        },
    }
    (path / companion_filename).write_text(json.dumps(companion), encoding="utf-8")
    covar = {
        "state_vect": _STATE if state is None else state,
        "covar": _COVARIANCE if covariance is None else covariance,
        "epoch": 2457696.686981,
    }
    (path / "covar.json").write_text(json.dumps(covar), encoding="utf-8")


def test_convert_real_short_arc_pair_preserves_same_epoch_state_and_covariance(
    tmp_path: Path,
) -> None:
    _write_bundle(tmp_path)

    result = convert_find_orb_bundle(tmp_path)

    assert len(result.orbits) == 1
    assert result.metadata.object_id == "t41f9bb"
    assert result.metadata.companion_file == "elem_short.json"
    assert result.metadata.covariance_epoch_jd_tt == 2457696.686981
    assert result.metadata.element_epoch_jd_tt == 2457702.5
    assert result.metadata.absolute_magnitude == 17.75
    assert result.metadata.slope_parameter == 0.15
    assert result.metadata.consistency_check == "epoch_bounded_elements_v2"
    np.testing.assert_array_equal(result.orbits.coordinates.values[0], _STATE)
    assert result.orbits.coordinates.time.jd()[0].as_py() == 2457696.686981
    np.testing.assert_array_equal(
        result.orbits.coordinates.covariance.to_matrix()[0],
        (np.asarray(_COVARIANCE) + np.asarray(_COVARIANCE).T) / 2.0,
    )


def test_rejects_conflicting_aggregate_identity(tmp_path: Path) -> None:
    _write_bundle(tmp_path, declared_id="other-object")

    with pytest.raises(FindOrbFormatError, match="declares ID"):
        convert_find_orb_bundle(tmp_path)


def test_rejects_stale_covariance_state(tmp_path: Path) -> None:
    stale_state = list(_STATE)
    stale_state[0] = 0.5
    _write_bundle(tmp_path, state=stale_state)

    with pytest.raises(FindOrbFormatError, match="inconsistent with the companion"):
        convert_find_orb_bundle(tmp_path)


def test_rejects_degenerate_state_with_nonfinite_invariants(tmp_path: Path) -> None:
    _write_bundle(tmp_path, state=[0.0] * 6)

    with pytest.raises(FindOrbFormatError, match="non-finite orbital invariants"):
        convert_find_orb_bundle(tmp_path)


def test_rejects_boolean_state_values(tmp_path: Path) -> None:
    state: list[object] = list(_STATE)
    state[0] = True
    _write_bundle(tmp_path, state=state)  # type: ignore[arg-type]

    with pytest.raises(FindOrbFormatError, match="must be numeric"):
        convert_find_orb_bundle(tmp_path)


def test_rejects_nonfinite_state(tmp_path: Path) -> None:
    state = list(_STATE)
    state[0] = float("nan")
    _write_bundle(tmp_path, state=state)

    with pytest.raises(FindOrbFormatError, match="must be finite"):
        convert_find_orb_bundle(tmp_path)


def test_rejects_asymmetric_covariance(tmp_path: Path) -> None:
    covariance = np.asarray(_COVARIANCE)
    covariance[0, 1] += 1e-4
    _write_bundle(tmp_path, covariance=covariance.tolist())

    with pytest.raises(FindOrbFormatError, match="not symmetric"):
        convert_find_orb_bundle(tmp_path)


def test_rejects_zero_covariance(tmp_path: Path) -> None:
    _write_bundle(tmp_path, covariance=np.zeros((6, 6)).tolist())

    with pytest.raises(FindOrbFormatError, match="must not be all zero"):
        convert_find_orb_bundle(tmp_path)


def test_rejects_non_psd_covariance(tmp_path: Path) -> None:
    covariance = np.eye(6)
    covariance[0, 0] = -1.0
    _write_bundle(tmp_path, covariance=covariance.tolist())

    with pytest.raises(FindOrbFormatError, match="positive semidefinite"):
        convert_find_orb_bundle(tmp_path)


def test_rejects_covariance_with_wrong_parameter_count(tmp_path: Path) -> None:
    _write_bundle(tmp_path, covariance=np.eye(7).tolist())

    with pytest.raises(FindOrbFormatError, match=r"shape \(6, 6\)"):
        convert_find_orb_bundle(tmp_path)


def test_rejects_conflicting_companions(tmp_path: Path) -> None:
    _write_bundle(tmp_path, companion_filename="elements.json")
    conflicting = dict(_ELEMENTS)
    conflicting["epoch"] = 2457703.5
    _write_bundle(tmp_path, elements=conflicting, companion_filename="total.json")

    with pytest.raises(FindOrbFormatError, match="companions disagree on 'epoch'"):
        convert_find_orb_bundle(tmp_path)


def test_rejects_missing_covariance(tmp_path: Path) -> None:
    _write_bundle(tmp_path)
    (tmp_path / "covar.json").unlink()

    with pytest.raises(FindOrbFormatError, match="requires covar.json"):
        convert_find_orb_bundle(tmp_path)


def test_rejects_missing_companion(tmp_path: Path) -> None:
    _write_bundle(tmp_path)
    (tmp_path / "elem_short.json").unlink()

    with pytest.raises(FindOrbFormatError, match="requires one companion"):
        convert_find_orb_bundle(tmp_path)


def test_rejects_multiple_companion_objects(tmp_path: Path) -> None:
    _write_bundle(tmp_path)
    path = tmp_path / "elem_short.json"
    companion = json.loads(path.read_text(encoding="utf-8"))
    companion["objects"]["second"] = companion["objects"]["t41f9bb"]
    path.write_text(json.dumps(companion), encoding="utf-8")

    with pytest.raises(FindOrbFormatError, match="exactly one object"):
        convert_find_orb_bundle(tmp_path)


def test_rejects_large_element_epoch_gap(tmp_path: Path) -> None:
    elements = dict(_ELEMENTS)
    elements["epoch"] = 2457800.5
    _write_bundle(tmp_path, elements=elements)

    with pytest.raises(FindOrbFormatError, match="epoch is too far"):
        convert_find_orb_bundle(tmp_path)


def test_rejects_unsupported_frame(tmp_path: Path) -> None:
    elements = dict(_ELEMENTS)
    elements["frame"] = "J2000 equatorial"
    _write_bundle(tmp_path, elements=elements)

    with pytest.raises(FindOrbFormatError, match="Unsupported Find_Orb element frame"):
        convert_find_orb_bundle(tmp_path)


def test_rejects_unsupported_central_body_and_reference(tmp_path: Path) -> None:
    elements = dict(_ELEMENTS)
    elements["central body"] = "Earth"
    _write_bundle(tmp_path, elements=elements)
    with pytest.raises(FindOrbFormatError, match="central body"):
        convert_find_orb_bundle(tmp_path)

    elements["central body"] = "Sun"
    elements["reference"] = "Other"
    _write_bundle(tmp_path, elements=elements)
    with pytest.raises(FindOrbFormatError, match="reference"):
        convert_find_orb_bundle(tmp_path)


def test_rejects_non_utf8_json(tmp_path: Path) -> None:
    _write_bundle(tmp_path)
    (tmp_path / "elem_short.json").write_bytes(b"\xff\xfe")

    with pytest.raises(FindOrbFormatError, match="Could not read"):
        convert_find_orb_bundle(tmp_path)


def test_rejects_oversized_and_deeply_nested_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_bundle(tmp_path)
    monkeypatch.setattr("adam_fo.conversions._MAX_JSON_BYTES", 10)
    with pytest.raises(FindOrbFormatError, match="32 MiB limit"):
        convert_find_orb_bundle(tmp_path)

    monkeypatch.setattr("adam_fo.conversions._MAX_JSON_BYTES", 32 * 1024 * 1024)
    (tmp_path / "covar.json").write_text("[" * 2000, encoding="utf-8")
    with pytest.raises(FindOrbFormatError, match="Could not read"):
        convert_find_orb_bundle(tmp_path)


def test_rejected_residual_requires_numeric_astrometry(tmp_path: Path) -> None:
    _write_bundle(tmp_path, companion_filename="total.json")
    path = tmp_path / "total.json"
    total = json.loads(path.read_text(encoding="utf-8"))
    total["objects"]["t41f9bb"]["observations"] = {"residuals": [{"incl": 0}]}
    path.write_text(json.dumps(total), encoding="utf-8")

    with pytest.raises(FindOrbFormatError, match="residual.JD"):
        rejected_observations_from_fo(str(tmp_path))


def test_missing_residual_diagnostics_are_optional(tmp_path: Path) -> None:
    _write_bundle(tmp_path, companion_filename="total.json")

    rejected = rejected_observations_from_fo(str(tmp_path))

    assert len(rejected) == 0
