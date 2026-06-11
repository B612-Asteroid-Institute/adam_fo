import pickle
import tempfile

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest
from adam_fo.build import main as build_fo
from adam_fo.config import check_build_exists

from adam_core.coordinates import CoordinateCovariances, SphericalCoordinates
from adam_core.coordinates.cartesian import CartesianCoordinates
from adam_core.coordinates.origin import Origin
from adam_core.observations.ades import ADESObservations
from adam_core.observers import Observers
from adam_core.orbits import Orbits
from adam_core.time import Timestamp
from adam_core.orbit_determination.evaluate import OrbitDeterminationObservations, OrbitDeterminationPhotometry
from adam_fo.find_orb_orbit_fitter import FindOrbOrbitFitter


@pytest.fixture
def real_data():
    # Actual observations for "2009 JY22"
    obstimes = Timestamp.from_kwargs(
        days=[54952, 54952, 54952, 54952, 54977, 54977, 54977, 54977, 56209, 56209],
        nanos=[
            15930432000000,
            16879968000000,
            17813088000000,
            18760032000000,
            14122080000000,
            14906592000000,
            15680736000000,
            16459200000000,
            31688237000000,
            32917536000000,
        ],
        scale="utc",
    )
    obscodes = ["G96", "G96", "G96", "G96", "G96", "G96", "G96", "G96", "F51", "F51"]
    lon = [
        173.174080,
        173.173500,
        173.173170,
        173.172420,
        174.067330,
        174.068380,
        174.069290,
        174.070500,
        20.118004,
        20.114975,
    ]
    lat = [
        7.762110,
        7.760780,
        7.759580,
        7.758310,
        4.412810,
        4.411390,
        4.410170,
        4.408890,
        8.454381,
        8.453956,
    ]
    obsids = [f"KG0CNl00000055470100001d{i}" for i in range(10)]
    bands = ["V", "V", "V", "V", "V", "V", "V", "V", "w", "w"]
    mags = [21.1, 20.5, 21.2, 21.1, 21.9, 21.2, 21.8, 21.4, 22.0, 21.9]

    # Per-observation 6x6 covariance with finite RA/Dec sigmas (1 arcsec).
    sigma_arcsec = 1.0
    sigma_deg = sigma_arcsec / 3600.0
    cov = np.zeros((10, 6, 6))
    cov[:, 1, 1] = sigma_deg ** 2
    cov[:, 2, 2] = sigma_deg ** 2

    coords = SphericalCoordinates.from_kwargs(
        lon=lon,
        lat=lat,
        time=obstimes,
        origin=Origin.from_kwargs(code=obscodes),
        frame="equatorial",
        covariance=CoordinateCovariances.from_matrix(cov),
    )
    observers = Observers.from_codes(codes=obscodes, times=obstimes)

    photometry = OrbitDeterminationPhotometry.from_kwargs(
        mag=mags,
        band=bands,
    )

    observations = OrbitDeterminationObservations.from_kwargs(
        id=obsids,
        coordinates=coords,
        observers=observers,
        photometry=photometry,
    )
    return observations


def force_adam_fo_install():
    try:
        check_build_exists()
    except RuntimeError:
        build_fo()


def test_pickle():
    results_dir = "/some/path"
    fitter = FindOrbOrbitFitter(fo_result_dir=results_dir)
    assert len(fitter.obscodes) > 0, "Obscodes should be loaded by the constructor"
    saved = pickle.dumps(fitter)

    new_fitter = pickle.loads(saved)
    assert new_fitter.fo_result_dir == results_dir
    assert len(new_fitter.obscodes) == len(fitter.obscodes)


def test_success(real_data):
    # force_adam_fo_install()
    observations = real_data
    out_dir = tempfile.TemporaryDirectory()
    fitter = FindOrbOrbitFitter(fo_result_dir=out_dir.name)
    object_id = "2009 JY22"
    fitted_orbit, fitted_members = fitter.initial_fit(object_id, observations)
    assert len(fitted_orbit) == 1
    assert fitted_orbit.object_id[0].as_py() == object_id
    assert len(fitted_members) == len(observations)
    outliers = fitted_members.outlier
    solution = fitted_members.solution
    assert pc.invert(outliers) == solution
    outlier_count = np.sum(outliers.to_pylist())
    # FO rejects the last two of the observations
    assert outlier_count > 0
    assert outlier_count < len(observations)


def test_returns_fitted_orbits_with_chi2_and_success(real_data):
    """Regression test for hw1: FindOrbOrbitFitter.initial_fit must return a
    real FittedOrbits with reduced_chi2 and success populated, not a plain
    Orbits. Pilot v10 was unusable because these columns silently became null
    in the cloud worker."""
    observations = real_data
    out_dir = tempfile.TemporaryDirectory()
    fitter = FindOrbOrbitFitter(fo_result_dir=out_dir.name)
    fitted_orbit, _ = fitter.initial_fit("2009 JY22", observations)

    assert len(fitted_orbit) == 1
    # Schema columns required by downstream callers
    assert "reduced_chi2" in fitted_orbit.table.column_names
    assert "success" in fitted_orbit.table.column_names
    # Values must be populated, not null/NaN
    assert fitted_orbit.reduced_chi2[0].is_valid
    assert np.isfinite(fitted_orbit.reduced_chi2[0].as_py())
    assert fitted_orbit.success[0].is_valid
    assert fitted_orbit.success[0].as_py() is True
    # to_orbits() is what core.py uses downstream — must work without error
    plain = fitted_orbit.to_orbits()
    assert len(plain) == 1


def test_not_enough_data(real_data):
    # force_adam_fo_install()
    # A FindOrb failure now returns a single-row placeholder with
    # success=False (and a member row per input obs) instead of empty tables,
    # so callers can detect the failure via success rather than by an empty
    # table that is easily mistaken for "nothing to process".
    observations = real_data[:2]
    out_dir = tempfile.TemporaryDirectory()
    fitter = FindOrbOrbitFitter(fo_result_dir=out_dir.name)
    object_id = "2009 JY22"
    fitted_orbit, fitted_members = fitter.initial_fit(object_id, observations)
    assert len(fitted_orbit) == 1
    assert fitted_orbit.success[0].as_py() is False
    assert fitted_orbit.object_id[0].as_py() == object_id
    assert len(fitted_members) == len(observations)


def test_findorb_failure_returns_placeholder_with_success_false(real_data, monkeypatch):
    """When Find_Orb exits without producing covar.json/total.json, the fitter
    must return a non-empty single-row ``FittedOrbits`` with ``success=False``
    (and a member row per input observation) rather than ``FittedOrbits.empty()``.

    This matches the failure contract of adam_core's reference fitter
    ``fit_least_squares``, which always returns a success-flagged row and never
    an empty table. An empty table is easily dropped by downstream
    ``len(...) == 0`` guards, making the failure invisible; the placeholder
    keeps it detectable via ``success``.
    """
    def fake_fo(ades_string, out_dir=None, clean_up=True, state_vec=None, **_):
        return (
            Orbits.empty(),
            ADESObservations.empty(),
            "Find_Orb failed, covar.json or total.json file not found",
        )

    monkeypatch.setattr("adam_fo.find_orb_orbit_fitter.fo", fake_fo)

    fitter = FindOrbOrbitFitter(fo_result_dir="/tmp/unused")
    object_id = "2009 JY22"
    fitted_orbit, fitted_members = fitter.initial_fit(object_id, real_data)

    # Single placeholder orbit row, marked as a failure
    assert len(fitted_orbit) == 1
    assert fitted_orbit.success[0].is_valid
    assert fitted_orbit.success[0].as_py() is False
    assert fitted_orbit.object_id[0].as_py() == object_id
    # reduced_chi2 must be present in the schema; value is NaN for failures
    assert "reduced_chi2" in fitted_orbit.table.column_names
    assert np.isnan(fitted_orbit.reduced_chi2[0].as_py())

    # One member row per input observation, so downstream consumers see the row
    assert len(fitted_members) == len(real_data)
    # All members carry the placeholder orbit's id, so the join with orbits works
    assert (
        fitted_members.orbit_id[0].as_py() == fitted_orbit.orbit_id[0].as_py()
    )
    # solution/outlier are null on a failed fit (the fit produced no solution)
    assert not fitted_members.solution[0].is_valid
    assert not fitted_members.outlier[0].is_valid


def test_rejected_obs_match_failure_skips_gracefully(real_data):
    """When FindOrb's rejected-obs list contains an entry whose station+time
    matches no input observation (ADES round-trip can drop sub-second
    precision, station codes can differ in case/whitespace), the matching
    helper must log a warning and skip just THAT entry rather than crashing
    the entire fit via ``assert len(original) == 1``. The previous behavior
    discarded the whole fit over a single unmatchable rejection.
    """
    fitter = FindOrbOrbitFitter(fo_result_dir="/tmp/unused")

    # Construct a single rejected ADES entry whose station ("ZZZ") matches no
    # input observation. The 1-second time window is also irrelevant since
    # station won't match.
    rejected = ADESObservations.from_kwargs(
        obsTime=real_data.coordinates.time[0:1],
        ra=[real_data.coordinates.lon[0].as_py()],
        dec=[real_data.coordinates.lat[0].as_py()],
        rmsRACosDec=[0.5],
        rmsDec=[0.5],
        stn=["ZZZ"],
        mode=["NA"],
        astCat=["NA"],
    )

    # Must not raise; must return a member row per input observation with
    # zero rows flagged as outlier (since the unmatched rejected entry is
    # skipped, not matched to any input obs).
    result = fitter._rejected_observations_to_fitted_members(
        real_data, rejected, orbit_id="test-orbit-id"
    )

    assert len(result) == len(real_data)
    outliers = result.outlier.to_pylist()
    assert sum(1 for o in outliers if o) == 0, (
        "unmatched rejected entry must be skipped, not matched to an input obs"
    )


def test_rejected_obs_matching_is_time_scale_invariant(real_data):
    """The rejected-obs join compares epochs in a common scale (UTC). A
    rejected entry whose ``obsTime`` carries a different scale (here TT,
    ~69 s offset from UTC if compared naively) must still match its input
    observation: both sides are rescaled to UTC before the 1-second window
    is applied. Without the rescale, every such entry would silently fail
    to match and be skipped, leaving outliers unflagged and inflating
    reduced_chi2 downstream.
    """
    fitter = FindOrbOrbitFitter(fo_result_dir="/tmp/unused")

    # Same instant as the first input observation, expressed in TT.
    rejected = ADESObservations.from_kwargs(
        obsTime=real_data.coordinates.time[0:1].rescale("tt"),
        ra=[real_data.coordinates.lon[0].as_py()],
        dec=[real_data.coordinates.lat[0].as_py()],
        rmsRACosDec=[0.5],
        rmsDec=[0.5],
        stn=[real_data.observers.code[0].as_py()],
        mode=["NA"],
        astCat=["NA"],
    )

    result = fitter._rejected_observations_to_fitted_members(
        real_data, rejected, orbit_id="test-orbit-id"
    )

    assert len(result) == len(real_data)
    outliers = result.outlier.to_pylist()
    assert sum(1 for o in outliers if o) == 1, (
        "a rejected entry at the same instant in a different time scale "
        "must match its input observation after rescaling to UTC"
    )
    # The flagged row must be the first observation specifically.
    flagged_idx = outliers.index(True)
    assert result.obs_id[flagged_idx].as_py() == real_data.id[0].as_py()


def _make_seed_orbit(jd_tdb: float, x: float, y: float, z: float,
                     vx: float, vy: float, vz: float) -> Orbits:
    """Build a single-row heliocentric ecliptic J2000 Orbits in TDB."""
    mjd = jd_tdb - 2400000.5
    days = int(np.floor(mjd))
    nanos = int(round((mjd - days) * 86_400 * 1_000_000_000))
    epoch = Timestamp.from_kwargs(days=[days], nanos=[nanos], scale="tdb")
    return Orbits.from_kwargs(
        orbit_id=["seed-1"],
        object_id=["TEST"],
        coordinates=CartesianCoordinates.from_kwargs(
            x=[x], y=[y], z=[z], vx=[vx], vy=[vy], vz=[vz],
            time=epoch,
            frame="ecliptic",
            origin=Origin.from_kwargs(code=["SUN"]),
        ),
    )


def test_initial_fit_omits_state_vec_without_seed(real_data, monkeypatch):
    """Backward compatibility: with reference_orbit unset, ``initial_fit`` must
    invoke the underlying ``fo`` wrapper without a ``state_vec`` (cold-start
    path preserved exactly as before bead 9sg).
    """
    captured = {}

    def fake_fo(ades_string, out_dir=None, clean_up=True, state_vec=None,
                **_):
        captured["state_vec"] = state_vec
        return Orbits.empty(), ADESObservations.empty(), "stop"

    monkeypatch.setattr("adam_fo.find_orb_orbit_fitter.fo", fake_fo)

    fitter = FindOrbOrbitFitter(fo_result_dir="/tmp/unused")
    fitter.initial_fit("2009 JY22", real_data)

    assert captured["state_vec"] is None


def test_initial_fit_forwards_reference_orbit_as_state_vec(real_data, monkeypatch):
    """When reference_orbit is supplied, the seed must reach Find_Orb's ``-v``
    flag in the format ``"<jd>,<x> <y> <z> <vx> <vy> <vz>[,<scale>]"`` parsed
    by ``extract_state_vect_from_text`` in ``find_orb/elem_out.cpp``. The seed
    is converted to heliocentric ecliptic J2000 cartesian (Find_Orb's default
    frame) and the time scale is appended so Find_Orb can convert internally.
    """
    captured = {}

    def fake_fo(ades_string, out_dir=None, clean_up=True, state_vec=None,
                **_):
        captured["state_vec"] = state_vec
        return Orbits.empty(), ADESObservations.empty(), "stop"

    monkeypatch.setattr("adam_fo.find_orb_orbit_fitter.fo", fake_fo)

    seed = _make_seed_orbit(
        jd_tdb=2459580.5,
        x=1.5, y=1.6, z=0.05,
        vx=-0.011, vy=0.0095, vz=-0.0005,
    )

    fitter = FindOrbOrbitFitter(fo_result_dir="/tmp/unused")
    fitter.initial_fit("2009 JY22", real_data, reference_orbit=seed)

    sv = captured["state_vec"]
    assert sv is not None, "state_vec must be passed when reference_orbit is supplied"
    # Format: "<jd>,<x> <y> <z> <vx> <vy> <vz>,TDB"
    epoch_part, state_part, scale_part = sv.split(",")
    assert abs(float(epoch_part) - 2459580.5) < 1e-6
    components = state_part.split()
    assert len(components) == 6
    nums = [float(c) for c in components]
    np.testing.assert_allclose(
        nums,
        [1.5, 1.6, 0.05, -0.011, 0.0095, -0.0005],
        rtol=1e-9,
        atol=1e-12,
    )
    assert scale_part == "TDB"


def test_warm_start_recovers_catastrophic_cold_start(real_data):
    """Regression test for bead 9sg + 39j: when Find_Orb's Gauss/Vaisala IOD
    is geometrically degenerate (a 4-observation, ~3-minute G96 tracklet from
    the real_data fixture), cold-start ``initial_fit`` returns an empty
    ``FittedOrbits`` — no covariance is produced and the worker has no orbit
    to hand back. Supplying ``reference_orbit`` makes Find_Orb take the ``-v``
    branch in ``fetch_previous_solution()`` (find_orb/elem_out.cpp:3244) and
    skip the IOD altogether, converging via least-squares from the seed.

    This exercises the catastrophic-cold-start path: without a seed, short
    held-in arcs produce either no orbit or an orbit so far from truth that
    downstream evaluation chi² explodes; with a seed, the fitter converges.
    """
    obs_full = real_data
    short = obs_full[:4]  # 3-minute G96 tracklet — Gauss IOD is degenerate

    out_full = tempfile.TemporaryDirectory()
    seed_orbit, _ = FindOrbOrbitFitter(fo_result_dir=out_full.name).initial_fit(
        "2009 JY22", obs_full
    )
    assert len(seed_orbit) == 1, "full fixture must converge cold to provide a seed"
    seed = seed_orbit.to_orbits()

    out_cold = tempfile.TemporaryDirectory()
    cold_fitter = FindOrbOrbitFitter(fo_result_dir=out_cold.name)
    cold_fit, _ = cold_fitter.initial_fit("2009 JY22", short)
    # A failed fit returns a single-row placeholder with success=False rather
    # than FittedOrbits.empty(). The semantic check ("cold-start fails on the
    # short tracklet") is preserved.
    assert len(cold_fit) == 1 and cold_fit.success[0].as_py() is False, (
        "cold-start on the 3-minute tracklet must fail (degenerate Gauss IOD); "
        "if this assertion fires the fixture has shifted and the warm-start "
        "comparison below is no longer measuring the catastrophic-recovery path"
    )

    out_warm = tempfile.TemporaryDirectory()
    warm_fitter = FindOrbOrbitFitter(fo_result_dir=out_warm.name)
    warm_fit, _ = warm_fitter.initial_fit("2009 JY22", short, reference_orbit=seed)
    assert len(warm_fit) == 1, "warm-start with seed must produce a fitted orbit"
    assert warm_fit.success[0].as_py() is True
    assert warm_fit.reduced_chi2[0].is_valid

    # The warm-start orbit must be physically sensible: main-belt heliocentric
    # distance, not a degenerate IOD result like r2 -> 0 or r2 -> infinity.
    warm_xyz = np.array([
        warm_fit.coordinates.x[0].as_py(),
        warm_fit.coordinates.y[0].as_py(),
        warm_fit.coordinates.z[0].as_py(),
    ])
    helio_dist = float(np.linalg.norm(warm_xyz))
    assert 1.0 < helio_dist < 5.0, (
        f"warm-start heliocentric distance {helio_dist:.3f} AU outside main-belt "
        f"range — seed did not anchor the LSQ"
    )
