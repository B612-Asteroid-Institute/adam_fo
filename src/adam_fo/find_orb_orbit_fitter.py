import json
import logging
import uuid
from typing import Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from mpc_obscodes import mpc_obscodes

from adam_core.coordinates.cartesian import CartesianCoordinates
from adam_core.coordinates.origin import Origin, OriginCodes
from adam_core.coordinates.transform import transform_coordinates
from adam_core.dynamics.propagation import propagate_2body
from adam_core.observations.ades import (
    ADES_to_string,
    ADESObservations,
    ObsContext,
    ObservatoryObsContext,
    SubmitterObsContext,
    TelescopeObsContext,
)
from adam_core.orbit_determination.evaluate import (
    FittedOrbitMembers,
    FittedOrbits,
    OrbitDeterminationObservations,
    evaluate_orbits,
)
from adam_core.orbit_determination.orbit_fitter import OrbitFitter
from adam_core.orbits import Orbits
from adam_core.propagator.propagator import Propagator

try:
    from adam_fo import fo
except ImportError:
    raise ImportError("Please install adam_fo to use this feature.")

logger = logging.getLogger(__name__)


class _TwoBodyPropagator(Propagator):
    """Default propagator used to evaluate hold-in chi2 when none is supplied."""

    def _propagate_orbits(self, orbits, times, max_iter=1000, tol=1e-14, **kwargs):
        return propagate_2body(orbits, times, max_iter=max_iter, tol=tol)

    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)


class FindOrbOrbitFitter(OrbitFitter):
    """Implementation of OrbitFitter using Find_Orb."""

    def __init__(
        self,
        *args: object,  # Generic type for arbitrary positional arguments
        fo_result_dir: str,
        clean_up_fo_dir: bool = True,
        propagator: Optional[Propagator] = None,
        **kwargs: object,  # Generic type for arbitrary keyword arguments
    ) -> None:
        """
        Parameters:
        -----------
        fo_result_dir: str
           directory to use to store FindOrb's inputs and outputs
        clean_up_fo_dir: bool, default True
           whether to clean up the contents of fo_result_dir after a run
        propagator: Propagator, optional
           Propagator used to evaluate the converged orbit so the returned
           ``FittedOrbits`` table has reduced_chi2 populated. Defaults to a
           2-body propagator. Callers should pass the same propagator used
           downstream so chi2 is consistent.
        """
        super().__init__(*args, **kwargs)
        self.fo_result_dir = fo_result_dir
        self.clean_up_fo_dir = clean_up_fo_dir
        self.propagator = propagator if propagator is not None else _TwoBodyPropagator()
        self._load_obscodes()

    def _load_obscodes(self):
        with open(mpc_obscodes) as mpc_file:
            self.obscodes = json.load(mpc_file)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("obscodes")
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._load_obscodes()

    def _short_observation_id(self, obs_id: str) -> str:
        """Convert obs_id from MPC observation to string matching `packed` field of total.json."""
        return obs_id[0:4] + obs_id[-4:]

    def _observations_to_ades(
        self, observations: OrbitDeterminationObservations
    ) -> Tuple[str, ADESObservations]:
        """
        Convert OrbitDeterminationObservations to ADES format string.

        Parameters
        ----------
        observations : OrbitDeterminationObservations
            Observations to convert

        Returns
        -------
        Tuple[str, ADESObservations]
            Tuple containing:
            - ADES format string
            - ADESObservations object
        """
        # "packed" field in total.json ends with this, but that field is currently not exposed
        # so this is more for future extensions
        orbit_ids = [self._short_observation_id(id.as_py()) for id in observations.id]

        # Convert uncertainties in RA and Dec to arcseconds
        # and adjust the RA uncertainty to account for the cosine of the declination
        # The convesion matches MPC values if mpc_to_od_observations was called with prevent_nans=False
        sigma_ra_cos_dec = (
            np.cos(
                np.radians(observations.coordinates.lat.to_numpy(zero_copy_only=False))
            )
            * observations.coordinates.covariance.sigmas[:, 1]
        )
        sigma_ra_cos_dec_arcseconds = pa.array(
            sigma_ra_cos_dec * 3600, type=pa.float64()
        )
        sigma_dec_arcseconds = pa.array(
            observations.coordinates.covariance.sigmas[:, 2] * 3600, type=pa.float64()
        )

        # Replace nans with nulls using pyarrow
        sigma_ra_cos_dec_arcseconds = pc.if_else(
            pc.is_nan(sigma_ra_cos_dec_arcseconds), None, sigma_ra_cos_dec_arcseconds
        )
        sigma_dec_arcseconds = pc.if_else(
            pc.is_nan(sigma_dec_arcseconds), None, sigma_dec_arcseconds
        )

        # Serialize observations to an ADES table
        ades_observations = ADESObservations.from_kwargs(
            trkSub=orbit_ids,
            obsTime=observations.coordinates.time,
            ra=observations.coordinates.lon,
            dec=observations.coordinates.lat,
            rmsRACosDec=sigma_ra_cos_dec_arcseconds,
            rmsDec=sigma_dec_arcseconds,
            mag=observations.photometry.mag,
            rmsMag=observations.photometry.rmsmag,
            band=observations.photometry.band,
            stn=observations.observers.code,
            mode=pa.repeat("NA", len(observations)),
            astCat=pa.repeat("NA", len(observations)),
        )

        codes = [v.as_py() for v in pc.unique(observations.observers.code)]

        telescope = TelescopeObsContext(
            name="Thing",
            design="Reflector",
            detector="CCD",
            aperture=40.0,
        )

        obs_contexts = {
            code: ObsContext(
                observatory=ObservatoryObsContext(
                    mpcCode=code, name=self.obscodes[code]["Name"]
                ),
                submitter=SubmitterObsContext(
                    name="J. Doe",
                    institution="B612 Asteroid Institute",
                ),
                observers=["J. Doe"],
                measurers=["J. Doe"],
                telescope=telescope,
            )
            for code in codes
        }

        ades_string = ADES_to_string(ades_observations, obs_contexts)
        return ades_string, ades_observations

    def _rejected_observations_to_fitted_members(
        self,
        observations: OrbitDeterminationObservations,
        rejected: ADESObservations,
        orbit_id: str,
    ) -> FittedOrbitMembers:
        """
        Convert a set of observations rejected by Find_Orb into fitted members.

        Parameters
        ----------
        observations: OrbitDeterminationObservations (N)
           All observations used as input to the fitter
        rejected: ADESObservations (M)
           Observations rejected by the fitter. 0<=M<=N
        orbit_id: str
           orbit_id to be used in the output fitted members

        Returns
        -------
        Set of fitted members corresponding to all input observations. The solution and outlier
        flags are set based on whether the corresponding observation is found in the rejected list.
        Residuals are NOT computed, because doing so would require covariance matrix in observations
        to have no NaNs.
        """

        assert len(observations) >= len(rejected)
        obs_ids_all = observations.id
        # Compare epochs in a common time scale. The rejected list is
        # reconstructed from Find_Orb's output in UTC, while the input table
        # may carry any scale (ADES serialization rescales to UTC on write, so
        # the fit itself is scale-agnostic). Rescaling both sides here keeps
        # the 1-second join below correct by construction; rescale() is a
        # no-op when the scale already matches.
        obs_mjd_utc = observations.coordinates.time.rescale("utc").mjd()
        rejected_ids = []
        for ades in rejected:
            # Observation id is not passed fully in the ADES format, so we need to match on fields.
            # Look for an observation from the same station within 1 second. There should be exactly 1
            rejected_mjd_utc = ades.obsTime.rescale("utc").mjd()[0]
            same_station = pc.equal(observations.observers.code, ades.stn[0])
            same_time = pc.less(
                pc.abs(pc.subtract(obs_mjd_utc, rejected_mjd_utc)),
                1.0 / 86400,
            )
            original = observations.apply_mask(pc.and_(same_station, same_time))
            if len(original) == 0:
                # ADES round-trip can lose sub-second precision and
                # station-code casing/whitespace can differ, so the strict
                # 1-second match may return zero rows. Log diagnostic
                # candidates and skip just this rejected entry rather than
                # aborting the whole fit (which would discard an otherwise
                # usable result over a single unmatchable rejection).
                wider_time = pc.less(
                    pc.abs(pc.subtract(obs_mjd_utc, rejected_mjd_utc)),
                    5.0 / 86400,
                )
                candidates = observations.apply_mask(wider_time)
                cand_stns = candidates.observers.code.to_pylist()
                cand_mjds = pc.filter(obs_mjd_utc, wider_time).to_pylist()
                logger.warning(
                    "Could not match FindOrb rejected observation "
                    f"(stn={ades.stn[0].as_py()!r}, mjd_utc={rejected_mjd_utc.as_py()}) "
                    f"to any input observation within 1s; candidates within 5s: "
                    f"stns={cand_stns}, mjds_utc={cand_mjds}. Skipping this rejected entry."
                )
                continue
            assert (
                len(original) == 1
            ), f"Expected 1 input observation for {ades.stn[0]} at {rejected_mjd_utc} MJD (UTC), got {original.observers.code} {pc.filter(obs_mjd_utc, pc.and_(same_station, same_time))}"
            rejected_ids.append(original.id[0].as_py())

        # To calculate residuals we need covariance matrix without NaNs, but
        # normal fitting prefers one with NaNs, so we can't compute residuals
        # here. We'll leave them null for now. Call evaluate_orbits later
        # to get residuals.

        # Invariant: every id we matched back to an input observation flags
        # exactly one row as an outlier. We compare against len(rejected_ids),
        # not len(rejected): unmatchable rejected entries are skipped above, so
        # rejected_ids may be shorter than the rejected list FindOrb returned.
        outlier = np.isin(obs_ids_all, rejected_ids)
        assert np.sum(outlier) == len(
            rejected_ids
        ), "Something failed in extracting rejected observations"

        od_orbit_members = FittedOrbitMembers.from_kwargs(
            orbit_id=np.full(len(obs_ids_all), orbit_id, dtype="object"),
            obs_id=obs_ids_all,
            # not setting residuals here
            solution=np.isin(obs_ids_all, rejected_ids, invert=True),
            outlier=outlier,
        )

        return od_orbit_members

    def _build_state_vec_arg(self, reference_orbit: Orbits) -> str:
        """Format the first row of ``reference_orbit`` for Find_Orb's ``-v`` flag.

        Find_Orb's ``extract_state_vect_from_text()`` (find_orb/elem_out.cpp:3024)
        expects ``"<epoch>,<x> <y> <z> <vx> <vy> <vz>[,<modifier>...]"`` with
        units AU and AU/day in heliocentric ecliptic J2000, default time scale
        TT. We coerce the seed orbit to that frame/origin and append a ``,TDB``
        or ``,UTC`` modifier so Find_Orb converts the epoch internally.
        """
        assert (
            len(reference_orbit) == 1
        ), f"Expected a single seed orbit, got {len(reference_orbit)}"
        coords = transform_coordinates(
            reference_orbit.coordinates[:1],
            representation_out=CartesianCoordinates,
            frame_out="ecliptic",
            origin_out=OriginCodes.SUN,
        )
        scale = coords.time.scale
        jd = float(coords.time.jd().to_numpy(zero_copy_only=False)[0])
        x, y, z, vx, vy, vz = coords.values[0]

        if scale == "tdb":
            modifier = ",TDB"
        elif scale == "utc":
            modifier = ",UTC"
        else:
            modifier = ""

        return (
            f"{jd:.10f},"
            f"{x:.12g} {y:.12g} {z:.12g} {vx:.12g} {vy:.12g} {vz:.12g}"
            f"{modifier}"
        )

    def _build_failure_placeholder(
        self,
        object_id: str | pa.LargeStringScalar,
        observations: OrbitDeterminationObservations,
    ) -> Tuple[FittedOrbits, FittedOrbitMembers]:
        """Build a single-row placeholder result for a FindOrb failure.

        This follows the failure contract of adam_core's reference fitter,
        ``orbit_determination.differential_correction.fit_least_squares``,
        which always returns a single-row ``FittedOrbits`` with ``success``
        set (True or False) — never an empty table — even when the fit does
        not converge. Returning ``FittedOrbits.empty()`` here would instead
        make a FindOrb failure indistinguishable from "no object processed":
        any consumer that guards on ``len(...) == 0`` silently drops the input
        rather than recording the failure.

        The placeholder carries ``success=False``, a NaN orbit state anchored
        at the first observation's time, and one member row per input
        observation (with ``solution``/``outlier`` left null, since no fit was
        produced), so callers detect the failure via ``success`` rather than by
        counting rows.
        """
        # Mint a fresh orbit_id: Find_Orb produced no orbit to take an id
        # from, and uuid4().hex is FittedOrbits.orbit_id's own column default.
        orbit_id = uuid.uuid4().hex

        # Anchor the placeholder coordinate at the first observation's time —
        # the orbit state is NaN since no fit converged.
        first_time = observations.coordinates.time[0:1]
        placeholder_coords = CartesianCoordinates.from_kwargs(
            x=[np.nan],
            y=[np.nan],
            z=[np.nan],
            vx=[np.nan],
            vy=[np.nan],
            vz=[np.nan],
            time=first_time,
            frame="ecliptic",
            origin=Origin.from_kwargs(code=["SUN"]),
        )

        # num_obs and arc_length follow evaluate_orbits' semantics: both are
        # computed over the observations *included in the fit*, and a failed
        # fit included none. The attempted set remains recoverable through the
        # member rows' obs_ids.
        placeholder_orbit = FittedOrbits.from_kwargs(
            orbit_id=[orbit_id],
            object_id=[object_id],
            coordinates=placeholder_coords,
            arc_length=[0.0],
            num_obs=[0],
            chi2=[np.nan],
            reduced_chi2=[np.nan],
            success=[False],
        )

        n = len(observations)
        placeholder_members = FittedOrbitMembers.from_kwargs(
            orbit_id=np.full(n, orbit_id, dtype="object"),
            obs_id=observations.id,
            solution=pa.array([None] * n, type=pa.bool_()),
            outlier=pa.array([None] * n, type=pa.bool_()),
        )

        return placeholder_orbit, placeholder_members

    def initial_fit(
        self,
        object_id: str | pa.LargeStringScalar,
        observations: OrbitDeterminationObservations,
        reference_orbit: Optional[Orbits] = None,
    ) -> Tuple[FittedOrbits, FittedOrbitMembers]:
        """Fit an initial orbit for a single object via Find_Orb.

        Returns a real ``FittedOrbits`` table with ``reduced_chi2`` populated by
        evaluating the converged orbit against the input observations and
        ``success`` set to True. Outliers in ``FittedOrbitMembers`` reflect the
        observations that Find_Orb rejected; residuals on members are not set.

        When ``reference_orbit`` is supplied, Find_Orb is warm-started from that
        cartesian state via the ``-v`` flag, bypassing its Gauss/Vaisala cold
        start. Pass ``None`` (the default) to preserve the legacy cold-start
        behavior.
        """
        if observations is None or len(observations) == 0:
            logger.error(f"No observation provided for object {object_id}")
            return FittedOrbits.empty(), FittedOrbitMembers.empty()
        logger.info(
            f"Initial propagation using Find_Orb for {object_id} using {len(observations)} observations"
        )
        ades_string, _ = self._observations_to_ades(observations)

        state_vec = None
        if reference_orbit is not None and len(reference_orbit) > 0:
            state_vec = self._build_state_vec_arg(reference_orbit)
            logger.info(f"Warm-starting Find_Orb for {object_id} with -v {state_vec}")

        orbit, rejected, error = fo(
            ades_string,
            out_dir=self.fo_result_dir,
            clean_up=self.clean_up_fo_dir,
            state_vec=state_vec,
        )
        if error is not None:
            # Return a success=False placeholder rather than an empty table so
            # callers can distinguish a failed fit from "nothing to process".
            # Matches adam_core's fit_least_squares, which likewise returns a
            # success-flagged row (never empty) on non-convergence. See
            # _build_failure_placeholder for the full rationale.
            logger.warning(
                f"FindOrb failed for object {object_id} with error {error}; "
                f"returning fit_success=False placeholder"
            )
            return self._build_failure_placeholder(object_id, observations)

        N = len(orbit)
        if isinstance(object_id, str):
            object_id = pa.scalar(object_id, type=pa.large_string())
        orbit = orbit.set_column("object_id", [object_id] * N)
        if len(rejected) > 0:
            logger.info(
                f"Find_Orb rejected {len(rejected)} input observations out of {len(observations)}"
            )
        orbit_id = orbit.orbit_id[0].as_py()
        fitted_members = self._rejected_observations_to_fitted_members(
            observations, rejected, orbit_id
        )

        # Evaluate the converged orbit so the returned FittedOrbits has
        # reduced_chi2 populated. Mirrors the pattern in
        # adam_core.orbit_determination.differential_correction.fit_least_squares.
        rejected_ids = (
            fitted_members.obs_id.filter(fitted_members.outlier).to_pylist()
            if len(fitted_members) > 0
            else None
        )
        fitted_orbit, _ = evaluate_orbits(
            orbit,
            observations,
            self.propagator,
            parameters=6,
            ignore=rejected_ids if rejected_ids else None,
        )
        fitted_orbit = fitted_orbit.set_column("success", [True])
        return fitted_orbit, fitted_members
