# adam-fo

A Python wrapper for Find_Orb orbit determination software, designed to work seamlessly with adam_core.

## Overview

`adam-fo` provides a Python interface to Bill Gray's Find_Orb orbit determination software, allowing you to:
- Determine orbits from ADES-formatted observations
- Get orbit state vectors and covariance matrices in adam_core format
- Identify rejected observations from the orbit determination process

## Installation

1. First, install the package using your preferred Python package manager:
```bash
pip install adam-fo
```

2. After installation, you need to build Find_Orb and its dependencies. Run:
```bash
build-fo
```

This will:
- Clone and build required repositories (lunar, sat_code, jpl_eph, find_orb, miscell)
- Download necessary ephemeris files
- Install everything in your XDG data directory (typically `~/.local/share/adam_fo`)

## Usage

Here's a basic example of using adam-fo with adam_core:

```python
from adam_fo import fo
from adam_core.observations import ADESObservations

# Load your ADES-formatted observations
ades_string = """
# Example ADES PSV format
# version=2017
permID|trkSub|mode|stn|obsTime|ra|dec|rmsRA|rmsDec|astCat
|2020_CD3|CCD|F51|2020-02-18T10:11:22.000|180.5876|10.4789|0.15|0.15|Gaia2
"""

# Run orbit determination
orbits, rejected_observations, error = fo(ades_string)

if error is None:
    print(f"Successfully determined {len(orbits)} orbit(s)")
    print(f"Number of rejected observations: {len(rejected_observations)}")
    
    # Access orbit parameters (in ecliptic coordinates)
    coords = orbits.coordinates
    print("State vector at epoch:")
    print(f"Position (AU): {coords.x[0]}, {coords.y[0]}, {coords.z[0]}")
    print(f"Velocity (AU/day): {coords.vx[0]}, {coords.vy[0]}, {coords.vz[0]}")
    
else:
    print(f"Orbit determination failed: {error}")
```

## Key Features

1. **ADES Format Support**: Works with ADES-formatted observations, the standard format for minor planet astrometry.

2. **adam_core Integration**: Returns results as adam_core objects:
   - Orbits in `adam_core.orbits.Orbits` format
   - Rejected observations in `adam_core.observations.ADESObservations` format

3. **Covariance Information**: Includes full 6x6 covariance matrices for orbit uncertainty analysis.

4. **Automatic Setup**: Handles Find_Orb installation and configuration automatically.

## Configuration

Find_Orb configuration is handled through environment variables and configuration files:

- Default configuration is stored in the installed package
- Custom configuration can be placed in `~/.local/share/adam_fo/find_orb/find_orb/environ.dat`
- Ephemeris files are automatically downloaded and configured

## Notes

- Orbit state vectors are returned in ecliptic coordinates centered on the Sun
- Times are in TT (Terrestrial Time) scale
- Positions are in AU and velocities in AU/day
- Covariance matrices correspond to the cartesian state vector

## Requirements

- Python 3.11+
- adam_core
- Build tools (gcc, make)
- Git

## License

This project is licensed under the GPL License to be compatible with Find_Orb itself.
