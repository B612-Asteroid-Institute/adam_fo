import pytest
import os
import tempfile

from adam_fo import fo


@pytest.fixture
def sample_ades_string():
    """Return a sample ADES string from the adam-thor-candidates.psv file."""
    import pathlib

    test_dir = pathlib.Path(__file__).parent
    with open(test_dir / "data/adam-thor-candidates-small.psv", "r") as f:
        return f.read()


def test_fo(sample_ades_string):
    orbits, rejected_obs, errors = fo(sample_ades_string)
    assert len(orbits) == 1
    assert len(rejected_obs) == 1
    assert errors is None

def test_fo_out_dir(sample_ades_string):
    with tempfile.TemporaryDirectory() as out_dir:
        orbits, rejected_obs, errors = fo(sample_ades_string, out_dir=out_dir)

        assert os.path.exists(out_dir)
        assert not os.path.exists(os.path.join(out_dir, "bc405.dat"))
        assert not os.path.exists(os.path.join(out_dir, "linux_p1550p2650.440t"))
        assert os.path.exists(os.path.join(out_dir, "total.json"))
        assert os.path.exists(os.path.join(out_dir, "covar.json"))
        assert os.path.exists(os.path.join(out_dir, "environ.dat"))
