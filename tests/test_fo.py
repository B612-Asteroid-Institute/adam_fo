import pytest

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
    assert len(errors) == 0
