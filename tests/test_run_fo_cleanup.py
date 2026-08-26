import importlib
import subprocess
from pathlib import Path

import pytest

run_fo_module = importlib.import_module("adam_fo.run_fo")


def test_fo_cleans_working_directory_after_process_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    working_directory = tmp_path / "fo_work"
    working_directory.mkdir()
    monkeypatch.setattr(run_fo_module, "_de440t_exists", lambda: None)
    monkeypatch.setattr(
        run_fo_module,
        "_create_fo_tmp_directory",
        lambda _temp_dir: str(working_directory),
    )
    monkeypatch.setattr(
        run_fo_module.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=[],
            returncode=1,
            stdout="",
            stderr="failed",
        ),
    )

    evidence_directory = tmp_path / "evidence"
    _orbits, _rejected, error = run_fo_module.fo(
        "ades", out_dir=str(evidence_directory)
    )

    assert error == "Find_Orb failed"
    assert not working_directory.exists()
    assert (evidence_directory / "observations.ades").is_file()


def test_fo_preserves_working_directory_when_cleanup_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    working_directory = tmp_path / "fo_work"
    working_directory.mkdir()
    monkeypatch.setattr(run_fo_module, "_de440t_exists", lambda: None)
    monkeypatch.setattr(
        run_fo_module,
        "_create_fo_tmp_directory",
        lambda _temp_dir: str(working_directory),
    )
    monkeypatch.setattr(
        run_fo_module.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=[],
            returncode=1,
            stdout="",
            stderr="failed",
        ),
    )

    run_fo_module.fo("ades", clean_up=False)

    assert working_directory.exists()


def test_fo_returns_validation_error_cleans_working_directory_and_keeps_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    working_directory = tmp_path / "fo_work"
    working_directory.mkdir()
    (working_directory / "covar.json").write_text("{}", encoding="utf-8")
    (working_directory / "total.json").write_text("{}", encoding="utf-8")
    out_directory = tmp_path / "evidence"
    monkeypatch.setattr(run_fo_module, "_de440t_exists", lambda: None)
    monkeypatch.setattr(
        run_fo_module,
        "_create_fo_tmp_directory",
        lambda _temp_dir: str(working_directory),
    )
    monkeypatch.setattr(
        run_fo_module.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        ),
    )
    monkeypatch.setattr(
        run_fo_module,
        "fo_to_adam_orbit_cov",
        lambda _path: (_ for _ in ()).throw(
            run_fo_module.FindOrbFormatError("ambiguous bundle")
        ),
    )

    orbits, rejected, error = run_fo_module.fo("ades", out_dir=str(out_directory))

    assert len(orbits) == 0
    assert len(rejected) == 0
    assert error == "ambiguous bundle"
    assert not working_directory.exists()
    assert (out_directory / "covar.json").is_file()
    assert (out_directory / "total.json").is_file()


def test_temporary_directory_is_removed_when_population_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fail_population(_working_directory: str) -> str:
        raise RuntimeError("missing runtime file")

    monkeypatch.setattr(run_fo_module, "_populate_fo_directory", fail_population)

    with pytest.raises(RuntimeError, match="missing runtime file"):
        run_fo_module._create_fo_tmp_directory(str(tmp_path))

    assert list(tmp_path.iterdir()) == []


def test_temporary_directory_is_removed_when_chmod_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fail_chmod(_path: str, _mode: int) -> None:
        raise OSError("chmod failed")

    monkeypatch.setattr(run_fo_module.os, "chmod", fail_chmod)

    with pytest.raises(OSError, match="chmod failed"):
        run_fo_module._create_fo_tmp_directory(str(tmp_path))

    assert list(tmp_path.iterdir()) == []
