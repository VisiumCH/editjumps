"""`.dvc/config`: the settings a cloud job depends on, which a stray `dvc config` could undo."""

from pathlib import Path

import pytest

DVC_CONFIG = Path(__file__).parents[2] / ".dvc" / "config"


def test_analytics_is_disabled_so_a_finished_job_cannot_fail_on_its_way_out() -> None:
    """DVC's exit-time analytics daemon aborts on the SkyPilot VMs and takes the exit code with it.

    gRPC's fork handlers bail out, the forked child dies before writing its pid, and dvc/daemon.py
    evaluates `int("")`. The stage has already succeeded and pushed by then, so the job is reported
    as failed on work that was in fact done (#184). `dvc config` rewrites this file, so the setting
    is easy to lose by accident.
    """
    assert "analytics = false" in DVC_CONFIG.read_text(), (
        ".dvc/config no longer disables analytics; a cloud job will exit non-zero after its stage "
        "succeeds"
    )


def test_dvc_itself_reports_analytics_as_off() -> None:
    """The file says so; check DVC agrees, rather than trusting the spelling of the key."""
    analytics = pytest.importorskip("dvc.analytics")
    assert analytics.is_enabled() is False


def test_the_remote_is_still_configured_and_names_nobody() -> None:
    """The comment block sits inside [core]; a malformed file would silently drop the remote."""
    text = DVC_CONFIG.read_text()
    assert "remote = gcs" in text
    # The bucket is templated so the committed file names no one's storage; the real value lives in
    # the gitignored .dvc/config.local.
    assert "${DVC_BUCKET}" in text
