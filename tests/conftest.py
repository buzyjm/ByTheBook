"""Keep the tests independent of the machine they run on.

The tools in this package write rig-specific files (rest pose, registration, camera modes)
under rig_state/. During a test run those paths point into a temporary directory, so
a rest pose recorded on the rig cannot change where the simulated arm parks.
"""
import pytest

from bythebook import rig


@pytest.fixture(autouse=True)
def isolated_rig_state(tmp_path, monkeypatch):
    state = tmp_path / "rig_state"
    monkeypatch.setattr(rig, "STATE_DIR", state)
    monkeypatch.setattr(rig, "HIRES_FILE", state / "hires.json")
    monkeypatch.setattr(rig, "REGISTRATION_FILE", state / "base_registration.json")
    monkeypatch.setattr(rig, "ARM_CALIB_DIR", state / "arm_calibration")
    monkeypatch.setattr(rig, "REST_POSE_FILE", state / "rest_pose.json")
