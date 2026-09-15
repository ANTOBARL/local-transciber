"""HTTP API tests with a fake engine (no model, no GPU)."""

import time

import pytest
from fastapi.testclient import TestClient

from scriba.api import create_app
from scriba.config import load_settings
from scriba.core.pipeline import TranscriptionService
from scriba.models import SpeakerTurn

from test_pipeline import FakeEngine, wav_file  # noqa: F401  (fixture reuse)


@pytest.fixture
def client(tmp_path):
    settings = load_settings(overrides={"app": {"output_root": str(tmp_path / "out")}})
    service = TranscriptionService()
    service.engine = FakeEngine()

    class FakeDiarizer:
        def diarize(self, *a, **k):
            return [SpeakerTurn(start=0, end=1.2, speaker="SPEAKER_00"),
                    SpeakerTurn(start=1.2, end=2.0, speaker="SPEAKER_01")]

    service.diarizer = FakeDiarizer()
    return TestClient(create_app(settings, service=service))


def wait_done(client, task_id, timeout=20):
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/api/tasks/{task_id}").json()
        if body["state"] in ("done", "error"):
            return body
        time.sleep(0.05)
    raise AssertionError("task did not finish")


def test_config_and_i18n(client):
    cfg = client.get("/api/config").json()
    assert cfg["defaults"]["language"] == "Italian" and "Italian" in cfg["languages"]
    texts = client.get("/api/i18n").json()["texts"]
    assert texts["it"]["transcribe"] == "TRASCRIVI" and texts["en"]["transcribe"] == "TRANSCRIBE"


def test_upload_rejects_unknown_type(client):
    r = client.post("/api/uploads", files={"file": ("notes.exe", b"xx")})
    assert r.status_code == 415


def test_full_flow_with_speaker_rename(client, wav_file):  # noqa: F811
    with wav_file.open("rb") as fh:
        up = client.post("/api/uploads", files={"file": ("sample.wav", fh, "audio/wav")}).json()
    assert abs(up["info"]["duration"] - 2.0) < 0.05

    defaults = client.get("/api/config").json()["defaults"]
    form = {**defaults, "diarize": True, "formats": ["txt", "srt"]}
    task_id = client.post("/api/tasks", json={"upload_id": up["upload_id"], "form": form}).json()["task_id"]
    body = wait_done(client, task_id)

    assert body["state"] == "done", body
    assert body["speakers"] == ["SPEAKER_00", "SPEAKER_01"]
    names = {f["name"] for f in body["files"]}
    assert names == {"transcript.json", "transcript.txt", "transcript.srt"}
    assert client.get(f"/api/tasks/{task_id}/files/transcript.srt").status_code == 200
    assert client.get(f"/api/tasks/{task_id}/files/job.json").status_code == 404  # only exports are served

    renamed = client.post(f"/api/tasks/{task_id}/speakers", json={"names": {"SPEAKER_00": "Luca"}}).json()
    assert "Luca" in renamed["preview"]


def test_invalid_backend_kwargs_is_400(client, wav_file):  # noqa: F811
    with wav_file.open("rb") as fh:
        up = client.post("/api/uploads", files={"file": ("sample.wav", fh)}).json()
    defaults = client.get("/api/config").json()["defaults"]
    r = client.post("/api/tasks", json={"upload_id": up["upload_id"], "form": {**defaults, "backend_kwargs": "{bad"}})
    assert r.status_code == 400
