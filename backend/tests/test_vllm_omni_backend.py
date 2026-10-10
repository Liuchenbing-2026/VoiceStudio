"""Real loopback HTTP tests of the client contract; no model/device simulation claim."""

import base64
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
from services.tts_backend import TTSInputError, get_backend_class


@pytest.fixture
def service(monkeypatch):
    from services import tts_backend

    monkeypatch.setattr(tts_backend, "_ENGINE_INSTANCES", {})
    buffer = io.BytesIO()
    sf.write(buffer, np.full(2400, 0.1, dtype=np.float32), 24000, format="WAV")
    state = SimpleNamespace(
        wav=buffer.getvalue(), status=200, calls=[], models=[{"id": "OmniVoice"}]
    )

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            state.calls.append((self.path, None))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({"data": state.models}).encode())

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state.calls.append((self.path, payload))
            self.send_response(state.status)
            self.end_headers()
            self.wfile.write(state.wav)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state.url = f"http://127.0.0.1:{server.server_port}"
    monkeypatch.setenv("OMNIVOICE_VLLM_OMNI_URL", state.url)
    monkeypatch.setenv("OMNIVOICE_VLLM_OMNI_MODEL", "OmniVoice")
    yield state
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def backend():
    return get_backend_class("vllm-omni")()


def test_registered_engine():
    cls = get_backend_class("vllm-omni")
    assert cls.id == "vllm-omni"
    assert cls.configured_identity() == "k2-fsa/OmniVoice"


def test_client_cache_preserves_model_licence_gate(monkeypatch):
    from services import model_acceptance, tts_backend

    checked = []
    monkeypatch.setattr(tts_backend, "_ENGINE_INSTANCES", {})
    monkeypatch.setattr(
        model_acceptance,
        "ensure_engine_accepted",
        lambda engine, identity: checked.append((engine, identity)),
    )
    first = tts_backend.get_engine_instance_for("vllm-omni")
    assert tts_backend.get_engine_instance_for("vllm-omni") is first
    assert checked == [("vllm-omni", "k2-fsa/OmniVoice")] * 2


def test_unconfigured_engine_does_not_probe(monkeypatch):
    monkeypatch.delenv("OMNIVOICE_VLLM_OMNI_URL", raising=False)
    assert get_backend_class("vllm-omni").is_available()[0] is False


def test_readiness_requires_served_model(service):
    cls = get_backend_class("vllm-omni")
    assert cls.is_available() == (True, "ready")
    service.models = [{"id": "other"}]
    assert cls.is_available()[0] is False
    assert all(path == "/v1/models" for path, _ in service.calls)


def test_speech_parameters_and_waveform(service):
    engine = backend()
    audio = engine.generate(
        "Hello",
        language="English",
        instruct="female",
        num_step=32,
        guidance_scale=0.0,
        speed=1.2,
        seed=42,
    )
    assert audio.shape == (1, 2400) and engine.sample_rate == 24000
    assert np.isfinite(audio.numpy()).all()
    path, payload = service.calls[-1]
    assert path == "/v1/audio/speech"
    assert payload == {
        "model": "OmniVoice",
        "input": "Hello",
        "response_format": "wav",
        "stream": False,
        "language": "English",
        "instructions": "female",
        "speed": 1.2,
        "seed": 42,
        "extra_params": {"num_inference_steps": 32, "guidance_scale": 0.0},
    }


def test_reference_is_uploaded_not_a_client_path(service, tmp_path):
    reference = tmp_path / "reference.wav"
    sf.write(reference, np.full(16000, 0.2), 16000)
    backend().generate(
        "Hello", ref_audio=str(reference), ref_text="Reference", language="Auto"
    )
    payload = service.calls[-1][1]
    prefix, encoded = payload["ref_audio"].split(",", 1)
    assert prefix == "data:audio/wav;base64" and str(reference) not in json.dumps(
        payload
    )
    samples, rate = sf.read(io.BytesIO(base64.b64decode(encoded)))
    assert rate == 16000 and samples.size == 16000
    original, _ = sf.read(reference, dtype="float32")
    np.testing.assert_array_equal(samples, original)
    assert payload["ref_text"] == "Reference" and "language" not in payload


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_step": 0},
        {"num_step": -1},
        {"num_step": 1.5},
        {"num_step": True},
        {"guidance_scale": float("nan")},
        {"guidance_scale": float("inf")},
        {"guidance_scale": True},
        {"duration": 1.0},
        {"denoise": False},
        {"t_shift": 0.8},
        {"guidance_scale": 10**1000},
    ],
)
def test_invalid_or_unsupported_controls_fail_before_http(service, kwargs):
    with pytest.raises(TTSInputError):
        backend().generate("Hello", **kwargs)
    assert not service.calls


@pytest.mark.parametrize(
    "status,error", [(400, TTSInputError), (422, TTSInputError), (500, RuntimeError)]
)
def test_http_errors_are_not_audio(service, status, error):
    service.status = status
    with pytest.raises(error):
        backend().generate("Hello")


def test_invalid_response_is_not_success(service):
    service.wav = b'{"error": "not audio"}'
    with pytest.raises(RuntimeError, match="invalid audio"):
        backend().generate("Hello")


def test_bad_reference_fails_before_http(service, tmp_path):
    with pytest.raises(TTSInputError, match="Reference audio"):
        backend().generate("Hello", ref_audio=str(tmp_path / "missing.wav"))
    assert not service.calls


def test_batch_preserves_per_item_conditioning(service):
    audio = backend().generate_batch(
        ["First", "Second"], language=["English", "Chinese"]
    )
    assert len(audio) == 2
    assert [p["language"] for _, p in service.calls] == ["English", "Chinese"]


def test_remote_device_is_not_reported_as_local_cpu():
    from core.device_caps import HostCaps

    profile = get_backend_class("vllm-omni").runtime_compute_profile(
        HostCaps(family="cpu", available_families=("cpu",))
    )
    assert profile["effective_device"] == "external"
    assert profile["runtime_device_verified"] is None


def test_voicestudio_speech_route_calls_the_service(service):
    from fastapi.testclient import TestClient
    from main import app

    client = TestClient(app, client=("127.0.0.1", 50000))
    response = client.post(
        "/v1/audio/speech",
        json={
            "model": "vllm-omni",
            "input": "Hello",
            "response_format": "wav",
            "language": "English",
            "num_step": 16,
        },
    )
    assert response.status_code == 200, response.text
    audio, rate = sf.read(io.BytesIO(response.content))
    assert rate == 24000 and audio.size > 0 and np.isfinite(audio).all()
    assert service.calls[-1][0] == "/v1/audio/speech"
    assert service.calls[-1][1]["input"] == "Hello"


def test_voicestudio_generate_route_calls_the_service(service):
    from core.db import init_db
    from fastapi.testclient import TestClient
    from main import app

    init_db()
    client = TestClient(app, client=("127.0.0.1", 50000))
    response = client.post(
        "/generate",
        data={
            "engine": "vllm-omni",
            "text": "Hello",
            "language": "English",
            "num_step": "16",
            "seed": "42",
        },
    )
    assert response.status_code == 200, response.text
    assert service.calls[-1][0] == "/v1/audio/speech"
    assert service.calls[-1][1]["seed"] == 42


@pytest.mark.parametrize(
    "option,value", [("t_shift", "0.8"), ("duration", "1"), ("denoise", "false")]
)
def test_generate_route_refuses_unsupported_controls_before_http(
    service, option, value
):
    from fastapi.testclient import TestClient
    from main import app

    client = TestClient(app, client=("127.0.0.1", 50000))
    response = client.post(
        "/generate",
        data={"engine": "vllm-omni", "text": "Hello", option: value},
    )
    assert response.status_code == 400, response.text
    assert not service.calls
