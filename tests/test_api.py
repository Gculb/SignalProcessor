import base64
import json
import urllib.request

import numpy as np


def _encode_samples(samples: np.ndarray) -> str:
    return base64.b64encode(samples.astype(np.float32).tobytes()).decode("ascii")


def _post_json(url: str, payload: dict) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def test_health_endpoint() -> None:
    payload = urllib.request.urlopen("http://127.0.0.1:8000/api/health", timeout=10).read().decode("utf-8")
    assert "ok" in payload
    assert "digital-signal-processor" in payload


def test_process_endpoint() -> None:
    signal = np.sin(2 * np.pi * 440 * np.linspace(0, 0.1, 4800, endpoint=False))
    payload = {
        "sampleRate": 48000,
        "signalType": "generic",
        "samples": _encode_samples(signal),
        "settings": {"highpass": 80, "lowpass": 3400, "detectNoise": True},
    }
    result = _post_json("http://127.0.0.1:8000/api/process", payload)

    assert result["sampleRate"] == 48000
    assert result["sampleCount"] == 4800
    assert isinstance(result["processed"], dict)
    assert isinstance(result.get("noisePeakDetails", []), list)
    for detail in result.get("noisePeakDetails", []):
        assert "frequency" in detail
        assert "ratio" in detail
        assert "persistence" in detail
        assert "bandwidthHz" in detail
    assert isinstance(result.get("spectrogram", []), list)
    assert all(isinstance(row, list) for row in result.get("spectrogram", []))
    assert "classification" in result
    assert result["classification"]["label"] in {"noise", "speech_like", "ecg_like"}


def test_classify_endpoint() -> None:
    signal = np.sin(2 * np.pi * 5 * np.linspace(0, 2.0, 9600, endpoint=False))
    payload = {
        "sampleRate": 48000,
        "signalType": "ecg",
        "samples": _encode_samples(signal),
    }
    result = _post_json("http://127.0.0.1:8000/api/classify", payload)

    assert "classification" in result
    assert result["classification"]["label"] in {"noise", "speech_like", "ecg_like"}
