"""Opt-in OmniVoice inference through a locally hosted vLLM-Omni service."""

from __future__ import annotations

import base64
import io
import json
import math
import os
from pathlib import Path

import torch
from services.outbound_http import EndpointHTTPError, open_trusted_endpoint
from services.tts_backend import TTSBackend, TTSInputError


class VLLMOmniBackend(TTSBackend):
    """Keep VoiceStudio's application pipeline, run OmniVoice in vLLM-Omni."""

    id = "vllm-omni"
    display_name = "OmniVoice (vLLM-Omni service)"
    instruct_vocabulary = "tags"
    supports_voice_design = True
    supports_generation_seed = True
    runs_out_of_process = True
    gpu_compat = ("cpu",)  # The client never loads weights on its own accelerator.
    _DEFAULT_SAMPLE_RATE = 24000

    def __init__(
        self,
        *,
        url: str | None = None,
        model: str | None = None,
        timeout: float = 300.0,
    ):
        self._url = (
            url if url is not None else os.environ.get("OMNIVOICE_VLLM_OMNI_URL", "")
        )
        self._served_model = model if model is not None else self._model_name()
        self._timeout = timeout

    @staticmethod
    def _model_name() -> str:
        return os.environ.get("OMNIVOICE_VLLM_OMNI_MODEL", "OmniVoice")

    @classmethod
    def configured_identity(cls) -> str:
        # Use the existing OmniVoice model-licence acceptance, including revocation.
        return "k2-fsa/OmniVoice"

    def model_identity(self) -> str:
        return self.configured_identity()

    def built_identity(self) -> str:
        return self.configured_identity()

    @property
    def sample_rate(self) -> int:
        return self._DEFAULT_SAMPLE_RATE

    @property
    def supported_languages(self) -> list[str]:
        return ["multi"]

    @classmethod
    def runtime_compute_profile(cls, caps) -> dict:
        profile = super().runtime_compute_profile(caps)
        # Client-side GPU counters cannot identify the service's device.
        profile.update(
            effective_device="external",
            routing_status="n/a",
            routing_reason=None,
            runtime_backend="vllm-omni",
        )
        return profile

    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        url = os.environ.get("OMNIVOICE_VLLM_OMNI_URL", "")
        if not url:
            return (
                False,
                "Configure OMNIVOICE_VLLM_OMNI_URL for a local vLLM-Omni service.",
            )
        try:
            with open_trusted_endpoint(
                url, method="GET", path="v1/models", timeout=2
            ) as response:
                data = json.loads(response.read())
            if not any(
                row.get("id") == cls._model_name() for row in data.get("data", [])
            ):
                return (
                    False,
                    "The configured OmniVoice model is not served by vLLM-Omni.",
                )
        except (OSError, ValueError, TypeError, AttributeError):
            return (
                False,
                "vLLM-Omni is unavailable; check its model, address and trusted-network settings.",
            )
        return True, "ready"

    @staticmethod
    def validate_generation_options(**kw) -> None:
        # The current diffusion pipeline has no duration/advanced overrides
        # and always uses denoise conditioning. Refuse rather than ignore.
        if kw.get("duration") is not None:
            raise TTSInputError(
                "vLLM-Omni OmniVoice does not support a fixed duration."
            )
        if kw.get("denoise", True) is not True:
            raise TTSInputError("vLLM-Omni OmniVoice currently requires denoise=True.")
        for name in (
            "t_shift",
            "layer_penalty_factor",
            "position_temperature",
            "class_temperature",
        ):
            if kw.get(name) is not None:
                raise TTSInputError(
                    f"vLLM-Omni OmniVoice does not support overriding {name}."
                )

    def generate(self, text: str, **kw) -> torch.Tensor:
        if not isinstance(text, str) or not text.strip():
            raise TTSInputError("Input text cannot be empty.")
        if not self._url:
            raise TTSInputError(
                "Configure OMNIVOICE_VLLM_OMNI_URL before selecting vllm-omni."
            )
        self.validate_generation_options(**kw)
        steps = kw.get("num_step", 16)
        if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
            raise TTSInputError("num_step must be a positive integer.")
        guidance = kw.get("guidance_scale", 2.0)
        try:
            valid_guidance = (
                not isinstance(guidance, bool)
                and isinstance(guidance, (int, float))
                and math.isfinite(guidance)
            )
        except OverflowError:
            valid_guidance = False
        if not valid_guidance:
            raise TTSInputError("guidance_scale must be a finite number.")

        payload = {
            "model": self._served_model,
            "input": text,
            "response_format": "wav",
            "speed": kw.get("speed", 1.0),
            "stream": False,
            "extra_params": {"num_inference_steps": steps, "guidance_scale": guidance},
        }
        language = kw.get("language")
        if language and language.casefold() != "auto":
            payload["language"] = language
        if kw.get("instruct"):
            payload["instructions"] = kw["instruct"]
        if kw.get("seed") is not None:
            payload["seed"] = kw["seed"]
        reference = kw.get("ref_audio")
        if reference:
            # Transfer bytes; the service must not depend on client-local paths.
            import soundfile as sf

            try:
                audio, rate = sf.read(Path(reference), dtype="float32", always_2d=True)
            except (OSError, ValueError, RuntimeError) as exc:
                raise TTSInputError("Reference audio could not be decoded.") from exc
            reference_wav = io.BytesIO()
            sf.write(reference_wav, audio, rate, format="WAV", subtype="FLOAT")
            payload["ref_audio"] = "data:audio/wav;base64," + base64.b64encode(
                reference_wav.getvalue()
            ).decode("ascii")
            if kw.get("ref_text"):
                payload["ref_text"] = kw["ref_text"]
        try:
            body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode(
                "utf-8"
            )
        except (TypeError, ValueError) as exc:
            raise TTSInputError(
                "Speech parameters must be finite JSON values."
            ) from exc
        try:
            with open_trusted_endpoint(
                self._url,
                method="POST",
                path="v1/audio/speech",
                timeout=self._timeout,
                body=body,
                content_type="application/json",
            ) as response:
                wav = response.read()
        except EndpointHTTPError as exc:
            if exc.status in {400, 422}:
                raise TTSInputError(
                    "vLLM-Omni rejected the speech parameters; check its logs."
                ) from exc
            raise RuntimeError(
                f"vLLM-Omni speech request failed with HTTP {exc.status}."
            ) from exc
        import numpy as np
        import soundfile as sf

        try:
            audio, rate = sf.read(io.BytesIO(wav), dtype="float32", always_2d=True)
        except (ValueError, RuntimeError) as exc:
            raise RuntimeError("vLLM-Omni returned invalid audio.") from exc
        if (
            rate != self.sample_rate
            or audio.shape[1] != 1
            or audio.size == 0
            or not np.isfinite(audio).all()
        ):
            raise RuntimeError(
                "vLLM-Omni must return finite, nonempty 24 kHz mono audio."
            )
        return torch.from_numpy(audio.T.copy())

    def unload(self) -> None:
        # Switching client engines must not stop the independently hosted service.
        pass
