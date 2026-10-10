# OmniVoice through vLLM-Omni

This opt-in engine keeps VoiceStudio's application, saved voices, effects and
synthetic-audio marking, and sends speech inference to an existing vLLM-Omni
OmniVoice server. The client does not load a second copy of the model. The
server can use Ascend NPU, CUDA or another device supported by its deployment.

Start the OmniVoice service separately, using its deployment instructions and
`--served-model-name OmniVoice`. Then start VoiceStudio with:

```bash
export OMNIVOICE_TTS_BACKEND=vllm-omni
export OMNIVOICE_VLLM_OMNI_URL=http://127.0.0.1:18091
export OMNIVOICE_VLLM_OMNI_MODEL=OmniVoice
python -m uvicorn backend.main:app --host 127.0.0.1 --port 3900
```

Use an HTTP(S) origin without `/v1`, credentials, query or fragment. Loopback is
trusted by default. An explicitly configured LAN service also needs its network
in `OMNIVOICE_TRUSTED_NETWORKS`; redirects and untrusted addresses are refused.
The client does not forward an API key. Use a loopback service or a local proxy
configured to authenticate upstream. SSH can forward a loopback service between
machines; a tunnel does not remove the service's authentication requirement.

Speech generation, voice-design tags, reference-audio cloning, language, speed,
step count, guidance and seed are forwarded to `/v1/audio/speech`. Reference
recordings are sent as float32 WAV bytes without another integer quantization,
so the server needs no access to client paths.
VoiceStudio's ordinary OmniVoice model-licence acceptance still applies.

Fixed duration, disabling denoise and custom advanced sampling controls are
unsupported by the current vLLM-Omni OmniVoice pipeline and are refused.
`postprocess_output` controls the native model's postprocessing; this engine
returns the server's audio and uses VoiceStudio's shared application mastering.
Long-reference selection, transcription, native output-postprocessing parity,
and full dubbing/audiobook workflows require separate integration validation.
The engine expects 24 kHz mono WAV responses and reports compute as external;
the client cannot verify the server's accelerator from its own GPU counters.
