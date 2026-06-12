"""Tests for the optional ML-wrapper modules.

The heavy backends (whisper, silero-vad, panns_inference, asteroid, speechbrain,
transformers, pyannote, madmom, rnnoise) are NOT installed in CI. Every test
here injects a *spec-faithful* fake that mirrors the REAL upstream signature, so
the tests double as API-drift guards: if a wrapper reverts to a wrong/outdated
upstream call, the fake will mismatch and the test will fail.
"""
import sys
import types
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
import torchaudio


# --------------------------------------------------------------------------- #
# ASR (Whisper)
# --------------------------------------------------------------------------- #
def test_asr_transcribe_with_fake_whisper(monkeypatch):
    import audiofeat.asr as asr

    class _Model:
        def transcribe(self, audio, language=None, **kwargs):
            return {"text": "hello", "language": language, "audio": audio, "kwargs": kwargs}

    fake_whisper = types.SimpleNamespace(load_model=lambda _: _Model())
    monkeypatch.setitem(sys.modules, "whisper", fake_whisper)

    out = asr.transcribe("dummy.wav", model_size="base", language="en", temperature=0.0)
    assert out["text"] == "hello"
    assert out["language"] == "en"


# --------------------------------------------------------------------------- #
# RNNoise denoise
# --------------------------------------------------------------------------- #
def test_denoise_rnn_with_fake_backend(monkeypatch):
    import audiofeat.denoise as denoise

    class _RNNoise:
        def __call__(self, waveform, sample_rate):
            return waveform * 0.5

    monkeypatch.setitem(sys.modules, "rnnoise_torch", types.SimpleNamespace(RNNoise=_RNNoise))
    x = torch.randn(4000)
    y = denoise.denoise_rnn(x, sample_rate=48000)
    assert y.shape == x.shape


# --------------------------------------------------------------------------- #
# VAD (Silero) — real API is load_silero_vad(), plus a pure-torch energy fallback
# --------------------------------------------------------------------------- #
def test_vad_is_speech_with_fake_silero(monkeypatch):
    import audiofeat.vad as vad

    vad._MODEL = None  # reset module-level cache

    class _Model:
        def __call__(self, waveform, sample_rate):
            return torch.tensor(0.8)

    # REAL Silero >=5 entry point is ``load_silero_vad`` (NOT get_silero_vad_model).
    fake_mod = types.SimpleNamespace(load_silero_vad=lambda: _Model())
    monkeypatch.setitem(sys.modules, "silero_vad", fake_mod)

    assert vad.is_speech(torch.randn(16000), 16000, threshold=0.5) is True
    # second call must reuse the cached model
    assert vad._MODEL is not None


def test_vad_model_is_cached(monkeypatch):
    import audiofeat.vad as vad

    vad._MODEL = None
    calls = {"n": 0}

    class _Model:
        def __call__(self, waveform, sample_rate):
            return torch.tensor(0.9)

    def _loader():
        calls["n"] += 1
        return _Model()

    fake_mod = types.SimpleNamespace(load_silero_vad=_loader)
    monkeypatch.setitem(sys.modules, "silero_vad", fake_mod)

    vad.is_speech(torch.randn(16000), 16000)
    vad.is_speech(torch.randn(16000), 16000)
    assert calls["n"] == 1  # loaded exactly once


def test_vad_energy_fallback_distinguishes_tone_vs_silence():
    import audiofeat.vad as vad

    sr = 16000
    t = torch.arange(sr, dtype=torch.float32) / sr
    tone = 0.5 * torch.sin(2 * torch.pi * 220.0 * t)
    silence = torch.zeros(sr)

    # No silero installed/required: pure-torch energy fallback.
    assert vad.is_speech_energy(tone, sr, threshold=0.01) is True
    assert vad.is_speech_energy(silence, sr, threshold=0.01) is False

    # is_speech(use_silero=False) routes to the same fallback.
    assert vad.is_speech(tone, sr, use_silero=False, energy_threshold=0.01) is True
    assert vad.is_speech(silence, sr, use_silero=False, energy_threshold=0.01) is False


def test_vad_gracefully_degrades_without_silero(monkeypatch):
    import audiofeat.vad as vad

    vad._MODEL = None
    # Ensure silero import fails.
    monkeypatch.setitem(sys.modules, "silero_vad", None)

    sr = 16000
    t = torch.arange(sr, dtype=torch.float32) / sr
    tone = 0.5 * torch.sin(2 * torch.pi * 220.0 * t)
    # use_silero=True but silero missing -> falls back to energy, no exception.
    assert vad.is_speech(tone, sr, energy_threshold=0.01) is True
    assert vad.is_speech(torch.zeros(sr), sr, energy_threshold=0.01) is False


# --------------------------------------------------------------------------- #
# Scene (PANNs) — inference(audio) returns (clipwise (1,527), emb) NumPy tuple
# --------------------------------------------------------------------------- #
def test_scene_classify_with_fake_panns(monkeypatch):
    import audiofeat.scene as scene

    scene._TAGGER = None  # reset cache

    n_classes = 527

    class _Tagger:
        def __init__(self, checkpoint_path=None, device="cpu"):
            self.device = device

        def inference(self, audio):
            # REAL signature: a single (batch, samples) NumPy array, NO sample_rate.
            assert isinstance(audio, np.ndarray), "PANNs expects a NumPy array"
            assert audio.ndim == 2, "PANNs expects (batch, samples)"
            clipwise = np.zeros((1, n_classes), dtype=np.float32)
            clipwise[0, 1] = 0.9  # 'speech'
            clipwise[0, 2] = 0.5  # 'music'
            embedding = np.zeros((1, 2048), dtype=np.float32)
            return clipwise, embedding  # already-sigmoid'd tuple

    labels = ["bg", "speech", "music"] + [f"c{i}" for i in range(n_classes - 3)]
    fake_panns = types.SimpleNamespace(AudioTagging=_Tagger, labels=labels)
    monkeypatch.setitem(sys.modules, "panns_inference", fake_panns)

    out = scene.classify_scene(torch.randn(16000), 16000, top_k=2)
    assert len(out) == 2
    assert out[0][0] == "speech"
    assert out[1][0] == "music"
    # probabilities must be the raw sigmoid values, NOT re-softmaxed.
    assert out[0][1] == pytest.approx(0.9, abs=1e-5)


# --------------------------------------------------------------------------- #
# Spatial / source separation (asteroid ConvTasNet) — forward, not .separate()
# --------------------------------------------------------------------------- #
def test_spatial_separate_sources_with_fake_asteroid(monkeypatch):
    import audiofeat.spatial as spatial

    class _Model:
        def __call__(self, waveform):
            # forward returns (batch, n_src, samples).
            mono = waveform.reshape(-1)
            return torch.stack([mono, mono * 0.1], dim=0).unsqueeze(0)

        def separate(self, waveform):  # present but must NOT be used
            raise AssertionError("separate() writes files; wrapper must use forward")

    class _ConvTasNet:
        @staticmethod
        def from_pretrained(_name):
            return _Model()

    fake_models = types.SimpleNamespace(ConvTasNet=_ConvTasNet)
    monkeypatch.setitem(sys.modules, "asteroid.models", fake_models)

    out = spatial.separate_sources(torch.randn(1, 16000), 16000)
    assert isinstance(out, list)
    assert len(out) == 2
    assert out[0].ndim == 1


# --------------------------------------------------------------------------- #
# SSL embeddings (transformers) — processor must receive a NumPy array
# --------------------------------------------------------------------------- #
def test_ssl_embed_with_fake_transformers(monkeypatch):
    import audiofeat.ssl as ssl

    ssl._load_transformer.cache_clear()

    class _Processor:
        @staticmethod
        def from_pretrained(_name):
            return _Processor()

        def __call__(self, waveform, sampling_rate, return_tensors="pt"):
            # HF feature extractors raise on a torch.Tensor; proves conversion.
            if isinstance(waveform, torch.Tensor):
                raise TypeError("HF processor expects a NumPy array, got torch.Tensor")
            assert isinstance(waveform, np.ndarray)
            return {"input_values": torch.from_numpy(np.asarray(waveform)).unsqueeze(0)}

    class _Model:
        @staticmethod
        def from_pretrained(_name):
            return _Model()

        def eval(self):
            return self

        def __call__(self, **inputs):
            _ = inputs
            return types.SimpleNamespace(last_hidden_state=torch.ones(1, 8, 4))

    fake_transformers = types.SimpleNamespace(AutoProcessor=_Processor, AutoModel=_Model)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    emb = ssl.embed(torch.randn(16000), 16000, backend="wav2vec2_base")
    assert emb.shape == (4,)
    ssl._load_transformer.cache_clear()


# --------------------------------------------------------------------------- #
# Streaming
# --------------------------------------------------------------------------- #
def test_streaming_feature_extractor_push():
    from audiofeat.streaming import StreamingFeatureExtractor

    def _fn(frame, _sr):
        return frame.mean()

    extractor = StreamingFeatureExtractor(_fn, sample_rate=16000, frame_ms=20, hop_ms=10)
    out = extractor.push(torch.randn(1600))
    assert "frames" in out
    assert len(out["frames"]) > 0


def test_streaming_flush_emits_trailing_frame():
    from audiofeat.streaming import StreamingFeatureExtractor

    def _fn(frame, _sr):
        return torch.tensor(float(frame.numel()))

    extractor = StreamingFeatureExtractor(_fn, sample_rate=1000, frame_ms=20, hop_ms=10)
    # frame=20 samples, hop=10 samples. Push 25 -> one full frame, 15 residual.
    out = extractor.push(torch.arange(25, dtype=torch.float32))
    assert len(out["frames"]) == 1
    flushed = extractor.flush()
    assert "frames" in flushed
    assert len(flushed["frames"]) == 1  # the trailing partial frame
    assert flushed["frames"][0].item() == 15.0
    # buffer is cleared; a second flush yields nothing.
    assert extractor.flush() == {}


def test_streaming_reset_and_lazy_dtype_device():
    from audiofeat.streaming import StreamingFeatureExtractor

    def _fn(frame, _sr):
        return frame.dtype

    extractor = StreamingFeatureExtractor(_fn, sample_rate=1000, frame_ms=20, hop_ms=10)
    chunk = torch.arange(40, dtype=torch.float64)
    out = extractor.push(chunk)
    # buffer inherited the chunk dtype, so frames are float64.
    assert out["frames"][0] == torch.float64
    extractor.reset()
    assert extractor.buffer is None


# --------------------------------------------------------------------------- #
# Diarization (pyannote) — auth_token + 3.x checkpoint
# --------------------------------------------------------------------------- #
def test_diarization_with_fake_pipeline(monkeypatch):
    import audiofeat.diarization as diarization

    class _Segment:
        def __init__(self, start, end):
            self.start = start
            self.end = end

    class _Annotation:
        def itertracks(self, yield_label=True):
            _ = yield_label
            return iter(
                [
                    (_Segment(1.0, 2.0), None, "B"),
                    (_Segment(0.0, 0.8), None, "A"),
                ]
            )

    class _PipelineObj:
        def __call__(self, path, num_speakers=None):
            _ = (path, num_speakers)
            return _Annotation()

    seen = {}

    class _Pipeline:
        @staticmethod
        def from_pretrained(_id, use_auth_token=None):
            seen["id"] = _id
            seen["token"] = use_auth_token
            return _PipelineObj()

    fake_pyannote_audio = types.SimpleNamespace(Pipeline=_Pipeline)
    monkeypatch.setitem(sys.modules, "pyannote.audio", fake_pyannote_audio)

    out = diarization.diarize("dummy.wav", auth_token="hf_test")
    assert out[0][0] <= out[1][0]
    assert out[0][2] == "A"
    # uses pyannote 3.x checkpoint and forwards the auth token.
    assert seen["id"] == "pyannote/speaker-diarization-3.1"
    assert seen["token"] == "hf_test"


def test_diarization_reads_hf_token_env(monkeypatch):
    import audiofeat.diarization as diarization

    class _Annotation:
        def itertracks(self, yield_label=True):
            return iter([])

    class _PipelineObj:
        def __call__(self, path, num_speakers=None):
            return _Annotation()

    seen = {}

    class _Pipeline:
        @staticmethod
        def from_pretrained(_id, use_auth_token=None):
            seen["token"] = use_auth_token
            return _PipelineObj()

    monkeypatch.setitem(sys.modules, "pyannote.audio", types.SimpleNamespace(Pipeline=_Pipeline))
    monkeypatch.setenv("HF_TOKEN", "hf_from_env")

    diarization.diarize("dummy.wav")
    assert seen["token"] == "hf_from_env"


def test_diarization_gated_model_error(monkeypatch):
    import audiofeat.diarization as diarization

    class _Pipeline:
        @staticmethod
        def from_pretrained(_id, use_auth_token=None):
            return None  # pyannote returns None for gated model w/o valid token

    monkeypatch.setitem(sys.modules, "pyannote.audio", types.SimpleNamespace(Pipeline=_Pipeline))
    monkeypatch.delenv("HF_TOKEN", raising=False)

    with pytest.raises(RuntimeError, match="gated"):
        diarization.diarize("dummy.wav")


# --------------------------------------------------------------------------- #
# Speaker embeddings (speechbrain) — speechbrain.inference, explicit savedir
# --------------------------------------------------------------------------- #
def test_embeddings_validate_sample_rate_for_tensor():
    import audiofeat.embeddings as embeddings

    embeddings._CLASSIFIER = None
    with pytest.raises(ValueError):
        embeddings.extract_speaker_embedding(torch.randn(16000), sample_rate=None)


def test_embeddings_with_fake_speechbrain_inference(monkeypatch):
    import audiofeat.embeddings as embeddings

    embeddings._CLASSIFIER = None
    seen = {}

    class _Classifier:
        @staticmethod
        def from_hparams(source, savedir=None):
            seen["source"] = source
            seen["savedir"] = savedir
            return _Classifier()

        def encode_batch(self, waveform):
            _ = waveform
            return torch.ones(1, 1, 192)  # real ECAPA output shape

    # speechbrain>=1.0: EncoderClassifier lives in speechbrain.inference.
    fake_inference = types.SimpleNamespace(EncoderClassifier=_Classifier)
    monkeypatch.setitem(sys.modules, "speechbrain.inference", fake_inference)

    emb = embeddings.extract_speaker_embedding(
        torch.randn(16000), sample_rate=16000, savedir="/tmp/sb_test"
    )
    assert emb.shape == (192,)  # reshape(-1) flattens (1,1,192) -> (192,)
    assert seen["source"] == "speechbrain/spkrec-ecapa-voxceleb"
    assert seen["savedir"] == "/tmp/sb_test"


def test_embeddings_falls_back_to_pretrained(monkeypatch):
    import audiofeat.embeddings as embeddings

    embeddings._CLASSIFIER = None

    class _Classifier:
        @staticmethod
        def from_hparams(source, savedir=None):
            return _Classifier()

        def encode_batch(self, waveform):
            return torch.ones(1, 1, 192)

    # speechbrain.inference missing -> wrapper must fall back to .pretrained.
    monkeypatch.setitem(sys.modules, "speechbrain.inference", None)
    monkeypatch.setitem(
        sys.modules, "speechbrain.pretrained", types.SimpleNamespace(EncoderClassifier=_Classifier)
    )

    emb = embeddings.extract_speaker_embedding(torch.randn(16000), sample_rate=16000)
    assert emb.shape == (192,)


# --------------------------------------------------------------------------- #
# Emotion (SSL) — model param, NumPy conversion
# --------------------------------------------------------------------------- #
def test_emotion_ssl_with_fake_transformers(monkeypatch):
    import audiofeat.emotion_ssl as emotion_ssl

    emotion_ssl._load.cache_clear()

    class _Processor:
        @staticmethod
        def from_pretrained(_name):
            return _Processor()

        def __call__(self, waveform, sampling_rate, return_tensors="pt"):
            if isinstance(waveform, torch.Tensor):
                raise TypeError("HF processor expects a NumPy array, got torch.Tensor")
            assert isinstance(waveform, np.ndarray)
            return {"input_values": torch.randn(1, 50)}

    class _Model:
        config = types.SimpleNamespace(id2label={0: "neutral", 1: "happy"})

        @staticmethod
        def from_pretrained(_name):
            return _Model()

        def eval(self):
            return self

        def __call__(self, **inputs):
            _ = inputs
            return types.SimpleNamespace(logits=torch.tensor([[0.1, 1.0]]))

    fake_tf = types.SimpleNamespace(
        AutoModelForAudioClassification=_Model,
        AutoProcessor=_Processor,
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_tf)

    label = emotion_ssl.detect_emotion_ssl(torch.randn(3200), 16000)
    assert label == "happy"
    emotion_ssl._load.cache_clear()


def test_emotion_ssl_accepts_model_param(monkeypatch):
    import audiofeat.emotion_ssl as emotion_ssl

    emotion_ssl._load.cache_clear()
    seen = {}

    class _Processor:
        @staticmethod
        def from_pretrained(name):
            seen.setdefault("ids", []).append(name)
            return _Processor()

        def __call__(self, waveform, sampling_rate, return_tensors="pt"):
            return {"input_values": torch.randn(1, 50)}

    class _Model:
        config = types.SimpleNamespace(id2label={0: "neutral"})

        @staticmethod
        def from_pretrained(name):
            return _Model()

        def eval(self):
            return self

        def __call__(self, **inputs):
            return types.SimpleNamespace(logits=torch.tensor([[1.0]]))

    fake_tf = types.SimpleNamespace(
        AutoModelForAudioClassification=_Model, AutoProcessor=_Processor
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_tf)

    emotion_ssl.detect_emotion_ssl(torch.randn(3200), 16000, model="my/custom-ser")
    assert "my/custom-ser" in seen["ids"]
    emotion_ssl._load.cache_clear()


# --------------------------------------------------------------------------- #
# Placeholder emotion MLP — must emit a UserWarning
# --------------------------------------------------------------------------- #
def test_emotion_placeholder_emits_userwarning():
    from audiofeat.emotion import detect_emotion

    sr = 22050
    t = torch.arange(sr, dtype=torch.float32) / sr
    x = torch.sin(2 * torch.pi * 220.0 * t)

    with pytest.warns(UserWarning, match="UNTRAINED"):
        label = detect_emotion(x, sample_rate=sr)
    assert isinstance(label, str)


# --------------------------------------------------------------------------- #
# Beat tracking (madmom)
# --------------------------------------------------------------------------- #
def test_beat_madmom_with_fake_processors(monkeypatch):
    import audiofeat.beat_madmom as beat_madmom

    class _RNNBeat:
        def __call__(self, path):
            _ = path
            return torch.tensor([0.1, 0.3])

    class _DBNBeat:
        def __init__(self, fps=100):
            self.fps = fps

        def __call__(self, act):
            _ = act
            return torch.tensor([0.5, 1.0])

    class _RNNDown:
        def __call__(self, path):
            _ = path
            return torch.tensor([[0.1, 0.9]])

    class _DBNDown:
        def __init__(self, beats_per_bar=None, fps=100):
            self.beats_per_bar = beats_per_bar
            self.fps = fps

        def __call__(self, act):
            _ = act
            return [(0.5, 1), (1.0, 2)]

    monkeypatch.setattr(beat_madmom, "RNNBeatProcessor", _RNNBeat)
    monkeypatch.setattr(beat_madmom, "DBNBeatTrackingProcessor", _DBNBeat)
    monkeypatch.setattr(beat_madmom, "RNNDownBeatProcessor", _RNNDown)
    monkeypatch.setattr(beat_madmom, "DBNDownBeatTrackingProcessor", _DBNDown)

    beats = beat_madmom.beat_track("x.wav")
    downbeats = beat_madmom.downbeat_track("x.wav")
    assert len(beats) == 2
    assert len(downbeats) == 2


def test_beat_madmom_missing_raises(monkeypatch):
    import audiofeat.beat_madmom as beat_madmom

    monkeypatch.setattr(beat_madmom, "RNNBeatProcessor", None)
    monkeypatch.setattr(beat_madmom, "RNNDownBeatProcessor", None)
    with pytest.raises(ModuleNotFoundError, match="audiofeat\\[beat\\]"):
        beat_madmom.beat_track("x.wav")
    with pytest.raises(ModuleNotFoundError, match="audiofeat\\[beat\\]"):
        beat_madmom.downbeat_track("x.wav")


# --------------------------------------------------------------------------- #
# Chord recognition — 24 templates; a real C-major triad must detect "C"
# --------------------------------------------------------------------------- #
def test_chord_has_all_24_templates():
    import audiofeat.chord as chord

    assert len(chord._CHORD_LABELS) == 24
    assert len(chord._CHORD_TEMPLATES) == 24
    # major and minor for every root present
    for root in ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]:
        assert root in chord._CHORD_LABELS
        assert f"{root}m" in chord._CHORD_LABELS


def test_chord_detect_c_major_triad(tmp_path: Path):
    import audiofeat.chord as chord

    sr = 22050
    dur = 2.0
    t = torch.arange(int(sr * dur), dtype=torch.float32) / sr
    # C major triad: C4 (261.63), E4 (329.63), G4 (392.00) Hz.
    x = (
        torch.sin(2 * torch.pi * 261.63 * t)
        + torch.sin(2 * torch.pi * 329.63 * t)
        + torch.sin(2 * torch.pi * 392.00 * t)
    )
    x = (x / x.abs().max()).unsqueeze(0)
    path = tmp_path / "cmaj.wav"
    torchaudio.save(str(path), x, sr)

    out = chord.detect_chords(str(path), hop_length=2048)
    assert len(out) > 0
    assert isinstance(out[0][1], str)
    # Majority-vote the per-frame labels; a clean C-major triad must read "C".
    from collections import Counter

    most_common = Counter(label for _, label in out).most_common(1)[0][0]
    assert most_common == "C"


def test_chord_detect_runs_on_simple_tone(tmp_path: Path):
    import audiofeat.chord as chord

    sr = 22050
    t = torch.arange(sr, dtype=torch.float32) / sr
    x = torch.sin(2 * torch.pi * 440.0 * t).unsqueeze(0)
    path = tmp_path / "tone.wav"
    torchaudio.save(str(path), x, sr)

    out = chord.detect_chords(str(path), hop_length=2048)
    assert len(out) > 0
    assert isinstance(out[0][1], str)


# --------------------------------------------------------------------------- #
# Safety property: importing audiofeat (and the wrappers) must NOT import the
# heavy optional deps.
# --------------------------------------------------------------------------- #
def test_importing_wrappers_does_not_load_heavy_deps():
    import importlib

    import audiofeat  # noqa: F401
    import audiofeat.vad  # noqa: F401
    import audiofeat.scene  # noqa: F401
    import audiofeat.chord  # noqa: F401

    importlib.import_module("audiofeat._optional")

    for heavy in (
        "whisper",
        "panns_inference",
        "asteroid",
        "speechbrain",
        "madmom",
        "pyannote.audio",
    ):
        mod = sys.modules.get(heavy)
        # If a prior test injected a fake (a real module/namespace), that's the
        # monkeypatched stand-in, not the genuine heavy import; None is also OK.
        assert mod is None or isinstance(mod, (types.SimpleNamespace, type(None)))


def test_optional_require_helper_message():
    from audiofeat._optional import require

    with pytest.raises(ModuleNotFoundError, match=r"audiofeat\[vad\]"):
        require("a_module_that_truly_does_not_exist_xyz", extra="vad")
