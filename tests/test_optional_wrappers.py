import sys
import types
from pathlib import Path

import pytest
import torch
import torchaudio


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


def test_denoise_rnn_with_fake_backend(monkeypatch):
    import audiofeat.denoise as denoise

    class _RNNoise:
        def __call__(self, waveform, sample_rate):
            return waveform * 0.5

    monkeypatch.setitem(sys.modules, "rnnoise_torch", types.SimpleNamespace(RNNoise=_RNNoise))
    x = torch.randn(4000)
    y = denoise.denoise_rnn(x, sample_rate=48000)
    assert y.shape == x.shape


def test_vad_is_speech_with_fake_model(monkeypatch):
    import audiofeat.vad as vad

    class _Model:
        def __call__(self, waveform, sample_rate):
            return torch.tensor(0.8)

    fake_mod = types.SimpleNamespace(get_silero_vad_model=lambda: _Model())
    monkeypatch.setitem(sys.modules, "silero_vad", fake_mod)
    assert vad.is_speech(torch.randn(16000), 16000, threshold=0.5) is True


def test_scene_classify_with_fake_panns(monkeypatch):
    import audiofeat.scene as scene

    class _Tagger:
        def __init__(self, checkpoint_path=None, device="cpu"):
            self.device = device

        def inference(self, waveform, sample_rate):
            return torch.tensor([[0.1, 2.0, 0.5]])

    monkeypatch.setattr(scene, "AudioTagging", _Tagger)
    monkeypatch.setattr(scene, "labels", ["bg", "speech", "music"])
    out = scene.classify_scene(torch.randn(16000), 16000, top_k=2)
    assert len(out) == 2
    assert out[0][0] in {"speech", "music"}


def test_spatial_separate_sources_with_fake_asteroid(monkeypatch):
    import audiofeat.spatial as spatial

    class _Model:
        def separate(self, waveform):
            return torch.stack([waveform.squeeze(0), waveform.squeeze(0) * 0.1], dim=0)

    class _ConvTasNet:
        @staticmethod
        def from_pretrained(_name):
            return _Model()

    fake_models = types.SimpleNamespace(ConvTasNet=_ConvTasNet)
    fake_utils = types.SimpleNamespace(torch_utils=object())
    monkeypatch.setitem(sys.modules, "asteroid.models", fake_models)
    monkeypatch.setitem(sys.modules, "asteroid.utils", fake_utils)

    out = spatial.separate_sources(torch.randn(1, 16000), 16000)
    assert isinstance(out, list)
    assert len(out) == 2


def test_ssl_embed_with_fake_transformers(monkeypatch):
    import audiofeat.ssl as ssl

    class _Processor:
        @staticmethod
        def from_pretrained(_name):
            return _Processor()

        def __call__(self, waveform, sampling_rate, return_tensors="pt"):
            return {"input_values": waveform.unsqueeze(0)}

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


def test_streaming_feature_extractor_push():
    from audiofeat.streaming import StreamingFeatureExtractor

    def _fn(frame, _sr):
        return frame.mean()

    extractor = StreamingFeatureExtractor(_fn, sample_rate=16000, frame_ms=20, hop_ms=10)
    out = extractor.push(torch.randn(1600))
    assert "frames" in out
    assert len(out["frames"]) > 0


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

    class _Pipeline:
        @staticmethod
        def from_pretrained(_id, use_auth_token=None):
            _ = use_auth_token
            return _PipelineObj()

    fake_pyannote_audio = types.SimpleNamespace(Pipeline=_Pipeline)
    monkeypatch.setitem(sys.modules, "pyannote.audio", fake_pyannote_audio)

    out = diarization.diarize("dummy.wav")
    assert out[0][0] <= out[1][0]
    assert out[0][2] == "A"


def test_embeddings_validate_sample_rate_for_tensor():
    import audiofeat.embeddings as embeddings

    with pytest.raises(ValueError):
        embeddings.extract_speaker_embedding(torch.randn(16000), sample_rate=None)


def test_embeddings_with_fake_speechbrain(monkeypatch):
    import audiofeat.embeddings as embeddings

    class _Classifier:
        @staticmethod
        def from_hparams(source):
            _ = source
            return _Classifier()

        def encode_batch(self, waveform):
            _ = waveform
            return torch.ones(1, 192)

    fake_pretrained = types.SimpleNamespace(EncoderClassifier=_Classifier)
    monkeypatch.setitem(sys.modules, "speechbrain.pretrained", fake_pretrained)

    emb = embeddings.extract_speaker_embedding(torch.randn(16000), sample_rate=16000)
    assert emb.shape == (192,)


def test_emotion_ssl_with_fake_transformers(monkeypatch):
    import audiofeat.emotion_ssl as emotion_ssl

    class _Processor:
        @staticmethod
        def from_pretrained(_name):
            return _Processor()

        def __call__(self, waveform, sampling_rate, return_tensors="pt"):
            _ = (waveform, sampling_rate, return_tensors)
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
