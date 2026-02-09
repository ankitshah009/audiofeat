import sys
import types

import pytest
import torch

from audiofeat.standards import (
    available_opensmile_feature_levels,
    available_opensmile_feature_sets,
    extract_opensmile_features,
)


class _FakeRow(dict):
    def to_dict(self):
        return dict(self)


class _FakeILoc:
    def __getitem__(self, idx):
        return _FakeRow({"egemaps_feat_1": 1.23, "egemaps_feat_2": 4.56})


class _FakeFrame:
    shape = (1, 2)
    iloc = _FakeILoc()

    def to_csv(self, *args, **kwargs):
        return "egemaps_feat_1,egemaps_feat_2\n1.23,4.56\n"


class _FakeSmile:
    def __init__(self, *args, **kwargs):
        pass

    def process_file(self, path):
        return _FakeFrame()

    def process_signal(self, signal, sample_rate):
        return _FakeFrame()


def _install_fake_opensmile(monkeypatch):
    fake_module = types.SimpleNamespace(
        FeatureSet=types.SimpleNamespace(eGeMAPSv02="eGeMAPSv02", ComParE_2016="ComParE_2016"),
        FeatureLevel=types.SimpleNamespace(
            Functionals="Functionals",
            LowLevelDescriptors="LowLevelDescriptors",
        ),
        Smile=_FakeSmile,
    )
    monkeypatch.setitem(sys.modules, "opensmile", fake_module)


def test_extract_opensmile_features_from_file(monkeypatch):
    _install_fake_opensmile(monkeypatch)
    out = extract_opensmile_features(
        "dummy.wav",
        feature_set="eGeMAPSv02",
        feature_level="Functionals",
        flatten=True,
    )
    assert isinstance(out, dict)
    assert "egemaps_feat_1" in out


def test_extract_opensmile_features_from_tensor(monkeypatch):
    _install_fake_opensmile(monkeypatch)
    wav = torch.randn(16000)
    out = extract_opensmile_features(
        wav,
        sample_rate=16000,
        feature_set="eGeMAPSv02",
        feature_level="Functionals",
        flatten=True,
    )
    assert isinstance(out, dict)
    assert out["egemaps_feat_2"] == 4.56


def test_opensmile_helpers_and_validation(monkeypatch):
    _install_fake_opensmile(monkeypatch)
    assert "ComParE_2016" in available_opensmile_feature_sets()
    assert "Functionals" in available_opensmile_feature_levels()

    wav = torch.randn(8000)
    with pytest.raises(ValueError):
        extract_opensmile_features(wav, sample_rate=8000, feature_set="DoesNotExist")
    with pytest.raises(ValueError):
        extract_opensmile_features(wav, sample_rate=8000, feature_level="DoesNotExist")
    with pytest.raises(ValueError):
        extract_opensmile_features(wav, feature_set="eGeMAPSv02", feature_level="Functionals")
