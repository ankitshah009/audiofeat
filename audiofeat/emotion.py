"""Placeholder emotion detection from MFCCs.

.. warning::

   :func:`detect_emotion` runs a **randomly initialised, untrained** MLP. Its
   output is therefore meaningless and must not be treated as a real prediction.
   It exists only as a structural demo of the feature → classifier wiring. For
   genuine speech-emotion recognition use
   :func:`audiofeat.emotion_ssl.detect_emotion_ssl`, which loads a pretrained
   transformer model (``pip install audiofeat[emotion]``).

The function emits a :class:`UserWarning` on every call so this limitation can
never silently ship as if it were a real classifier.
"""
from __future__ import annotations

import warnings

import torch
from torch import nn

from audiofeat.spectral import mfcc  # reuse existing MFCC for input features

_PLACEHOLDER_WARNING = (
    "audiofeat.emotion.detect_emotion uses a randomly-initialised, UNTRAINED "
    "MLP: its output is meaningless and must not be used as a real prediction. "
    "Use audiofeat.emotion_ssl.detect_emotion_ssl (pip install audiofeat[emotion]) "
    "for genuine speech-emotion recognition."
)


class EmotionDetector(nn.Module):
    """Toy MLP mapping 40 MFCCs to 7 emotion logits (weights are untrained)."""

    def __init__(self):
        super().__init__()
        # Simple MLP for demo; in practice load a pretrained model instead.
        self.fc1 = nn.Linear(40, 128)  # 40 MFCC coefficients in
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 7)  # neutral, happy, sad, angry, fear, disgust, surprise
        self.relu = nn.ReLU()

    def forward(self, features):
        x = self.relu(self.fc1(features))
        x = self.relu(self.fc2(x))
        return self.fc3(x)  # logits for emotion classes


def detect_emotion(waveform: torch.Tensor, sample_rate: int) -> str:
    """Return an emotion label from MFCCs (PLACEHOLDER — see module docstring).

    .. warning::
       The underlying network is untrained; the returned label is arbitrary.
       A :class:`UserWarning` is emitted on every call.
    """
    warnings.warn(_PLACEHOLDER_WARNING, UserWarning, stacklevel=2)

    # Extract features (e.g. MFCC).
    mfcc_features = mfcc(waveform, sample_rate)  # (n_mfcc, n_frames)
    if mfcc_features.ndim == 2:
        mfcc_features = mfcc_features.mean(dim=1)
    elif mfcc_features.ndim != 1:
        mfcc_features = mfcc_features.reshape(-1)

    if mfcc_features.numel() != 40:
        if mfcc_features.numel() > 40:
            mfcc_features = mfcc_features[:40]
        else:
            mfcc_features = torch.nn.functional.pad(
                mfcc_features,
                (0, 40 - mfcc_features.numel()),
            )

    mfcc_features = mfcc_features.unsqueeze(0)
    model = EmotionDetector()  # untrained: instantiated fresh, no weights loaded
    with torch.no_grad():
        output = model(mfcc_features)
        predicted = torch.argmax(output, dim=1)
        emotions = ["neutral", "happy", "sad", "angry", "fear", "disgust", "surprise"]
        return emotions[predicted.item()]
