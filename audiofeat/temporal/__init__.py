"""Temporal audio features.

Each submodule defines an explicit ``__all__`` listing only its public
feature functions (helpers such as ``frame_signal``/``hann_window`` and
imported modules like ``torch`` are intentionally excluded). Import order
below resolves two historical name collisions deterministically:

* ``temporal_centroid`` resolves to the canonical MPEG-7 whole-signal
  descriptor in :mod:`audiofeat.temporal.centroid` (energy-weighted mean
  time in seconds). The frame-based variant remains available as
  ``temporal_centroid_framewise`` (and, for backward compatibility, as
  ``audiofeat.temporal.rhythm.temporal_centroid``).
* ``beat_track`` resolves to the beat-*times* tracker in
  :mod:`audiofeat.temporal.rhythm_features`. The richer
  ``(tempo, beat_frames)`` tracker is exported as ``beat_track_with_tempo``
  (and remains available as ``audiofeat.temporal.beat.beat_track``).
"""

from .rms import *  # rms, short_time_energy
from .zcr import *  # zero_crossing_count, zero_crossing_rate
from .amplitude import *  # amplitude_modulation_depth
from .energy import *  # energy
from .energy_entropy import *  # entropy_of_energy
from .centroid import *  # temporal_centroid (MPEG-7, canonical)
from .rhythm import *  # breath_group_duration, speech_rate, temporal_centroid_framewise
from .attack import *  # log_attack_time
from .decay import *  # decay_time
from .teager import *  # teager_energy_operator
from .loudness import *  # loudness
from .onset import *  # onset_detect
from .tristimulus import *  # tristimulus
from .beat import *  # beat_track_with_tempo
from .rhythm_features import *  # tempo, beat_track (beat times)

__all__ = [
    # rms.py
    "rms",
    "short_time_energy",
    # zcr.py
    "zero_crossing_count",
    "zero_crossing_rate",
    # amplitude.py
    "amplitude_modulation_depth",
    # energy.py
    "energy",
    # energy_entropy.py
    "entropy_of_energy",
    # centroid.py (canonical MPEG-7 temporal centroid)
    "temporal_centroid",
    # rhythm.py
    "breath_group_duration",
    "speech_rate",
    "temporal_centroid_framewise",
    # attack.py
    "log_attack_time",
    # decay.py
    "decay_time",
    # teager.py
    "teager_energy_operator",
    # loudness.py
    "loudness",
    # onset.py
    "onset_detect",
    # tristimulus.py
    "tristimulus",
    # beat.py (rich tracker)
    "beat_track_with_tempo",
    # rhythm_features.py
    "tempo",
    "beat_track",
]
