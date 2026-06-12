
import torch

def vocal_tract_length(F1: float, F2: float = None, c: float = 35000.0):
    """Estimate vocal tract length (VTL) in centimetres.

    The default uses the canonical single-formant uniform-tube model, in which
    the first formant of a tube closed at one end is ``F1 = c / (4 L)``, giving::

        L = c / (4 * F1)

    For ``F1 = 500`` Hz and ``c = 35000`` cm/s (speed of sound at body
    temperature) this yields ~17.2 cm, the textbook adult VTL.

    If ``F2`` is provided, a formant-spacing estimate ``L = c / (2 * (F2 - F1))``
    is averaged in (the spacing between consecutive tube resonances is
    ``c / (2 L)``); the single-formant estimate remains the physically correct
    default and is always used when ``F2`` is ``None``.

    Args:
        F1 (float): First formant frequency in Hz (must be > 0).
        F2 (float, optional): Second formant frequency in Hz.
        c (float): Speed of sound in cm/s (default 35000).

    Returns:
        float: Estimated vocal tract length in centimetres.
    """
    if F1 is None or F1 <= 0:
        raise ValueError("F1 must be > 0 Hz to estimate vocal tract length.")

    l_f1 = c / (4.0 * F1)
    if F2 is None:
        return float(l_f1)

    spacing = F2 - F1
    if spacing <= 0:
        return float(l_f1)
    l_spacing = c / (2.0 * spacing)
    return float(0.5 * (l_f1 + l_spacing))
