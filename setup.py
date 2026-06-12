"""Compatibility shim.

All project metadata lives in ``pyproject.toml`` (PEP 621). This file exists
only for tooling that still invokes ``setup.py`` directly. Do not duplicate
metadata here.
"""

from setuptools import setup

setup()
