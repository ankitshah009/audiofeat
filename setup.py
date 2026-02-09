
from setuptools import setup, find_packages

# Read the contents of README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Get version from _version.py
import sys
sys.path.insert(0, str(this_directory / "audiofeat"))
from _version import __version__

setup(
    name="audiofeat",
    version=__version__,
    author="Ankit Shah",
    author_email="ankit.tronix@gmail.com",
    description="A comprehensive PyTorch-based audio feature extraction library for machine learning, research, and audio analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ankitshah009/audiofeat",
    project_urls={
        "Bug Reports": "https://github.com/ankitshah009/audiofeat/issues",
        "Source": "https://github.com/ankitshah009/audiofeat",
        "Documentation": "https://github.com/ankitshah009/audiofeat#readme",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research", 

        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Multimedia :: Sound/Audio",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="audio feature-extraction signal-processing pytorch machine-learning speech voice spectrogram mfcc spectral-features temporal-features pitch-detection audio-analysis music-information-retrieval mir dsp audio-processing",
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0", 
        "torch>=1.9.0",
        "torchaudio>=0.9.0",
    ],
    extras_require={
        "models": [
            "pyannote.audio>=3.1.1",
            "speechbrain>=1.0.0",
            "openai-whisper>=202311",
            "torchaudio>=2.1.0",
            "huggingface-hub>=0.21.0",
            "silero-vad==0.4.0",
            "asteroid>=0.5.0",
            "panns_inference>=0.1.0",
            "madmom>=0.16.1",
        ],
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "isort",
            "flake8",
            "mypy",
        ],
        "examples": [
            "matplotlib",
            "librosa", 
            "soundfile",
        ],
        "validation": [
            "praat-parselmouth>=0.4.3",
        ],
        "standards": [
            "opensmile>=2.5.0",
        ],
        "full": [
            "matplotlib",
            "librosa",
            "soundfile",
            "praat-parselmouth>=0.4.3",
            "opensmile>=2.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "audiofeat=audiofeat.cli:main",
        ]
    },
    include_package_data=True,
    zip_safe=False,
)
